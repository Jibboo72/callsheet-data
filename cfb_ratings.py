"""
College Football Power Ratings Engine
=====================================
Opponent-adjusted OFFENSE and DEFENSE ratings for every FBS team, expressed
in POINTS, so that

    projected home margin = (home Net) - (away Net) + home-field advantage
    projected total       = 2 * league avg + home Off + away Def + away Off + home Def

where Net = Off - Def, Off = points scored above an average FBS team vs an
average defense, and Def = points ALLOWED above average (negative = good).

METHOD
------
1. Every completed game with at least one FBS team becomes two rows:
       home points ~ avg + Off[home] + Def[away] + HFA/2
       away points ~ avg + Off[away] + Def[home] - HFA/2
2. Blowouts are tamed before fitting. Margin beyond `margin_cap` only counts
   at `tail_keep` strength (total points kept), so beating an FCS team by 63
   isn't worth twice beating them by 31. This is the "not all production is
   equal" part -- garbage-time points stop inflating ratings.
3. A WEIGHTED RIDGE REGRESSION fits all teams at once. That is the opponent
   adjustment: every team's numbers are measured net of everyone it played,
   and everyone THEY played, simultaneously.
4. The ridge pulls each team toward a PRIOR, not toward zero: last season's
   final rating, regressed by `carryover`. In Week 2 the prior dominates; by
   Week 8 the season's own games have taken over.
5. Recent games count more (exponential decay, `half_life_weeks`).
6. FCS opponents get their own ratings, shrunk toward an FCS group level
   that's fit freely -- so beating North Dakota State counts for more than
   beating a winless FCS team.
7. Win probability = Normal CDF(spread / sd_margin). sd_margin comes from the
   backtest residuals, not a guess.

Every tunable knob is in CONFIG and is set by backtest_cfb.py.

    python3 cfb_ratings.py 2026            # ratings through the latest week
"""

import math
import os
import sys

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main"
SCHED_URL = BASE + "/schedules/parquet/cfb_schedules_{season}.parquet"
LINES_URL = BASE + "/betting/parquet/cfb_line_odds.parquet"
CACHE = os.environ.get("CFB_CACHE", "cache")

CONFIG = dict(
    # Tuned by backtest_cfb.py (coordinate descent on 2021-2024, scored on
    # MAE vs actual margin; 2025 held out). Held-out 2025: model MAE 12.16
    # vs closing line 11.82.
    margin_cap=21.0,        # margin beyond this...
    tail_keep=0.75,         # ...only counts at 75% strength
    half_life_weeks=40.0,   # recency barely matters inside one season
    alpha=2.0,              # prior strength, in "games" of evidence per team
    alpha_fcs=0.5,          # FCS teams have few games vs FBS -> lean on group
    carryover=1.0,          # last year's rating carried into this year's prior
    carry2=0.3,             # share of the prior taken from TWO seasons back
                            # (program level -- recruiting, coaching, depth)
    hfa=2.5,                # home-field advantage, points
    sd_margin=15.8,         # game-margin SD for win % (backtest residual sd)
)

# The FCS model: same engine over every Division I game (FBS and FCS), so
# FCS teams are rated off their whole schedule, not just their FBS games.
# Knobs tuned separately by backtest_fcs.py against FCS-vs-FCS closing lines.
# Tuned on 2022-2024 FCS-vs-FCS games; held-out 2025: MAE 12.15 vs line 11.86.
FCS_CONFIG = dict(CONFIG, scope="di", alpha=1.0, alpha_fcs=2.0, hfa=2.0)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _read(url, name):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = pd.read_parquet(url)
    try:
        os.makedirs(CACHE, exist_ok=True)
        df.to_parquet(path)
    except Exception:
        pass
    return df


EXTRA_GAMES = {}   # season -> DataFrame of games from another source (CFBD), merged in


def load_schedule(season, scope="fbs"):
    """Games with a sortable week index (postseason = 16+).

    scope="fbs": every game with at least one FBS team (the FBS model).
    scope="di":  every game between two Division I teams, FBS or FCS (the
                 FCS model). Games against Division II/III teams are dropped.
    """
    df = _read(SCHED_URL.format(season=season), f"cfb_schedules_{season}.parquet")
    if season in EXTRA_GAMES and len(EXTRA_GAMES[season]):
        x = EXTRA_GAMES[season]
        df = pd.concat([df, x[~x.game_id.isin(df.game_id)]], ignore_index=True)
        # a CFBD row has the score even when the free file hasn't caught up
        sc = x.set_index("game_id")[["home_points", "away_points"]]
        miss = df.home_points.isna() & df.game_id.isin(sc.index)
        df.loc[miss, ["home_points", "away_points"]] = sc.loc[df.loc[miss, "game_id"]].values
    df = df[df.season_type.isin(["regular", "postseason"])].copy()
    if scope == "di":
        di = {"fbs", "fcs"}
        df = df[df.home_division.isin(di) & df.away_division.isin(di)]
    else:
        df = df[(df.home_division == "fbs") | (df.away_division == "fbs")]
    df["wk"] = np.where(df.season_type == "postseason", 16 + df.week, df.week).astype(int)
    df["neutral"] = df.neutral_site.fillna(False).astype(bool)
    df["conf_game"] = df.conference_game.fillna(False).astype(bool)
    df["played"] = df.home_points.notna() & df.away_points.notna()
    keep = ["game_id", "season", "wk", "week", "season_type", "start_date",
            "home_team", "away_team", "home_division", "away_division",
            "home_conference", "away_conference", "home_points", "away_points",
            "neutral", "conf_game", "played", "notes"]
    return df[keep].sort_values(["wk", "start_date"]).reset_index(drop=True)


def load_lines(season=None):
    """Consensus CLOSING home spread + total per game (median across books)."""
    b = _read(LINES_URL, "cfb_line_odds.parquet")
    if season is not None:
        b = b[b.season == season]
    out = {}
    for mkt, col in (("spread", "line_spread"), ("total", "line_total")):
        x = b[b.market_type == mkt][["game_id", "abbr", "lines", "opening_lines", "book",
                                     "season"]].dropna(subset=["lines"])
        out[mkt] = x
    sp = out["spread"]
    if not len(sp):
        return pd.DataFrame(columns=["game_id", "line_spread", "open_spread", "line_total"])
    sched = pd.concat([load_schedule(int(s), "di")[["game_id", "home_team"]]
                       for s in sorted(sp.season.unique())])
    sp = sp.merge(sched, on="game_id")
    home = sp.abbr == sp.home_team
    sp = sp.assign(hs=np.where(home, sp.lines, -sp.lines),
                   ho=np.where(home, sp.opening_lines, -sp.opening_lines))
    g = sp.groupby("game_id").agg(line_spread=("hs", "median"), open_spread=("ho", "median"))
    tot = out["total"].groupby("game_id").lines.median().rename("line_total")
    g = g.join(tot, how="left").reset_index()
    g["game_id"] = g.game_id.astype("int64")
    return g


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def _tamed_scores(hp, ap, cfg):
    m = hp - ap
    t = hp + ap
    cap = cfg["margin_cap"]
    am = np.abs(m)
    m2 = np.where(am > cap, np.sign(m) * (cap + (am - cap) * cfg["tail_keep"]), m)
    return (t + m2) / 2.0, (t - m2) / 2.0


def fit(games, cfg=None, prior=None, as_of_wk=None):
    """
    Fit ratings on the PLAYED games in `games` (one season), using games with
    wk < as_of_wk (all played games if None).

    prior: dict team -> (off, def) from last season, or None.
    Returns dict with 'teams' DataFrame (team, div, off, def, net), 'avg', cfg.
    """
    cfg = {**CONFIG, **(cfg or {})}
    g = games[games.played]
    if as_of_wk is not None:
        g = g[g.wk < as_of_wk]
    now = (as_of_wk if as_of_wk is not None else (g.wk.max() + 1 if len(g) else 1))

    # every team in the season's schedule gets a rating, played or not
    div = {}
    for _, r in games.iterrows():
        div[r.home_team] = r.home_division
        div[r.away_team] = r.away_division
    teams = sorted(div)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    # columns: [0..n) off, [n..2n) def, 2n avg, 2n+1 fcs_off, 2n+2 fcs_def
    P = 2 * n + 3
    is_fcs = np.array([div[t] != "fbs" for t in teams])

    rows, y, w = [], [], []
    if len(g):
        hp, ap = _tamed_scores(g.home_points.values.astype(float),
                               g.away_points.values.astype(float), cfg)
        lam = math.log(2) / cfg["half_life_weeks"]
        wt = np.exp(-lam * (now - g.wk.values))
        hfa = np.where(g.neutral.values, 0.0, cfg["hfa"])
        for k, (h, a) in enumerate(zip(g.home_team.values, g.away_team.values)):
            ih, ia = idx[h], idx[a]
            for off, dfn, pts, sgn in ((ih, ia, hp[k], 1), (ia, ih, ap[k], -1)):
                rows.append((off, dfn))
                y.append(pts - sgn * hfa[k] / 2.0)
                w.append(wt[k])

    m = len(rows)
    X = np.zeros((m, P))
    for r, (o, d) in enumerate(rows):
        X[r, o] = 1.0
        X[r, n + d] = 1.0
        X[r, 2 * n] = 1.0
        if is_fcs[o]:
            X[r, 2 * n + 1] = 1.0
        if is_fcs[d]:
            X[r, 2 * n + 2] = 1.0
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    # prior means and penalties
    b0 = np.zeros(P)
    pen = np.zeros(P)
    for t, i in idx.items():
        a = cfg["alpha_fcs"] if is_fcs[i] else cfg["alpha"]
        pen[i] = pen[n + i] = a
        if prior and t in prior:
            po, pd_ = prior[t]
            b0[i] = cfg["carryover"] * po
            b0[n + i] = cfg["carryover"] * pd_
    # intercept / FCS group: essentially free, but a whisper of ridge keeps
    # the system solvable in Week 1 when there's no data
    b0[2 * n] = 28.0
    pen[2 * n] = 1e-3 if m else 1.0
    b0[2 * n + 1], b0[2 * n + 2] = -14.0, 14.0
    pen[2 * n + 1] = pen[2 * n + 2] = 0.5

    A = X.T @ (X * w[:, None]) + np.diag(pen)
    rhs = X.T @ (w * y) + pen * b0
    beta = np.linalg.solve(A, rhs)

    # what carries into next season's prior: FBS teams their rating, FCS
    # teams their distance from the FCS group (the group level is refit)
    pri_o = beta[:n].copy()
    pri_d = beta[n:2 * n].copy()
    off = beta[:n] + np.where(is_fcs, beta[2 * n + 1], 0.0)
    dfn = beta[n:2 * n] + np.where(is_fcs, beta[2 * n + 2], 0.0)
    # center on the FBS average so 0 = average FBS team; the intercept
    # absorbs the shift so predictions are unchanged
    fbs = ~is_fcs
    oc, dc = off[fbs].mean(), dfn[fbs].mean()
    off, dfn = off - oc, dfn - dc
    pri_o = np.where(is_fcs, pri_o, off)
    pri_d = np.where(is_fcs, pri_d, dfn)
    tdf = pd.DataFrame({"team": teams, "div": [div[t] for t in teams],
                        "off": off, "def": dfn, "net": off - dfn,
                        "pri_o": pri_o, "pri_d": pri_d})
    return {"teams": tdf, "avg": float(beta[2 * n] + oc + dc), "cfg": cfg,
            "as_of_wk": int(now), "n_games": int(len(g))}


def to_prior(model, older=None, cfg=None):
    """Final ratings -> next season's prior. `older` is the prior that went
    INTO this season (i.e. built from the season before); blending it in
    with carry2 gives a two-year program level, which steadies Weeks 1-3."""
    cfg = {**CONFIG, **(cfg or model.get("cfg", {}))}
    t = model["teams"]
    if cfg.get("scope", "fbs") != "di":
        t = t[t["div"] == "fbs"]
    w2 = cfg.get("carry2", 0.0)
    out = {}
    for _, r in t.iterrows():
        o, d = r.pri_o, r.pri_d
        if older and w2 and r.team in older:
            # both are raw ratings; carryover is applied later, in fit()
            po, pd_ = older[r.team]
            o = (1 - w2) * o + w2 * po
            d = (1 - w2) * d + w2 * pd_
        out[r.team] = (o, d)
    return out


def project(model, home, away, neutral=False):
    t = model["teams"].set_index("team")
    cfg = model["cfg"]
    h, a = t.loc[home], t.loc[away]
    hfa = 0.0 if neutral else cfg["hfa"]
    spread = h.net - a.net + hfa            # + = home favored by this many
    total = 2 * model["avg"] + h.off + a["def"] + a.off + h["def"]
    wp = 0.5 * (1 + math.erf(spread / (cfg["sd_margin"] * math.sqrt(2))))
    return {"margin": float(spread), "total": float(total), "home_wp": float(wp)}


def season_prior(season, cfg=None, first=2018):
    """Walk the chain of seasons from `first` to season-1, final ratings each."""
    cfg = {**CONFIG, **(cfg or {})}
    scope = cfg.get("scope", "fbs")
    prior = None
    for s in range(first, season):
        prior = to_prior(fit(load_schedule(s, scope), cfg, prior), prior, cfg)
    return prior


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    prior = season_prior(season)
    m = fit(load_schedule(season), prior=prior)
    t = m["teams"][m["teams"]["div"] == "fbs"].sort_values("net", ascending=False)
    print(f"{season}: {m['n_games']} games, avg {m['avg']:.1f} pts/team")
    print(t.head(25).to_string(index=False, float_format=lambda v: f"{v:6.1f}"))
