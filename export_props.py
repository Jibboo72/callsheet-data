#!/usr/bin/env python3
"""
Player-prop models for the Call Sheet's Props tab.

    python3 export_props.py 2026 data/props-2026.json

WHAT IT MAKES
  For every skill player on a team playing in the next unplayed week, a
  full probability distribution for five markets:

      atd     anytime touchdown (rushing or receiving)      one number, P(>=1)
      ptd     passing touchdowns (QBs)                       P(X = 0,1,2,...)
      car     rushing attempts                               P(X = 0,1,2,...)
      lrush   longest rush                                   P(X >= 1,2,3,...)
      combo   rushing + receiving yards                      P(X >= x) on a grid

  The app turns a distribution into the probability of ANY line the book
  posts, so nothing here is tied to one number.

HOW EACH ONE IS BUILT (kept simple on purpose -- every number has to be
explainable, and none of this has been tested against historical prop prices
because those do not exist for us; the app tracks its own picks going forward)
  * Player form is an exponentially weighted average of recent games (half
    life ~3 games, carried across the season boundary), shrunk toward the
    position average so a two-game sample can't masquerade as a role.
  * Only games the player was ACTIVE for (at least one touch) are used. Books
    void a prop when a player is inactive, so "chance given he plays" is the
    right thing to model.
  * Game script comes from the book's own spread and total: the favourite
    runs more and scores more. Team implied points = total/2 +- spread/2.
  * ptd  Poisson, rate = QB's recent share of his team's points that came
         through the air, times implied points.
  * car  negative binomial, mean = form x game-script factor (fit on 2025).
  * lrush  per-carry chance of a run of L+ yards (shrunk to position) mixed
         over the carries distribution.
  * combo  gamma, mean = form x script factor, spread fit on 2025.
  * Depth charts (nflverse, ESPN snapshots): each player's spot at his
    position in the last snapshot BEFORE kickoff -- starter, 3rd string or
    deeper, not listed, moved up or down since his last game. The model's
    recent form only learns a role change a game or two late; this knows at
    kickoff. Backtest (fit on the season before): carries error -1.6% (2025)
    / -4.3% (2026), yardage -1.9% / -2.4%, and 6-24% on games where the
    player's depth spot had changed.
  * Expected yards (ffopportunity): the yards a player's targets and carries
    were worth on average, weighted over recent games, against his actual
    yards. Adds about 2% more on yardage. Both are optional: if either file
    can't be reached the model fits without it.
  * atd  logistic on touchdown usage: expected TDs (every carry and target
         priced by field position, rates from the prior season) over a
         weighted long window and the last 5 games, the player's share of
         his team's xTD, team implied points, targets and carries. Each
         player also gets a "tdu" block for the app's TD scorer card (last-5
         xTD, TDs, red-zone / goal-line / end-zone looks, usage score).

  Everything is checked out of sample before it is trusted: models are fit on
  data BEFORE the season being scored and the calibration table is written to
  the output ("calib") so the app can show it.

If the sandbox cannot reach GitHub (or a season has no play-by-play yet) this
exits without writing anything, so a bad run never overwrites a good file.
"""
import json
import math
import zlib
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

NV = "https://github.com/nflverse/nflverse-data/releases/download/"
PBP = NV + "pbp/play_by_play_{season}.parquet"
PLAYERS = NV + "players/players.parquet"
DEPTH = NV + "depth_charts/depth_charts_{season}.parquet"
SCHED = NV + "schedules/games.parquet"
# expected yards per player-game, from nflverse's sister project ffopportunity
FFOPP = "https://github.com/ffverse/ffopportunity/releases/download/latest-data/ep_weekly_{season}.parquet"

HALF_LIFE = 3.0                  # games
LOOKBACK = 10                    # most recent active games considered
POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
SKILL = {"QB", "RB", "WR", "TE"}
ROUND = 4

# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------


def load_pbp(season, cache=None):
    cols = ["season", "week", "season_type", "game_id", "home_team", "away_team",
            "posteam", "defteam", "play_type", "yardline_100", "yards_gained",
            "rush_attempt", "pass_attempt", "complete_pass", "qb_kneel", "sack",
            "rusher_player_id", "receiver_player_id", "passer_player_id",
            "rushing_yards", "receiving_yards", "pass_touchdown", "rush_touchdown",
            "td_player_id", "spread_line", "total_line", "qb_scramble",
            "air_yards", "two_point_attempt"]
    path = (cache or {}).get(season)
    if path and os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(PBP.format(season=season))
    df = df[[c for c in cols if c in df.columns]]
    return df[df.season_type == "REG"].copy()


def load_positions(local=None):
    df = pd.read_parquet(local if local and os.path.exists(local) else PLAYERS)
    col = "gsis_id"
    pos = df.set_index(col)["position"].to_dict()
    name = df.set_index(col)["display_name"].to_dict()
    return pos, name


def player_games(pbp, pos):
    """One row per player per game he was active (>=1 target or carry)."""
    d = pbp
    ctx = (d.groupby(["season", "week", "game_id"])
             .agg(home=("home_team", "first"), away=("away_team", "first"),
                  spread=("spread_line", "first"), total=("total_line", "first"))
             .reset_index())

    # two-point tries are not carries, targets or yards in the box score, so
    # no book counts them (2026-11-13: about 70 "carries" a season were)
    if "two_point_attempt" in d.columns:
        d = d[d.two_point_attempt != 1]

    # --- rushing
    ru = d[(d.rush_attempt == 1) & d.rusher_player_id.notna()].copy()
    ru["yds"] = ru.rushing_yards.fillna(ru.yards_gained).fillna(0)
    ru["rztd"] = ((ru.rush_touchdown == 1) & (ru.td_player_id == ru.rusher_player_id)).astype(int)
    ru["rz10"] = (ru.yardline_100 <= 10).astype(int)
    ru["gl"] = (ru.yardline_100 <= 3).astype(int)
    g_ru = ru.groupby(["season", "week", "game_id", "posteam", "rusher_player_id"]).agg(
        car=("yds", "size"), rushyds=("yds", "sum"), lrush=("yds", "max"),
        rz10=("rz10", "sum"), gl=("gl", "sum"), rutd=("rztd", "sum")).reset_index()
    g_ru = g_ru.rename(columns={"rusher_player_id": "pid"})

    # --- receiving
    re = d[(d.pass_attempt == 1) & d.receiver_player_id.notna()].copy()
    re["rec"] = re.complete_pass.fillna(0)
    re["ry"] = re.receiving_yards.fillna(0) * re.rec
    re["rz20"] = (re.yardline_100 <= 20).astype(int)
    re["retd"] = ((re.pass_touchdown == 1) & (re.td_player_id == re.receiver_player_id)).astype(int)
    g_re = re.groupby(["season", "week", "game_id", "posteam", "receiver_player_id"]).agg(
        tgt=("rec", "size"), rec=("rec", "sum"), recyds=("ry", "sum"),
        rz20=("rz20", "sum"), retd=("retd", "sum")).reset_index()
    g_re = g_re.rename(columns={"receiver_player_id": "pid"})

    # --- passing (QBs): attempts, TDs
    pa = d[(d.pass_attempt == 1) & d.passer_player_id.notna()].copy()
    pa["ptd"] = (pa.pass_touchdown == 1).astype(int)
    g_pa = pa.groupby(["season", "week", "game_id", "posteam", "passer_player_id"]).agg(
        patt=("ptd", "size"), ptd=("ptd", "sum")).reset_index()
    g_pa = g_pa.rename(columns={"passer_player_id": "pid"})

    key = ["season", "week", "game_id", "posteam", "pid"]
    pg = g_ru.merge(g_re, on=key, how="outer").merge(g_pa, on=key, how="outer")
    for c in ["car", "rushyds", "rz10", "gl", "rutd", "tgt", "rec", "recyds", "rz20",
              "retd", "patt", "ptd"]:
        pg[c] = pg[c].fillna(0)
    pg["lrush"] = pg.lrush.fillna(0)
    pg["td"] = ((pg.rutd + pg.retd) > 0).astype(int)
    pg["combo"] = pg.rushyds + pg.recyds
    pg["pos"] = pg.pid.map(pos).map(POS_GROUP)
    pg = pg.merge(ctx, on=["season", "week", "game_id"], how="left")
    pg["is_home"] = (pg.posteam == pg.home).astype(int)
    # team-perspective spread: + means this player's team is favoured
    pg["tspread"] = np.where(pg.is_home == 1, pg.spread, -pg.spread)
    pg["imp"] = pg.total / 2 + pg.tspread / 2
    pg["opp"] = np.where(pg.is_home == 1, pg.away, pg.home)
    pg["active"] = ((pg.car + pg.tgt) > 0) | (pg.patt >= 5)
    pg = pg[pg.active & pg.pos.isin(SKILL)].copy()
    pg["t"] = pg.season * 100 + pg.week
    return pg.sort_values(["pid", "t"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# form: exponentially weighted trailing averages
# ----------------------------------------------------------------------------
FORM = ["car", "rushyds", "tgt", "recyds", "rz10", "rz20", "gl", "td", "ptd", "patt", "combo"]


def pos_prior(pg):
    """League-average value of each FORM stat by position (training data)."""
    return pg.groupby("pos")[FORM].mean().to_dict("index")


def ew(vals, hl=HALF_LIFE):
    n = len(vals)
    if n == 0:
        return None, 0.0
    w = 0.5 ** (np.arange(n - 1, -1, -1) / hl)
    return w, w.sum()


def add_form(pg, prior, shrink=1.5):
    """Attach form_<stat> = EW average over the player's previous active games,
    shrunk toward the position average by `shrink` pseudo-games."""
    out = {c: np.full(len(pg), np.nan) for c in FORM}
    nprev = np.zeros(len(pg))
    for pid, idx in pg.groupby("pid").indices.items():
        sub = pg.iloc[idx]
        pos = sub.pos.iloc[0]
        pr = prior.get(pos, {})
        vals = {c: sub[c].to_numpy(float) for c in FORM}
        for j in range(len(sub)):
            lo = max(0, j - LOOKBACK)
            n = j - lo
            nprev[idx[j]] = n
            for c in FORM:
                if n == 0:
                    out[c][idx[j]] = pr.get(c, 0.0)
                    continue
                w = 0.5 ** (np.arange(n - 1, -1, -1) / HALF_LIFE)
                out[c][idx[j]] = (np.dot(w, vals[c][lo:j]) + shrink * pr.get(c, 0.0)) / (w.sum() + shrink)
    for c in FORM:
        pg["f_" + c] = out[c]
    pg["nprev"] = nprev
    return pg


# ----------------------------------------------------------------------------
# small stats helpers (numpy only -- the workflow does not install scipy)
# ----------------------------------------------------------------------------


def irls_logit(X, y, l2=1.0, iters=40):
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X1.shape[1])
    pen = np.eye(X1.shape[1]) * l2
    pen[0, 0] = 0
    for _ in range(iters):
        z = X1 @ b
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        H = X1.T @ (X1 * W[:, None]) + pen
        g = X1.T @ (y - p) - pen @ b
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def irls_pois(X, y, offset, iters=40, l2=0.5):
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X1.shape[1])
    b[0] = np.log(max(y.sum(), 1) / max(np.exp(offset).sum(), 1e-9))
    pen = np.eye(X1.shape[1]) * l2
    pen[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(X1 @ b + offset, -8, 8)
        mu = np.exp(eta)
        H = X1.T @ (X1 * mu[:, None]) + pen
        g = X1.T @ (y - mu) - pen @ b
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def nb_pmf(mu, r, kmax):
    """Negative binomial pmf, mean mu, variance mu + mu^2/r. r<=0 or huge -> Poisson."""
    mu = max(mu, 1e-6)
    out = np.zeros(kmax + 1)
    if r is None or r > 1e5:
        out[0] = math.exp(-mu)
        for k in range(kmax):
            out[k + 1] = out[k] * mu / (k + 1)
        return out
    p0 = (r / (r + mu)) ** r
    out[0] = p0
    q = mu / (r + mu)
    for k in range(kmax):
        out[k + 1] = out[k] * (k + r) / (k + 1) * q
    return out


def fit_disp(mu, y):
    """Method-of-moments NB size r for var = mu + mu^2/r, pooled."""
    mu = np.asarray(mu, float)
    y = np.asarray(y, float)
    excess = ((y - mu) ** 2 - mu).sum()
    denom = (mu ** 2).sum()
    if excess <= 0:
        return 1e6
    return float(denom / excess)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
ATD_COLS = ["f_tgt", "f_car", "u_xew", "u_x5", "u_shew", "u_tdew"]


# ----------------------------------------------------------------------------
# touchdown usage: every carry and target priced by where it happened
# ----------------------------------------------------------------------------
# Expected TDs (xTD) price each carry and target at the league's TD rate for
# that spot on the field: carries split QB / everyone else (sneaks score far
# more), targets split by whether the throw reached the end zone. Rates come
# from the PRIOR season only, so a backtest never sees its own answers.
# Over a 5-game window this is the M.C. Analytics "Touchdown Cheat Sheet"
# number; the model below uses it, a longer weighted window, the player's
# share of his team's xTD and the team's implied points.
GL_YD, RZ_YD = 5, 20
USAGE_HL = 4.0          # games, half-life of the long usage window
USAGE_N = 5             # the short window (the cheat-sheet window)


def td_events(pbp, pos):
    d = pbp[(pbp.get("two_point_attempt", 0) != 1) & (pbp.qb_kneel != 1)]
    ru = d[(d.rush_attempt == 1) & d.rusher_player_id.notna()]
    ru = pd.DataFrame({"season": ru.season, "game_id": ru.game_id, "posteam": ru.posteam,
                       "pid": ru.rusher_player_id, "kind": "rush", "yl": ru.yardline_100,
                       "td": ((ru.rush_touchdown == 1) & (ru.td_player_id == ru.rusher_player_id)).astype(int),
                       "ez": 0})
    re = d[(d.pass_attempt == 1) & d.receiver_player_id.notna()]
    re = pd.DataFrame({"season": re.season, "game_id": re.game_id, "posteam": re.posteam,
                       "pid": re.receiver_player_id, "kind": "tgt", "yl": re.yardline_100,
                       "td": ((re.pass_touchdown == 1) & (re.td_player_id == re.receiver_player_id)).astype(int),
                       "ez": (re.air_yards >= re.yardline_100).fillna(False).astype(int)})
    ev = pd.concat([ru, re], ignore_index=True)
    ev = ev[ev.yl.notna()].copy()
    ev["yl"] = ev.yl.clip(1, 99).astype(int)
    ev["qb"] = (ev.pid.map(pos) == "QB").astype(int)
    ev["grp"] = np.where(ev.kind == "rush", np.where(ev.qb == 1, "rush_qb", "rush"),
                         np.where(ev.ez == 1, "tgt_ez", "tgt"))
    return ev


def td_rate_tables(ev):
    """TD rate by yard line for each play group, smoothed over 5 yards except
    right at the goal line where every yard matters."""
    T = {}
    for grp, sub in ev.groupby("grp"):
        g = sub.groupby("yl").td.agg(["sum", "count"]).reindex(range(1, 100), fill_value=0)
        s = g["sum"].rolling(5, center=True, min_periods=1).sum()
        c = g["count"].rolling(5, center=True, min_periods=1).sum()
        narrow = 5 if grp.startswith("rush") else 3
        r = np.where(g.index <= narrow, (g["sum"] + 0.5) / (g["count"] + 1), (s + 0.5) / (c + 1))
        T[grp] = r
    return T


def add_usage(pg, ev, T):
    """Per game: xtd, red-zone / goal-line counts, team xTD. Then the pre-game
    windows (u_*) and the after-game windows (n_*, what the NEXT game uses)."""
    ev = ev.copy()
    ev["x"] = [T.get(g, T["tgt"])[y - 1] for g, y in zip(ev.grp, ev.yl)]
    ev["rzr"] = ((ev.kind == "rush") & (ev.yl <= RZ_YD)).astype(int)
    ev["glr"] = ((ev.kind == "rush") & (ev.yl <= GL_YD)).astype(int)
    ev["rzt"] = ((ev.kind == "tgt") & (ev.yl <= RZ_YD)).astype(int)
    ev["ezt"] = ((ev.kind == "tgt") & (ev.ez == 1)).astype(int)
    g = ev.groupby(["game_id", "pid"]).agg(xtd=("x", "sum"), rzr=("rzr", "sum"), glr=("glr", "sum"),
                                           rzt=("rzt", "sum"), ezt=("ezt", "sum")).reset_index()
    tm = ev.groupby(["game_id", "posteam"]).x.sum().rename("txtd").reset_index()
    pg = pg.drop(columns=[c for c in ("xtd", "rzr", "glr", "rzt", "ezt", "txtd") if c in pg], errors="ignore")
    pg = pg.merge(g, on=["game_id", "pid"], how="left").merge(tm, on=["game_id", "posteam"], how="left")
    for c in ("xtd", "rzr", "glr", "rzt", "ezt"):
        pg[c] = pg[c].fillna(0.0)
    pg["share"] = pg.xtd / pg.txtd.fillna(0).clip(lower=0.5)
    pg = pg.sort_values(["pid", "t"]).reset_index(drop=True)
    by = pg.groupby("pid")
    for c, src in (("xew", "xtd"), ("shew", "share"), ("tdew", "td")):
        n = by[src].transform(lambda v: v.ewm(halflife=USAGE_HL).mean())
        pg["n_" + c] = n
        pg["u_" + c] = n.groupby(pg.pid).shift(1)
    n5 = by.xtd.transform(lambda v: v.rolling(USAGE_N, min_periods=1).mean())
    pg["n_x5"] = n5
    pg["u_x5"] = n5.groupby(pg.pid).shift(1)
    return pg


# ----------------------------------------------------------------------------
# depth charts + expected yards (tested 2026-11: see docstring)
# ----------------------------------------------------------------------------
SKILL_DEPTH = {"QB", "RB", "WR", "TE", "FB", "HB"}


def load_depth(seasons):
    """Pre-game depth rank per (season, week, team, pid) and the latest
    snapshot per (team, pid) for the coming games. Handles both nflverse
    formats: weekly NFL charts (through 2024) and ESPN snapshots (2025+)."""
    try:
        sch = pd.read_parquet(SCHED)
    except Exception as e:
        print(f"  depth: no schedule ({e})", file=sys.stderr)
        return None, {}
    sch = sch[(sch.season.isin(seasons)) & (sch.game_type == "REG")]
    sch = sch.assign(kick=pd.to_datetime(sch.gameday + " " + sch.gametime.fillna("13:00"))
                     .dt.tz_localize("America/New_York").dt.tz_convert("UTC"))
    rows, now = [], {}
    for y in seasons:
        try:
            d = pd.read_parquet(DEPTH.format(season=y))
        except Exception as e:
            print(f"  depth {y}: unavailable ({e})", file=sys.stderr)
            continue
        if "dt" in d.columns:                                   # ESPN snapshots
            d = d[d.pos_abb.isin(SKILL_DEPTH) & d.gsis_id.notna()].copy()
            d["dt"] = pd.to_datetime(d.dt, utc=True)
            d["pos_rank"] = pd.to_numeric(d.pos_rank, errors="coerce")
            snaps = pd.DatetimeIndex(sorted(d.dt.unique()))
            best = d.groupby(["dt", "team", "gsis_id"]).pos_rank.min().reset_index()
            for _, g in sch[sch.season == y].iterrows():
                i = snaps.searchsorted(g.kick) - 1              # last snapshot before kickoff
                if i < 0:
                    continue
                t = snaps[i]
                x = best[(best.dt == t) & best.team.isin([g.home_team, g.away_team])]
                rows.append(pd.DataFrame({"season": y, "week": g.week, "posteam": x.team.values,
                                          "pid": x.gsis_id.values, "depth": x.pos_rank.values}))
            last = best[best.dt == snaps[-1]]
            now.update({(t, p): float(r) for t, p, r in zip(last.team, last.gsis_id, last.pos_rank)})
        else:                                                   # weekly NFL charts
            d = d[(d.formation == "Offense") & d.depth_position.isin(SKILL_DEPTH) & (d.game_type == "REG")]
            d = d.assign(depth=pd.to_numeric(d.depth_team, errors="coerce"))
            x = d.groupby(["week", "club_code", "gsis_id"]).depth.min().reset_index()
            rows.append(pd.DataFrame({"season": y, "week": x.week.values, "posteam": x.club_code.values,
                                      "pid": x.gsis_id.values, "depth": x.depth.values}))
    if not rows:
        return None, now
    dep = pd.concat(rows, ignore_index=True).drop_duplicates(["season", "week", "posteam", "pid"])
    dep["week"] = dep.week.astype(int)
    return dep, now


def load_ffopp(seasons):
    out = []
    for y in seasons:
        try:
            f = pd.read_parquet(FFOPP.format(season=y))
        except Exception as e:
            print(f"  expected yards {y}: unavailable ({e})", file=sys.stderr)
            continue
        f = f.assign(ff_xyds=f.rush_yards_gained_exp.fillna(0) + f.rec_yards_gained_exp.fillna(0))
        out.append(f.rename(columns={"player_id": "pid"})[["game_id", "pid", "ff_xyds"]])
    return pd.concat(out).drop_duplicates(["game_id", "pid"]) if out else None


def add_extras(pg, dep, ff):
    """depth (pre-game), dep_prev (his depth at his previous game), and the
    expected-yards windows u_ff_xyds (pre-game) / n_ff_xyds (for next game)."""
    pg = pg.drop(columns=[c for c in ("depth", "ff_xyds") if c in pg], errors="ignore")
    if dep is not None:
        pg = pg.merge(dep, on=["season", "week", "posteam", "pid"], how="left")
    else:
        pg["depth"] = np.nan
    if ff is not None:
        pg = pg.merge(ff, on=["game_id", "pid"], how="left")
    else:
        pg["ff_xyds"] = np.nan
    pg = pg.sort_values(["pid", "t"]).reset_index(drop=True)
    pg["dep_prev"] = pg.groupby("pid").depth.shift(1)
    if pg.ff_xyds.notna().any():
        n = pg.groupby("pid").ff_xyds.transform(lambda v: v.fillna(0).ewm(halflife=USAGE_HL).mean())
        n[pg.groupby("pid").ff_xyds.transform(lambda v: v.notna().cumsum()) == 0] = np.nan
    else:
        n = pd.Series(np.nan, index=pg.index)
    pg["n_ff_xyds"] = n
    pg["u_ff_xyds"] = n.groupby(pg.pid).shift(1)
    return pg


def _col(d, c):
    return np.asarray(d[c], float) if c in d else np.full(len(d), np.nan)


def depth_X(d):
    dp, pv = _col(d, "depth"), _col(d, "dep_prev")
    seen = ~np.isnan(pv)
    with np.errstate(invalid="ignore"):
        return np.column_stack([np.nan_to_num(dp == 1), np.nan_to_num(dp >= 3), np.isnan(dp),
                                np.nan_to_num((dp < pv) & seen), np.nan_to_num((dp > pv) & seen)]).astype(float)


def ffyds_X(d):
    u, fc = _col(d, "u_ff_xyds"), np.clip(_col(d, "f_combo"), 1, None)
    r = np.log((u + 5) / (fc + 5))
    return np.nan_to_num(r)[:, None]                      # missing -> 0 (= no signal)


def car_X(d):
    return np.column_stack([np.asarray(d.tspread, float) / 7.0, np.asarray(d.pos.eq("QB"), float),
                            np.log(np.asarray(d.f_car, float)), depth_X(d), ffyds_X(d)])


def combo_X(d):
    return np.column_stack([np.asarray(d.tspread, float) / 7.0, np.log(np.asarray(d.f_combo, float)),
                            depth_X(d), ffyds_X(d)])


def _lg(x):
    return np.log(np.clip(np.asarray(x, float), 0.01, None))
LRUSH_T = list(range(1, 51))          # longest-rush thresholds exported (P(longest >= L), L = 1..50)
COMBO_X = list(range(0, 301, 5))      # combo-yards grid (P(total >= x), every 5 yards)
CAR_MAX = 40
PTD_MAX = 7
Q_THR = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80]


def atd_X(d):
    """Anytime TD, logistic. Tested walk-forward on 2025 and 2026 (fit on the
    season before): log loss 0.4736 / 0.4707 vs 0.4754 / 0.4738 for the old
    counts-based version and 0.4949 / 0.4903 for the raw 5-game xTD read as a
    probability. Red-zone counts, goal-line counts and the opponent's TDs
    allowed added nothing once xTD and implied points were in."""
    imp = np.asarray(d.imp, float)
    return np.column_stack([_lg(d.u_xew), _lg(d.u_x5), (imp - 22.5) / 5,
                            _lg(np.asarray(d.u_shew, float) * imp / 7), d.u_tdew,
                            d.pos.eq("RB"), d.pos.eq("TE"), d.pos.eq("QB"),
                            d.f_tgt, d.f_car, depth_X(d)]).astype(float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def recal(surv, cal):
    """Apply a backtest correction (logit a + b*z) to a list of P(X >= line)
    values and keep it from rising. Certainties (0 or 1) are left alone."""
    if not cal:
        return list(surv)
    out = []
    for p in surv:
        if 0.001 < p < 0.999:
            p = 1 / (1 + math.exp(-(cal[0] + cal[1] * math.log(p / (1 - p)))))
        out.append(p)
    return np.minimum.accumulate(np.array(out)).tolist()


def fit_models(pg, carries):
    """pg: player-game rows with form attached (training period only).
       carries: DataFrame [pid, pos, yds] one row per carry (training period)."""
    M = {}
    d = pg[pg.nprev >= 1]
    # --- anytime TD
    a = d[d.u_xew.notna()]
    M["atd"] = irls_logit(atd_X(a), a.td.values, l2=1.0)
    # --- passing TDs: Poisson, offset = log(form), covariate = log(implied pts)
    q = d[(d.pos == "QB") & (d.patt >= 8)]
    off = np.log(np.clip(q.f_ptd.values, 0.15, None))
    X = np.column_stack([np.log(q.imp.values / 22.5)])
    M["ptd"] = irls_pois(X, q.ptd.values.astype(float), off)
    # --- rushing attempts: Poisson mean, NB spread per position group
    r = d[(d.car >= 1) | (d.pos.isin(["RB", "QB"]))]
    r = r[r.f_car > 0.3]
    off = np.zeros(len(r))
    X = car_X(r)
    M["car"] = irls_pois(X, r.car.values.astype(float), off)
    mu = np.exp(np.column_stack([np.ones(len(r)), X]) @ M["car"] + off)
    M["car_r"] = {}
    for g, mask in (("QB", r.pos.eq("QB").values), ("RB", r.pos.isin(["RB"]).values),
                    ("WR", r.pos.isin(["WR", "TE"]).values)):
        if mask.sum() > 30:
            M["car_r"][g] = fit_disp(mu[mask], r.car.values[mask])
    # --- combo yards: log-link mean, gamma spread: var = a*mu + b*mu^2
    c = d[d.f_combo > 3]
    off = np.zeros(len(c))
    X = combo_X(c)
    M["combo"] = irls_pois(X, c.combo.clip(lower=0).values.astype(float), off)
    mu = np.exp(np.column_stack([np.ones(len(c)), X]) @ M["combo"] + off)
    res2 = (c.combo.values - mu) ** 2
    A = np.column_stack([mu, mu ** 2])
    coef = np.linalg.lstsq(A, res2, rcond=None)[0]
    M["combo_var"] = [max(float(coef[0]), 1.0), max(float(coef[1]), 0.05)]
    # --- per-carry big-run rates by position, for longest rush
    M["q_prior"] = {}
    for g, grp in (("QB", ["QB"]), ("RB", ["RB"]), ("WR", ["WR", "TE"])):
        y = carries[carries.pos.isin(grp)].yds.values
        if len(y) < 50:
            y = carries.yds.values
        M["q_prior"][g] = {str(t): float((y >= t).mean()) for t in Q_THR}
    return M


def qgroup(pos):
    return "QB" if pos == "QB" else ("RB" if pos == "RB" else "WR")


def predict_row(M, row, car_stats=None):
    """Distributions for one player-game. `row` has f_*, imp, tspread, pos.
       car_stats = (n_carries, {thr: count>=thr}) for this player."""
    out = {}
    pos = row["pos"]
    g = qgroup(pos)
    # atd
    x = atd_X(pd.DataFrame([row]))
    out["atd"] = float(sigmoid(M["atd"][0] + x @ M["atd"][1:])[0])
    # ptd
    if pos == "QB" and row["f_patt"] >= 8:
        lam = math.exp(M["ptd"][0] + M["ptd"][1] * math.log(row["imp"] / 22.5)
                       + math.log(max(row["f_ptd"], 0.15)))
        out["ptd"] = [round(v, ROUND) for v in nb_pmf(lam, None, PTD_MAX)]
        out["ptd_mu"] = round(lam, 3)
    # car
    mu_car = None
    if row["f_car"] > 0.3 and (pos in ("RB", "QB") or row["f_car"] >= 1.0):
        beta = M["car"]
        mu_car = float(math.exp(beta[0] + (car_X(pd.DataFrame([row])) @ beta[1:])[0]))
        r = M["car_r"].get(g) or M["car_r"].get("RB") or 20.0
        pm = nb_pmf(mu_car, r, CAR_MAX)
        if M.get("car_cal"):
            # P(carries >= L) corrected, then turned back into P(carries = k)
            sv = np.cumsum(pm[::-1])[::-1].tolist()
            sv[0] = 1.0                              # P(carries >= 0) is certain
            sv = recal(sv, M["car_cal"])
            pc = np.array(sv) - np.array(sv[1:] + [0.0])
            out["car"] = [round(float(v), ROUND) for v in pc]
        else:
            out["car"] = [round(v, ROUND) for v in pm]
        out["car_mu"] = round(mu_car, 2)
        # longest rush: mix over carries, per-carry big-run chance shrunk to position
        n_c, hits = car_stats if car_stats else (0, {})
        m = 40.0
        qs = []
        prior = M["q_prior"][g]
        thr = [int(t) for t in prior]
        qv = []
        for t in thr:
            qv.append((hits.get(t, 0) + m * prior[str(t)]) / (n_c + m))
        qv = np.minimum.accumulate(np.array(qv))           # survival must not rise
        surv = []
        for L in LRUSH_T:
            # interpolate q at L (log space) across the threshold grid
            if L <= thr[0]:
                qL = 1.0 if L <= 1 else qv[0] + (1 - qv[0]) * (thr[0] - L) / (thr[0] - 1)
            elif L >= thr[-1]:
                qL = qv[-1] * math.exp(-(L - thr[-1]) / 25.0)
            else:
                k = max(i for i, t in enumerate(thr) if t <= L)
                t0, t1 = thr[k], thr[k + 1]
                a, b2 = max(qv[k], 1e-6), max(qv[k + 1], 1e-6)
                qL = math.exp(math.log(a) + (math.log(b2) - math.log(a)) * (L - t0) / (t1 - t0))
            qL = min(max(qL, 0.0), 1.0)
            # P(longest >= L) = 1 - E[(1-q)^N]
            pk = 1 - float(np.dot(pm, (1 - qL) ** np.arange(len(pm))))
            cal = M.get("lrush_cal")
            if cal and 0.001 < pk < 0.999:
                z = math.log(pk / (1 - pk))
                pk = 1 / (1 + math.exp(-(cal[0] + cal[1] * z)))
            surv.append(pk)
        # recalibration can break monotonicity slightly; survival must not rise
        surv = np.minimum.accumulate(np.array(surv)).tolist()
        out["lrush"] = surv
    # combo
    if row["f_combo"] > 3:
        mu = float(math.exp(M["combo"][0] + (combo_X(pd.DataFrame([row])) @ M["combo"][1:])[0]))
        a, b = M["combo_var"]
        var = a * mu + b * mu * mu
        k = mu * mu / var
        th = var / mu
        rng = np.random.default_rng(zlib.crc32(str(row.get("pid")).encode()))
        s = rng.gamma(k, th, 6000)
        out["combo"] = [round(float(v), ROUND) for v in
                        recal([float((s >= x).mean()) for x in COMBO_X], M.get("combo_cal"))]
        out["combo_mu"] = round(mu, 1)
    return out


# ----------------------------------------------------------------------------
# calibration summary (honest, out of sample) -- shipped in the file
# ----------------------------------------------------------------------------
def _brier(p, a):
    p = np.asarray(p, float)
    a = np.asarray(a, float)
    if len(p) == 0:
        return None
    return {"n": int(len(p)), "brier": round(float(((p - a) ** 2).mean()), 4),
            "base": round(float(((a.mean() - a) ** 2).mean()), 4),
            "pred": round(float(p.mean()), 3), "act": round(float(a.mean()), 3)}


def backtest(pg, carries, train_season, test_season):
    """Fit on `train_season` only, score `test_season`. Lines are placed where a
    book would put them (about the model's own median), because nobody posts a
    line far from the middle."""
    tr = pg[pg.season == train_season]
    te = pg[(pg.season == test_season) & (pg.nprev >= 1)]
    if len(tr) < 500 or len(te) < 100:
        return None
    M = fit_models(tr, carries[carries.season == train_season] if "season" in carries else carries)
    # each player's carries in game order, so a game is scored on the carries
    # he had BEFORE it. Until 2026-11-13 this used every carry in the file,
    # including the game being scored and the ones after it, which flattered
    # the longest-rush backtest (2026 log loss 0.634 with the peek, 0.642 without).
    by = {}
    if "t" in carries:
        for pid, g in carries.sort_values("t").groupby("pid"):
            by[pid] = (g.t.values, g.yds.values)

    def before(pid, t):
        if pid not in by:
            return None
        tt, yy = by[pid]
        y = yy[:int(np.searchsorted(tt, t, side="left"))]
        return (len(y), {q: int((y >= q).sum()) for q in Q_THR})

    P = {"atd": ([], []), "ptd": ([], []), "car": ([], []), "lrush": ([], []), "combo": ([], [])}
    LRC = ([], [])
    CARC, COMC = ([], []), ([], [])      # carries / yards at three lines each, for the corrections
    for _, r in te.iterrows():
        o = predict_row(M, r.to_dict(), before(r.pid, r.t))
        P["atd"][0].append(o["atd"]); P["atd"][1].append(r.td)
        if "ptd" in o:
            pm = np.array(o["ptd"])
            for L in (1, 2):
                P["ptd"][0].append(pm[L:].sum()); P["ptd"][1].append(float(r.ptd >= L))
        if "car" in o:
            pm = np.array(o["car"])
            L = int(math.floor(o["car_mu"])) + 1
            P["car"][0].append(pm[L:].sum()); P["car"][1].append(float(r.car >= L))
            for LL in {max(L - 2, 1), L, L + 2}:
                CARC[0].append(pm[LL:].sum()); CARC[1].append(float(r.car >= LL))
            s = np.array(o["lrush"])
            L = int(np.argmin(abs(s - .5))) + 1
            P["lrush"][0].append(s[L - 1]); P["lrush"][1].append(float(r.lrush >= L))
            for LL in (max(L - 3, 2), L + 4):
                LRC[0].append(s[LL - 1]); LRC[1].append(float(r.lrush >= LL))
            LRC[0].append(s[L - 1]); LRC[1].append(float(r.lrush >= L))
        if "combo" in o:
            s = np.array(o["combo"])
            x = min(int(round(o["combo_mu"] / 5)), len(s) - 1)
            P["combo"][0].append(s[x]); P["combo"][1].append(float(r.combo >= COMBO_X[x]))
            for xx in {max(x - 3, 1), x, min(x + 3, len(s) - 1)}:
                COMC[0].append(s[xx]); COMC[1].append(float(r.combo >= COMBO_X[xx]))
    res = {k: _brier(*v) for k, v in P.items()}
    # anytime-TD calibration by predicted chance: long shots are where a small
    # overrating turns into a big (fake) EV, so check them on their own
    pa, aa = np.array(P["atd"][0]), np.array(P["atd"][1])
    res["atd_bins"] = []
    for lo, hi in ((0, .06), (.06, .10), (.10, .15), (.15, .25), (.25, .40), (.40, 1.01)):
        m = (pa >= lo) & (pa < hi)
        if m.sum():
            res["atd_bins"].append({"lo": lo, "hi": hi, "n": int(m.sum()),
                                    "pred": round(float(pa[m].mean()), 4), "act": round(float(aa[m].mean()), 4)})
    p = np.clip(np.array(LRC[0]), 0.02, 0.98)
    if len(p) > 200:
        b = irls_logit(np.log(p / (1 - p))[:, None], np.array(LRC[1]), l2=1.0)
        res["_lrush_cal"] = [float(b[0]), float(b[1])]
    # Carries and rush+rec yards get the same kind of correction (2026-11-13).
    # Two seasons running the model said "over" too often on carries (44% said,
    # 40-43% happened at its own middle line) and too rarely on yards (37-38%
    # said, 39-40% happened). Fit on 2025 and applied to 2026: log loss
    # 0.6238 -> 0.6201 (carries), 0.6147 -> 0.6139 (yards).
    for key, (pp, aa2) in (("_car_cal", CARC), ("_combo_cal", COMC)):
        p = np.clip(np.array(pp, float), 0.02, 0.98)
        if len(p) > 200:
            b = irls_logit(np.log(p / (1 - p))[:, None], np.array(aa2), l2=1.0)
            res[key] = [float(b[0]), float(b[1])]
    return res


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def trim(a, nd=3, eps=0.0015):
    """Round and drop the tail that is effectively zero (keeps the file small)."""
    a = [round(float(x), nd) for x in a]
    while len(a) > 3 and a[-1] < eps:
        a.pop()
    return a


def mkey(s):
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def carries_table(pbp, pos):
    ru = pbp[(pbp.rush_attempt == 1) & pbp.rusher_player_id.notna()].copy()
    if "two_point_attempt" in ru.columns:
        ru = ru[ru.two_point_attempt != 1]
    ru["yds"] = ru.rushing_yards.fillna(ru.yards_gained).fillna(0)
    ru["pid"] = ru.rusher_player_id
    ru["pos"] = ru.pid.map(pos).map(POS_GROUP)
    ru["t"] = ru.season * 100 + ru.week          # same game clock as player_games
    return ru[["season", "t", "pid", "pos", "yds"]].dropna()


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    out = sys.argv[2] if len(sys.argv) > 2 else f"data/props-{season}.json"
    d = os.path.dirname(out) or "."
    games_path = os.path.join(d, f"games-{season}.json")
    roster_path = os.path.join(d, f"roster-{season}.json")
    cache = {}
    for y in (season - 1, season):
        for cand in (f"pbp{y}.parquet",):
            if os.path.exists(cand):
                cache[y] = cand
    try:
        pos, name = load_positions("players.parquet" if os.path.exists("players.parquet") else None)
        frames = []
        for y in (season - 1, season):
            try:
                frames.append(load_pbp(y, cache))
            except Exception as e:                       # prior season missing is survivable
                print(f"  no play-by-play for {y}: {e}", file=sys.stderr)
        pbp = pd.concat(frames)
    except Exception as e:
        raise SystemExit(f"could not load data, leaving the old file alone: {e}")
    cur = pbp[pbp.season == season]
    if cur.empty:
        raise SystemExit("no games played yet this season; nothing to model")
    last_wk = int(cur.week.max())

    pg = player_games(pbp, pos)
    ev = td_events(pbp, pos)
    tab_ev = ev[ev.season < season] if (ev.season < season).any() else ev
    T = td_rate_tables(tab_ev)
    pg = add_usage(pg, ev, T)
    dep, dep_now = load_depth([season - 1, season])
    ff = load_ffopp([season - 1, season])
    pg = add_extras(pg, dep, ff)
    print(f"  depth charts: {'yes' if dep is not None else 'no'} · expected yards: {'yes' if ff is not None else 'no'}", file=sys.stderr)
    # usage score: 100 = the best 10-game xTD stretch on record (cheat-sheet scale)
    r10 = pg.groupby("pid").xtd.transform(lambda v: v.rolling(10).mean())
    best10 = float(np.nanmax(r10.values)) if r10.notna().any() else 1.25
    prior = pos_prior(pg[pg.season < season] if (pg.season < season).any() else pg)
    carries = carries_table(pbp, pos)

    # ---- honest backtest (previous season -> this season) for the file
    calib = None
    try:
        pgb = add_form(pg.copy(), prior)
        calib = backtest(pgb, carries, season - 1, season)
    except Exception as e:
        print(f"  backtest skipped: {e}", file=sys.stderr)

    # ---- upcoming games
    G = json.load(open(games_path))["games"] if os.path.exists(games_path) else []
    today = datetime.now(timezone.utc).date().isoformat()
    yday = (pd.Timestamp(today) - pd.Timedelta(days=1)).date().isoformat()
    up = [g for g in G if g.get("hs") is None and str(g.get("date", "")) >= yday]
    if not up:
        raise SystemExit("no upcoming games in the games file; leaving the old file alone")
    # only the next slate: games within 9 days of the earliest unplayed one
    first = min(str(g["date"]) for g in up)
    cutoff = (pd.Timestamp(first) + pd.Timedelta(days=9)).date().isoformat()
    up = [g for g in up if str(g["date"]) <= cutoff]

    R = json.load(open(roster_path))["players"] if os.path.exists(roster_path) else {}
    tm_now = {pid: v for pid, v in R.items()}

    # ---- fit on everything played so far
    pgf = add_form(pg.copy(), prior)
    M = fit_models(pgf, carries)
    if calib and calib.get("_lrush_cal"):
        M["lrush_cal"] = calib.pop("_lrush_cal")
    for k in ("car_cal", "combo_cal"):
        if calib and calib.get("_" + k):
            M[k] = calib.pop("_" + k)
    cs = {}
    recent = carries[carries.season >= season - 1]
    for pid, g in recent.groupby("pid"):
        y = g.yds.values
        cs[pid] = (len(y), {t: int((y >= t).sum()) for t in Q_THR})

    # who is realistic to play for each team: active in 2 of their last 4
    last_rows = pg.sort_values("t").groupby("pid").tail(4)
    cand = {}
    for pid, grp in last_rows.groupby("pid"):
        t_now = (tm_now.get(pid) or {}).get("team") or grp.posteam.iloc[-1]
        cand.setdefault(t_now, []).append((pid, grp))

    players = []
    seen = set()
    for g in up:
        for side, tm, opp in (("home", g["home"], g["away"]), ("away", g["away"], g["home"])):
            sp, tot = g.get("spread"), g.get("total")
            if sp is None or tot is None:
                if g.get("ptot") is None or g.get("proj") is None:
                    continue
                tot = g["ptot"]
                sp = g["proj"]
            tsp = sp if side == "home" else -sp
            imp = tot / 2 + tsp / 2
            for pid, grp in cand.get(tm, []):
                if (pid, g.get("id")) in seen:
                    continue
                info = tm_now.get(pid) or {}
                if info.get("is") in ("Out", "Doubtful") or info.get("st") in ("IR", "CUT", "NONE", "RET", "PS"):
                    continue
                if len(grp) < 2 or grp.t.max() < (season - 1) * 100 + 15:
                    continue
                hist = pgf[pgf.pid == pid]
                if hist.empty:
                    continue
                ps = pos.get(pid) or info.get("pos")
                pg_pos = POS_GROUP.get(ps)
                if pg_pos is None or pg_pos not in SKILL:
                    continue
                fut = {"pid": pid, "pos": pg_pos}
                w = 0.5 ** (np.arange(min(len(hist), LOOKBACK))[::-1] / HALF_LIFE)
                tail = hist.tail(LOOKBACK)
                pr = prior.get(pg_pos, {})
                for c in FORM:
                    fut["f_" + c] = (np.dot(w, tail[c].to_numpy(float)) + 1.5 * pr.get(c, 0.0)) / (w.sum() + 1.5)
                fut.update({"imp": imp, "tspread": tsp, "pos": pg_pos})
                last = hist.iloc[-1]
                for c in ("xew", "x5", "shew", "tdew"):
                    fut["u_" + c] = float(last["n_" + c])
                # depth now (latest snapshot) vs his depth at his last game; expected-yards window
                fut["depth"] = dep_now.get((tm, pid), np.nan) if dep_now else np.nan
                fut["dep_prev"] = float(last["depth"]) if "depth" in last and pd.notna(last["depth"]) else np.nan
                fut["u_ff_xyds"] = float(last["n_ff_xyds"]) if "n_ff_xyds" in last and pd.notna(last["n_ff_xyds"]) else np.nan
                l5 = hist.tail(USAGE_N)
                tdl5 = int((l5.rutd + l5.retd).sum())
                usage = {"g": int(len(l5)), "x": round(float(l5.xtd.sum()), 2), "td": tdl5,
                         "rzr": int(l5.rzr.sum()), "glr": int(l5.glr.sum()),
                         "rzt": int(l5.rzt.sum()), "ezt": int(l5.ezt.sum()),
                         "sh": round(float(last.n_shew), 3), "xpg": round(float(last.n_xew), 3),
                         "us": int(min(100, round(100 * float(l5.xtd.mean()) / best10)))}
                o = predict_row(M, fut, cs.get(pid))
                # drop what nobody prices: tiny roles make a big file and no bets
                if o.get("combo_mu", 99) < 20:
                    o.pop("combo", None); o.pop("combo_mu", None)
                if o.get("car_mu", 99) < 3:
                    for k in ("car", "car_mu", "lrush"):
                        o.pop(k, None)
                atd_full = o.get("atd")
                if o.get("atd", 1) < 0.07:        # the app never tiers long shots
                    o.pop("atd", None)
                if not any(k in o for k in ("atd", "ptd", "car", "combo")):
                    continue
                seen.add((pid, g.get("id")))
                ent = {"pid": pid, "name": info.get("name") or name.get(pid),
                       "key": mkey(info.get("name") or name.get(pid)),
                       "team": tm, "opp": opp, "pos": pg_pos, "date": g["date"], "gid": g.get("id"),
                       "n": int(len(hist)), "imp": round(imp, 1), "tsp": round(tsp, 1)}
                if atd_full is not None and usage["x"] > 0:
                    usage["p"] = round(atd_full, 3)
                    ent["tdu"] = usage
                for k, v in o.items():
                    ent[k] = (trim(v, 2 if k in ("lrush", "combo") else 3) if isinstance(v, list)
                              else (round(v, 3) if isinstance(v, float) else v))
                players.append(ent)

    # ---- what actually happened, for grading saved picks
    results = {}
    for wk in range(max(1, last_wk - 2), last_wk + 1):
        rows = pg[(pg.season == season) & (pg.week == wk)]
        results[str(wk)] = {r.pid: {"td": int(r.td), "ptd": int(r.ptd), "car": int(r.car),
                                    "lrush": int(r.lrush), "combo": round(float(r.combo), 1),
                                    "team": r.posteam}
                            for r in rows.itertuples()}

    doc = {"season": season, "kind": "props",
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "through_week": last_wk, "calib": calib, "td_best10": round(best10, 3), "players": players, "results": results}
    os.makedirs(d, exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {out}: {len(players)} player-games across {len(up)} games; "
          f"results through week {last_wk}", file=sys.stderr)
    if calib:
        for k, v in calib.items():
            print(f"  calib {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
