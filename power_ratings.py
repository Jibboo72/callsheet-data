"""
NFL Power Ratings Engine
========================
Opponent-adjusted, luck-aware, recency-weighted team ratings expressed in POINTS,
built from nflverse play-by-play.

METHOD (the short version)
--------------------------
1. Filter plays to competitive, non-garbage, non-clock-kill situations.
2. Split into four "phases": pass offense, rush offense, pass defense, rush defense.
3. For each phase, fit a RIDGE REGRESSION on play-level EPA:
       epa ~ offense_team_dummies + defense_team_dummies + home
   The offense coefficient is that team's effect on EPA/play after removing the
   quality of every defense it faced. That IS the opponent adjustment, done
   properly (simultaneous, not iterative approximation).
4. Ridge's L2 penalty shrinks every team toward league average. That shrinkage is
   the Bayesian prior -- in Week 3 everyone is near 0 because the data hasn't
   earned a strong opinion yet. This is what stops early-season ratings from
   being garbage.
5. Repeat the same regression with SUCCESS RATE as the target. Success rate is
   noisier-resistant; EPA is fat-tailed. Blend them.
6. Recency-weight plays (exponential decay) and discount prior seasons.
7. Convert the blended EPA/play index into points/game, calibrated so that
   (rating_A - rating_B + HFA) is on the same scale as a real point spread.

Author: built for Josh
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG -- every tunable knob lives here
# --------------------------------------------------------------------------

CONFIG = dict(
    # --- sample filtering ---
    wp_low=0.05,            # drop plays outside this win-probability band (garbage time)
    wp_high=0.95,
    max_qtr=5,

    # --- recency ---
    half_life_weeks=8.0,    # in-season decay: a game 9 weeks ago counts half
    prior_season_weight=0.45,  # last season's plays enter at 25% weight
    prior_season_decay=0.35,   # season before that: 0.25*0.35

    # --- shrinkage (the prior strength) ---
    alpha_pass=200.0,
    alpha_rush=200.0,

    # --- blending ---
    w_epa=0.60,             # EPA vs success-rate blend
    w_sr=0.40,
    off_pass_weight=0.50,   # passing is ~2x more stable/predictive than rushing
    off_rush_weight=0.50,
    def_pass_weight=0.50,
    def_rush_weight=0.50,

    # --- points conversion ---
    plays_per_game=62.0,
    st_weight=1.0,          # special teams multiplier
)

TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


# --------------------------------------------------------------------------
# LOADING
# --------------------------------------------------------------------------

def load_pbp(seasons, cache_dir="."):
    """Load nflverse play-by-play for a list of seasons."""
    frames = []
    for s in seasons:
        path = f"{cache_dir}/pbp{s}.parquet"
        try:
            df = pd.read_parquet(path)
        except Exception:
            url = ("https://github.com/nflverse/nflverse-data/releases/"
                   f"download/pbp/play_by_play_{s}.parquet")
            df = pd.read_parquet(url)
            try:
                df.to_parquet(path)
            except Exception:
                pass
        frames.append(df)
    pbp = pd.concat(frames, ignore_index=True)
    for c in ("posteam", "defteam", "home_team", "away_team"):
        if c in pbp.columns:
            pbp[c] = pbp[c].replace(TEAM_FIX)
    return pbp


def load_games(cache_dir="."):
    try:
        g = pd.read_csv(f"{cache_dir}/games.csv")
    except Exception:
        g = pd.read_csv("https://raw.githubusercontent.com/nflverse/"
                        "nfldata/master/data/games.csv")
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIX)
    return g


# --------------------------------------------------------------------------
# FILTERING
# --------------------------------------------------------------------------

def scrimmage_plays(pbp, cfg=CONFIG):
    """Competitive, meaningful offense-vs-defense snaps."""
    d = pbp
    m = (
        d["posteam"].notna() & d["defteam"].notna()
        & d["epa"].notna()
        & (d["qb_kneel"] != 1) & (d["qb_spike"] != 1)
        & (d["aborted_play"] != 1)
        & (d["special"] != 1)
        & d["play_type"].isin(["pass", "run"])
        & (d["qtr"] <= cfg["max_qtr"])
    )
    d = d.loc[m].copy()
    # garbage-time filter on win probability (fall back to wp if vegas_wp missing)
    wp = d["vegas_wp"].fillna(d["wp"])
    d = d.loc[wp.between(cfg["wp_low"], cfg["wp_high"]) | wp.isna()].copy()
    d["is_pass"] = (d["qb_dropback"] == 1)
    return d


def special_teams_plays(pbp):
    d = pbp
    m = (d["special"] == 1) & d["epa"].notna() & d["posteam"].notna()
    return d.loc[m].copy()


# --------------------------------------------------------------------------
# WEIGHTS
# --------------------------------------------------------------------------

def play_weights(d, as_of_season, as_of_week, cfg=CONFIG):
    """Exponential recency decay in weeks, plus prior-season discount."""
    # weeks elapsed: approximate a season as 22 weeks
    weeks_ago = ((as_of_season - d["season"]) * 22.0) + (as_of_week - d["week"])
    weeks_ago = np.maximum(weeks_ago, 0.0)
    lam = np.log(2.0) / cfg["half_life_weeks"]
    w = np.exp(-lam * weeks_ago)

    season_gap = (as_of_season - d["season"]).values
    mult = np.ones(len(d))
    mult[season_gap == 1] = cfg["prior_season_weight"]
    mult[season_gap >= 2] = cfg["prior_season_weight"] * cfg["prior_season_decay"]
    return (w.values * mult)


# --------------------------------------------------------------------------
# THE RIDGE MODEL -- opponent adjustment
# --------------------------------------------------------------------------

def _weighted_ridge(off, dfn, home, y, w, k, alpha):
    """
    Weighted ridge on the design [off dummies | def dummies | home | intercept],
    penalising every coefficient except the intercept -- identical to
    sklearn Ridge(alpha, fit_intercept=True) with sample_weight.

    Builds the normal equations directly from the one-hot structure, so it
    never materialises the full design matrix.
    """
    m = 2 * k + 2
    A = np.zeros((m, m))
    b = np.zeros(m)

    wh = w * home
    so = np.bincount(off, weights=w, minlength=k)
    sd = np.bincount(dfn, weights=w, minlength=k)
    cross = np.bincount(off * k + dfn, weights=w, minlength=k * k).reshape(k, k)

    A[:k, :k] = np.diag(so)
    A[k:2*k, k:2*k] = np.diag(sd)
    A[:k, k:2*k] = cross
    A[k:2*k, :k] = cross.T

    A[:k, 2*k] = A[2*k, :k] = np.bincount(off, weights=wh, minlength=k)
    A[k:2*k, 2*k] = A[2*k, k:2*k] = np.bincount(dfn, weights=wh, minlength=k)
    A[:k, 2*k+1] = A[2*k+1, :k] = so
    A[k:2*k, 2*k+1] = A[2*k+1, k:2*k] = sd

    A[2*k, 2*k] = (w * home * home).sum()
    A[2*k, 2*k+1] = A[2*k+1, 2*k] = wh.sum()
    A[2*k+1, 2*k+1] = w.sum()

    wy = w * y
    b[:k] = np.bincount(off, weights=wy, minlength=k)
    b[k:2*k] = np.bincount(dfn, weights=wy, minlength=k)
    b[2*k] = (wy * home).sum()
    b[2*k+1] = wy.sum()

    P = np.eye(m) * alpha
    P[-1, -1] = 0.0                      # intercept unpenalised
    return np.linalg.solve(A + P, b)[:2*k+1]

def fit_phase(d, teams, target, alpha, weights):
    """Return (offense_effect, defense_effect) Series in target units."""
    if len(d) < 200:
        z = pd.Series(0.0, index=teams)
        return z.copy(), z.copy()

    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    off = d["posteam"].map(idx).to_numpy()
    dfn = d["defteam"].map(idx).to_numpy()
    home = (d["posteam"].values == d["home_team"].values).astype(float)
    y = d[target].astype(float).to_numpy()
    w = np.asarray(weights, dtype=float)

    coef = _weighted_ridge(off, dfn, home, y, w, k, alpha)
    off_s = pd.Series(coef[:k], index=teams)
    dfn_s = pd.Series(coef[k:2 * k], index=teams)
    # center so league average is exactly 0
    return off_s - off_s.mean(), dfn_s - dfn_s.mean()


# --------------------------------------------------------------------------
# SPECIAL TEAMS
# --------------------------------------------------------------------------

def special_teams_rating(pbp, teams, as_of_season, as_of_week, cfg=CONFIG):
    st = special_teams_plays(pbp)
    st = st[(st["season"] < as_of_season) |
            ((st["season"] == as_of_season) & (st["week"] < as_of_week))]
    if len(st) == 0:
        return pd.Series(0.0, index=teams)
    w = play_weights(st, as_of_season, as_of_week, cfg)
    st = st.assign(_w=w)
    # EPA credited to the team with possession; kicking team on FG/punt/KO
    num = st.groupby("posteam").apply(
        lambda g: np.average(g["epa"], weights=g["_w"]) if g["_w"].sum() > 0 else 0.0
    )
    cnt = st.groupby("posteam")["_w"].sum()
    # shrink toward 0 based on sample
    shrunk = num * (cnt / (cnt + 250.0))
    out = shrunk.reindex(teams).fillna(0.0)
    return out - out.mean()


# --------------------------------------------------------------------------
# LUCK / REGRESSION DIAGNOSTICS
# --------------------------------------------------------------------------

def luck_flags(pbp, as_of_season, as_of_week):
    """Things that happened but probably won't keep happening."""
    d = pbp[(pbp["season"] == as_of_season) & (pbp["week"] < as_of_week)]
    d = d[d["posteam"].notna()]
    rows = []
    teams = sorted(set(d["posteam"].dropna()) | set(d["defteam"].dropna()))
    for t in teams:
        off = d[d["posteam"] == t]
        dfn = d[d["defteam"] == t]

        # fumble recovery luck: own fumbles kept + forced fumbles recovered
        own_fum = off["fumble"].sum() if "fumble" in off else 0
        own_lost = off["fumble_lost"].sum() if "fumble_lost" in off else 0
        forced = dfn["fumble"].sum() if "fumble" in dfn else 0
        forced_rec = dfn["fumble_lost"].sum() if "fumble_lost" in dfn else 0
        tot_fum = own_fum + forced
        rec = (own_fum - own_lost) + forced_rec
        fum_rate = rec / tot_fum if tot_fum > 0 else 0.5

        # red zone TD rate (offense)
        rz = off[(off["yardline_100"] <= 20) & off["play_type"].isin(["pass", "run"])]
        rz_drives = rz["fixed_drive"].nunique() if len(rz) else 0
        rz_td = rz[rz["touchdown"] == 1]["fixed_drive"].nunique() if len(rz) else 0
        rz_rate = rz_td / rz_drives if rz_drives else np.nan

        # 3rd down conversion vs. league baseline for same distance
        rows.append(dict(team=t, fumble_recovery_rate=fum_rate,
                         fumbles_in_play=int(tot_fum),
                         rz_td_rate=rz_rate))
    out = pd.DataFrame(rows).set_index("team")
    lg_rz = out["rz_td_rate"].mean()
    out["rz_td_rate_vs_lg"] = out["rz_td_rate"] - lg_rz
    out["fumble_luck"] = out["fumble_recovery_rate"] - 0.50
    return out


# --------------------------------------------------------------------------
# MAIN RATING BUILD
# --------------------------------------------------------------------------

def build_ratings(pbp, as_of_season, as_of_week, cfg=CONFIG, teams=None):
    """Ratings using ONLY data strictly before (as_of_season, as_of_week)."""
    d = scrimmage_plays(pbp, cfg)
    d = d[(d["season"] < as_of_season) |
          ((d["season"] == as_of_season) & (d["week"] < as_of_week))]
    if teams is None:
        teams = sorted(set(d["posteam"].dropna()))
    w_all = play_weights(d, as_of_season, as_of_week, cfg)
    d = d.assign(_w=w_all)

    dp = d[d["is_pass"]]
    dr = d[~d["is_pass"]]

    po_e, pd_e = fit_phase(dp, teams, "epa", cfg["alpha_pass"], dp["_w"].values)
    ro_e, rd_e = fit_phase(dr, teams, "epa", cfg["alpha_rush"], dr["_w"].values)
    po_s, pd_s = fit_phase(dp, teams, "success", cfg["alpha_pass"], dp["_w"].values)
    ro_s, rd_s = fit_phase(dr, teams, "success", cfg["alpha_rush"], dr["_w"].values)

    def blend(e, s):
        se, ss = e.std(), s.std()
        s_scaled = s * (se / ss) if ss > 1e-9 else s * 0.0
        return cfg["w_epa"] * e + cfg["w_sr"] * s_scaled

    pass_off = blend(po_e, po_s)
    rush_off = blend(ro_e, ro_s)
    pass_def = blend(pd_e, pd_s)
    rush_def = blend(rd_e, rd_s)

    off_idx = cfg["off_pass_weight"] * pass_off + cfg["off_rush_weight"] * rush_off
    def_idx = cfg["def_pass_weight"] * pass_def + cfg["def_rush_weight"] * rush_def

    ppg = cfg["plays_per_game"]
    st = special_teams_rating(pbp, teams, as_of_season, as_of_week, cfg) * 8.0

    out = pd.DataFrame({
        "off_pass": pass_off * ppg,
        "off_rush": rush_off * ppg,
        "def_pass": -pass_def * ppg,   # flip so positive = good defense
        "def_rush": -rush_def * ppg,
        "offense": off_idx * ppg,
        "defense": -def_idx * ppg,
        "special": st * cfg["st_weight"],
    })
    out["rating"] = out["offense"] + out["defense"] + out["special"]
    out = out.sort_values("rating", ascending=False)
    out.index.name = "team"
    return out


# --------------------------------------------------------------------------
# PROJECTION
# --------------------------------------------------------------------------

def project_spread(ratings, home, away, hfa=1.6, scale=1.0,
                   home_adj=0.0, away_adj=0.0):
    """Return projected margin from the HOME team's perspective.
    Positive = home favored by that many points."""
    h = ratings.loc[home, "rating"] + home_adj
    a = ratings.loc[away, "rating"] + away_adj
    return scale * (h - a) + hfa


def project_matchup(ratings, home, away, hfa=1.6, scale=1.0):
    """Full matchup view: each offense against the opposing defense."""
    h, a = ratings.loc[home], ratings.loc[away]
    return dict(
        home=home, away=away,
        home_off_vs_away_def=h["offense"] - a["defense"],
        away_off_vs_home_def=a["offense"] - h["defense"],
        home_pass_edge=h["off_pass"] - a["def_pass"],
        home_rush_edge=h["off_rush"] - a["def_rush"],
        away_pass_edge=a["off_pass"] - h["def_pass"],
        away_rush_edge=a["off_rush"] - h["def_rush"],
        st_edge=h["special"] - a["special"],
        projected_margin=project_spread(ratings, home, away, hfa, scale),
    )
