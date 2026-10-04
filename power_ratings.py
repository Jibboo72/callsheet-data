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
5b. STEP 2 ("tighten the formula"): a third signal, Next Gen Stats CPOE
    (completion % above expectation) and RYOE (rush yards over expectation
    per attempt), is blended in alongside EPA/success rate -- same idea,
    different stat. NGS is weekly player-level data, not play-level, so it's
    first aggregated to team-week grain (attempt-weighted) and opponent-
    adjusted with its OWN ridge fit (fit_ngs_phase, below) that reuses the
    exact same _weighted_ridge machinery as the EPA/success phases -- one
    team-week row stands in for "one play". The weights that control how
    much this counts (w_cpoe_off/def, w_ryoe_off/def) all default to 0.0, so
    nothing about the live ratings changes until these are backtested and
    turned on (see test_cpoe_ryoe.py).
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

    # --- final combination ---
    # Equal 1:1 weighting by default. Research (Brian Burke's predictivity
    # work; nfelo's EPA tiers, which use 1.6 offense : 1.0 defense) says
    # offensive efficiency is a stickier, more predictive signal than
    # defensive efficiency -- defense is noisier year-over-year and
    # week-to-week. Don't just adopt that number: see test_formula_weights.py,
    # which sweeps this honestly against the walk-forward backtest before
    # touching the default here.
    rating_off_weight=1.0,
    rating_def_weight=1.0,

    # --- NGS CPOE / RYOE: a 3rd signal in the offense/defense blend ---
    # fit_ngs_phase() opponent-adjusts CPOE (passing) and RYOE (rushing) the
    # same way EPA/success are adjusted, just at team-week grain instead of
    # play grain (see fit_ngs_phase below). alpha_cpoe/alpha_ryoe are the
    # ridge shrinkage for THOSE fits -- much lower than alpha_pass/alpha_rush
    # because there are ~17 team-week rows per team per season, not
    # thousands of plays, so far less shrinkage is needed to avoid overfit.
    # The w_* weights are how much CPOE/RYOE count in the final blend, each
    # scaled to EPA's own spread before being applied (see blend() in
    # build_ratings) so a weight of e.g. 0.15 means "15% as much swing as
    # EPA gets", not a raw unit. All four default to 0.0 -- OFF -- until
    # test_cpoe_ryoe.py's walk-forward backtest says otherwise.
    alpha_cpoe=25.0,
    alpha_ryoe=25.0,
    w_cpoe_off=0.0,
    w_cpoe_def=0.0,
    w_ryoe_off=0.0,
    w_ryoe_def=0.0,

    # --- step 3: charted pass-pressure rate (PFR, free) ---
    # Same idea and same fit_ngs_phase() machinery as CPOE/RYOE, one more
    # team-week signal blended into the pass offense/defense phases. See
    # load_pfr_pressure_team_week() for why PFR and not FTN. Defaults to
    # 0.0 -- OFF -- until test_pressure.py's walk-forward backtest says
    # otherwise.
    alpha_prss=25.0,
    w_prss_off=0.0,
    w_prss_def=0.0,

    # --- step 4: starter-availability (injury) discount ---
    # A per-GAME point adjustment (via project_spread()'s home_adj/
    # away_adj), not a persistent team-quality signal, so it does NOT flow
    # through build_ratings()'s ridge blend like steps 1-3 -- see
    # load_injury_discount_team_week(). injury_trailing_weeks is how many
    # of a player's own prior games (this season) their snap share is
    # averaged over to judge how much losing them matters.
    # injury_weight (0.0 by default) is what test_injury_discount.py
    # sweeps.
    injury_trailing_weeks=6,
    injury_weight=0.0,
)

TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}

# --------------------------------------------------------------------------
# STEP 4: starter-availability (injury) discount -- see
# load_injury_discount_team_week() below. Unlike steps 1-3 these are fixed
# weights, not something test_injury_discount.py sweeps: the backtest
# sweeps the OVERALL scale (CONFIG["injury_weight"]), because getting the
# relative ordering between positions roughly right (QB >> OL/WR/CB/EDGE >
# TE/S/LB > RB/DT > kicker) matters more than its precise number, and a
# full per-position regression is a bigger project than this step needs
# before even knowing whether the signal helps at all.
# --------------------------------------------------------------------------

POSITION_INJURY_WEIGHT = {
    "QB": 1.00,
    "T": 0.35, "OT": 0.35, "G": 0.35, "OG": 0.35, "C": 0.35, "OL": 0.35,
    "WR": 0.30, "CB": 0.30, "DE": 0.30, "EDGE": 0.30, "OLB": 0.30,
    "TE": 0.20, "S": 0.20, "SS": 0.20, "FS": 0.20,
    "LB": 0.20, "ILB": 0.20, "MLB": 0.20,
    "RB": 0.15, "DT": 0.15, "NT": 0.15,
    "K": 0.05, "P": 0.05, "LS": 0.05,
}
DEFAULT_POSITION_INJURY_WEIGHT = 0.10

# How often each report-status tier actually ends up playing, roughly --
# Out essentially never plays, Doubtful rarely does, Questionable mostly
# does play (hence the small weight).
STATUS_INJURY_WEIGHT = {"Out": 1.00, "Doubtful": 0.75, "Questionable": 0.25}


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


def load_ngs(stat_type, cache_dir="."):
    """Load nflverse Next Gen Stats, player-week grain, ALL seasons in one
    file (stat_type: 'passing' or 'rushing' -- 'receiving' also exists but
    isn't used here). Drops the week=0 rows, which are season-long totals,
    not point-in-time data -- using them would be a look-ahead leak."""
    path = f"{cache_dir}/ngs_{stat_type}.parquet"
    try:
        df = pd.read_parquet(path)
    except Exception:
        url = ("https://github.com/nflverse/nflverse-data/releases/download/"
               f"nextgen_stats/ngs_{stat_type}.parquet")
        df = pd.read_parquet(url)
        try:
            df.to_parquet(path)
        except Exception:
            pass
    df = df[df["week"] > 0].copy()
    df["team_abbr"] = df["team_abbr"].replace(TEAM_FIX)
    return df


def _team_week_agg(ngs, value_col, weight_col):
    """Collapse player-week NGS rows to one attempt/rush-weighted row per
    team-week -- the grain the opponent-adjustment ridge needs.

    NGS has real NaNs in value_col (e.g. ~1k rows of early-era
    rush_yards_over_expected_per_att, before NGS backfilled that stat) --
    a plain weighted average propagates one NaN player into the whole
    team-week, and from there into the whole ridge solve the moment a
    weight is non-zero (every team's coefficient goes NaN, not just that
    team's -- that's what a SVD-did-not-converge crash downstream means).
    So NaN rows are dropped before averaging, using only the players who
    actually have a value, same as scrimmage_plays() dropping epa.isna()."""
    def _wavg(x):
        ok = x[value_col].notna()
        w = x.loc[ok, weight_col].to_numpy(dtype=float)
        v = x.loc[ok, value_col].to_numpy(dtype=float)
        tot = w.sum()
        # if NOTHING on this team-week has a real value, report weight 0 too
        # (not the raw attempt count) -- fit_ngs_phase multiplies by this
        # weight, so a 0 here correctly drops the row instead of feeding
        # the ridge a fake "0.0, fully trusted" observation.
        return pd.Series({value_col: (np.average(v, weights=w) if tot > 0 else 0.0),
                          weight_col: (x[weight_col].sum() if tot > 0 else 0.0)})
    g = (ngs.groupby(["season", "week", "team_abbr"])
            .apply(_wavg)
            .reset_index())
    return g


def _attach_opponent(tw, games):
    """Join each team-week row to its opponent (defteam) and the game's real
    home team, via the schedule -- the columns fit_phase()/fit_ngs_phase()
    need (posteam/defteam/home_team), same shape as play-level pbp rows."""
    g = games[["season", "week", "home_team", "away_team"]]
    home = tw.merge(g, left_on=["season", "week", "team_abbr"],
                    right_on=["season", "week", "home_team"], how="inner")
    home["defteam"] = home["away_team"]
    away = tw.merge(g, left_on=["season", "week", "team_abbr"],
                    right_on=["season", "week", "away_team"], how="inner")
    away["defteam"] = away["home_team"]
    out = pd.concat([home, away], ignore_index=True)
    return out.rename(columns={"team_abbr": "posteam"})


def load_ngs_pass_team_week(seasons=None, cache_dir="."):
    """Opponent-joined, team-week CPOE table -- pass ngs_pass= into
    build_ratings() to turn CPOE on (still gated by w_cpoe_off/w_cpoe_def)."""
    ngs = load_ngs("passing", cache_dir)
    if seasons is not None:
        ngs = ngs[ngs["season"].isin(seasons)]
    tw = _team_week_agg(ngs, "completion_percentage_above_expectation", "attempts")
    return _attach_opponent(tw, load_games(cache_dir))


def load_ngs_rush_team_week(seasons=None, cache_dir="."):
    """Opponent-joined, team-week RYOE-per-attempt table -- pass ngs_rush=
    into build_ratings() to turn RYOE on (gated by w_ryoe_off/w_ryoe_def)."""
    ngs = load_ngs("rushing", cache_dir)
    if seasons is not None:
        ngs = ngs[ngs["season"].isin(seasons)]
    tw = _team_week_agg(ngs, "rush_yards_over_expected_per_att", "rush_attempts")
    return _attach_opponent(tw, load_games(cache_dir))


def load_pfr_pressure_team_week(pbp, seasons=None, cache_dir="."):
    """Step 3: opponent-joined, team-week pass-pressure-rate table -- pass
    pfr_prss= into build_ratings() to turn it on (gated by
    w_prss_off/w_prss_def).

    SOURCE: PFR's weekly advanced passing stats (free, same nflverse
    pipeline as everything else here) -- its times_pressured column, a
    real charted "was the QB pressured on this dropback" outcome. FTN's
    own charting release was checked first (that was the original plan --
    "charted pressure rate ... cross-checked with PFR"), but FTN's
    ftn_charting files only have pass-rush CONTEXT (n_pass_rushers,
    n_blitzers) with no pressure OUTCOME column at all, so it can't
    produce a rate. PFR is the one source of the two that actually has
    one, so it's the only one wired in here.

    Player rows (one per passer-week, PFR's own grain) are summed to
    team-week counts, then divided by that TEAM's dropback count from pbp
    -- the same dropback definition scrimmage_plays() uses everywhere
    else in this file -- rather than trusting an un-published PFR
    denominator. Negated before returning, so higher = LESS pressure
    allowed = better offense, matching the "higher is better" polarity
    every other signal here uses (po_e, po_s, cpoe_off, ...) -- the sign
    flip for the defense side (allowed -> generated) happens downstream
    in combine_components(), same as EPA's pass defense always has.

    Starts 2018 -- PFR's advstats release doesn't go back further;
    seasons before that are skipped here, and any as-of date whose whole
    training window predates 2018 just gets zero signal (fit_ngs_phase's
    w.sum()<=0 guard), not a crash -- same handling RYOE already needed
    for its own pre-2018 gap.
    """
    want = seasons if seasons is not None else range(2018, 2027)
    frames = []
    for s in want:
        if s < 2018:
            continue
        path = f"{cache_dir}/pfr_pass_{s}.parquet"
        try:
            df = pd.read_parquet(path)
        except Exception:
            url = ("https://github.com/nflverse/nflverse-data/releases/"
                   f"download/pfr_advstats/advstats_week_pass_{s}.parquet")
            try:
                df = pd.read_parquet(url)
            except Exception:
                continue        # season not covered by PFR advstats
            try:
                df.to_parquet(path)
            except Exception:
                pass
        frames.append(df)

    empty_cols = ["season", "week", "team_abbr", "pressure_rate", "dropbacks",
                 "defteam", "home_team", "away_team"]
    if not frames:
        return pd.DataFrame(columns=empty_cols)

    p = pd.concat(frames, ignore_index=True)
    p["team"] = p["team"].replace(TEAM_FIX)
    pressed = (p.groupby(["season", "week", "team"], as_index=False)["times_pressured"]
                .sum().rename(columns={"team": "team_abbr"}))

    dp = scrimmage_plays(pbp, CONFIG)
    dp = dp[dp["is_pass"]]
    drop = (dp.groupby(["season", "week", "posteam"]).size()
              .rename("dropbacks").reset_index()
              .rename(columns={"posteam": "team_abbr"}))

    tw = pressed.merge(drop, on=["season", "week", "team_abbr"], how="inner")
    tw = tw[tw["dropbacks"] > 0].copy()
    if len(tw) == 0:
        return pd.DataFrame(columns=empty_cols)
    tw["pressure_rate"] = -(tw["times_pressured"] / tw["dropbacks"])
    return _attach_opponent(tw[["season", "week", "team_abbr", "pressure_rate", "dropbacks"]],
                            load_games(cache_dir))


def load_injury_discount_team_week(seasons=None, cache_dir="."):
    """Step 4: team-week RAW (unscaled) starter-availability discount,
    built from free nflverse injury reports + snap counts + the player-ID
    crosswalk (gsis_id <-> pfr_id). Multiply by CONFIG['injury_weight']
    (0.0 by default) and ADD to a game's raw rating-diff before scale/HFA
    calibration -- see test_injury_discount.py. This is a per-GAME
    adjustment, not a persistent team-quality signal like steps 1-3, so it
    deliberately does NOT go through build_ratings()'s ridge blend.

    For each team-week, every player listed Questionable/Doubtful/Out on
    that week's FINAL injury report is weighted by:
      - POSITION_INJURY_WEIGHT (a QB out matters far more than a punter)
      - STATUS_INJURY_WEIGHT (Out counts fully, Questionable barely at
        all -- matches how often each tier actually ends up playing)
      - that player's own TRAILING snap share: the average of their
        offense/defense/ST snap % over their most recent games this
        season, as of the LAST game they actually recorded snaps in
        before the injury-report week (merge_asof, so a player out for
        several straight weeks keeps the share from their last healthy
        stretch rather than silently dropping to 0) -- a Questionable
        WR2 who plays 90% of snaps matters more than a Questionable WR2
        who plays 20%.
    Summed per team-week into one raw discount number (higher = worse for
    that team that week; week 1 of a season, with no trailing games yet,
    correctly comes back near 0 -- no data, no signal, same rule every
    other step here follows). The ABSOLUTE scale is deliberately not
    calibrated here -- that is what CONFIG['injury_weight'] and
    test_injury_discount.py's sweep are for.

    Both injuries and snap counts go back to 2015, so unlike RYOE
    (2018+) and pressure (2018+) there's no pre-coverage gap to guard
    against here.
    """
    want = list(seasons) if seasons is not None else list(range(2015, 2027))

    def _load_series(tag_dir, prefix):
        frames = []
        for s in want:
            path = f"{cache_dir}/{prefix}_{s}.parquet"
            try:
                df = pd.read_parquet(path)
            except Exception:
                url = ("https://github.com/nflverse/nflverse-data/releases/"
                       f"download/{tag_dir}/{tag_dir}_{s}.parquet")
                try:
                    df = pd.read_parquet(url)
                except Exception:
                    continue
                try:
                    df.to_parquet(path)
                except Exception:
                    pass
            frames.append(df)
        return frames

    inj_frames = _load_series("injuries", "injuries")
    snap_frames = _load_series("snap_counts", "snap_counts")
    empty_cols = ["season", "week", "team_abbr", "discount"]
    if not inj_frames or not snap_frames:
        return pd.DataFrame(columns=empty_cols)

    inj = pd.concat(inj_frames, ignore_index=True)
    # some seasons' injuries files carry season/week as float64 (NaN rows
    # present in the source), others int -- merge_asof below requires
    # identical dtypes on both sides of the join, so force both frames to
    # the same int64 here rather than trust whatever each file shipped.
    inj = inj.dropna(subset=["season", "week"])
    inj["season"] = inj["season"].astype("int64")
    inj["week"] = inj["week"].astype("int64")
    inj = inj[inj["report_status"].isin(list(STATUS_INJURY_WEIGHT))].copy()
    inj["team"] = inj["team"].replace(TEAM_FIX)

    snaps = pd.concat(snap_frames, ignore_index=True)
    snaps = snaps.dropna(subset=["season", "week"])
    snaps["season"] = snaps["season"].astype("int64")
    snaps["week"] = snaps["week"].astype("int64")
    snaps["team"] = snaps["team"].replace(TEAM_FIX)
    snaps["snap_pct"] = snaps[["offense_pct", "defense_pct", "st_pct"]].max(axis=1)

    path = f"{cache_dir}/players.parquet"
    try:
        players = pd.read_parquet(path)
    except Exception:
        players = pd.read_parquet("https://github.com/nflverse/nflverse-data/"
                                  "releases/download/players/players.parquet")
        try:
            players.to_parquet(path)
        except Exception:
            pass
    xwalk = players[["gsis_id", "pfr_id"]].dropna()
    inj = inj.merge(xwalk, on="gsis_id", how="left")
    inj = inj.dropna(subset=["pfr_id"])
    if len(inj) == 0:
        return pd.DataFrame(columns=empty_cols)

    tw = CONFIG["injury_trailing_weeks"]
    snaps_sorted = snaps.sort_values(["pfr_player_id", "season", "week"]).copy()
    snaps_sorted["trail_share"] = (
        snaps_sorted.groupby(["pfr_player_id", "season"])["snap_pct"]
        .transform(lambda s: s.rolling(tw, min_periods=1).mean())
    )
    trail = (snaps_sorted[["pfr_player_id", "season", "week", "trail_share"]]
             .rename(columns={"pfr_player_id": "pfr_id"})
             .sort_values("week"))

    # as-of merge: each injury row gets the trailing share from this
    # player's most recent PRIOR (or same, if no games missed yet) week
    # with recorded snaps, within the same season -- not a strict
    # week-before-week match, since an injured player may have several
    # consecutive weeks with no snap_counts row at all.
    merged = pd.merge_asof(inj.sort_values("week"), trail,
                           on="week", by=["pfr_id", "season"], direction="backward")
    merged["trail_share"] = merged["trail_share"].fillna(0.0)
    pos_w = merged["position"].map(POSITION_INJURY_WEIGHT).fillna(DEFAULT_POSITION_INJURY_WEIGHT)
    stat_w = merged["report_status"].map(STATUS_INJURY_WEIGHT).fillna(0.0)
    merged["contrib"] = pos_w * stat_w * merged["trail_share"]

    out = (merged.groupby(["season", "week", "team"], as_index=False)["contrib"]
           .sum().rename(columns={"team": "team_abbr", "contrib": "discount"}))
    return out


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
    """Exponential recency decay in weeks, plus prior-season discount.
    Generic over grain -- only needs season/week columns, so it works
    unchanged on play-level pbp rows AND on team-week NGS rows."""
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

def fit_phase(d, teams, target, alpha, weights, min_rows=200):
    """Return (offense_effect, defense_effect) Series in target units.
    Grain-agnostic: needs posteam/defteam/home_team + a numeric target
    column. Called on play-level pbp rows for EPA/success (min_rows=200,
    the default -- thousands of plays is normal), and on team-week NGS
    rows for CPOE/RYOE (fit_ngs_phase, below, passes a much lower min_rows
    since a team-week row is a whole game's aggregate, not one play) --
    same math either way, one row just means something different."""
    if len(d) < min_rows:
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


def fit_ngs_phase(tw, teams, value_col, weight_col, alpha, as_of_season, as_of_week, cfg=CONFIG):
    """Opponent-adjusted team-week ridge for a Next Gen Stats signal (CPOE or
    RYOE-per-attempt). Reuses fit_phase()'s exact ridge machinery: each
    team-week row (one team's aggregate CPOE/RYOE for one game, already
    attempt-weighted across its players) stands in for "one play" in the
    original EPA model. Needs STRICTLY prior data, same no-look-ahead rule
    as build_ratings() itself -- tw is filtered to (season, week) < as_of
    right here so every caller gets that for free.

    tw: output of load_ngs_pass_team_week() / load_ngs_rush_team_week()
        (has posteam, defteam, home_team, season, week, value_col, weight_col).
    Returns (offense_effect, defense_effect) in value_col's own units --
    e.g. for CPOE, offense_effect is "this team's opponent-adjusted
    completion % above expectation when passing"; defense_effect is the
    same thing allowed, when defending (higher = worse pass defense, same
    direction as the EPA-allowed convention elsewhere in this file).
    """
    d = tw[(tw["season"] < as_of_season) |
          ((tw["season"] == as_of_season) & (tw["week"] < as_of_week))]
    if len(d) < 20:
        z = pd.Series(0.0, index=teams)
        return z.copy(), z.copy()
    rw = play_weights(d, as_of_season, as_of_week, cfg)
    w = rw * d[weight_col].to_numpy(dtype=float)
    # RYOE-per-attempt specifically: NGS didn't start computing it until
    # 2018 -- 2016-17 is 100% NaN, which _team_week_agg() already turns
    # into weight 0 per the comment there. If EVERY row in the training
    # window is one of those (as_of date whose whole prior window predates
    # real data), w is all zero, which makes the ridge's intercept term
    # 0/0 -- an unpenalized, dataless row -- and np.linalg.solve raises
    # LinAlgError: Singular matrix. Bail to "no signal yet" instead,
    # exactly like the len(d)<20 case just above.
    if w.sum() <= 0:
        z = pd.Series(0.0, index=teams)
        return z.copy(), z.copy()
    # min_rows=20, not fit_phase's play-level default of 200 -- a team-week
    # row here is a whole game's attempt-weighted aggregate, not one play,
    # so 20 of them is already a real sample (the len(d)<20 check above
    # covers the "basically nothing yet" case).
    return fit_phase(d.rename(columns={value_col: "_target"}), teams, "_target",
                     alpha, w, min_rows=20)


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

def _fit_components(pbp, as_of_season, as_of_week, cfg=CONFIG, teams=None,
                    ngs_pass=None, ngs_rush=None, pfr_prss=None):
    """Everything build_ratings() needs, BEFORE the final blend/combine
    step -- split out so a backtest can fit this ONCE per (season, week)
    and then cheaply try many blend weights against it (the same trick
    test_formula_weights.py uses for rating_off_weight/rating_def_weight).
    Returns a dict of team-indexed Series plus 'teams'.

    ngs_pass / ngs_rush / pfr_prss: optional pre-built team-week tables
    (see load_ngs_pass_team_week() / load_ngs_rush_team_week() /
    load_pfr_pressure_team_week()). Omit them (the default) and
    cpoe_off/cpoe_def/ryoe_off/ryoe_def/prss_off/prss_def all come back as
    0 -- build_ratings() is then byte-for-byte what it was before step 2.
    """
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

    zero = pd.Series(0.0, index=teams)
    cpoe_off = cpoe_def = ryoe_off = ryoe_def = prss_off = prss_def = zero
    if ngs_pass is not None:
        cpoe_off, cpoe_def = fit_ngs_phase(
            ngs_pass, teams, "completion_percentage_above_expectation",
            "attempts", cfg["alpha_cpoe"], as_of_season, as_of_week, cfg)
    if ngs_rush is not None:
        ryoe_off, ryoe_def = fit_ngs_phase(
            ngs_rush, teams, "rush_yards_over_expected_per_att",
            "rush_attempts", cfg["alpha_ryoe"], as_of_season, as_of_week, cfg)
    if pfr_prss is not None:
        prss_off, prss_def = fit_ngs_phase(
            pfr_prss, teams, "pressure_rate",
            "dropbacks", cfg["alpha_prss"], as_of_season, as_of_week, cfg)

    st = special_teams_rating(pbp, teams, as_of_season, as_of_week, cfg) * 8.0

    return dict(teams=teams, po_e=po_e, pd_e=pd_e, ro_e=ro_e, rd_e=rd_e,
                po_s=po_s, pd_s=pd_s, ro_s=ro_s, rd_s=rd_s,
                cpoe_off=cpoe_off, cpoe_def=cpoe_def,
                ryoe_off=ryoe_off, ryoe_def=ryoe_def,
                prss_off=prss_off, prss_def=prss_def, st=st)


def _scaled(base, extra):
    """Scale extra to base's own spread, so a blend weight means 'this much
    swing relative to EPA' rather than a raw, incomparable unit. Shared by
    build_ratings() and any backtest that recombines _fit_components()."""
    sb, se = base.std(), extra.std()
    return extra * (sb / se) if se > 1e-9 else extra * 0.0


def combine_components(c, cfg=CONFIG):
    """The blend/combine step split out of build_ratings() -- turns the raw
    fitted components into the same points-scale DataFrame build_ratings()
    returns. Pulled out so test_cpoe_ryoe.py (and test_formula_weights.py's
    successor, if it's ever extended) can call this directly per weight
    candidate without re-running _fit_components()."""
    def blend(e, s, *extras):
        """extras: any number of (series, weight) pairs -- each one scaled
        to e's own spread (see _scaled) before being added in, so a 0
        weight is always a true no-op regardless of how many signals are
        stacked on."""
        out = cfg["w_epa"] * e + cfg["w_sr"] * _scaled(e, s)
        for extra, w_extra in extras:
            if w_extra:
                out = out + w_extra * _scaled(e, extra)
        return out

    pass_off = blend(c["po_e"], c["po_s"],
                     (c["cpoe_off"], cfg["w_cpoe_off"]),
                     (c["prss_off"], cfg["w_prss_off"]))
    rush_off = blend(c["ro_e"], c["ro_s"],
                     (c["ryoe_off"], cfg["w_ryoe_off"]))
    pass_def = blend(c["pd_e"], c["pd_s"],
                     (c["cpoe_def"], cfg["w_cpoe_def"]),
                     (c["prss_def"], cfg["w_prss_def"]))
    rush_def = blend(c["rd_e"], c["rd_s"],
                     (c["ryoe_def"], cfg["w_ryoe_def"]))

    off_idx = cfg["off_pass_weight"] * pass_off + cfg["off_rush_weight"] * rush_off
    def_idx = cfg["def_pass_weight"] * pass_def + cfg["def_rush_weight"] * rush_def

    ppg = cfg["plays_per_game"]
    teams = c["teams"]

    out = pd.DataFrame({
        "off_pass": pass_off * ppg,
        "off_rush": rush_off * ppg,
        "def_pass": -pass_def * ppg,   # flip so positive = good defense
        "def_rush": -rush_def * ppg,
        "offense": off_idx * ppg,
        "defense": -def_idx * ppg,
        "special": c["st"] * cfg["st_weight"],
    }, index=teams)
    out["rating"] = (cfg["rating_off_weight"] * out["offense"]
                      + cfg["rating_def_weight"] * out["defense"]
                      + out["special"])
    out = out.sort_values("rating", ascending=False)
    out.index.name = "team"
    return out


def build_ratings(pbp, as_of_season, as_of_week, cfg=CONFIG, teams=None,
                  ngs_pass=None, ngs_rush=None, pfr_prss=None):
    """Ratings using ONLY data strictly before (as_of_season, as_of_week).

    ngs_pass / ngs_rush / pfr_prss: optional pre-built team-week tables
    (see load_ngs_pass_team_week() / load_ngs_rush_team_week() /
    load_pfr_pressure_team_week() above) that add CPOE, RYOE, and charted
    pass-pressure rate as extra signals in the offense/defense blend. Omit
    them (the default) and nothing changes -- CONFIG's
    w_cpoe_*/w_ryoe_*/w_prss_* all default to 0.0, so even passing them in
    changes nothing until those weights are turned on and backtested
    (test_cpoe_ryoe.py, test_pressure.py).
    """
    c = _fit_components(pbp, as_of_season, as_of_week, cfg, teams,
                        ngs_pass, ngs_rush, pfr_prss)
    return combine_components(c, cfg)


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
