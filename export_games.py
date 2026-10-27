"""
Game lines, totals, results and a calibrated projection for the Sides board.

    python3 export_games.py 2026 data/games-2026.json

WHERE THE LINES COME FROM
  nflverse games.csv carries spread_line, total_line, moneylines and weather
  from 1999 forward, and posts upcoming lines weeks ahead. That replaces the
  SBR scraper entirely for this purpose -- no blocked IPs, no HTML parsing.

  One thing it is NOT: an opening line. These are closing numbers (and, for
  unplayed games, the current number). Opening-vs-closing work still needs
  the SBR archive; anything about closing line value here is measured
  against where the number sits now, not where it opened.

CALIBRATION
  A raw rating difference is not a point spread. Ratings are regressed onto
  actual margins from COMPLETED seasons only, giving scale and home-field
  constants, exactly as test_vs_open.py does. For an unplayed season the fit
  comes from the prior season, so nothing looks ahead.

INJURY ADJUSTMENT (step 4 of "tighten the formula")
  Unlike the ratings themselves (one snapshot per build, as of
  played_through+1), the starter-availability discount is looked up PER
  GAME at that game's own week -- a missing starter in week 9 says nothing
  about week 10, so this can't be baked into the one shared ratings
  snapshot the way CPOE/pressure/etc. are. Each game's projection becomes
  scale * (rating_diff + injury_weight * (away_discount - home_discount)) + hfa,
  exactly the formula test_injury_discount.py backtested honestly before
  injury_weight got a nonzero default in power_ratings.py. See
  load_injury_discount_team_week()'s docstring there for what the discount
  itself measures. A missing/unreachable injuries or snap-counts file for
  this season degrades gracefully to a 0 discount (no adjustment), same as
  every other optional signal in this codebase.

LIVE INJURIES FOR THE UPCOMING WEEK (added 2026-10-23)
  nflverse's injury file runs days behind -- in practice the week about to
  be played has no rows at all when this runs, so the one part of the
  projection that tested as real (the injury adjustment) was missing from
  every edge shown before kickoff. For the next unplayed week only, teams
  with no nflverse rows get their discount from ESPN's public injury report
  instead, scored exactly the same way (position weight x status weight x
  the player's trailing snap share, matched to snap counts by name + team).
  ESPN unreachable or shaped differently -> no adjustment, as before.
  Every game carries inj_src ("nflverse" / "espn" / none) and inj_adj (the
  points the adjustment moved the projection, home perspective).

LAST SEASON'S PLAYS (added 2026-10-23)
  Ratings are built from this season AND last season's play-by-play (last
  season enters at CONFIG's prior_season_weight), and calibration is fit
  the same way one season back. Before this, only the current season was
  loaded: week 1 had no projection and weeks 2-4 ran off one to three
  games. 2021-2025 walk-forward, with the injury adjustment: RMSE 13.22 ->
  13.06, ATS at 2+ pts 51.6% -> 52.1%, at 3+ 50.4% -> 53.6%; weeks 2-4 at
  2+ went from 45% to 49%.

FROZEN PRE-KICKOFF PROJECTIONS (added 2026-10-23)
  Once a game is played, its proj/ptot/inj values are the ones this file
  carried the last time it was built BEFORE kickoff (read back from the
  previous copy of the output file), flagged frz=1 -- so the Scorecard
  grades the number that was on the board, not one recomputed afterwards
  with injury data that arrived late. Games played before this existed (or
  with no earlier build) fall back to the walk-forward number.
"""
import json
import os
import re
import sys
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

from power_ratings import (load_pbp, load_games, build_ratings,
                           load_injury_discount_team_week, CONFIG,
                           POSITION_INJURY_WEIGHT, DEFAULT_POSITION_INJURY_WEIGHT,
                           STATUS_INJURY_WEIGHT, TEAM_FIX)

FIRST_FIT_WEEK = 5          # ratings before this are mostly prior


def _n(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), nd)


def _pbp_two(season):
    """This season's play-by-play plus last season's (last season enters the
    ratings at CONFIG['prior_season_weight']). A missing prior season just
    means this season alone, as before."""
    frames = []
    for s in (season - 1, season):
        try:
            frames.append(load_pbp([s]))
        except Exception as e:
            print(f"warning: no play-by-play for {s} ({e})", file=sys.stderr)
    if not frames:
        raise SystemExit(f"no play-by-play for {season}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# live injury report (ESPN) for the upcoming week
# ---------------------------------------------------------------------------
ESPN_INJ = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_ABBR = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}
TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def _nm(s):
    """Name key for matching ESPN to snap counts: no accents, punctuation or
    suffixes, lower case."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())


def _fix_team(ab):
    ab = str(ab or "").upper()
    ab = ESPN_ABBR.get(ab, ab)
    return TEAM_FIX.get(ab, ab)


def _espn_status(inj):
    """'Out' / 'Doubtful' / 'Questionable', or None for anything else (IR,
    PUP, suspended: those players are off the weekly report and already out
    of the ratings, the same rule the nflverse path follows)."""
    cands = [inj.get("status")]
    t = inj.get("type") or {}
    cands += [t.get("description"), t.get("name"), t.get("abbreviation")]
    for c in cands:
        c = str(c or "").lower()
        if not c:
            continue
        if "reserve" in c or c in ("ir", "pup", "susp", "suspension"):
            return None
        if "doubt" in c or c == "d":
            return "Doubtful"
        if "question" in c or c == "q":
            return "Questionable"
        if c == "out" or c == "o" or c.endswith("_out"):
            return "Out"
    return None


def espn_injury_rows():
    """Current injury designations from ESPN's public feed, as rows of
    (team, name key, position, status). Empty list on any failure."""
    try:
        req = urllib.request.Request(ESPN_INJ, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"warning: ESPN injuries unavailable ({e})", file=sys.stderr)
        return []
    rows = []
    for tm in data.get("injuries") or []:
        team = _fix_team(tm.get("abbreviation")) if tm.get("abbreviation") else \
            TEAM_NAMES.get(tm.get("displayName") or tm.get("name") or "", "")
        for inj in tm.get("injuries") or []:
            a = inj.get("athlete") or {}
            t = team or _fix_team((a.get("team") or {}).get("abbreviation"))
            st = _espn_status(inj)
            nm = a.get("displayName") or " ".join(
                x for x in (a.get("firstName"), a.get("lastName")) if x)
            if not t or not st or not nm:
                continue
            pos = ((a.get("position") or {}).get("abbreviation")) or ""
            rows.append(dict(team=t, key=_nm(nm), name=nm, position=pos, status=st))
    print(f"ESPN injuries: {len(rows)} Out/Doubtful/Questionable rows across "
          f"{len(set(r['team'] for r in rows))} teams", file=sys.stderr)
    return rows


def espn_discounts(season, before_week):
    """{team: raw discount} for the upcoming week from ESPN, scored the same
    way as load_injury_discount_team_week(): position weight x status weight
    x trailing snap share (mean of his last CONFIG['injury_trailing_weeks']
    games this season, from snap counts before `before_week`)."""
    rows = espn_injury_rows()
    if not rows:
        return {}
    try:
        sn = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/"
                             f"download/snap_counts/snap_counts_{season}.parquet")
    except Exception as e:
        print(f"warning: snap counts unavailable ({e})", file=sys.stderr)
        return {}
    sn = sn.dropna(subset=["week"])
    sn = sn[sn.week.astype(int) < before_week].copy()
    sn["team"] = sn["team"].replace(TEAM_FIX)
    sn["share"] = sn[["offense_pct", "defense_pct", "st_pct"]].max(axis=1)
    sn["key"] = sn["player"].map(_nm)
    sn = sn.sort_values("week")
    tw = CONFIG["injury_trailing_weeks"]
    trail = (sn.groupby(["team", "key"])["share"]
               .apply(lambda x: float(x.tail(tw).mean())).to_dict())
    anyteam = (sn.groupby("key")["share"]
                 .apply(lambda x: float(x.tail(tw).mean())).to_dict())
    out, matched = {}, 0
    for r in rows:
        share = trail.get((r["team"], r["key"]))
        if share is None:
            share = anyteam.get(r["key"], 0.0)   # traded mid-season
        else:
            matched += 1
        pw = POSITION_INJURY_WEIGHT.get(r["position"].upper(), DEFAULT_POSITION_INJURY_WEIGHT)
        out[r["team"]] = out.get(r["team"], 0.0) + pw * STATUS_INJURY_WEIGHT[r["status"]] * share
    print(f"ESPN injuries: {matched}/{len(rows)} matched to snap counts", file=sys.stderr)
    return out


def fit_scale_hfa(season):
    """
    Regress actual margin on raw rating difference for one completed season.
    Returns (scale, hfa, n). Walk-forward: each week is projected from
    ratings built only on earlier weeks.
    """
    pbp = _pbp_two(season)          # same inputs the live ratings get
    g = load_games()
    g = g[(g.season == season) & (g.game_type == "REG")]
    g = g.dropna(subset=["home_score", "away_score"])
    if g.empty:
        return None

    cur = pbp[(pbp.season == season) & (pbp.season_type == "REG")]
    maxwk = int(cur.week.max())
    raw, act = [], []
    tx, ty = [], []      # team-points rows: [offense, opposing defense, home]
    for w in range(FIRST_FIT_WEEK, maxwk + 1):
        r = build_ratings(pbp, season, w, CONFIG)
        for _, x in g[g.week == w].iterrows():
            if x.home_team not in r.index or x.away_team not in r.index:
                continue
            # neutral-site games (London, Germany, Brazil...) have no home
            # field, so they'd only muddy the home-field fit
            if str(x.get("location", "Home")) == "Neutral":
                continue
            raw.append(float(r.loc[x.home_team, "rating"] -
                             r.loc[x.away_team, "rating"]))
            act.append(float(x.home_score - x.away_score))
            h, a = x.home_team, x.away_team
            tx.append([float(r.loc[h, "offense"]), float(r.loc[a, "defense"]), 1.0])
            ty.append(float(x.home_score))
            tx.append([float(r.loc[a, "offense"]), float(r.loc[h, "defense"]), 0.0])
            ty.append(float(x.away_score))
    if len(raw) < 50:
        return None
    b = np.polyfit(np.array(raw), np.array(act), 1)
    # points a team scores = const + a*its offense + b*opposing defense + c*home
    X = np.c_[np.ones(len(tx)), np.array(tx)]
    pts_coef = np.linalg.lstsq(X, np.array(ty), rcond=None)[0]
    return float(b[0]), float(b[1]), len(raw), [float(v) for v in pts_coef]


def build(season, cache_dir=None, prev_path=None):
    season = int(season)
    g = load_games()
    g = g[(g.season == season) & (g.game_type == "REG")]
    if g.empty:
        raise SystemExit(f"no schedule for {season}")

    played = g.dropna(subset=["home_score"])
    played_through = int(played.week.max()) if len(played) else 0

    # --- ratings as they stand now -----------------------------------
    # This season's plays plus last season's (see the module docstring); before
    # week 1 that is last season alone, at prior-season weight -- the same
    # ordering, just scaled, as carrying last season's final ratings forward.
    pbp = _pbp_two(season)
    ratings = build_ratings(pbp, season, played_through + 1, CONFIG)

    ratings = ratings.copy()
    ratings["rating_rank"] = ratings.rating.rank(ascending=False).astype(int)
    ratings["off_rank"] = ratings.offense.rank(ascending=False).astype(int)
    ratings["def_rank"] = ratings.defense.rank(ascending=False).astype(int)

    teams = {}
    for t, row in ratings.iterrows():
        teams[t] = {k: _n(row[k]) for k in
                    ("rating", "offense", "defense", "off_pass", "off_rush",
                     "def_pass", "def_rush", "special")}
        teams[t].update({k: int(row[k]) for k in
                         ("rating_rank", "off_rank", "def_rank")})

    # --- calibration from a COMPLETED season -------------------------
    # Always calibrate on the PRIOR season. Fitting on the season being
    # projected inflates everything: in-sample, 2025 graded out near 67%
    # against the spread; fit on 2024 and tested honestly on 2025 it was
    # 53.8%, with model RMSE (12.89) worse than the closing line's (12.32).
    # The market is hard to beat and the numbers shipped here should say so.
    fit_season = season - 1
    fit = fit_scale_hfa(fit_season)
    if fit:
        scale, hfa, nfit, pts_coef = fit
    else:
        scale, hfa, nfit, pts_coef = 1.0, 1.6, 0, None
    print(f"calibration: scale={scale:.3f} hfa={hfa:+.2f} "
          f"(fit on {fit_season}, n={nfit})", file=sys.stderr)

    # --- starter-availability (injury) discount, per game/week ---------
    # Free nflverse injury reports + snap counts, same pipeline as
    # everything else here. Looked up per (season, week, team) rather than
    # baked into the shared `ratings` snapshot above, since it's a
    # one-off per-game adjustment (see the module docstring). Falls back
    # to an empty table (every lookup -> 0.0, i.e. no adjustment) if the
    # season's injuries/snap-counts files can't be loaded at all.
    try:
        discount = load_injury_discount_team_week([season])
    except Exception as e:
        print(f"warning: couldn't load injury discount ({e}); "
              f"proceeding with no injury adjustment", file=sys.stderr)
        discount = pd.DataFrame(columns=["season", "week", "team_abbr", "discount"])
    disc_idx = discount.set_index(["season", "week", "team_abbr"])["discount"]

    # each team's NEXT unplayed game: nflverse has nothing for it yet, so use
    # ESPN's live report (which always describes a team's next game). Keyed
    # per team rather than one league-wide week, so a Monday-night game still
    # pending doesn't hold back every other team's next-week report.
    unplayed = g[g.home_score.isna()]
    next_wk = int(unplayed.week.min()) if len(unplayed) else None
    next_by_team = {}
    for _, u in unplayed.sort_values("week").iterrows():
        next_by_team.setdefault(u.home_team, int(u.week))
        next_by_team.setdefault(u.away_team, int(u.week))
    have = {t for t, w in next_by_team.items() if (season, w, t) in disc_idx.index}
    live = {}
    if next_by_team and len(have) < 20:
        # snap shares from every game played so far (unplayed games have none)
        live = espn_discounts(season, max(next_by_team.values()))
        live = {t: v for t, v in live.items() if t not in have}

    def _injury_discount(wk, team):
        """(discount, source)"""
        try:
            return float(disc_idx.loc[(season, wk, team)]), "nflverse"
        except KeyError:
            pass
        if live and next_by_team.get(team) == wk:
            return float(live.get(team, 0.0)), "espn"   # 0 = on ESPN's feed, nobody listed
        return 0.0, None

    # --- previous copy of this file: pre-kickoff numbers to freeze ------
    prev = {}
    if prev_path and os.path.exists(prev_path):
        try:
            with open(prev_path) as f:
                pj = json.load(f)
            if int(pj.get("season", 0)) == season:
                prev = {x["id"]: x for x in pj.get("games", [])}
        except Exception as e:
            print(f"warning: couldn't read previous {prev_path} ({e})", file=sys.stderr)

    # --- walk-forward ratings for games already played ----------------
    # A played game must be projected from ratings built BEFORE it, not from
    # today's ratings (which already contain its result) — otherwise every
    # past projection looks psychic and the Scorecard flatters the model.
    wk_ratings = {}
    if played_through:
        for w in range(1, played_through + 1):
            try:
                wk_ratings[w] = build_ratings(pbp, season, w, CONFIG)
            except Exception as e:
                print(f"warning: week {w} pre-game ratings failed ({e}); "
                      f"using current ratings for it", file=sys.stderr)

    # --- games -------------------------------------------------------
    games = []
    for _, x in g.sort_values(["week", "gameday"]).iterrows():
        h, a = x.home_team, x.away_team
        proj = None
        ptot = None
        inj_home = inj_away = inj_adj = inj_src = None
        R = wk_ratings.get(int(x.week), ratings)
        # neutral site (international games etc.): no home-field points.
        # Before 2026-10-26 every game got them, including PIT-NO in Week 7.
        neutral = str(x.get("location", "Home")) == "Neutral"
        hfa_g = 0.0 if neutral else hfa
        if h in R.index and a in R.index:
            diff = float(R.loc[h, "rating"] - R.loc[a, "rating"])
            inj_home, s1 = _injury_discount(int(x.week), h)
            inj_away, s2 = _injury_discount(int(x.week), a)
            inj_src = s1 or s2
            diff_adj = diff + CONFIG["injury_weight"] * (inj_away - inj_home)
            proj = round(scale * diff_adj + hfa_g, 2)
            inj_adj = scale * CONFIG["injury_weight"] * (inj_away - inj_home)
            if pts_coef:
                # the model's own game total: both teams' projected points,
                # from offense vs the other side's defense (fit on the prior
                # season, walk-forward). Team points are then this total split
                # by the margin above: home = (total + proj)/2.
                c0, co, cd, ch = pts_coef
                hp = c0 + co * float(R.loc[h, "offense"]) + cd * float(R.loc[a, "defense"]) + (0.0 if neutral else ch)
                ap = c0 + co * float(R.loc[a, "offense"]) + cd * float(R.loc[h, "defense"])
                ptot = round(hp + ap, 1)

        # games.csv spread_line is from the HOME perspective, positive =
        # home favored. Keep that convention throughout.
        rec = {
            "id": x.game_id,
            "wk": int(x.week),
            "date": str(x.gameday),
            "away": a, "home": h,
            "spread": _n(x.spread_line, 1),
            "total": _n(x.total_line, 1),
            "ml_home": _n(x.home_moneyline, 0),
            "ml_away": _n(x.away_moneyline, 0),
            "roof": None if pd.isna(x.roof) else str(x.roof),
            "wind": _n(x.wind, 0),
            "temp": _n(x.temp, 0),
            "div": bool(x.div_game) if not pd.isna(x.div_game) else None,
            "neutral": neutral,
            "proj": proj,
            "ptot": ptot,
            # raw (unweighted) starter-availability discount, higher =
            # worse for that team that week -- exposed mainly so the UI
            # CAN show "adjusted for injuries" context if it wants to;
            # proj above already has CONFIG["injury_weight"] applied.
            "inj_home": _n(inj_home, 2),
            "inj_away": _n(inj_away, 2),
            "inj_adj": _n(inj_adj, 2),
            "inj_src": inj_src,
        }
        if not pd.isna(x.home_score):
            # freeze what the board showed before kickoff (module docstring)
            p0 = prev.get(x.game_id)
            if p0 and (p0.get("frz") or "margin" not in p0) and p0.get("proj") is not None:
                for k in ("proj", "ptot", "inj_home", "inj_away", "inj_adj", "inj_src"):
                    rec[k] = p0.get(k)
                rec["frz"] = 1
        if not pd.isna(x.home_score):
            rec.update({
                "hs": int(x.home_score), "as": int(x.away_score),
                "margin": int(x.home_score - x.away_score),
                "pts": int(x.home_score + x.away_score),
            })
        games.append(rec)

    return {
        "season": season,
        "through_week": played_through,
        "calibration": {"scale": round(scale, 4), "hfa": round(hfa, 3),
                        "fit_season": int(fit_season), "n": nfit,
                        "pts_coef": [round(v, 4) for v in pts_coef] if pts_coef else None},
        "teams": teams,
        "games": games,
    }


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2026"
    out = sys.argv[2] if len(sys.argv) > 2 else f"games-{season}.json"
    data = build(season, cache_dir=os.path.dirname(out) or ".", prev_path=out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    withline = sum(1 for x in data["games"] if x["spread"] is not None)
    print(f"wrote {out}: {len(data['games'])} games, {withline} with lines, "
          f"{os.path.getsize(out)/1024:.0f} KB", file=sys.stderr)
