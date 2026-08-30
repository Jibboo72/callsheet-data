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
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from power_ratings import load_pbp, load_games, build_ratings, CONFIG

FIRST_FIT_WEEK = 5          # ratings before this are mostly prior


def _n(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), nd)


def fit_scale_hfa(season):
    """
    Regress actual margin on raw rating difference for one completed season.
    Returns (scale, hfa, n). Walk-forward: each week is projected from
    ratings built only on earlier weeks.
    """
    pbp = load_pbp([season])
    g = load_games()
    g = g[(g.season == season) & (g.game_type == "REG")]
    g = g.dropna(subset=["home_score", "away_score"])
    if g.empty:
        return None

    maxwk = int(pbp[pbp.season_type == "REG"].week.max())
    raw, act = [], []
    for w in range(FIRST_FIT_WEEK, maxwk + 1):
        r = build_ratings(pbp, season, w, CONFIG)
        for _, x in g[g.week == w].iterrows():
            if x.home_team not in r.index or x.away_team not in r.index:
                continue
            raw.append(float(r.loc[x.home_team, "rating"] -
                             r.loc[x.away_team, "rating"]))
            act.append(float(x.home_score - x.away_score))
    if len(raw) < 50:
        return None
    b = np.polyfit(np.array(raw), np.array(act), 1)
    return float(b[0]), float(b[1]), len(raw)


def build(season, cache_dir=None):
    season = int(season)
    g = load_games()
    g = g[(g.season == season) & (g.game_type == "REG")]
    if g.empty:
        raise SystemExit(f"no schedule for {season}")

    played = g.dropna(subset=["home_score"])
    played_through = int(played.week.max()) if len(played) else 0

    # --- ratings as they stand now -----------------------------------
    pbp = load_pbp([season]) if played_through else None
    if pbp is not None and played_through:
        ratings = build_ratings(pbp, season, played_through + 1, CONFIG)
    else:
        # season hasn't started: carry last season's final ratings forward
        prev = load_pbp([season - 1])
        last = int(prev[prev.season_type == "REG"].week.max())
        ratings = build_ratings(prev, season - 1, last + 1, CONFIG)

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
        scale, hfa, nfit = fit
    else:
        scale, hfa, nfit = 1.0, 1.6, 0
    print(f"calibration: scale={scale:.3f} hfa={hfa:+.2f} "
          f"(fit on {fit_season}, n={nfit})", file=sys.stderr)

    # --- games -------------------------------------------------------
    games = []
    for _, x in g.sort_values(["week", "gameday"]).iterrows():
        h, a = x.home_team, x.away_team
        proj = None
        if h in ratings.index and a in ratings.index:
            diff = float(ratings.loc[h, "rating"] - ratings.loc[a, "rating"])
            proj = round(scale * diff + hfa, 2)

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
            "proj": proj,
        }
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
                        "fit_season": int(fit_season), "n": nfit},
        "teams": teams,
        "games": games,
    }


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2026"
    out = sys.argv[2] if len(sys.argv) > 2 else f"games-{season}.json"
    data = build(season, cache_dir=os.path.dirname(out) or ".")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    withline = sum(1 for x in data["games"] if x["spread"] is not None)
    print(f"wrote {out}: {len(data['games'])} games, {withline} with lines, "
          f"{os.path.getsize(out)/1024:.0f} KB", file=sys.stderr)
