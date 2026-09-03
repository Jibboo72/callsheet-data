#!/usr/bin/env python3
"""
College football schedule for the Saturday board.

    python3 export_cfb.py 2026 data/cfb-2026.json

WHAT THIS DOES AND DOESN'T HAVE
  Schedules, kickoff times, conferences and final scores — free, no account.

  It does NOT have spreads or totals. Unlike the NFL, where nflverse
  games.csv ships lines going back to 1999, college lines live behind the
  CollegeFootballData API and need a (free) registered key. Rather than
  make you register for one, the app lets you type the line in when you
  place the bet. If you ever want lines pulled automatically, that's the
  one thing that would change.

  No ratings and no projection either — this is a board for seeing games
  and logging bets, not a model.
"""
import json
import os
import sys

import pandas as pd

SCHED = ("https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/"
         "main/schedules/parquet/cfb_schedules_{season}.parquet")


def _s(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def build(season):
    df = pd.read_parquet(SCHED.format(season=season))
    df = df[df.season_type.isin(["regular", "postseason"])]
    # FBS only — 3,800 games including every FCS matchup is not a board,
    # it's a phone book.
    df = df[(df.home_division == "fbs") & (df.away_division == "fbs")]

    games = []
    for _, x in df.sort_values(["week", "start_date"]).iterrows():
        rec = {
            "id": str(x.game_id),
            "wk": int(x.week),
            "date": _s(x.start_date)[:10],
            "kick": _s(x.start_date)[11:16],
            "away": _s(x.away_team), "home": _s(x.home_team),
            "aconf": _s(x.away_conference), "hconf": _s(x.home_conference),
            "neutral": bool(x.neutral_site) if not pd.isna(x.neutral_site) else False,
            "conf_game": bool(x.conference_game) if not pd.isna(x.conference_game) else False,
        }
        if not pd.isna(x.home_points):
            rec.update({"hs": int(x.home_points), "as": int(x.away_points),
                        "margin": int(x.home_points - x.away_points),
                        "pts": int(x.home_points + x.away_points)})
        games.append(rec)

    return {"season": int(season), "kind": "cfb", "count": len(games),
            "games": games}


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2026"
    out = sys.argv[2] if len(sys.argv) > 2 else f"cfb-{season}.json"
    data = build(season)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    played = sum(1 for g in data["games"] if "hs" in g)
    print(f"wrote {out}: {data['count']} FBS games, {played} played, "
          f"{os.path.getsize(out)/1024:.0f} KB", file=sys.stderr)
