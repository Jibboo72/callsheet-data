#!/usr/bin/env python3
"""
Current rosters for the call sheet — names, teams, and bio only.

    python3 export_roster.py 2026 data/roster-2026.json

WHY THIS IS SEPARATE FROM build_receivers.py
  Receiving logs come from play-by-play, which doesn't exist until games
  are played. Rosters are published months earlier. Between the last game
  of one season and the first of the next, every player's team in the
  receivers file is stale — after the 2025 season, 48 of 251 receivers
  had changed teams. A wrong team means a wrong next opponent and a wrong
  pass-defense read, so this keeps that current in the gap.

  The file deliberately carries NO "log" key. The app only overwrites a
  player's game log when the incoming file actually has one, so importing
  this updates team and bio while leaving every logged game intact.
"""
import json
import os
import sys

import pandas as pd

ROSTER = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "rosters/roster_{season}.csv")

# Pass catchers only. Nobody is betting receptions on a long snapper.
POSITIONS = {"WR", "TE", "RB", "FB", "QB"}


def _height(v):
    try:
        n = int(float(v))
        return f"{n // 12}-{n % 12}" if 60 <= n <= 90 else str(v)
    except (TypeError, ValueError):
        return "" if pd.isna(v) else str(v)


def _s(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def build(season):
    r = pd.read_csv(ROSTER.format(season=season), low_memory=False)
    r = r.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    r = r[r.position.isin(POSITIONS)]

    players = {}
    for _, row in r.iterrows():
        players[row.gsis_id] = {
            "name": _s(row.get("full_name")),
            "team": _s(row.get("team")),
            "pos": _s(row.get("position")),
            "ht": _height(row.get("height")),
            "wt": _s(row.get("weight")).replace(".0", ""),
            "college": _s(row.get("college")),
            "exp": _s(row.get("years_exp")).replace(".0", ""),
        }
    return {"season": int(season), "kind": "roster",
            "count": len(players), "players": players}


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2026"
    out = sys.argv[2] if len(sys.argv) > 2 else f"roster-{season}.json"
    data = build(season)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out}: {data['count']} players, "
          f"{os.path.getsize(out)/1024:.0f} KB", file=sys.stderr)
