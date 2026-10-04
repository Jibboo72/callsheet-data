"""
Export full NFL power ratings for the Call Sheet's Power tab.

    python3 export_ratings.py 2025 12        # season, week (ratings AS OF)
    python3 export_ratings.py                # latest available

Writes ratings.json -- one entry per team, sorted best to worst.

Per team:
  rating                overall power rating, in points
  rank                   1 = best overall
  offense, defense       phase totals (defense: higher = better defense)
  off_pass, off_rush     offense split
  def_pass, def_rush     defense split (higher = better defense)
  special                special teams rating

This is the same engine that already drives the Sides tab's projected
spread (project_spread) and the Props tab's matchup grades
(export_matchup.py) -- this script just exposes the team table itself,
which neither of those surfaces.
"""
import json
import sys

import pandas as pd
from power_ratings import load_pbp, build_ratings, CONFIG


def build_export(season, week):
    pbp = load_pbp([season])
    r = build_ratings(pbp, season, week, CONFIG)

    cols = ['rating', 'offense', 'defense', 'off_pass', 'off_rush',
            'def_pass', 'def_rush', 'special']

    teams = []
    for rank, (t, row) in enumerate(r[cols].iterrows(), start=1):
        entry = {'team': t, 'rank': rank}
        entry.update({c: round(float(row[c]), 2) for c in cols})
        teams.append(entry)

    return {'kind': 'ratings', 'season': int(season), 'through_week': int(week),
            'n_teams': len(teams), 'teams': teams}


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        season, week = int(sys.argv[1]), int(sys.argv[2])
    else:
        season = 2025
        pbp = load_pbp([season])
        week = int(pbp[pbp.season_type == 'REG'].week.max()) + 1

    out = build_export(season, week)
    with open('ratings.json', 'w') as f:
        json.dump(out, f, indent=1, sort_keys=True)

    print(f"wrote ratings.json  ({out['n_teams']} teams, "
          f"{out['season']} through week {out['through_week']})")

    d = pd.DataFrame(out['teams'])
    print('\ntop 5 overall:')
    print(d.nsmallest(5, 'rank')[['team', 'rating', 'rank']].to_string(index=False))
    print('\nbottom 5 overall:')
    print(d.nlargest(5, 'rank')[['team', 'rating', 'rank']].to_string(index=False))
