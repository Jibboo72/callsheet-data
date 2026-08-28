"""
Export team matchup context for the Call Sheet.

    python3 export_matchup.py 2025 12        # season, week (ratings AS OF)
    python3 export_matchup.py                # latest available

Writes matchup.json -- one entry per team, keyed by nflverse abbreviation
so it joins straight onto defteam / posteam in build_receivers.py.

Per team:
  def_pass, def_pass_rank    pass defense rating (higher = better defense)
  def_rush, def_rush_rank
  off_pass, off_rush         offense side, for opponent context
  rating,   rating_rank      overall team strength
  plays_pg                   neutral-script plays per game  (pace)
  pass_rate                  neutral-script pass rate       (volume tilt)

The two pace fields matter as much as the ratings for receptions and
longest-play props: a fast, pass-heavy opponent inflates everyone's
target volume regardless of how good the defense is.

Neutral script = win probability 20-80%, 1st-3rd down, excludes the
final two minutes of each half, so it measures intent rather than
score-chasing.
"""
import json
import sys

import numpy as np
import pandas as pd
from power_ratings import load_pbp, build_ratings, CONFIG


def pace_and_tendency(pbp, season, upto_week):
    """Neutral-script plays per game and pass rate, per offense."""
    d = pbp[(pbp.season == season) & (pbp.week < upto_week)]
    d = d[(d.qb_dropback == 1) | (d.rush == 1)]
    d = d[d.wp.between(0.20, 0.80)]
    d = d[d.down.isin([1, 2, 3])]
    d = d[d.half_seconds_remaining > 120]
    d = d[d.qb_kneel != 1]

    g = d.groupby('posteam')
    games = g.game_id.nunique()
    out = pd.DataFrame({
        'plays_pg': g.size() / games,
        'pass_rate': g.qb_dropback.mean(),
    })
    return out


def build_export(season, week):
    pbp = load_pbp([season])
    r = build_ratings(pbp, season, week, CONFIG)
    pace = pace_and_tendency(pbp, season, week)

    df = r.join(pace, how='left')

    # rank: 1 = best. defense ratings are already "higher is better"
    df['def_pass_rank'] = df.def_pass.rank(ascending=False).astype(int)
    df['def_rush_rank'] = df.def_rush.rank(ascending=False).astype(int)
    df['rating_rank'] = df.rating.rank(ascending=False).astype(int)

    cols = ['def_pass', 'def_pass_rank', 'def_rush', 'def_rush_rank',
            'off_pass', 'off_rush', 'rating', 'rating_rank',
            'plays_pg', 'pass_rate']

    teams = {}
    for t, row in df[cols].iterrows():
        teams[t] = {c: (None if pd.isna(row[c])
                        else (int(row[c]) if c.endswith('_rank')
                              else round(float(row[c]), 3)))
                    for c in cols}

    return {'season': int(season), 'through_week': int(week),
            'n_teams': len(teams), 'teams': teams}


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        season, week = int(sys.argv[1]), int(sys.argv[2])
    else:
        season = 2025
        pbp = load_pbp([season])
        week = int(pbp[pbp.season_type == 'REG'].week.max()) + 1

    out = build_export(season, week)
    with open('matchup.json', 'w') as f:
        json.dump(out, f, indent=1, sort_keys=True)

    print(f"wrote matchup.json  ({out['n_teams']} teams, "
          f"{out['season']} through week {out['through_week']})")

    d = pd.DataFrame(out['teams']).T
    print('\ntoughest pass defenses:')
    print(d.nsmallest(5, 'def_pass_rank')[['def_pass', 'def_pass_rank']]
          .to_string())
    print('\nsoftest pass defenses:')
    print(d.nlargest(5, 'def_pass_rank')[['def_pass', 'def_pass_rank']]
          .to_string())
    print('\nfastest / most pass-happy (neutral script):')
    print(d.nlargest(5, 'plays_pg')[['plays_pg', 'pass_rate']].to_string())
