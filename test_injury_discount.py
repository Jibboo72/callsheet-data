"""
Test the starter-availability (injury) discount -- step 4 of "tighten the
formula" -- the same honest walk-forward way the other three steps were
tested.

    python3 test_injury_discount.py

WHY
---
Steps 1-3 (offense/defense weighting, CPOE/RYOE, charted pressure) all
came back the same way: no real out-of-sample edge, because they're each
mostly re-explaining information the EPA+success-rate model already
captures from the plays themselves. A missing starter is different in
kind -- it's not an outcome stat computed from plays that were run, it's
information the play-level model has NO way to see at all (next week's
Week-19 Josh Allen isn't in this week's pbp). So this one has an actual
shot at being additive rather than redundant.

HOW THIS DIFFERS FROM STEPS 1-3
--------------------------------
CPOE/RYOE/pressure are persistent TEAM-QUALITY signals, opponent-adjusted
and blended into build_ratings() itself. An injury discount is the
opposite: a one-off, game-specific point adjustment -- "Mahomes is out
THIS week" says nothing about next week. So it is NOT blended into the
ridge at all. Instead it rides on project_spread()'s existing home_adj/
away_adj parameters (already there in power_ratings.py, unused until now)
as an ADDITIVE term on top of the normal rating diff, before scale/HFA
calibration -- same no-look-ahead rolling calibration test_vs_open.py
uses, just with one extra predictor.

raw_for_weight(games, injury_weight) =
    (home.rating - away.rating) + injury_weight * (away.discount - home.discount)

A team having its OWN injuries subtracts from its own side (discount_home
makes home worse), which is why away.discount is added and home.discount
is subtracted -- more away-team injuries help the home side, exactly like
a real spread adjustment would.

Needs nfl_open_close.csv (from run_all.py) in the working directory.
2020 excluded throughout. See load_injury_discount_team_week() in
power_ratings.py for the discount itself.
"""
import numpy as np
import pandas as pd
from power_ratings import (load_pbp, load_games, build_ratings,
                           load_injury_discount_team_week, CONFIG)

EXCLUDE = {2020}
ALL_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021]
TEST = [2017, 2018, 2019, 2021]
WINDOW = 3
FIRST_WEEK = 5

# Discount values run roughly 0-2.6 with a median around 0.4 (see
# load_injury_discount_team_week's docstring/output) -- a weight of 1.0
# here means "a typical week's worth of starter-availability gap is worth
# about half a rating point", which is a small, conservative starting
# scale.
#
# First pass (0.0-5.0) came back with EVERY metric -- RMSE, CLV, ATS-all,
# ATS-edge>=2, and even the edge>=2 sample size -- improving together,
# monotonically, all the way out to 5.0, the top of that range. That's
# unlike steps 1-3, which just bounced around noisily. But 5.0 was the
# edge of what got tested, with RMSE already flattening while CLV/ATS
# kept climbing -- so the true peak (or reversal) was still unknown.
# Extended much further out to find it: if this keeps improving forever,
# something is off (an unbounded "better" weight isn't a real result,
# it's a sign the backtest itself has a leak or degenerate case worth
# checking); if it peaks and comes back down, THAT peak is the real
# candidate, not whatever the edge of the old range happened to be.
INJURY_WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]


# ------------------------------------------------------------------ data
def load_lines(path='nfl_open_close.csv'):
    ln = pd.read_csv(path)
    ln = ln[~ln.season.isin(EXCLUDE)]
    ln = ln[~ln.neutral.fillna(False)]
    ln['open_home'] = -ln.open_spread      # flip to home-favored-positive
    ln['close_home'] = -ln.close_spread
    ln['actual'] = ln.home_score - ln.away_score
    return ln.dropna(subset=['open_home', 'close_home', 'actual'])


def walk_forward_games(pbp, lines, discount, seasons):
    """Per game: raw rating diff (scale=1, hfa=0, exactly like
    test_vs_open.py's walk_forward_raw) PLUS each side's injury discount,
    kept separate so injury_weight can be swept after the fact without
    re-fitting the ridge."""
    disc = discount.set_index(['season', 'week', 'team_abbr'])['discount']

    def look_up(season, week, team):
        try:
            return float(disc.loc[(season, week, team)])
        except KeyError:
            return 0.0

    rows = []
    for s in seasons:
        maxwk = int(pbp[pbp.season == s].week.max())
        for w in range(FIRST_WEEK, maxwk + 1):
            r = build_ratings(pbp, s, w, CONFIG)
            g = lines[(lines.season == s) & (lines.week == w)]
            for _, x in g.iterrows():
                if x.home not in r.index or x.away not in r.index:
                    continue
                rows.append({
                    'season': s, 'week': w,
                    'rating_diff': r.loc[x.home, 'rating'] - r.loc[x.away, 'rating'],
                    'disc_home': look_up(s, w, x.home),
                    'disc_away': look_up(s, w, x.away),
                    'open': x.open_home, 'close': x.close_home, 'actual': x.actual,
                })
    return pd.DataFrame(rows)


def raw_for_weight(df, injury_weight):
    return df.rating_diff + injury_weight * (df.disc_away - df.disc_home)


def fit(raw, actual):
    b = np.polyfit(raw.to_numpy(), actual.to_numpy(), 1)
    return float(b[0]), float(b[1])


# ------------------------------------------------------------------ scoring
def score(df, injury_weight):
    parts = []
    for s in TEST:
        prior = [p for p in ALL_SEASONS if p < s][-WINDOW:]
        train = df[df.season.isin(prior)]
        sB, hB = fit(raw_for_weight(train, injury_weight), train.actual)
        d = df[df.season == s].copy()
        d['proj'] = raw_for_weight(d, injury_weight) * sB + hB
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)

    edge = out.proj - out.open
    cover = out.actual - out.open
    mv = out.close - out.open

    rmse = float(np.sqrt(((out.proj - out.actual) ** 2).mean()))
    clv = float((np.sign(edge[mv != 0]) == np.sign(mv[mv != 0])).mean())

    def ats_at(lo):
        m = edge.abs() >= lo
        if m.sum() < 25:
            return None, int(m.sum())
        won = np.where(edge[m] > 0, cover[m] > 0, cover[m] < 0)
        return float(won.mean()), int(m.sum())

    ats0, n0 = ats_at(0)
    ats2, n2 = ats_at(2)
    return dict(injury_weight=injury_weight, rmse=rmse, clv=clv,
                ats0=ats0, n0=n0, ats2=ats2, n2=n2)


if __name__ == '__main__':
    lines = load_lines()
    g = load_games()[['season', 'week', 'home_team', 'away_team']]
    lines = lines.merge(g, left_on=['season', 'home', 'away'],
                        right_on=['season', 'home_team', 'away_team'],
                        how='inner')
    print(f'lines joined to schedule: {len(lines)} games')

    pbp = load_pbp(ALL_SEASONS)
    print('loading injuries + snap counts (free, same nflverse pipeline)...')
    discount = load_injury_discount_team_week(ALL_SEASONS)
    print(f'{len(discount)} team-week discount rows')

    print('building walk-forward games (one ridge fit per season/week -- '
          'this is the slow part, and only happens once)...')
    games = walk_forward_games(pbp, lines, discount, ALL_SEASONS)
    print(f'{len(games)} test games with both a rating and a line\n')

    results = [score(games, w) for w in INJURY_WEIGHTS]

    print(f"{'weight':>7}  {'RMSE':>6}  {'CLV':>6}  {'ATS all':>9}  {'n':>4}  "
          f"{'ATS edge>=2':>12}  {'n':>4}")
    for r in results:
        ats0 = f'{r["ats0"]*100:7.1f}%' if r['ats0'] is not None else '     —'
        ats2 = f'{r["ats2"]*100:7.1f}%' if r['ats2'] is not None else '     —'
        print(f"{r['injury_weight']:7.2f}  {r['rmse']:6.2f}  {r['clv']*100:5.1f}%  "
              f"{ats0:>9}  {r['n0']:4d}  {ats2:>12}  {r['n2']:4d}")

    cur = next(r for r in results if r['injury_weight'] == 0.0)
    best_rmse = min(results, key=lambda r: r['rmse'])
    scored = [r for r in results if r['ats2'] is not None and r['n2'] >= 25]
    best_ats = max(scored, key=lambda r: r['ats2']) if scored else None

    print(f"\ncurrent default (injury_weight=0.0, i.e. off): RMSE {cur['rmse']:.2f}")
    print(f"lowest RMSE: injury_weight={best_rmse['injury_weight']:.2f} "
          f"(RMSE {best_rmse['rmse']:.2f}, {best_rmse['rmse']-cur['rmse']:+.2f} vs current)")
    if best_ats:
        print(f"best ATS @ edge>=2: injury_weight={best_ats['injury_weight']:.2f} "
              f"({best_ats['ats2']*100:.1f}%, n={best_ats['n2']})")
    print(f"\nbreak-even on a -110 bet is 52.4% -- only trust an ATS number "
          f"with a real sample size (n), and remember this table is picking "
          f"the best RESULT across {len(INJURY_WEIGHTS)} candidates on a "
          f"fixed test set, which overstates how good the winner will look "
          f"on NEW seasons. Treat the winner as a hypothesis for next "
          f"season to re-check, not a settled answer.")
