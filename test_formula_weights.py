"""
Test offense-vs-defense weighting in the final power rating, the same
honest walk-forward way test_vs_open.py tests scale and HFA.

    python3 test_formula_weights.py

WHY
---
build_ratings() currently combines offense and defense 1:1 into the final
`rating` (CONFIG["rating_off_weight"] = CONFIG["rating_def_weight"] = 1.0).
But offensive efficiency is a stickier, more predictive signal than
defensive efficiency:
  - Brian Burke's predictivity research found offensive pass efficiency is
    the single most predictive team stat, with defensive stats much
    noisier year-over-year.
  - nfelo's own EPA tiers weight offensive EPA 1.6x against defensive
    EPA's 1.0x for exactly this reason.
This script checks whether up-weighting offense actually improves
OUT-OF-SAMPLE prediction for THIS model, rather than assuming some other
site's number transfers over.

HOW
---
build_ratings() is called ONCE per (season, week) in the walk-forward
loop -- the ridge regression itself doesn't change with the weight being
tested. What changes is cheap: Offense and Defense are kept as SEPARATE
columns and recombined with different weights after the fact, so testing
a dozen weight candidates costs about the same as testing one.

Each weight candidate then gets its OWN rolling scale/HFA calibration
(prior seasons only, never the test season itself) -- the same no-look-
ahead discipline as test_vs_open.py's version B, because a weighting that
only looks good with hindsight-fit scale/HFA isn't a real result.

Needs nfl_open_close.csv (produced by run_all.py) in the working
directory -- same input test_vs_open.py uses. 2020 excluded throughout.
"""
import numpy as np
import pandas as pd
from power_ratings import load_pbp, load_games, build_ratings, CONFIG

EXCLUDE = {2020}
ALL_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021]
TEST = [2017, 2018, 2019, 2021]
WINDOW = 3            # how many prior seasons to calibrate scale/HFA on
FIRST_WEEK = 5
# Defense held at 1.0 throughout -- only testing how much MORE offense
# should count relative to it, matching how nfelo frames their own 1.6:1.
OFF_WEIGHTS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]


# ------------------------------------------------------------------ data
def load_lines(path='nfl_open_close.csv'):
    ln = pd.read_csv(path)
    ln = ln[~ln.season.isin(EXCLUDE)]
    ln = ln[~ln.neutral.fillna(False)]
    ln['open_home'] = -ln.open_spread      # flip to home-favored-positive
    ln['close_home'] = -ln.close_spread
    ln['actual'] = ln.home_score - ln.away_score
    return ln.dropna(subset=['open_home', 'close_home', 'actual'])


def walk_forward_components(pbp, lines, seasons):
    """Per game: home/away offense, defense, special -- kept SEPARATE so
    weighting can be swept after the fact without re-fitting anything."""
    rows = []
    for s in seasons:
        maxwk = int(pbp[pbp.season == s].week.max())
        for w in range(FIRST_WEEK, maxwk + 1):
            r = build_ratings(pbp, s, w, CONFIG)   # one fit, reused for every weight
            g = lines[(lines.season == s) & (lines.week == w)]
            for _, x in g.iterrows():
                if x.home not in r.index or x.away not in r.index:
                    continue
                h, a = r.loc[x.home], r.loc[x.away]
                rows.append({
                    'season': s, 'week': w, 'home': x.home, 'away': x.away,
                    'h_off': h.offense, 'h_def': h.defense, 'h_st': h.special,
                    'a_off': a.offense, 'a_def': a.defense, 'a_st': a.special,
                    'open': x.open_home, 'close': x.close_home, 'actual': x.actual,
                })
    return pd.DataFrame(rows)


def raw_for_weight(df, w_off, w_def=1.0):
    """Recombine the stored components at a given offense weight -- this
    is the whole reason the components were kept separate above."""
    return (w_off * (df.h_off - df.a_off) + w_def * (df.h_def - df.a_def)
            + (df.h_st - df.a_st))


def fit(raw, actual):
    b = np.polyfit(raw.to_numpy(), actual.to_numpy(), 1)
    return float(b[0]), float(b[1])


# ------------------------------------------------------------------ scoring
def score(df, w_off):
    """Rolling-calibrated walk-forward score for one weight candidate."""
    parts = []
    for s in TEST:
        prior = [p for p in ALL_SEASONS if p < s][-WINDOW:]
        train = df[df.season.isin(prior)]
        sB, hB = fit(raw_for_weight(train, w_off), train.actual)
        d = df[df.season == s].copy()
        d['proj'] = raw_for_weight(d, w_off) * sB + hB
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
    return dict(w_off=w_off, rmse=rmse, clv=clv,
                ats0=ats0, n0=n0, ats2=ats2, n2=n2)


if __name__ == '__main__':
    lines = load_lines()
    g = load_games()[['season', 'week', 'home_team', 'away_team']]
    lines = lines.merge(g, left_on=['season', 'home', 'away'],
                        right_on=['season', 'home_team', 'away_team'],
                        how='inner')
    print(f'lines joined to schedule: {len(lines)} games')

    pbp = load_pbp(ALL_SEASONS)
    print('building walk-forward components (one ridge fit per season/week '
          '-- this is the slow part, and only happens once)...')
    comp = walk_forward_components(pbp, lines, ALL_SEASONS)
    print(f'{len(comp)} test games with both a rating and a line\n')

    results = [score(comp, w) for w in OFF_WEIGHTS]

    print(f"{'off wt':>7}  {'RMSE':>6}  {'CLV':>6}  {'ATS all':>9}  {'n':>4}  "
          f"{'ATS edge>=2':>12}  {'n':>4}")
    for r in results:
        ats0 = f'{r["ats0"]*100:7.1f}%' if r['ats0'] is not None else '     —'
        ats2 = f'{r["ats2"]*100:7.1f}%' if r['ats2'] is not None else '     —'
        print(f"{r['w_off']:7.2f}  {r['rmse']:6.2f}  {r['clv']*100:5.1f}%  "
              f"{ats0:>9}  {r['n0']:4d}  {ats2:>12}  {r['n2']:4d}")

    cur = next(r for r in results if r['w_off'] == 1.0)
    best_rmse = min(results, key=lambda r: r['rmse'])
    scored = [r for r in results if r['ats2'] is not None and r['n2'] >= 25]
    best_ats = max(scored, key=lambda r: r['ats2']) if scored else None

    print(f"\ncurrent default (1.0, equal weight): RMSE {cur['rmse']:.2f}")
    print(f"lowest RMSE: offense weight {best_rmse['w_off']:.2f} "
          f"(RMSE {best_rmse['rmse']:.2f}, {best_rmse['rmse']-cur['rmse']:+.2f} vs current)")
    if best_ats:
        print(f"best ATS @ edge>=2: offense weight {best_ats['w_off']:.2f} "
              f"({best_ats['ats2']*100:.1f}%, n={best_ats['n2']})")
    print(f"\nbreak-even on a -110 bet is 52.4% -- only trust an ATS number "
          f"with a real sample size (n), and remember this whole table is "
          f"picking the best RESULT across {len(OFF_WEIGHTS)} candidates on "
          f"a fixed test set, which overstates how good the winner will look "
          f"on NEW seasons. Treat the winner as a hypothesis for next "
          f"season to re-check, not a settled answer.")
