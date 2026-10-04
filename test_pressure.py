"""
Test charted pass-pressure rate (step 3 of "tighten the formula") the same
honest walk-forward way test_cpoe_ryoe.py tests CPOE/RYOE.

    python3 test_pressure.py

WHY
---
PFR's weekly advanced passing stats have a real charted outcome -- was the
QB pressured on this dropback -- free, same nflverse pipeline as
everything else. (FTN's own charting release was checked first, since the
original plan was "charted pressure rate, cross-checked with PFR" -- but
FTN's files only have pass-rush CONTEXT (n_pass_rushers, n_blitzers), no
pressure OUTCOME column, so there's nothing there to cross-check against.
See load_pfr_pressure_team_week()'s docstring in power_ratings.py.)

power_ratings.py opponent-adjusts this the same way CPOE/RYOE are (team-
week grain, fit_ngs_phase()) and can blend it into both pass offense
(pressure allowed -- pass-block quality) and pass defense (pressure
generated -- pass-rush quality). Both weights -- w_prss_off, w_prss_def --
default to 0.0, so nothing about the live ratings has changed yet. This
script checks whether turning either on actually helps OUT-OF-SAMPLE
prediction for THIS model, same discipline as steps 1 and 2.

HOW
---
Same shape as test_cpoe_ryoe.py: _fit_components() (the ridge fits,
including the new pressure fit) runs ONCE per (season, week) and is
cached; combine_components() (cheap array math, no refit) is re-run once
per weight candidate against that cache. Each candidate gets its own
rolling scale/HFA calibration (prior seasons only).

The two weights are swept ONE AT A TIME, holding the other (and every
other signal's weight) at CONFIG's current defaults.

Needs nfl_open_close.csv (from run_all.py) in the working directory.
2020 excluded throughout. PFR's advstats only goes back to 2018 -- 2015-17
contribute no pressure signal, fit_ngs_phase() just returns 0 for those
weeks rather than failing.
"""
import numpy as np
import pandas as pd
from power_ratings import (load_pbp, load_games, load_pfr_pressure_team_week,
                           _fit_components, combine_components, CONFIG)

EXCLUDE = {2020}
ALL_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021]
TEST = [2017, 2018, 2019, 2021]
WINDOW = 3
FIRST_WEEK = 5

PRSS_OFF_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
PRSS_DEF_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


# ------------------------------------------------------------------ data
def load_lines(path='nfl_open_close.csv'):
    ln = pd.read_csv(path)
    ln = ln[~ln.season.isin(EXCLUDE)]
    ln = ln[~ln.neutral.fillna(False)]
    ln['open_home'] = -ln.open_spread      # flip to home-favored-positive
    ln['close_home'] = -ln.close_spread
    ln['actual'] = ln.home_score - ln.away_score
    return ln.dropna(subset=['open_home', 'close_home', 'actual'])


def build_component_cache(pbp, pfr_prss, seasons):
    """{(season, week): components dict} -- the one expensive pass. Every
    weight candidate below reuses this unchanged."""
    cache = {}
    for s in seasons:
        maxwk = int(pbp[pbp.season == s].week.max())
        for w in range(FIRST_WEEK, maxwk + 1):
            cache[(s, w)] = _fit_components(pbp, s, w, CONFIG, pfr_prss=pfr_prss)
    return cache


def games_for_weight(cache, lines, cfg):
    """Recombine the cached components at this cfg and attach the matching
    lines -- the cheap step, repeated once per weight candidate."""
    rows = []
    for (s, w), c in cache.items():
        r = combine_components(c, cfg)
        g = lines[(lines.season == s) & (lines.week == w)]
        for _, x in g.iterrows():
            if x.home not in r.index or x.away not in r.index:
                continue
            rows.append({
                'season': s, 'week': w,
                'raw': r.loc[x.home, 'rating'] - r.loc[x.away, 'rating'],
                'open': x.open_home, 'close': x.close_home, 'actual': x.actual,
            })
    return pd.DataFrame(rows)


def fit(raw, actual):
    b = np.polyfit(raw.to_numpy(), actual.to_numpy(), 1)
    return float(b[0]), float(b[1])


# ------------------------------------------------------------------ scoring
def score(df):
    """Rolling-calibrated walk-forward score -- same shape as the other
    backtests -- for one already-recombined (season, week, raw, open,
    close, actual) table."""
    parts = []
    for s in TEST:
        prior = [p for p in ALL_SEASONS if p < s][-WINDOW:]
        train = df[df.season.isin(prior)]
        sB, hB = fit(train.raw, train.actual)
        d = df[df.season == s].copy()
        d['proj'] = d.raw * sB + hB
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
    return dict(rmse=rmse, clv=clv, ats0=ats0, n0=n0, ats2=ats2, n2=n2)


def sweep(label, key, weights, cache, lines):
    print(f'\n--- {label} ({key}) ---')
    print(f"{'weight':>7}  {'RMSE':>6}  {'CLV':>6}  {'ATS all':>9}  {'n':>4}  "
          f"{'ATS edge>=2':>12}  {'n':>4}")
    results = []
    for val in weights:
        cfg = dict(CONFIG)
        cfg[key] = val
        df = games_for_weight(cache, lines, cfg)
        r = score(df)
        r[key] = val
        results.append(r)
        ats0 = f'{r["ats0"]*100:7.1f}%' if r['ats0'] is not None else '     —'
        ats2 = f'{r["ats2"]*100:7.1f}%' if r['ats2'] is not None else '     —'
        print(f"{val:7.2f}  {r['rmse']:6.2f}  {r['clv']*100:5.1f}%  "
              f"{ats0:>9}  {r['n0']:4d}  {ats2:>12}  {r['n2']:4d}")

    cur = next(r for r in results if r[key] == 0.0)
    best_rmse = min(results, key=lambda r: r['rmse'])
    scored = [r for r in results if r['ats2'] is not None and r['n2'] >= 25]
    best_ats = max(scored, key=lambda r: r['ats2']) if scored else None
    print(f"current default ({key}=0.0): RMSE {cur['rmse']:.2f}")
    print(f"lowest RMSE: {key}={best_rmse[key]:.2f} "
          f"(RMSE {best_rmse['rmse']:.2f}, {best_rmse['rmse']-cur['rmse']:+.2f} vs current)")
    if best_ats:
        print(f"best ATS @ edge>=2: {key}={best_ats[key]:.2f} "
              f"({best_ats['ats2']*100:.1f}%, n={best_ats['n2']})")
    return results


if __name__ == '__main__':
    lines = load_lines()
    g = load_games()[['season', 'week', 'home_team', 'away_team']]
    lines = lines.merge(g, left_on=['season', 'home', 'away'],
                        right_on=['season', 'home_team', 'away_team'],
                        how='inner')
    print(f'lines joined to schedule: {len(lines)} games')

    pbp = load_pbp(ALL_SEASONS)
    print('loading PFR charted pressure rate (free, same nflverse pipeline '
          'as play-by-play)...')
    pfr_prss = load_pfr_pressure_team_week(pbp, ALL_SEASONS)
    print(f'{len(pfr_prss)} team-week pressure rows (PFR starts 2018 -- '
          '2015-17 contribute none, by design)')

    print('fitting components (one ridge fit per season/week, including the '
          'pressure fit -- this is the only slow part, and only happens '
          'once)...')
    cache = build_component_cache(pbp, pfr_prss, ALL_SEASONS)
    print(f'{len(cache)} (season, week) fits cached')

    sweep('pressure, offense (pass-block quality)', 'w_prss_off',
          PRSS_OFF_WEIGHTS, cache, lines)
    sweep('pressure, defense (pass-rush quality)', 'w_prss_def',
          PRSS_DEF_WEIGHTS, cache, lines)

    print(f"\nbreak-even on a -110 bet is 52.4% -- only trust an ATS number "
          f"with a real sample size (n). Each sweep above picks the best "
          f"RESULT across its own candidates on a fixed test set, which "
          f"overstates how good the winner will look on NEW seasons -- and "
          f"offense/defense were tested ONE AT A TIME, holding the other at "
          f"0, so a win on one axis doesn't guarantee it still helps once "
          f"stacked with the other. Treat any winner as a hypothesis for "
          f"next season to re-check, not a settled answer.")
