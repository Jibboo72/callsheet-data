"""
Test CPOE/RYOE (step 2 of "tighten the formula") the same honest walk-
forward way test_formula_weights.py tests the offense/defense weighting.

    python3 test_cpoe_ryoe.py

WHY
---
Next Gen Stats CPOE (completion % above expectation) and RYOE (rush yards
over expectation per attempt) are free -- same nflverse pipeline already
used for play-by-play, no extra cost. power_ratings.py now opponent-adjusts
them the same way EPA/success rate are adjusted (fit_ngs_phase(), at
team-week grain instead of play grain) and can blend them in as a third
signal. All four new weights -- w_cpoe_off, w_cpoe_def, w_ryoe_off,
w_ryoe_def -- default to 0.0 in CONFIG, so nothing about the live ratings
has changed yet. This script checks whether turning any of them up
actually helps OUT-OF-SAMPLE prediction for THIS model, same discipline as
step 1 (test_formula_weights.py), before any default gets touched.

HOW
---
_fit_components() -- the ridge fits, including the new NGS fit -- is the
slow part and only runs ONCE per (season, week). combine_components() --
blending the fitted components into a final rating -- is cheap (array math
over ~32 teams, no refit), so it's re-run once per weight candidate per
(season, week) against the SAME cached fit. Each candidate then gets its
own rolling scale/HFA calibration (prior seasons only), the same no-look-
ahead discipline as the other two backtests.

The four weights are swept ONE AT A TIME, holding the other three (and
rating_off_weight/rating_def_weight) at CONFIG's current defaults -- this
isolates what each signal does on its own before anyone stacks them.

Needs nfl_open_close.csv (from run_all.py) in the working directory.
2020 excluded throughout. NGS only goes back to 2016 -- 2015 contributes no
CPOE/RYOE signal, fit_ngs_phase() just returns 0 for those weeks rather
than failing.
"""
import numpy as np
import pandas as pd
from power_ratings import (load_pbp, load_games, load_ngs_pass_team_week,
                           load_ngs_rush_team_week, _fit_components,
                           combine_components, CONFIG)

EXCLUDE = {2020}
ALL_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021]
TEST = [2017, 2018, 2019, 2021]
WINDOW = 3
FIRST_WEEK = 5

CPOE_OFF_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
CPOE_DEF_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
RYOE_OFF_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
RYOE_DEF_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


# ------------------------------------------------------------------ data
def load_lines(path='nfl_open_close.csv'):
    ln = pd.read_csv(path)
    ln = ln[~ln.season.isin(EXCLUDE)]
    ln = ln[~ln.neutral.fillna(False)]
    ln['open_home'] = -ln.open_spread      # flip to home-favored-positive
    ln['close_home'] = -ln.close_spread
    ln['actual'] = ln.home_score - ln.away_score
    return ln.dropna(subset=['open_home', 'close_home', 'actual'])


def build_component_cache(pbp, ngs_pass, ngs_rush, seasons):
    """{(season, week): components dict} -- the one expensive pass. Every
    weight candidate below reuses this unchanged."""
    cache = {}
    for s in seasons:
        maxwk = int(pbp[pbp.season == s].week.max())
        for w in range(FIRST_WEEK, maxwk + 1):
            cache[(s, w)] = _fit_components(pbp, s, w, CONFIG,
                                            ngs_pass=ngs_pass, ngs_rush=ngs_rush)
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
    """Rolling-calibrated walk-forward score -- same shape as the other two
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
    print('loading Next Gen Stats (CPOE/RYOE -- free, same nflverse pipeline '
          'as play-by-play)...')
    ngs_pass = load_ngs_pass_team_week(ALL_SEASONS)
    ngs_rush = load_ngs_rush_team_week(ALL_SEASONS)

    print('fitting components (one ridge fit per season/week, including the '
          'NGS fit -- this is the only slow part, and only happens once)...')
    cache = build_component_cache(pbp, ngs_pass, ngs_rush, ALL_SEASONS)
    print(f'{len(cache)} (season, week) fits cached')

    sweep('CPOE, offense', 'w_cpoe_off', CPOE_OFF_WEIGHTS, cache, lines)
    sweep('CPOE, defense', 'w_cpoe_def', CPOE_DEF_WEIGHTS, cache, lines)
    sweep('RYOE, offense', 'w_ryoe_off', RYOE_OFF_WEIGHTS, cache, lines)
    sweep('RYOE, defense', 'w_ryoe_def', RYOE_DEF_WEIGHTS, cache, lines)

    print(f"\nbreak-even on a -110 bet is 52.4% -- only trust an ATS number "
          f"with a real sample size (n). Each sweep above picks the best "
          f"RESULT across its own candidates on a fixed test set, which "
          f"overstates how good the winner will look on NEW seasons -- and "
          f"these four were tested ONE AT A TIME, holding the others at 0, "
          f"so a win on each axis separately doesn't guarantee they still "
          f"help once stacked together. Treat any winner as a hypothesis "
          f"for next season to re-check, not a settled answer.")
