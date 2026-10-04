"""
Walk-forward test against OPENING lines, with rolling recalibration.

    python3 test_vs_open.py

What changed from the first version:
  The old test fit scale and HFA once on 2015-16 and froze them through
  2021. Home advantage fell sharply over that period, so every later
  projection carried a stale home bias -- which would produce losses that
  widen with edge size, exactly what we saw.

  Now scale and HFA are refit before each test season using only the
  seasons before it (rolling window). No look-ahead: season N is always
  projected with constants fit on data that ended before season N started.

  Both versions print, so the frozen numbers and the rolling numbers sit
  side by side and we can see whether the fix actually mattered.

2020 excluded throughout.
"""
import numpy as np
import pandas as pd
from power_ratings import (load_pbp, load_games, build_ratings,
                           project_spread, CONFIG)

EXCLUDE = {2020}
ALL_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021]
TEST = [2017, 2018, 2019, 2021]
WINDOW = 3            # how many prior seasons to calibrate on
FIRST_WEEK = 5


# ------------------------------------------------------------------ data
def load_lines(path='nfl_open_close.csv'):
    ln = pd.read_csv(path)
    ln = ln[~ln.season.isin(EXCLUDE)]
    ln = ln[~ln.neutral.fillna(False)]
    ln['open_home'] = -ln.open_spread      # flip to home-favored-positive
    ln['close_home'] = -ln.close_spread
    ln['actual'] = ln.home_score - ln.away_score
    return ln.dropna(subset=['open_home', 'close_home', 'actual'])


def walk_forward_raw(pbp, lines, seasons):
    """Raw projected margin (scale=1, hfa=0) using only prior-week data."""
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
                    'season': s, 'week': w, 'home': x.home, 'away': x.away,
                    'raw': project_spread(r, x.home, x.away,
                                          hfa=0.0, scale=1.0),
                    'open': x.open_home, 'close': x.close_home,
                    'actual': x.actual,
                })
    return pd.DataFrame(rows)


def fit(df):
    """Regress actual margin on raw projection -> (scale, hfa)."""
    b = np.polyfit(df.raw.to_numpy(), df.actual.to_numpy(), 1)
    return float(b[0]), float(b[1])


# ------------------------------------------------------------------ report
def ols_t(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    s2 = (r ** 2).sum() / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return coef, se


def report(df, label):
    edge = df.proj - df.open
    mv = df.close - df.open
    cover = df.actual - df.open

    print(f'\n{"="*58}\n{label}   n={len(df)}\n{"="*58}')
    print(f'RMSE  model {np.sqrt(((df.proj-df.actual)**2).mean()):.2f}'
          f'   opener {np.sqrt((cover**2).mean()):.2f}'
          f'   closer {np.sqrt(((df.close-df.actual)**2).mean()):.2f}')

    ok = mv.abs() > 0
    b = np.polyfit(edge, mv, 1)
    resid = mv - (b[0] * edge + b[1])
    se = np.sqrt((resid**2).sum() / (len(mv)-2) / ((edge-edge.mean())**2).sum())
    print(f'CLV   moves our way {(np.sign(edge[ok])==np.sign(mv[ok])).mean():.1%}'
          f'   beta={b[0]:.3f} t={b[0]/se:.2f}')

    X = np.column_stack([df.open, df.proj, np.ones(len(df))])
    coef, ses = ols_t(X, df.actual.to_numpy())
    print(f'ENCOMP opener t={coef[0]/ses[0]:5.2f}   '
          f'model t={coef[1]/ses[1]:5.2f}')

    print('ATS vs opener (break-even 52.4%)')
    for lo in range(0, 6):
        m = edge.abs() >= lo
        if m.sum() < 25:
            continue
        won = np.where(edge[m] > 0, cover[m] > 0, cover[m] < 0)
        n, w = len(won), int(won.sum())
        print(f'   edge>={lo}: {w:4d}-{n-w:<4d} {100*w/n:5.1f}%  '
              f'(+/-{100*np.sqrt(0.25/n):.1f} 1sd, n={n})')

    # directional split -- a stale HFA shows up as one side losing badly
    for side, m in [('picked HOME', edge > 0), ('picked AWAY', edge < 0)]:
        if m.sum() < 25:
            continue
        won = np.where(edge[m] > 0, cover[m] > 0, cover[m] < 0)
        print(f'   {side}: {won.mean():.1%}  (n={int(m.sum())})')


if __name__ == '__main__':
    lines = load_lines()
    g = load_games()[['season', 'week', 'home_team', 'away_team']]
    lines = lines.merge(g, left_on=['season', 'home', 'away'],
                        right_on=['season', 'home_team', 'away_team'],
                        how='inner')
    print(f'lines joined to schedule: {len(lines)} games')

    pbp = load_pbp(ALL_SEASONS)
    raw = walk_forward_raw(pbp, lines, ALL_SEASONS)

    # how much did home advantage actually move?
    print('\nempirical home margin by season:')
    for s, gg in raw.groupby('season'):
        print(f'   {s}: {gg.actual.mean():+.2f} pts   (opener implies '
              f'{gg.open.mean():+.2f})')

    # ---- A: frozen constants, the original approach -------------------
    sA, hA = fit(raw[raw.season.isin([2015, 2016])])
    print(f'\nfrozen calibration on 2015-16: scale={sA:.3f} hfa={hA:.2f}')
    a = raw[raw.season.isin(TEST)].copy()
    a['proj'] = a.raw * sA + hA
    report(a, 'A: FROZEN 2015-16 constants')

    # ---- B: rolling recalibration -------------------------------------
    parts = []
    print('\nrolling calibration actually used:')
    for s in TEST:
        prior = [p for p in ALL_SEASONS if p < s][-WINDOW:]
        sB, hB = fit(raw[raw.season.isin(prior)])
        print(f'   {s}: fit on {prior} -> scale={sB:.3f} hfa={hB:.2f}')
        d = raw[raw.season == s].copy()
        d['proj'] = d.raw * sB + hB
        parts.append(d)
    report(pd.concat(parts, ignore_index=True), 'B: ROLLING recalibration')
