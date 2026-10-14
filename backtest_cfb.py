#!/usr/bin/env python3
"""
Walk-forward backtest of cfb_ratings.py against real CLOSING lines.

Every game is predicted using ONLY games played before its week, plus last
season's final ratings as the prior -- exactly what the model would have
known on that Saturday. No peeking.

    python3 backtest_cfb.py            # report with current CONFIG
    python3 backtest_cfb.py tune       # coordinate-descent tune on 2021-2024,
                                       # then report on 2025 (held out)

What the report tells you
  MAE model / MAE line   how far each misses the real margin. The closing
                         line is the toughest benchmark in sports; getting
                         within ~0.5 pt of it is good for a scores-only model.
  ATS by edge bucket     record betting the model's side when it disagrees
                         with the line by at least N points. 52.4% breaks
                         even at -110.
  sd_margin              the residual spread used to turn spreads into win %.
"""
import math
import sys

import numpy as np
import pandas as pd

import cfb_ratings as cr

TRAIN = [2021, 2022, 2023, 2024]
TEST = [2025]


def season_preds(season, cfg, prior, lines):
    g = cr.load_schedule(season)
    out = []
    for wk in sorted(g[g.played].wk.unique()):
        m = cr.fit(g, cfg, prior, as_of_wk=wk)
        t = m["teams"].set_index("team")
        hfa = cfg["hfa"]
        wg = g[(g.wk == wk) & g.played]
        for _, r in wg.iterrows():
            h, a = t.loc[r.home_team], t.loc[r.away_team]
            mg = h.net - a.net + (0 if r.neutral else hfa)
            tot = 2 * m["avg"] + h.off + a["def"] + a.off + h["def"]
            out.append(dict(season=season, wk=wk, game_id=int(r.game_id),
                            fbs2=(r.home_division == "fbs") and (r.away_division == "fbs"),
                            pred=mg, pred_tot=tot,
                            actual=r.home_points - r.away_points,
                            actual_tot=r.home_points + r.away_points))
    final = cr.fit(g, cfg, prior)
    return pd.DataFrame(out), cr.to_prior(final, prior, cfg)


def run(cfg, seasons, lines, first=2018):
    cfg = {**cr.CONFIG, **cfg}
    prior = None
    for s in range(first, min(seasons)):
        prior = cr.to_prior(cr.fit(cr.load_schedule(s), cfg, prior), prior, cfg)
    frames = []
    for s in seasons:
        df, prior = season_preds(s, cfg, prior, lines)
        frames.append(df)
    df = pd.concat(frames).merge(lines, on="game_id", how="left")
    return df


def score(df):
    d = df[df.line_spread.notna() & df.fbs2]
    # line_spread is the HOME spread in betting convention (-7 = home favored
    # by 7); the line's projected home margin is its negative.
    line_m = -d.line_spread
    mae_m = (d.pred - d.actual).abs().mean()
    mae_l = (line_m - d.actual).abs().mean()
    return mae_m, mae_l, len(d)


def ats_table(df):
    d = df[df.line_spread.notna() & df.fbs2].copy()
    line_m = -d.line_spread
    d["edge"] = d.pred - line_m                 # + = model likes home more than line
    d["cover"] = d.actual - line_m               # + = home covered
    d = d[d.cover != 0]
    rows = []
    for lo in (0, 2, 3, 5, 7, 10):
        x = d[d.edge.abs() >= lo]
        win = (np.sign(x.edge) == np.sign(x.cover)).sum()
        n = len(x)
        rows.append((lo, n, win, win / n if n else float("nan")))
    return rows


def summary(df, label):
    mm, ml, n = score(df)
    d = df[df.line_spread.notna() & df.fbs2]
    return dict(label=label, games=int(n), mae_model=round(mm, 2), mae_line=round(ml, 2),
                sd=round(float((d.pred - d.actual).std()), 1),
                ats=[dict(edge=lo, n=int(nn), w=int(w), pct=round(p, 3))
                     for lo, nn, w, p in ats_table(df)])


def report(df, label):
    mm, ml, n = score(df)
    d = df[df.line_spread.notna() & df.fbs2]
    print(f"\n== {label}: {n} FBS-vs-FBS games with a closing line ==")
    print(f"MAE  model {mm:5.2f}   closing line {ml:5.2f}   gap {mm - ml:+.2f}")
    blend = 0.5 * d.pred + 0.5 * (-d.line_spread)
    print(f"MAE  50/50 model+line {(blend - d.actual).abs().mean():5.2f}")
    sd = (d.pred - d.actual).std()
    print(f"residual sd (-> sd_margin): {sd:.1f}")
    t = d[d.line_total.notna()]
    if len(t):
        print(f"Totals MAE  model {(t.pred_tot - t.actual_tot).abs().mean():5.2f}"
              f"   line {(t.line_total - t.actual_tot).abs().mean():5.2f}")
    print(" edge>=  games   wins   ATS%")
    for lo, nn, w, p in ats_table(df):
        print(f"  {lo:>4}  {nn:6d} {w:6d}  {p:6.1%}")
    by = d.assign(e=(d.pred - d.actual).abs()).groupby(
        pd.cut(d.wk, [0, 3, 6, 9, 30], labels=["wk1-3", "wk4-6", "wk7-9", "wk10+"]),
        observed=True).e.mean()
    print("MAE by week:", "  ".join(f"{k} {v:.2f}" for k, v in by.items()))


GRID = dict(
    margin_cap=[21, 24, 28, 35, 45],
    tail_keep=[0.35, 0.5, 0.6, 0.75, 0.9],
    half_life_weeks=[12, 20, 40, 60, 200],
    alpha=[1, 1.5, 2, 3],
    alpha_fcs=[0.5, 1, 2],
    carryover=[0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    carry2=[0.0, 0.15, 0.3, 0.45],
    hfa=[2.0, 2.5, 3.0],
)


def tune(lines):
    cfg = dict(cr.CONFIG)
    best = score(run(cfg, TRAIN, lines))[0]
    print(f"start MAE {best:.3f}")
    for rnd in range(2):
        for k, vals in GRID.items():
            for v in vals:
                if v == cfg[k]:
                    continue
                trial = {**cfg, k: v}
                s = score(run(trial, TRAIN, lines))[0]
                if s < best - 1e-4:
                    best, cfg = s, trial
                    print(f"  r{rnd} {k}={v}  MAE {s:.3f}", flush=True)
    return cfg


if __name__ == "__main__":
    lines = cr.load_lines()
    cfg = dict(cr.CONFIG)
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        cfg = tune(lines)
        print("\nTUNED CONFIG:", {k: cfg[k] for k in GRID})
    tr = run(cfg, TRAIN, lines)
    report(tr, f"TRAIN {TRAIN[0]}-{TRAIN[-1]}")
    te = run(cfg, TEST, lines)
    report(te, f"HELD-OUT {TEST[0]}")
    import json, os
    os.makedirs("data", exist_ok=True)
    with open("data/cfb-backtest.json", "w") as f:
        json.dump(dict(train=summary(tr, f"{TRAIN[0]}-{TRAIN[-1]}"),
                       test=summary(te, f"{TEST[0]} (held out)")), f)
    print("wrote data/cfb-backtest.json")
