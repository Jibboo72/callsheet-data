#!/usr/bin/env python3
"""
Walk-forward backtest of the FCS model against FCS-vs-FCS closing lines.

Same engine as the FBS model (cfb_ratings.py) run over every Division I game,
so an FCS team is rated off its whole schedule. Graded only on games between
two FCS teams -- those are the ones the FCS tab is for.

    python3 backtest_fcs.py           # report with cfb_ratings.FCS_CONFIG
    python3 backtest_fcs.py tune      # tune on 2022-2024, report 2025 held out

Lines for FCS-vs-FCS games exist in the free file from 2022 on (~550-660 a
season, mostly ESPN Bet / William Hill), so that's the test window.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import cfb_ratings as cr

TRAIN = [2022, 2023, 2024]
TEST = [2025]


def run(cfg, seasons, lines, first=2018):
    cfg = {**cr.FCS_CONFIG, **cfg}
    prior = None
    for s in range(first, min(seasons)):
        prior = cr.to_prior(cr.fit(cr.load_schedule(s, "di"), cfg, prior), prior, cfg)
    out = []
    for s in seasons:
        g = cr.load_schedule(s, "di")
        for wk in sorted(g[g.played].wk.unique()):
            m = cr.fit(g, cfg, prior, as_of_wk=wk)
            t = m["teams"].set_index("team")
            for _, r in g[(g.wk == wk) & g.played].iterrows():
                if r.home_division != "fcs" or r.away_division != "fcs":
                    continue
                h, a = t.loc[r.home_team], t.loc[r.away_team]
                out.append(dict(season=s, wk=wk, game_id=int(r.game_id),
                                pred=h.net - a.net + (0 if r.neutral else cfg["hfa"]),
                                pred_tot=2 * m["avg"] + h.off + a["def"] + a.off + h["def"],
                                actual=r.home_points - r.away_points,
                                actual_tot=r.home_points + r.away_points))
        prior = cr.to_prior(cr.fit(g, cfg, prior), prior, cfg)
    return pd.DataFrame(out).merge(lines, on="game_id", how="left")


def lined(df):
    return df[df.line_spread.notna()]


def score(df):
    d = lined(df)
    return ((d.pred - d.actual).abs().mean(), (-d.line_spread - d.actual).abs().mean(), len(d))


def ats(df):
    d = lined(df).copy()
    lm = -d.line_spread
    d["edge"], d["cover"] = d.pred - lm, d.actual - lm
    d = d[d.cover != 0]
    rows = []
    for lo in (0, 2, 3, 5, 7, 10):
        x = d[d.edge.abs() >= lo]
        w = int((np.sign(x.edge) == np.sign(x.cover)).sum())
        rows.append(dict(edge=lo, n=len(x), w=w, pct=round(w / len(x), 3) if len(x) else None))
    return rows


def summary(df, label):
    mm, ml, n = score(df)
    d = lined(df)
    return dict(label=label, games=int(n), mae_model=round(mm, 2), mae_line=round(ml, 2),
                sd=round(float((d.pred - d.actual).std()), 1),
                su_model=round(float((np.sign(d.pred) == np.sign(d.actual)).mean()), 3),
                su_line=round(float((np.sign(-d.line_spread) == np.sign(d.actual)).mean()), 3),
                ats=ats(df))


def report(s):
    print(f"\n== {s['label']}: {s['games']} FCS-vs-FCS games with a line ==")
    print(f"MAE  model {s['mae_model']}   line {s['mae_line']}   sd {s['sd']}")
    print(f"winners  model {s['su_model']:.1%}   line {s['su_line']:.1%}")
    for r in s["ats"]:
        pct = "—" if r["pct"] is None else f"{r['pct']:.1%}"
        print(f"  edge>={r['edge']:>2}  {r['n']:5d}  {r['w']:5d}  {pct}")


GRID = dict(
    alpha=[1, 2, 3, 5],
    alpha_fcs=[1, 2, 3, 5],
    carryover=[0.7, 0.85, 1.0],
    carry2=[0.0, 0.3, 0.5],
    hfa=[2.0, 2.5, 3.0, 3.5],
    margin_cap=[17, 21, 28],
    tail_keep=[0.5, 0.75, 1.0],
)


def tune(lines):
    cfg = dict(cr.FCS_CONFIG)
    best = score(run(cfg, TRAIN, lines))[0]
    print(f"start MAE {best:.3f}", flush=True)
    for rnd in range(2):
        for k, vals in GRID.items():
            for v in vals:
                if v == cfg[k]:
                    continue
                s = score(run({**cfg, k: v}, TRAIN, lines))[0]
                if s < best - 1e-4:
                    best, cfg = s, {**cfg, k: v}
                    print(f"  r{rnd} {k}={v}  MAE {s:.3f}", flush=True)
    return cfg


if __name__ == "__main__":
    lines = cr.load_lines()
    cfg = dict(cr.FCS_CONFIG)
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        cfg = tune(lines)
        print("\nTUNED FCS CONFIG:", {k: cfg[k] for k in GRID})
    tr = summary(run(cfg, TRAIN, lines), f"{TRAIN[0]}-{TRAIN[-1]}")
    te = summary(run(cfg, TEST, lines), f"{TEST[0]} (held out)")
    report(tr)
    report(te)
    os.makedirs("data", exist_ok=True)
    with open("data/fcs-backtest.json", "w") as f:
        json.dump(dict(train=tr, test=te), f)
    print("wrote data/fcs-backtest.json")
