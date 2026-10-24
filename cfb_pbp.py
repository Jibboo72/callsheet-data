#!/usr/bin/env python3
"""
Play-by-play layer for the college game projections.

The scores-only ratings (cfb_ratings.py) stay the base. On top of them each
FBS-vs-FBS game's projected margin adds two things the final score hides:

  dS   opponent-adjusted SUCCESS RATE: a weighted ridge regression on every
       scrimmage snap (success = cfbfastR's down-and-distance success flag),
       offense and defense dummies for every FBS team plus one shared rating
       for all non-FBS opponents, home-field term. Current-season snaps decay
       with a 40-week half-life (so barely); last season's snaps count at 10%.
       Snaps with win probability under 2% or over 98% are dropped.
  fl   FUMBLE-RECOVERY LUCK: fumbles a team recovered (its own and the
       opponent's) minus half of all fumbles in its games, per game, this
       season. Recovering fumbles is close to a coin flip, so a team that
       has won a lot of them has been lucky and the scores flatter it.

    margin = c1 * scores-model net gap + c2 * home + c3 * dS + c4 * fl

with a separate set of c's for weeks 1-3, 4-6 and 7+ (the scores model is
mostly last season early on). Every setting was chosen on 2019-2022 only:
the ridge settings by fitting the blend on 2019-20 and scoring 2021-22, then
tested once on 2023-2025 against closing lines. Held-out result, all FBS
games with a line (2,398):

    miss per game   scores only 12.50   with this layer 12.45   line 12.00
    weeks 1-3       13.13 -> 13.03      weeks 7+ 12.57 -> 12.50
    model side, edge 3+ pts: 49.7% -> 51.0%

It helped in each of the three test seasons. It does not beat the closing
line. Tried and dropped: raw EPA per play (made things worse, alone or
blended), explosiveness, pass/rush success split, a QB-change flag.

The coefficients shipped below were refit on 2019-2025 with the same
settings. Data: cfbfastR play-by-play (sportsdataverse), free, no key.
"""
import math
import os
import sys

import numpy as np
import pandas as pd

PBP_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/"
           "download/cfbfastR_cfb_pbp/play_by_play_{season}.parquet")
CACHE = os.environ.get("CFB_CACHE", "cache")

SR_CFG = dict(alpha=50.0, hl=40.0, p1=0.1, gt=0.02)
# [scores-model net gap, home (not neutral), success-rate gap, fumble luck gap]
COEF = {0: [0.9371, 3.0847, 24.9706, -0.245],      # weeks 1-3
        1: [0.7335, 2.3878, 25.4665, -1.9721],     # weeks 4-6
        2: [0.7249, 2.2490, 33.1284, -2.3273]}     # weeks 7+ and bowls

COLS = ["game_id", "pos_team", "def_pos_team", "rush", "pass", "EPA", "success",
        "wp_before", "play_type"]
FUM_OWN = {"Fumble Recovery (Own)"}
FUM_LOST = {"Fumble Recovery (Opponent)", "Fumble Recovery (Opponent) Touchdown",
            "Fumble Return Touchdown"}


def group(wk):
    return 0 if wk <= 3 else (1 if wk <= 6 else 2)


def load_pbp(season):
    """cfbfastR play-by-play for one season, only the columns used here.
    Cached as a slim parquet; the current season is re-downloaded each run."""
    os.makedirs(CACHE, exist_ok=True)
    slim = os.path.join(CACHE, f"cfb_pbp_slim_{season}.parquet")
    fresh = os.environ.get("CFB_PBP_FRESH", "")
    if os.path.exists(slim) and str(season) not in fresh.split(","):
        return pd.read_parquet(slim)
    raw = os.path.join(CACHE, f"cfb_pbp_raw_{season}.parquet")
    import urllib.request
    req = urllib.request.Request(PBP_URL.format(season=season),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(raw, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    import pyarrow.parquet as pq
    have = [c for c in COLS if c in pq.ParquetFile(raw).schema.names]
    df = pd.read_parquet(raw, columns=have)
    df.to_parquet(slim)
    try:
        os.remove(raw)
    except Exception:
        pass
    return df


def _ridge(off, dfn, home, y, w, k, alpha):
    """Weighted ridge on [off dummies | def dummies | home | intercept], all but
    the intercept penalised (same as power_ratings._weighted_ridge)."""
    m = 2 * k + 2
    A = np.zeros((m, m))
    b = np.zeros(m)
    wh = w * home
    so = np.bincount(off, weights=w, minlength=k)
    sd = np.bincount(dfn, weights=w, minlength=k)
    cross = np.bincount(off * k + dfn, weights=w, minlength=k * k).reshape(k, k)
    A[:k, :k] = np.diag(so)
    A[k:2 * k, k:2 * k] = np.diag(sd)
    A[:k, k:2 * k] = cross
    A[k:2 * k, :k] = cross.T
    A[:k, 2 * k] = A[2 * k, :k] = np.bincount(off, weights=wh, minlength=k)
    A[k:2 * k, 2 * k] = A[2 * k, k:2 * k] = np.bincount(dfn, weights=wh, minlength=k)
    A[:k, 2 * k + 1] = A[2 * k + 1, :k] = so
    A[k:2 * k, 2 * k + 1] = A[2 * k + 1, k:2 * k] = sd
    A[2 * k, 2 * k] = (w * home * home).sum()
    A[2 * k, 2 * k + 1] = A[2 * k + 1, 2 * k] = wh.sum()
    A[2 * k + 1, 2 * k + 1] = w.sum()
    wy = w * y
    b[:k] = np.bincount(off, weights=wy, minlength=k)
    b[k:2 * k] = np.bincount(dfn, weights=wy, minlength=k)
    b[2 * k] = (wy * home).sum()
    b[2 * k + 1] = wy.sum()
    P = np.eye(m) * alpha
    P[-1, -1] = 0.0
    return np.linalg.solve(A + P, b)


class PbpLayer:
    """Success-rate ratings and fumble luck as of any week of `season`.

    sched / prev_sched: cr.load_schedule() for this season and last (games
    with at least one FBS team; `wk` is the sortable week, bowls 16+)."""

    def __init__(self, season, sched, prev_sched=None, pbp=None, prev_pbp=None):
        self.season = season
        fbs = (set(sched.loc[sched.home_division == "fbs", "home_team"])
               | set(sched.loc[sched.away_division == "fbs", "away_team"]))
        self.fbs = fbs
        self.teams = sorted(fbs | {"_FCS"})
        self.idx = {t: i for i, t in enumerate(self.teams)}
        if pbp is None:
            pbp = load_pbp(season)
        self.cur = self._plays(sched, pbp)
        self.prev = None
        if prev_sched is not None and SR_CFG["p1"] > 0:
            try:
                pp = prev_pbp if prev_pbp is not None else load_pbp(season - 1)
                self.prev = self._plays(prev_sched, pp)
            except Exception as e:
                print(f"pbp layer: no {season - 1} play-by-play ({e})", file=sys.stderr)
        self._fum = self._fumbles(sched, pbp)
        self._sr = {}

    def _plays(self, sched, p):
        g = sched[["game_id", "wk", "home_team", "neutral"]]
        p = p[p.game_id.isin(g.game_id)]
        sc = (((p["rush"] == 1) | (p["pass"] == 1)) & p.EPA.notna()
              & p.pos_team.notna() & p.def_pos_team.notna())
        p = p[sc].merge(g, on="game_id", how="left")
        home = ((p.pos_team == p.home_team) & ~p.neutral.astype(bool)).astype(float)
        pos = p.pos_team.where(p.pos_team.isin(self.fbs), "_FCS")
        dfn = p.def_pos_team.where(p.def_pos_team.isin(self.fbs), "_FCS")
        return pd.DataFrame({"wk": p.wk.astype(int).values,
                             "oi": pos.map(self.idx).astype(int).values,
                             "di": dfn.map(self.idx).astype(int).values,
                             "home": home.values,
                             "y": p.success.astype(float).fillna(0.0).values,
                             "wp": p.wp_before.astype(float).values})

    def _fumbles(self, sched, p):
        g = sched[["game_id", "wk"]]
        p = p[p.game_id.isin(g.game_id)]
        pt = p.play_type.fillna("")
        a = pd.DataFrame({"game_id": p.game_id, "team": p.pos_team,
                          "own": pt.isin(FUM_OWN).astype(int),
                          "lost": pt.isin(FUM_LOST).astype(int)})
        base = a.groupby(["game_id", "team"]).agg(own=("own", "sum"),
                                                  lost=("lost", "sum")).reset_index()
        m = base.merge(base, on="game_id", suffixes=("", "_o"))
        m = m[m.team != m.team_o]
        m["fl"] = (m.own + m.lost_o) - 0.5 * (m.own + m.lost + m.own_o + m.lost_o)
        return m.merge(g, on="game_id")[["team", "wk", "fl"]]

    def sr_net(self, wk):
        """team -> opponent-adjusted success-rate net (offense minus defense
        allowed), from snaps before week `wk`."""
        if wk in self._sr:
            return self._sr[wk]
        c = SR_CFG
        lam = math.log(2) / c["hl"]
        cur = self.cur[self.cur.wk < wk]
        parts = [(cur, np.exp(-lam * (wk - cur.wk.values)))]
        if self.prev is not None:
            parts.append((self.prev, np.full(len(self.prev), c["p1"])))
        d = pd.concat([x for x, _ in parts], ignore_index=True)
        w = np.concatenate([x for _, x in parts])
        if c["gt"] > 0:
            keep = ((d.wp >= c["gt"]) & (d.wp <= 1 - c["gt"])) | d.wp.isna()
            d, w = d[keep.values], w[keep.values]
        k = len(self.teams)
        if len(d) < 50:
            out = {t: 0.0 for t in self.teams}
        else:
            beta = _ridge(d.oi.values, d.di.values, d.home.values, d.y.values, w, k, c["alpha"])
            o, df = beta[:k], beta[k:2 * k]
            net = (o - o.mean()) - (df - df.mean())
            out = {t: float(net[i]) for t, i in self.idx.items()}
        self._sr[wk] = out
        return out

    def fum_luck(self, team, wk):
        x = self._fum[(self._fum.team == team) & (self._fum.wk < wk)]
        return float(x.fl.mean()) if len(x) else 0.0

    def margin(self, home, away, neutral, wk, score_margin, hfa):
        """Projected home margin with the play-by-play layer. score_margin is
        cfb_ratings.project()'s margin (which already includes hfa)."""
        c = COEF[group(int(wk))]
        hi = 0.0 if neutral else 1.0
        snet = score_margin - hfa * hi
        sr = self.sr_net(int(wk))
        h = home if home in self.fbs else "_FCS"
        a = away if away in self.fbs else "_FCS"
        ds = sr.get(h, 0.0) - sr.get(a, 0.0)
        fl = self.fum_luck(home, wk) - self.fum_luck(away, wk)
        return c[0] * snet + c[1] * hi + c[2] * ds + c[3] * fl
