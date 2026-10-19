#!/usr/bin/env python3
"""
Player-prop models for the Call Sheet's Props tab.

    python3 export_props.py 2026 data/props-2026.json

WHAT IT MAKES
  For every skill player on a team playing in the next unplayed week, a
  full probability distribution for five markets:

      atd     anytime touchdown (rushing or receiving)      one number, P(>=1)
      ptd     passing touchdowns (QBs)                       P(X = 0,1,2,...)
      car     rushing attempts                               P(X = 0,1,2,...)
      lrush   longest rush                                   P(X >= 1,2,3,...)
      combo   rushing + receiving yards                      P(X >= x) on a grid

  The app turns a distribution into the probability of ANY line the book
  posts, so nothing here is tied to one number.

HOW EACH ONE IS BUILT (kept simple on purpose -- every number has to be
explainable, and none of this has been tested against historical prop prices
because those do not exist for us; the app tracks its own picks going forward)
  * Player form is an exponentially weighted average of recent games (half
    life ~3 games, carried across the season boundary), shrunk toward the
    position average so a two-game sample can't masquerade as a role.
  * Only games the player was ACTIVE for (at least one touch) are used. Books
    void a prop when a player is inactive, so "chance given he plays" is the
    right thing to model.
  * Game script comes from the book's own spread and total: the favourite
    runs more and scores more. Team implied points = total/2 +- spread/2.
  * ptd  Poisson, rate = QB's recent share of his team's points that came
         through the air, times implied points.
  * car  negative binomial, mean = form x game-script factor (fit on 2025).
  * lrush  per-carry chance of a run of L+ yards (shrunk to position) mixed
         over the carries distribution.
  * combo  gamma, mean = form x script factor, spread fit on 2025.
  * atd  logistic on usage (targets, red-zone looks, carries inside the 10,
         recent TD rate) and implied points, fit on prior data.

  Everything is checked out of sample before it is trusted: models are fit on
  data BEFORE the season being scored and the calibration table is written to
  the output ("calib") so the app can show it.

If the sandbox cannot reach GitHub (or a season has no play-by-play yet) this
exits without writing anything, so a bad run never overwrites a good file.
"""
import json
import math
import zlib
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

NV = "https://github.com/nflverse/nflverse-data/releases/download/"
PBP = NV + "pbp/play_by_play_{season}.parquet"
PLAYERS = NV + "players/players.parquet"

HALF_LIFE = 3.0                  # games
LOOKBACK = 10                    # most recent active games considered
POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
SKILL = {"QB", "RB", "WR", "TE"}
ROUND = 4

# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------


def load_pbp(season, cache=None):
    cols = ["season", "week", "season_type", "game_id", "home_team", "away_team",
            "posteam", "defteam", "play_type", "yardline_100", "yards_gained",
            "rush_attempt", "pass_attempt", "complete_pass", "qb_kneel", "sack",
            "rusher_player_id", "receiver_player_id", "passer_player_id",
            "rushing_yards", "receiving_yards", "pass_touchdown", "rush_touchdown",
            "td_player_id", "spread_line", "total_line", "qb_scramble"]
    path = (cache or {}).get(season)
    if path and os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(PBP.format(season=season))
    df = df[[c for c in cols if c in df.columns]]
    return df[df.season_type == "REG"].copy()


def load_positions(local=None):
    df = pd.read_parquet(local if local and os.path.exists(local) else PLAYERS)
    col = "gsis_id"
    pos = df.set_index(col)["position"].to_dict()
    name = df.set_index(col)["display_name"].to_dict()
    return pos, name


def player_games(pbp, pos):
    """One row per player per game he was active (>=1 target or carry)."""
    d = pbp
    ctx = (d.groupby(["season", "week", "game_id"])
             .agg(home=("home_team", "first"), away=("away_team", "first"),
                  spread=("spread_line", "first"), total=("total_line", "first"))
             .reset_index())

    # --- rushing
    ru = d[(d.rush_attempt == 1) & d.rusher_player_id.notna()].copy()
    ru["yds"] = ru.rushing_yards.fillna(ru.yards_gained).fillna(0)
    ru["rztd"] = ((ru.rush_touchdown == 1) & (ru.td_player_id == ru.rusher_player_id)).astype(int)
    ru["rz10"] = (ru.yardline_100 <= 10).astype(int)
    ru["gl"] = (ru.yardline_100 <= 3).astype(int)
    g_ru = ru.groupby(["season", "week", "game_id", "posteam", "rusher_player_id"]).agg(
        car=("yds", "size"), rushyds=("yds", "sum"), lrush=("yds", "max"),
        rz10=("rz10", "sum"), gl=("gl", "sum"), rutd=("rztd", "sum")).reset_index()
    g_ru = g_ru.rename(columns={"rusher_player_id": "pid"})

    # --- receiving
    re = d[(d.pass_attempt == 1) & d.receiver_player_id.notna()].copy()
    re["rec"] = re.complete_pass.fillna(0)
    re["ry"] = re.receiving_yards.fillna(0) * re.rec
    re["rz20"] = (re.yardline_100 <= 20).astype(int)
    re["retd"] = ((re.pass_touchdown == 1) & (re.td_player_id == re.receiver_player_id)).astype(int)
    g_re = re.groupby(["season", "week", "game_id", "posteam", "receiver_player_id"]).agg(
        tgt=("rec", "size"), rec=("rec", "sum"), recyds=("ry", "sum"),
        rz20=("rz20", "sum"), retd=("retd", "sum")).reset_index()
    g_re = g_re.rename(columns={"receiver_player_id": "pid"})

    # --- passing (QBs): attempts, TDs
    pa = d[(d.pass_attempt == 1) & d.passer_player_id.notna()].copy()
    pa["ptd"] = (pa.pass_touchdown == 1).astype(int)
    g_pa = pa.groupby(["season", "week", "game_id", "posteam", "passer_player_id"]).agg(
        patt=("ptd", "size"), ptd=("ptd", "sum")).reset_index()
    g_pa = g_pa.rename(columns={"passer_player_id": "pid"})

    key = ["season", "week", "game_id", "posteam", "pid"]
    pg = g_ru.merge(g_re, on=key, how="outer").merge(g_pa, on=key, how="outer")
    for c in ["car", "rushyds", "rz10", "gl", "rutd", "tgt", "rec", "recyds", "rz20",
              "retd", "patt", "ptd"]:
        pg[c] = pg[c].fillna(0)
    pg["lrush"] = pg.lrush.fillna(0)
    pg["td"] = ((pg.rutd + pg.retd) > 0).astype(int)
    pg["combo"] = pg.rushyds + pg.recyds
    pg["pos"] = pg.pid.map(pos).map(POS_GROUP)
    pg = pg.merge(ctx, on=["season", "week", "game_id"], how="left")
    pg["is_home"] = (pg.posteam == pg.home).astype(int)
    # team-perspective spread: + means this player's team is favoured
    pg["tspread"] = np.where(pg.is_home == 1, pg.spread, -pg.spread)
    pg["imp"] = pg.total / 2 + pg.tspread / 2
    pg["opp"] = np.where(pg.is_home == 1, pg.away, pg.home)
    pg["active"] = ((pg.car + pg.tgt) > 0) | (pg.patt >= 5)
    pg = pg[pg.active & pg.pos.isin(SKILL)].copy()
    pg["t"] = pg.season * 100 + pg.week
    return pg.sort_values(["pid", "t"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# form: exponentially weighted trailing averages
# ----------------------------------------------------------------------------
FORM = ["car", "rushyds", "tgt", "recyds", "rz10", "rz20", "gl", "td", "ptd", "patt", "combo"]


def pos_prior(pg):
    """League-average value of each FORM stat by position (training data)."""
    return pg.groupby("pos")[FORM].mean().to_dict("index")


def ew(vals, hl=HALF_LIFE):
    n = len(vals)
    if n == 0:
        return None, 0.0
    w = 0.5 ** (np.arange(n - 1, -1, -1) / hl)
    return w, w.sum()


def add_form(pg, prior, shrink=1.5):
    """Attach form_<stat> = EW average over the player's previous active games,
    shrunk toward the position average by `shrink` pseudo-games."""
    out = {c: np.full(len(pg), np.nan) for c in FORM}
    nprev = np.zeros(len(pg))
    for pid, idx in pg.groupby("pid").indices.items():
        sub = pg.iloc[idx]
        pos = sub.pos.iloc[0]
        pr = prior.get(pos, {})
        vals = {c: sub[c].to_numpy(float) for c in FORM}
        for j in range(len(sub)):
            lo = max(0, j - LOOKBACK)
            n = j - lo
            nprev[idx[j]] = n
            for c in FORM:
                if n == 0:
                    out[c][idx[j]] = pr.get(c, 0.0)
                    continue
                w = 0.5 ** (np.arange(n - 1, -1, -1) / HALF_LIFE)
                out[c][idx[j]] = (np.dot(w, vals[c][lo:j]) + shrink * pr.get(c, 0.0)) / (w.sum() + shrink)
    for c in FORM:
        pg["f_" + c] = out[c]
    pg["nprev"] = nprev
    return pg


# ----------------------------------------------------------------------------
# small stats helpers (numpy only -- the workflow does not install scipy)
# ----------------------------------------------------------------------------


def irls_logit(X, y, l2=1.0, iters=40):
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X1.shape[1])
    pen = np.eye(X1.shape[1]) * l2
    pen[0, 0] = 0
    for _ in range(iters):
        z = X1 @ b
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        H = X1.T @ (X1 * W[:, None]) + pen
        g = X1.T @ (y - p) - pen @ b
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def irls_pois(X, y, offset, iters=40, l2=0.5):
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X1.shape[1])
    b[0] = np.log(max(y.sum(), 1) / max(np.exp(offset).sum(), 1e-9))
    pen = np.eye(X1.shape[1]) * l2
    pen[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(X1 @ b + offset, -8, 8)
        mu = np.exp(eta)
        H = X1.T @ (X1 * mu[:, None]) + pen
        g = X1.T @ (y - mu) - pen @ b
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def nb_pmf(mu, r, kmax):
    """Negative binomial pmf, mean mu, variance mu + mu^2/r. r<=0 or huge -> Poisson."""
    mu = max(mu, 1e-6)
    out = np.zeros(kmax + 1)
    if r is None or r > 1e5:
        out[0] = math.exp(-mu)
        for k in range(kmax):
            out[k + 1] = out[k] * mu / (k + 1)
        return out
    p0 = (r / (r + mu)) ** r
    out[0] = p0
    q = mu / (r + mu)
    for k in range(kmax):
        out[k + 1] = out[k] * (k + r) / (k + 1) * q
    return out


def fit_disp(mu, y):
    """Method-of-moments NB size r for var = mu + mu^2/r, pooled."""
    mu = np.asarray(mu, float)
    y = np.asarray(y, float)
    excess = ((y - mu) ** 2 - mu).sum()
    denom = (mu ** 2).sum()
    if excess <= 0:
        return 1e6
    return float(denom / excess)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
ATD_COLS = ["f_tgt", "f_rz20", "f_car", "f_rz10", "f_gl", "f_td"]
LRUSH_T = list(range(1, 51))          # longest-rush thresholds exported (P(longest >= L), L = 1..50)
COMBO_X = list(range(0, 301, 5))      # combo-yards grid (P(total >= x), every 5 yards)
CAR_MAX = 40
PTD_MAX = 7
Q_THR = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80]


def atd_X(d):
    return np.column_stack([d.f_tgt, d.f_rz20, d.f_car, d.f_rz10, d.f_gl,
                            d.f_td * 5, (d.imp - 22.5) / 5,
                            d.pos.eq("RB"), d.pos.eq("TE"), d.pos.eq("QB")]).astype(float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_models(pg, carries):
    """pg: player-game rows with form attached (training period only).
       carries: DataFrame [pid, pos, yds] one row per carry (training period)."""
    M = {}
    d = pg[pg.nprev >= 1]
    # --- anytime TD
    M["atd"] = irls_logit(atd_X(d), d.td.values, l2=2.0)
    # --- passing TDs: Poisson, offset = log(form), covariate = log(implied pts)
    q = d[(d.pos == "QB") & (d.patt >= 8)]
    off = np.log(np.clip(q.f_ptd.values, 0.15, None))
    X = np.column_stack([np.log(q.imp.values / 22.5)])
    M["ptd"] = irls_pois(X, q.ptd.values.astype(float), off)
    # --- rushing attempts: Poisson mean, NB spread per position group
    r = d[(d.car >= 1) | (d.pos.isin(["RB", "QB"]))]
    r = r[r.f_car > 0.3]
    off = np.zeros(len(r))
    X = np.column_stack([r.tspread.values / 7.0, r.pos.eq("QB").values.astype(float),
                         np.log(r.f_car.values)])
    M["car"] = irls_pois(X, r.car.values.astype(float), off)
    mu = np.exp(np.column_stack([np.ones(len(r)), X]) @ M["car"] + off)
    M["car_r"] = {}
    for g, mask in (("QB", r.pos.eq("QB").values), ("RB", r.pos.isin(["RB"]).values),
                    ("WR", r.pos.isin(["WR", "TE"]).values)):
        if mask.sum() > 30:
            M["car_r"][g] = fit_disp(mu[mask], r.car.values[mask])
    # --- combo yards: log-link mean, gamma spread: var = a*mu + b*mu^2
    c = d[d.f_combo > 3]
    off = np.zeros(len(c))
    X = np.column_stack([c.tspread.values / 7.0, np.log(c.f_combo.values)])
    M["combo"] = irls_pois(X, c.combo.clip(lower=0).values.astype(float), off)
    mu = np.exp(np.column_stack([np.ones(len(c)), X]) @ M["combo"] + off)
    res2 = (c.combo.values - mu) ** 2
    A = np.column_stack([mu, mu ** 2])
    coef = np.linalg.lstsq(A, res2, rcond=None)[0]
    M["combo_var"] = [max(float(coef[0]), 1.0), max(float(coef[1]), 0.05)]
    # --- per-carry big-run rates by position, for longest rush
    M["q_prior"] = {}
    for g, grp in (("QB", ["QB"]), ("RB", ["RB"]), ("WR", ["WR", "TE"])):
        y = carries[carries.pos.isin(grp)].yds.values
        if len(y) < 50:
            y = carries.yds.values
        M["q_prior"][g] = {str(t): float((y >= t).mean()) for t in Q_THR}
    return M


def qgroup(pos):
    return "QB" if pos == "QB" else ("RB" if pos == "RB" else "WR")


def predict_row(M, row, car_stats=None):
    """Distributions for one player-game. `row` has f_*, imp, tspread, pos.
       car_stats = (n_carries, {thr: count>=thr}) for this player."""
    out = {}
    pos = row["pos"]
    g = qgroup(pos)
    # atd
    x = atd_X(pd.DataFrame([row]))
    out["atd"] = float(sigmoid(M["atd"][0] + x @ M["atd"][1:])[0])
    # ptd
    if pos == "QB" and row["f_patt"] >= 8:
        lam = math.exp(M["ptd"][0] + M["ptd"][1] * math.log(row["imp"] / 22.5)
                       + math.log(max(row["f_ptd"], 0.15)))
        out["ptd"] = [round(v, ROUND) for v in nb_pmf(lam, None, PTD_MAX)]
        out["ptd_mu"] = round(lam, 3)
    # car
    mu_car = None
    if row["f_car"] > 0.3 and (pos in ("RB", "QB") or row["f_car"] >= 1.0):
        beta = M["car"]
        mu_car = math.exp(beta[0] + beta[1] * row["tspread"] / 7.0
                          + beta[2] * (1.0 if pos == "QB" else 0.0)
                          + beta[3] * math.log(row["f_car"]))
        r = M["car_r"].get(g) or M["car_r"].get("RB") or 20.0
        pm = nb_pmf(mu_car, r, CAR_MAX)
        out["car"] = [round(v, ROUND) for v in pm]
        out["car_mu"] = round(mu_car, 2)
        # longest rush: mix over carries, per-carry big-run chance shrunk to position
        n_c, hits = car_stats if car_stats else (0, {})
        m = 40.0
        qs = []
        prior = M["q_prior"][g]
        thr = [int(t) for t in prior]
        qv = []
        for t in thr:
            qv.append((hits.get(t, 0) + m * prior[str(t)]) / (n_c + m))
        qv = np.minimum.accumulate(np.array(qv))           # survival must not rise
        surv = []
        for L in LRUSH_T:
            # interpolate q at L (log space) across the threshold grid
            if L <= thr[0]:
                qL = 1.0 if L <= 1 else qv[0] + (1 - qv[0]) * (thr[0] - L) / (thr[0] - 1)
            elif L >= thr[-1]:
                qL = qv[-1] * math.exp(-(L - thr[-1]) / 25.0)
            else:
                k = max(i for i, t in enumerate(thr) if t <= L)
                t0, t1 = thr[k], thr[k + 1]
                a, b2 = max(qv[k], 1e-6), max(qv[k + 1], 1e-6)
                qL = math.exp(math.log(a) + (math.log(b2) - math.log(a)) * (L - t0) / (t1 - t0))
            qL = min(max(qL, 0.0), 1.0)
            # P(longest >= L) = 1 - E[(1-q)^N]
            pk = 1 - float(np.dot(pm, (1 - qL) ** np.arange(len(pm))))
            cal = M.get("lrush_cal")
            if cal and 0.001 < pk < 0.999:
                z = math.log(pk / (1 - pk))
                pk = 1 / (1 + math.exp(-(cal[0] + cal[1] * z)))
            surv.append(pk)
        # recalibration can break monotonicity slightly; survival must not rise
        surv = np.minimum.accumulate(np.array(surv)).tolist()
        out["lrush"] = surv
    # combo
    if row["f_combo"] > 3:
        mu = math.exp(M["combo"][0] + M["combo"][1] * row["tspread"] / 7.0
                      + M["combo"][2] * math.log(row["f_combo"]))
        a, b = M["combo_var"]
        var = a * mu + b * mu * mu
        k = mu * mu / var
        th = var / mu
        rng = np.random.default_rng(zlib.crc32(str(row.get("pid")).encode()))
        s = rng.gamma(k, th, 6000)
        out["combo"] = [round(float((s >= x).mean()), ROUND) for x in COMBO_X]
        out["combo_mu"] = round(mu, 1)
    return out


# ----------------------------------------------------------------------------
# calibration summary (honest, out of sample) -- shipped in the file
# ----------------------------------------------------------------------------
def _brier(p, a):
    p = np.asarray(p, float)
    a = np.asarray(a, float)
    if len(p) == 0:
        return None
    return {"n": int(len(p)), "brier": round(float(((p - a) ** 2).mean()), 4),
            "base": round(float(((a.mean() - a) ** 2).mean()), 4),
            "pred": round(float(p.mean()), 3), "act": round(float(a.mean()), 3)}


def backtest(pg, carries, train_season, test_season):
    """Fit on `train_season` only, score `test_season`. Lines are placed where a
    book would put them (about the model's own median), because nobody posts a
    line far from the middle."""
    tr = pg[pg.season == train_season]
    te = pg[(pg.season == test_season) & (pg.nprev >= 1)]
    if len(tr) < 500 or len(te) < 100:
        return None
    M = fit_models(tr, carries[carries.season == train_season] if "season" in carries else carries)
    cs = {}
    for pid, g in carries.groupby("pid"):
        y = g.yds.values
        cs[pid] = (len(y), {t: int((y >= t).sum()) for t in Q_THR})
    P = {"atd": ([], []), "ptd": ([], []), "car": ([], []), "lrush": ([], []), "combo": ([], [])}
    LRC = ([], [])
    for _, r in te.iterrows():
        o = predict_row(M, r.to_dict(), cs.get(r.pid))
        P["atd"][0].append(o["atd"]); P["atd"][1].append(r.td)
        if "ptd" in o:
            pm = np.array(o["ptd"])
            for L in (1, 2):
                P["ptd"][0].append(pm[L:].sum()); P["ptd"][1].append(float(r.ptd >= L))
        if "car" in o:
            pm = np.array(o["car"])
            L = int(math.floor(o["car_mu"])) + 1
            P["car"][0].append(pm[L:].sum()); P["car"][1].append(float(r.car >= L))
            s = np.array(o["lrush"])
            L = int(np.argmin(abs(s - .5))) + 1
            P["lrush"][0].append(s[L - 1]); P["lrush"][1].append(float(r.lrush >= L))
            for LL in (max(L - 3, 2), L + 4):
                LRC[0].append(s[LL - 1]); LRC[1].append(float(r.lrush >= LL))
            LRC[0].append(s[L - 1]); LRC[1].append(float(r.lrush >= L))
        if "combo" in o:
            s = np.array(o["combo"])
            x = min(int(round(o["combo_mu"] / 5)), len(s) - 1)
            P["combo"][0].append(s[x]); P["combo"][1].append(float(r.combo >= COMBO_X[x]))
    res = {k: _brier(*v) for k, v in P.items()}
    p = np.clip(np.array(LRC[0]), 0.02, 0.98)
    if len(p) > 200:
        b = irls_logit(np.log(p / (1 - p))[:, None], np.array(LRC[1]), l2=1.0)
        res["_lrush_cal"] = [float(b[0]), float(b[1])]
    return res


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def trim(a, nd=3, eps=0.0015):
    """Round and drop the tail that is effectively zero (keeps the file small)."""
    a = [round(float(x), nd) for x in a]
    while len(a) > 3 and a[-1] < eps:
        a.pop()
    return a


def mkey(s):
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def carries_table(pbp, pos):
    ru = pbp[(pbp.rush_attempt == 1) & pbp.rusher_player_id.notna()].copy()
    ru["yds"] = ru.rushing_yards.fillna(ru.yards_gained).fillna(0)
    ru["pid"] = ru.rusher_player_id
    ru["pos"] = ru.pid.map(pos).map(POS_GROUP)
    return ru[["season", "pid", "pos", "yds"]].dropna()


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    out = sys.argv[2] if len(sys.argv) > 2 else f"data/props-{season}.json"
    d = os.path.dirname(out) or "."
    games_path = os.path.join(d, f"games-{season}.json")
    roster_path = os.path.join(d, f"roster-{season}.json")
    cache = {}
    for y in (season - 1, season):
        for cand in (f"pbp{y}.parquet",):
            if os.path.exists(cand):
                cache[y] = cand
    try:
        pos, name = load_positions("players.parquet" if os.path.exists("players.parquet") else None)
        frames = []
        for y in (season - 1, season):
            try:
                frames.append(load_pbp(y, cache))
            except Exception as e:                       # prior season missing is survivable
                print(f"  no play-by-play for {y}: {e}", file=sys.stderr)
        pbp = pd.concat(frames)
    except Exception as e:
        raise SystemExit(f"could not load data, leaving the old file alone: {e}")
    cur = pbp[pbp.season == season]
    if cur.empty:
        raise SystemExit("no games played yet this season; nothing to model")
    last_wk = int(cur.week.max())

    pg = player_games(pbp, pos)
    prior = pos_prior(pg[pg.season < season] if (pg.season < season).any() else pg)
    carries = carries_table(pbp, pos)

    # ---- honest backtest (previous season -> this season) for the file
    calib = None
    try:
        pgb = add_form(pg.copy(), prior)
        calib = backtest(pgb, carries, season - 1, season)
    except Exception as e:
        print(f"  backtest skipped: {e}", file=sys.stderr)

    # ---- upcoming games
    G = json.load(open(games_path))["games"] if os.path.exists(games_path) else []
    today = datetime.now(timezone.utc).date().isoformat()
    yday = (pd.Timestamp(today) - pd.Timedelta(days=1)).date().isoformat()
    up = [g for g in G if g.get("hs") is None and str(g.get("date", "")) >= yday]
    if not up:
        raise SystemExit("no upcoming games in the games file; leaving the old file alone")
    # only the next slate: games within 9 days of the earliest unplayed one
    first = min(str(g["date"]) for g in up)
    cutoff = (pd.Timestamp(first) + pd.Timedelta(days=9)).date().isoformat()
    up = [g for g in up if str(g["date"]) <= cutoff]

    R = json.load(open(roster_path))["players"] if os.path.exists(roster_path) else {}
    tm_now = {pid: v for pid, v in R.items()}

    # ---- fit on everything played so far
    pgf = add_form(pg.copy(), prior)
    M = fit_models(pgf, carries)
    if calib and calib.get("_lrush_cal"):
        M["lrush_cal"] = calib.pop("_lrush_cal")
    cs = {}
    recent = carries[carries.season >= season - 1]
    for pid, g in recent.groupby("pid"):
        y = g.yds.values
        cs[pid] = (len(y), {t: int((y >= t).sum()) for t in Q_THR})

    # who is realistic to play for each team: active in 2 of their last 4
    last_rows = pg.sort_values("t").groupby("pid").tail(4)
    cand = {}
    for pid, grp in last_rows.groupby("pid"):
        t_now = (tm_now.get(pid) or {}).get("team") or grp.posteam.iloc[-1]
        cand.setdefault(t_now, []).append((pid, grp))

    players = []
    seen = set()
    for g in up:
        for side, tm, opp in (("home", g["home"], g["away"]), ("away", g["away"], g["home"])):
            sp, tot = g.get("spread"), g.get("total")
            if sp is None or tot is None:
                if g.get("ptot") is None or g.get("proj") is None:
                    continue
                tot = g["ptot"]
                sp = g["proj"]
            tsp = sp if side == "home" else -sp
            imp = tot / 2 + tsp / 2
            for pid, grp in cand.get(tm, []):
                if (pid, g.get("id")) in seen:
                    continue
                info = tm_now.get(pid) or {}
                if info.get("is") in ("Out", "Doubtful") or info.get("st") in ("IR", "CUT", "NONE", "RET", "PS"):
                    continue
                if len(grp) < 2 or grp.t.max() < (season - 1) * 100 + 15:
                    continue
                hist = pgf[pgf.pid == pid]
                if hist.empty:
                    continue
                ps = pos.get(pid) or info.get("pos")
                pg_pos = POS_GROUP.get(ps)
                if pg_pos is None or pg_pos not in SKILL:
                    continue
                fut = {"pid": pid, "pos": pg_pos}
                w = 0.5 ** (np.arange(min(len(hist), LOOKBACK))[::-1] / HALF_LIFE)
                tail = hist.tail(LOOKBACK)
                pr = prior.get(pg_pos, {})
                for c in FORM:
                    fut["f_" + c] = (np.dot(w, tail[c].to_numpy(float)) + 1.5 * pr.get(c, 0.0)) / (w.sum() + 1.5)
                fut.update({"imp": imp, "tspread": tsp, "pos": pg_pos})
                o = predict_row(M, fut, cs.get(pid))
                # drop what nobody prices: tiny roles make a big file and no bets
                if o.get("combo_mu", 99) < 20:
                    o.pop("combo", None); o.pop("combo_mu", None)
                if o.get("car_mu", 99) < 3:
                    for k in ("car", "car_mu", "lrush"):
                        o.pop(k, None)
                if o.get("atd", 1) < 0.07:        # the app never tiers long shots
                    o.pop("atd", None)
                if not any(k in o for k in ("atd", "ptd", "car", "combo")):
                    continue
                seen.add((pid, g.get("id")))
                ent = {"pid": pid, "name": info.get("name") or name.get(pid),
                       "key": mkey(info.get("name") or name.get(pid)),
                       "team": tm, "opp": opp, "pos": pg_pos, "date": g["date"], "gid": g.get("id"),
                       "n": int(len(hist)), "imp": round(imp, 1), "tsp": round(tsp, 1)}
                for k, v in o.items():
                    ent[k] = (trim(v, 2 if k in ("lrush", "combo") else 3) if isinstance(v, list)
                              else (round(v, 3) if isinstance(v, float) else v))
                players.append(ent)

    # ---- what actually happened, for grading saved picks
    results = {}
    for wk in range(max(1, last_wk - 2), last_wk + 1):
        rows = pg[(pg.season == season) & (pg.week == wk)]
        results[str(wk)] = {r.pid: {"td": int(r.td), "ptd": int(r.ptd), "car": int(r.car),
                                    "lrush": int(r.lrush), "combo": round(float(r.combo), 1),
                                    "team": r.posteam}
                            for r in rows.itertuples()}

    doc = {"season": season, "kind": "props",
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "through_week": last_wk, "calib": calib, "players": players, "results": results}
    os.makedirs(d, exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {out}: {len(players)} player-games across {len(up)} games; "
          f"results through week {last_wk}", file=sys.stderr)
    if calib:
        for k, v in calib.items():
            print(f"  calib {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
