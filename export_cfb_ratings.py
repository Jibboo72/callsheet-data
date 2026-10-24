#!/usr/bin/env python3
"""
College football power ratings, weekly board and season outlook for the
Power tab's College section.

    python3 export_cfb_ratings.py 2026 data/cfb-ratings-2026.json

Writes one JSON file with:
  teams      every FBS team: Off / Def / Net ratings and ranks, record,
             strength of schedule, rating trend by week, and the season
             outlook from 10,000 simulations (expected record, conference
             title-game reach / win %, CFP %)
  games      every FBS game this season. Played games carry the score and
             what the model said BEFORE kickoff (walk-forward, no hindsight).
             Upcoming games carry the model spread, total and win %.
             Where a line is known, EDGE = model spread vs the line.
  confs      projected title game per conference
  backtest   the walk-forward record from backtest_cfb.py's last run

LINES
  Past seasons come from the free cfbfastR betting file. Current-season
  lines need a free CollegeFootballData key: put it in the CFBD_KEY
  environment variable (GitHub: Settings > Secrets > Actions > CFBD_KEY).
  Without it the file still builds -- the app then uses the line you type
  on the College Saturday board to work out the edge.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import cfb_ratings as cr

N_SIMS = 10_000
TEAM_SHOCK_SD = 4.0     # how unsure we are of a team's TRUE rating, in points
CFP_LOSS_PTS = 3.0      # committee proxy: each win-minus-loss is worth this much rating
CFP_CHAMP_BONUS = 3.0
FCS_WIN = 0.95          # FBS vs FCS game the model can't rate well -> flat 95%

TEAM_INFO_URL = cr.BASE + "/team_info/parquet/cfb_team_info_{season}.parquet"


def _et(ts):
    """Kickoff in Eastern time -> (YYYY-MM-DD, HH:MM)."""
    try:
        t = pd.Timestamp(ts)
        t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert("America/New_York")
        return t.strftime("%Y-%m-%d"), t.strftime("%H:%M")
    except Exception:
        return str(ts)[:10], ""


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def team_info(season):
    try:
        t = cr._read(TEAM_INFO_URL.format(season=season), f"cfb_team_info_{season}.parquet")
    except Exception:
        return {}
    out = {}
    for _, r in t.iterrows():
        out[r.school] = dict(abbr=r.abbreviation if isinstance(r.abbreviation, str) else "",
                             color=r.color if isinstance(r.color, str) else "",
                             alt=r.alt_color if isinstance(r.alt_color, str) else "",
                             logo=r.logos_7 if isinstance(r.logos_7, str) else
                             (r.logo if isinstance(r.logo, str) else ""),
                             division=r.division if isinstance(r.division, str) else "",
                             mascot=r.mascot if isinstance(r.mascot, str) else "")
    return out


_LINES_CACHE = {}


def cfbd_games(season):
    """This season's FCS games from CollegeFootballData (needs CFBD_KEY), in
    the same column layout as the free cfbfastR schedule file. The free file
    only carries games with an FBS team in them; this fills in FCS-vs-FCS."""
    key = os.environ.get("CFBD_KEY", "").strip()
    if not key:
        return pd.DataFrame()
    import requests
    rows = []
    for st in ("regular", "postseason"):
        try:
            r = requests.get("https://api.collegefootballdata.com/games",
                             params={"year": season, "seasonType": st, "classification": "fcs"},
                             headers={"Authorization": f"Bearer {key}"}, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"CFBD games ({st}) failed: {e}", file=sys.stderr)
            continue
        for g in r.json():
            rows.append(dict(
                game_id=int(g["id"]), season=int(g.get("season") or season),
                week=int(g.get("week") or 0), season_type=g.get("seasonType") or st,
                start_date=g.get("startDate"), neutral_site=bool(g.get("neutralSite")),
                conference_game=bool(g.get("conferenceGame")),
                home_team=g.get("homeTeam"), away_team=g.get("awayTeam"),
                home_division=g.get("homeClassification"), away_division=g.get("awayClassification"),
                home_conference=g.get("homeConference"), away_conference=g.get("awayConference"),
                home_points=g.get("homePoints"), away_points=g.get("awayPoints"),
                notes=g.get("notes")))
    df = pd.DataFrame(rows)
    if len(df):
        for c in ("home_points", "away_points"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        print(f"CFBD: {len(df)} FCS games, {df.home_points.notna().sum()} with scores", file=sys.stderr)
    return df


def cfbd_lines(season):
    """Current-season consensus lines from CollegeFootballData (needs CFBD_KEY).
    Covers FBS and FCS games alike -- the response carries every division."""
    if season in _LINES_CACHE:
        return _LINES_CACHE[season]
    key = os.environ.get("CFBD_KEY", "").strip()
    if not key:
        return pd.DataFrame()
    import requests
    rows = []
    for st in ("regular", "postseason"):
        try:
            r = requests.get("https://api.collegefootballdata.com/lines",
                             params={"year": season, "seasonType": st},
                             headers={"Authorization": f"Bearer {key}"}, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"CFBD lines ({st}) failed: {e}", file=sys.stderr)
            continue
        for g in r.json():
            ls = g.get("lines") or []
            pick = [x for x in ls if "draftkings" in str(x.get("provider", "")).lower()] or ls
            sp = [x.get("spread") for x in pick if x.get("spread") is not None]
            ot = [x.get("overUnder") for x in pick if x.get("overUnder") is not None]
            op = [x.get("spreadOpen") for x in pick if x.get("spreadOpen") is not None]
            if not sp:
                continue
            rows.append(dict(game_id=int(g["id"]), line_spread=float(np.median(sp)),
                             open_spread=float(np.median(op)) if op else np.nan,
                             line_total=float(np.median(ot)) if ot else np.nan,
                             book=pick[0].get("provider", "")))
    out = pd.DataFrame(rows).drop_duplicates("game_id") if rows else pd.DataFrame()
    print(f"CFBD: {len(out)} games with lines", file=sys.stderr)
    _LINES_CACHE[season] = out
    return out


def _norm(x):
    x = str(x or "").lower().replace("&", " and ")
    for ch in "().'’,":
        x = x.replace(ch, "")
    x = x.replace("-", " ").replace("state", "st").replace("saint ", "st ")
    return " ".join(x.split())


def _alias_map(season):
    """Every way a school might be written -> the schedule's name for it.
    FBS first so a shared abbreviation resolves to the FBS team."""
    try:
        t = cr._read(TEAM_INFO_URL.format(season=season), f"cfb_team_info_{season}.parquet")
    except Exception:
        return {}
    t = pd.concat([t[t.classification == "fbs"], t[t.classification != "fbs"]])
    out = {}
    for _, r in t.iterrows():
        school = r.school
        cands = [school, r.abbreviation, r.alt_name1, r.alt_name2, r.alt_name3]
        if isinstance(r.mascot, str):
            cands.append(f"{school} {r.mascot}")
        for c in cands:
            if isinstance(c, str) and c.strip():
                out.setdefault(_norm(c), school)
    return out


def sgo_lines(season, sched):
    """Upcoming-game lines from the SportsGameOdds college pull
    (data/cfb-odds-SEASON.json, written by pull_cfb_odds.py), matched to the
    schedule by team names + date. Returns a DataFrame shaped like
    load_lines()'s, plus team totals."""
    path = os.path.join(os.path.dirname(__file__) or ".", "data", f"cfb-odds-{season}.json")
    if not os.path.exists(path):
        return pd.DataFrame(), None
    try:
        doc = json.load(open(path))
    except Exception as e:
        print(f"SGO college odds unreadable: {e}", file=sys.stderr)
        return pd.DataFrame(), None
    alias = _alias_map(season)

    def resolve(nm):
        for k in ("long", "medium", "location", "short", "abbr", "teamID"):
            v = nm.get(k)
            if v and _norm(v) in alias:
                return alias[_norm(v)]
        # "Alabama Crimson Tide" with a mascot we don't know: try dropping words
        v = _norm(nm.get("long") or nm.get("medium") or "")
        parts = v.split()
        for cut in range(len(parts) - 1, 0, -1):
            if " ".join(parts[:cut]) in alias:
                return alias[" ".join(parts[:cut])]
        return None

    up = sched[~sched.played].copy()
    up["d"] = pd.to_datetime(up.start_date, utc=True, errors="coerce")
    by_pair = {}
    for _, r in up.iterrows():
        by_pair.setdefault(frozenset((r.home_team, r.away_team)), []).append(r)
    rows, missed = [], []
    for ev in doc.get("games", []):
        h, a = resolve(ev.get("home") or {}), resolve(ev.get("away") or {})
        cand = by_pair.get(frozenset((h, a))) if h and a else None
        if not cand:
            missed.append(f"{(ev.get('away') or {}).get('long') or (ev.get('away') or {}).get('medium')} @ "
                          f"{(ev.get('home') or {}).get('long') or (ev.get('home') or {}).get('medium')}"
                          f" -> {a} @ {h}")
            continue
        when = pd.to_datetime(ev.get("start"), utc=True, errors="coerce")
        r = min(cand, key=lambda x: abs((x.d - when).total_seconds()) if pd.notna(when) and pd.notna(x.d) else 0)
        flip = r.home_team != h                      # SGO has home/away the other way
        L = ev.get("lines") or {}
        sp = (L.get("spread") or {}).get("home")
        tt = L.get("team_total") or {}
        th, ta = (tt.get("away"), tt.get("home")) if flip else (tt.get("home"), tt.get("away"))
        rows.append(dict(game_id=int(r.game_id),
                         line_spread=(-sp if flip else sp) if sp is not None else np.nan,
                         open_spread=np.nan,
                         line_total=(L.get("total") or {}).get("line", np.nan),
                         tt_home=(th or {}).get("line"), tt_away=(ta or {}).get("line"),
                         book=(L.get("spread") or L.get("total") or {}).get("book", "")))
    if missed:
        print(f"SGO: {len(missed)} college games not matched to the schedule:", file=sys.stderr)
        for m in missed[:40]:
            print("   ", m, file=sys.stderr)
    print(f"SGO: matched {len(rows)} upcoming college games", file=sys.stderr)
    df = pd.DataFrame(rows)
    return df, doc.get("pulled")


def _merge_sgo(lines, season, sched):
    """game_id -> line row. Finished games: CollegeFootballData (the record
    of the closing number). Upcoming games: SportsGameOdds when it has the
    game -- it's the live board and carries team totals -- else CFBD."""
    lmap = {int(r.game_id): r for _, r in lines.iterrows()} if len(lines) else {}
    sg, _ = sgo_lines(season, sched)
    for _, r in (sg.iterrows() if len(sg) else []):
        if pd.isna(r.line_spread) and pd.isna(r.line_total):
            continue
        old = lmap.get(int(r.game_id))
        if old is not None and pd.isna(r.line_spread):
            r = r.copy(); r["line_spread"] = old.line_spread
        r = r.copy()
        r["sgo"] = True
        lmap[int(r.game_id)] = r
    return lmap


def get_lines(season):
    lines = cfbd_lines(season)
    if len(lines):
        return lines, "CollegeFootballData (DraftKings where posted)"
    try:
        lines = cr.load_lines(season)
        if len(lines):
            return lines, "cfbfastR closing consensus"
    except Exception:
        pass
    return pd.DataFrame(), ""


# --------------------------------------------------------------------------

def build(season):
    cfg = dict(cr.CONFIG)
    sched = cr.load_schedule(season)
    prior = cr.season_prior(season, cfg)
    # METHOD 9 in cfb_ratings.py: returning production / talent / recruiting
    prior = cr.offseason_adjust(
        prior, season,
        sorted(set(sched.loc[sched.home_division == "fbs", "home_team"])
               | set(sched.loc[sched.away_division == "fbs", "away_team"])))
    info = team_info(season)

    lines, line_src = get_lines(season)
    lmap = _merge_sgo(lines, season, sched)
    if any(bool(v.get("sgo", False)) for v in lmap.values()):
        line_src = (line_src + "; " if line_src else "") + "upcoming games: SportsGameOdds"

    # closing line -> expected home margin, so the ratings can mark a team that
    # lands far below it (cfb_ratings.py, METHOD 8)
    sched["exp_margin"] = [
        (-float(lmap[int(i)].line_spread)
         if int(i) in lmap and pd.notna(lmap[int(i)].line_spread) else np.nan)
        for i in sched.game_id]

    played_wks = sorted(sched[sched.played].wk.unique())
    through = int(max(played_wks)) if played_wks else 0
    next_wk = int(sched[~sched.played].wk.min()) if (~sched.played).any() else through + 1

    # walk-forward fits: model as it stood before each week was played
    fits = {wk: cr.fit(sched, cfg, prior, as_of_wk=wk) for wk in played_wks}
    cur = cr.fit(sched, cfg, prior)            # through all played games
    T = cur["teams"].set_index("team")
    fbs = T[T["div"] == "fbs"].copy()

    conf_of, div_of = {}, {}
    for _, r in sched.iterrows():
        for side in ("home", "away"):
            t = r[f"{side}_team"]
            if r[f"{side}_division"] == "fbs" and isinstance(r[f"{side}_conference"], str):
                conf_of[t] = r[f"{side}_conference"]
    for t in fbs.index:
        div_of[t] = info.get(t, {}).get("division", "")

    # ---- games -------------------------------------------------------------
    games = []
    for _, r in sched.iterrows():
        h, a = r.home_team, r.away_team
        if not ((r.home_division == "fbs") or (r.away_division == "fbs")):
            continue
        fbs2 = r.home_division == "fbs" and r.away_division == "fbs"
        m = fits[r.wk] if r.played else cur
        p = cr.project(m, h, a, bool(r.neutral))
        g = dict(id=str(int(r.game_id)), wk=int(r.wk), week=int(r.week),
                 post=r.season_type == "postseason",
                 date=_et(r.start_date)[0], kick=_et(r.start_date)[1],
                 home=h, away=a, neutral=bool(r.neutral), conf_game=bool(r.conf_game),
                 hconf=r.home_conference if isinstance(r.home_conference, str) else "FCS",
                 aconf=r.away_conference if isinstance(r.away_conference, str) else "FCS",
                 fcs=not fbs2,
                 pem=round(-p["margin"], 1),          # betting convention: home spread
                 pem_total=round(p["total"], 1),
                 home_wp=round(p["home_wp"], 3) if fbs2 else (FCS_WIN if r.home_division == "fbs" else 1 - FCS_WIN))
        if isinstance(r.notes, str) and r.notes:
            g["notes"] = r.notes
        L = lmap.get(int(r.game_id))
        if L is not None and not pd.isna(L.line_spread):
            g["line"] = round(float(L.line_spread), 1)
            if not pd.isna(L.get("line_total", np.nan)):
                g["line_total"] = round(float(L.line_total), 1)
            for k in ("tt_home", "tt_away"):           # team totals (SGO only)
                v = L.get(k, None)
                if v is not None and not pd.isna(v):
                    g[k] = round(float(v), 1)
            g["edge"] = round(abs(g["pem"] - g["line"]), 1)
            # side the model takes against the number
            g["pick"] = "home" if g["pem"] < g["line"] else ("away" if g["pem"] > g["line"] else "")
        if r.played:
            hs, as_ = int(r.home_points), int(r.away_points)
            g.update(hs=hs, as_=as_)
            if "line" in g and g.get("pick"):
                cover = (hs - as_) + g["line"]        # >0 home covered
                if cover == 0:
                    g["ats"] = "P"
                else:
                    home_cov = cover > 0
                    g["ats"] = "W" if (g["pick"] == "home") == home_cov else "L"
        games.append(g)

    # ---- records, SOS, trend ----------------------------------------------
    rec = {t: dict(w=0, l=0, cw=0, cl=0, opp=[]) for t in fbs.index}
    for g in games:
        if "hs" not in g:
            continue
        for me, op, mine, theirs in ((g["home"], g["away"], g["hs"], g["as_"]),
                                     (g["away"], g["home"], g["as_"], g["hs"])):
            if me not in rec:
                continue
            won = mine > theirs
            rec[me]["w" if won else "l"] += 1
            if g["conf_game"]:
                rec[me]["cw" if won else "cl"] += 1
            rec[me]["opp"].append(op)
    net_all = T["net"]
    trend = {t: [] for t in fbs.index}
    for wk in played_wks[1:] + [through + 1]:
        mt = (fits.get(wk) or cur)["teams"].set_index("team")
        for t in fbs.index:
            trend[t].append(round(float(mt.loc[t, "net"]), 1))

    # ---- season simulation -------------------------------------------------
    sim = simulate(sched, cur, fbs, conf_of, div_of, rec, cfg)

    teams = []
    fbs = fbs.sort_values("net", ascending=False)
    fbs["rank"] = np.arange(1, len(fbs) + 1)
    fbs["off_rank"] = fbs["off"].rank(ascending=False, method="min").astype(int)
    fbs["def_rank"] = fbs["def"].rank(ascending=True, method="min").astype(int)
    for t, r in fbs.iterrows():
        opp = rec[t]["opp"]
        sos = float(np.mean([net_all.get(o, 0.0) for o in opp])) if opp else 0.0
        rem = [g for g in games if "hs" not in g and t in (g["home"], g["away"]) and not g["post"]]
        rsos = float(np.mean([net_all.get(g["away"] if g["home"] == t else g["home"], 0.0)
                              for g in rem])) if rem else 0.0
        s = sim.get(t, {})
        i = info.get(t, {})
        teams.append(dict(
            team=t, abbr=i.get("abbr", ""), color=i.get("color", ""), alt=i.get("alt", ""),
            logo=i.get("logo", ""), mascot=i.get("mascot", ""),
            conf=conf_of.get(t, ""), cdiv=div_of.get(t, ""),
            rank=int(r["rank"]), off=round(float(r.off), 1), deff=round(float(-r["def"]), 1),
            net=round(float(r.net), 1), off_rank=int(r.off_rank), def_rank=int(r.def_rank),
            w=rec[t]["w"], l=rec[t]["l"], cw=rec[t]["cw"], cl=rec[t]["cl"],
            sos=round(sos, 1), rsos=round(rsos, 1), trend=trend[t],
            scar=round(float(T.loc[t, "scar"]), 2) if "scar" in T.columns else 0.0,
            exp_w=round(s.get("exp_w", 0), 1), exp_l=round(s.get("exp_l", 0), 1),
            reach=round(s.get("reach", 0), 4), win=round(s.get("win", 0), 4),
            cfp=round(s.get("cfp", 0), 4)))
    by = {t["team"]: t for t in teams}
    for c in set(conf_of.values()):
        ms = sorted([t for t in teams if t["conf"] == c], key=lambda x: -x["net"])
        for i, t in enumerate(ms):
            t["conf_rank"] = i + 1
    for t in teams:
        t["sos_rank"] = 1 + sum(1 for o in teams if o["sos"] > t["sos"])

    confs = []
    for c, tg in sim.get("_title", {}).items():
        if not tg:
            continue
        (a, b), share = tg
        p = cr.project(cur, a, b, neutral=True)
        confs.append(dict(conf=c, a=a, b=b, share=round(share, 3),
                          pem=round(-p["margin"], 1), a_wp=round(p["home_wp"], 3)))

    bt = {}
    try:
        with open(os.path.join(os.path.dirname(__file__) or ".", "data", "cfb-backtest.json")) as f:
            bt = json.load(f)
    except Exception:
        pass

    return dict(kind="cfb_ratings", season=int(season), through_week=through,
                next_week=next_wk,
                generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                lines_source=line_src, n_sims=N_SIMS,
                model=dict(hfa=cfg["hfa"], sd=cfg["sd_margin"], avg=round(cur["avg"], 1)),
                teams=teams, games=games, confs=sorted(confs, key=lambda x: x["conf"]),
                backtest=bt)


def simulate(sched, cur, fbs, conf_of, div_of, rec, cfg, n=N_SIMS, seed=7):
    """Monte Carlo of the rest of the regular season, conference title games
    and a 12-team CFP field. Team strength is itself uncertain, so every sim
    draws a shock per team that lasts the whole season -- that's what makes
    a 4-point favorite's next five games correlated, the way real seasons are."""
    rng = np.random.default_rng(seed)
    T = cur["teams"].set_index("team")
    names = list(fbs.index)
    ix = {t: i for i, t in enumerate(names)}
    K = len(names)
    net = np.array([T.loc[t, "net"] for t in names])
    shock = rng.normal(0, TEAM_SHOCK_SD, size=(n, K))
    true = net[None, :] + shock
    eps_sd = math.sqrt(max(cfg["sd_margin"] ** 2 - 2 * TEAM_SHOCK_SD ** 2, 25.0))

    W = np.tile(np.array([rec[t]["w"] for t in names], float), (n, 1))
    L = np.tile(np.array([rec[t]["l"] for t in names], float), (n, 1))
    CW = np.tile(np.array([rec[t]["cw"] for t in names], float), (n, 1))
    CL = np.tile(np.array([rec[t]["cl"] for t in names], float), (n, 1))
    exp_w = W[0].copy()

    rem = sched[(~sched.played) & (sched.season_type == "regular")]
    for _, r in rem.iterrows():
        h, a = r.home_team, r.away_team
        hi, ai = ix.get(h), ix.get(a)
        if hi is None and ai is None:
            continue
        if hi is None or ai is None:            # FBS vs FCS
            k = hi if hi is not None else ai
            won = rng.random(n) < FCS_WIN
            W[:, k] += won
            L[:, k] += ~won
            exp_w[k] += FCS_WIN
            continue
        hfa = 0.0 if r.neutral else cfg["hfa"]
        mu = true[:, hi] - true[:, ai] + hfa
        hw = (mu + rng.normal(0, eps_sd, n)) > 0
        exp_w[hi] += _phi((net[hi] - net[ai] + hfa) / cfg["sd_margin"])
        exp_w[ai] += 1 - _phi((net[hi] - net[ai] + hfa) / cfg["sd_margin"])
        W[:, hi] += hw; L[:, hi] += ~hw
        W[:, ai] += ~hw; L[:, ai] += hw
        if r.conf_game:
            CW[:, hi] += hw; CL[:, hi] += ~hw
            CW[:, ai] += ~hw; CL[:, ai] += hw

    reach = np.zeros(K)
    win = np.zeros(K)
    champ = np.zeros((n, K), bool)
    title = {}
    conf_list = sorted({c for c in conf_of.values() if c and c != "FBS Independents"})
    pct = CW / np.maximum(CW + CL, 1)
    # tiebreak where the real one (head-to-head, common opponents...) isn't
    # knowable in a sim: the team's simulated true strength
    key = pct + 1e-4 * true
    rows = np.arange(n)
    for c in conf_list:
        members = [ix[t] for t in names if conf_of.get(t) == c]
        if len(members) < 2:
            continue
        divs = sorted({div_of.get(names[m], "") for m in members} - {""})
        if len(divs) == 2:
            picks = []
            for d in divs:
                dm = np.array([m for m in members if div_of.get(names[m]) == d])
                picks.append(dm[np.argmax(key[:, dm], axis=1)])
            A, B = picks
        else:
            mm = np.array(members)
            order = np.argsort(-key[:, mm], axis=1)
            A, B = mm[order[:, 0]], mm[order[:, 1]]
        np.add.at(reach, A, 1)
        np.add.at(reach, B, 1)
        aw = (true[rows, A] - true[rows, B] + rng.normal(0, eps_sd, n)) > 0
        Wn = np.where(aw, A, B)
        Ln = np.where(aw, B, A)
        np.add.at(win, Wn, 1)
        champ[rows, Wn] = True
        W[rows, Wn] += 1
        L[rows, Ln] += 1
        pairs = pd.Series(list(zip(np.minimum(A, B), np.maximum(A, B)))).value_counts()
        (p0, p1), cnt = pairs.index[0], pairs.iloc[0]
        a_, b_ = names[p0], names[p1]
        if T.loc[b_, "net"] > T.loc[a_, "net"]:
            a_, b_ = b_, a_
        title[c] = ((a_, b_), cnt / n)

    # 12-team field: five highest-rated conference champions + seven at-large
    score = true + CFP_LOSS_PTS * (W - L) + CFP_CHAMP_BONUS * champ
    cfp = np.zeros(K)
    cs = np.where(champ, score, -np.inf)
    top5 = np.argsort(-cs, axis=1)[:, :5]
    inn = np.zeros((n, K), bool)
    inn[rows[:, None], top5] = True
    rest = np.where(inn, -np.inf, score)
    top7 = np.argsort(-rest, axis=1)[:, :7]
    inn[rows[:, None], top7] = True
    cfp = inn.mean(axis=0)

    out = {"_title": title}
    for t, i in ix.items():
        out[t] = dict(exp_w=float(exp_w[i]), exp_l=0.0, reach=reach[i] / n,
                      win=win[i] / n, cfp=float(cfp[i]))
    # expected losses = scheduled regular-season games - expected wins
    nreg = {t: rec[t]["w"] + rec[t]["l"] for t in names}
    for _, r in rem.iterrows():
        for t in (r.home_team, r.away_team):
            if t in nreg:
                nreg[t] += 1
    for t in names:
        out[t]["exp_l"] = float(nreg[t] - out[t]["exp_w"])
    return out


# --------------------------------------------------------------------------
# FCS -- its own file, its own tab
# --------------------------------------------------------------------------

def _game_rec(r, m, lmap, fbs_side=False):
    h, a = r.home_team, r.away_team
    p = cr.project(m, h, a, bool(r.neutral))
    g = dict(id=str(int(r.game_id)), wk=int(r.wk), week=int(r.week),
             post=r.season_type == "postseason",
             date=_et(r.start_date)[0], kick=_et(r.start_date)[1],
             home=h, away=a, neutral=bool(r.neutral), conf_game=bool(r.conf_game),
             hconf=r.home_conference if isinstance(r.home_conference, str) else "",
             aconf=r.away_conference if isinstance(r.away_conference, str) else "",
             hdiv=r.home_division, adiv=r.away_division,
             pem=round(-p["margin"], 1), pem_total=round(p["total"], 1),
             home_wp=round(p["home_wp"], 3))
    if isinstance(r.notes, str) and r.notes:
        g["notes"] = r.notes
    L = lmap.get(int(r.game_id))
    if L is not None and not pd.isna(L.line_spread):
        g["line"] = round(float(L.line_spread), 1)
        if not pd.isna(L.get("line_total", np.nan)):
            g["line_total"] = round(float(L.line_total), 1)
        for k in ("tt_home", "tt_away"):           # team totals (SGO only)
            v = L.get(k, None)
            if v is not None and not pd.isna(v):
                g[k] = round(float(v), 1)
        g["edge"] = round(abs(g["pem"] - g["line"]), 1)
        g["pick"] = "home" if g["pem"] < g["line"] else ("away" if g["pem"] > g["line"] else "")
    if r.played:
        hs, as_ = int(r.home_points), int(r.away_points)
        g.update(hs=hs, as_=as_)
        if "line" in g and g.get("pick"):
            c = (hs - as_) + g["line"]
            g["ats"] = "P" if c == 0 else ("W" if (g["pick"] == "home") == (c > 0) else "L")
    return g


def build_fcs(season):
    """Ratings for every FCS team, rated off ALL its Division I games, plus a
    weekly board of FCS games. Ratings are shown relative to the average FCS
    team (0 = average FCS team) -- spreads are the same either way. No season
    sims here: the 24-team FCS playoff is picked by committee with auto-bids
    from a dozen conferences, too loose to simulate honestly."""
    cfg = dict(cr.FCS_CONFIG)
    extra = cfbd_games(season)
    if len(extra):
        cr.EXTRA_GAMES[season] = extra
    sched = cr.load_schedule(season, "di")
    prior = cr.season_prior(season, cfg)
    info = team_info(season)
    lines, line_src = get_lines(season)
    lmap = _merge_sgo(lines, season, sched)
    if any(bool(v.get("sgo", False)) for v in lmap.values()):
        line_src = (line_src + "; " if line_src else "") + "upcoming games: SportsGameOdds"

    played_wks = sorted(sched[sched.played].wk.unique())
    through = int(max(played_wks)) if played_wks else 0
    next_wk = int(sched[~sched.played].wk.min()) if (~sched.played).any() else through + 1
    fits = {wk: cr.fit(sched, cfg, prior, as_of_wk=wk) for wk in played_wks}
    cur = cr.fit(sched, cfg, prior)
    T = cur["teams"].set_index("team")
    F = T[T["div"] == "fcs"].copy()
    # re-center on the average FCS team
    oc, dc = F["off"].mean(), F["def"].mean()
    fcs_shift = float((F["net"]).mean())

    games, rec = [], {t: dict(w=0, l=0, cw=0, cl=0, opp=[], fbsw=0, fbsl=0) for t in F.index}
    for _, r in sched.iterrows():
        if r.home_division != "fcs" and r.away_division != "fcs":
            continue
        m = fits[r.wk] if r.played else cur
        g = _game_rec(r, m, lmap)
        games.append(g)
        if "hs" in g:
            for me, op, mine, theirs, odiv in ((g["home"], g["away"], g["hs"], g["as_"], g["adiv"]),
                                               (g["away"], g["home"], g["as_"], g["hs"], g["hdiv"])):
                if me not in rec:
                    continue
                won = mine > theirs
                rec[me]["w" if won else "l"] += 1
                if g["conf_game"]:
                    rec[me]["cw" if won else "cl"] += 1
                if odiv == "fbs":
                    rec[me]["fbsw" if won else "fbsl"] += 1
                rec[me]["opp"].append(op)

    trend = {t: [] for t in F.index}
    for wk in played_wks[1:] + [through + 1]:
        mt = (fits.get(wk) or cur)["teams"].set_index("team")
        sh = float(mt[mt["div"] == "fcs"]["net"].mean())
        for t in F.index:
            trend[t].append(round(float(mt.loc[t, "net"]) - sh, 1))

    F = F.sort_values("net", ascending=False)
    F["rank"] = np.arange(1, len(F) + 1)
    F["off_rank"] = F["off"].rank(ascending=False, method="min").astype(int)
    F["def_rank"] = F["def"].rank(ascending=True, method="min").astype(int)
    conf_of = {}
    for _, r in sched.iterrows():
        for side in ("home", "away"):
            if r[f"{side}_division"] == "fcs" and isinstance(r[f"{side}_conference"], str):
                conf_of[r[f"{side}_team"]] = r[f"{side}_conference"]
    teams = []
    for t, r in F.iterrows():
        opp = rec[t]["opp"]
        sos = float(np.mean([T.loc[o, "net"] - fcs_shift for o in opp if o in T.index])) if opp else 0.0
        rem = [g for g in games if "hs" not in g and t in (g["home"], g["away"]) and not g["post"]]
        exp_w = rec[t]["w"] + sum((g["home_wp"] if g["home"] == t else 1 - g["home_wp"]) for g in rem)
        i = info.get(t, {})
        teams.append(dict(
            team=t, abbr=i.get("abbr", ""), color=i.get("color", ""), logo=i.get("logo", ""),
            mascot=i.get("mascot", ""), conf=conf_of.get(t, ""), cdiv="",
            rank=int(r["rank"]), off=round(float(r.off - oc), 1), deff=round(float(-(r["def"] - dc)), 1),
            net=round(float(r.net - fcs_shift), 1), off_rank=int(r.off_rank), def_rank=int(r.def_rank),
            w=rec[t]["w"], l=rec[t]["l"], cw=rec[t]["cw"], cl=rec[t]["cl"],
            fbsw=rec[t]["fbsw"], fbsl=rec[t]["fbsl"],
            sos=round(sos, 1), rsos=0.0, trend=trend[t],
            exp_w=round(exp_w, 1), exp_l=round(rec[t]["w"] + rec[t]["l"] + len(rem) - exp_w, 1),
            reach=None, win=None, cfp=None))
    for c in set(conf_of.values()):
        ms = sorted([x for x in teams if x["conf"] == c], key=lambda x: -x["net"])
        for k, x in enumerate(ms):
            x["conf_rank"] = k + 1
    for x in teams:
        x["sos_rank"] = 1 + sum(1 for o in teams if o["sos"] > x["sos"])

    # what an FBS team looks like on this scale, for reference on team pages
    fbs_avg = round(float(T[T["div"] == "fbs"]["net"].mean() - fcs_shift), 1)
    n_fcs2 = sum(1 for g in games if g["hdiv"] == "fcs" and g["adiv"] == "fcs")
    bt = {}
    try:
        with open(os.path.join(os.path.dirname(__file__) or ".", "data", "fcs-backtest.json")) as f:
            bt = json.load(f)
    except Exception:
        pass
    return dict(kind="fcs_ratings", season=int(season), through_week=through, next_week=next_wk,
                generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                lines_source=line_src, full_fcs=bool(len(extra)) or n_fcs2 > 0,
                fbs_avg=fbs_avg,
                model=dict(hfa=cfg["hfa"], sd=cfg["sd_margin"], avg=round(cur["avg"], 1)),
                teams=teams, games=games, backtest=bt)


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2026"
    out = sys.argv[2] if len(sys.argv) > 2 else f"cfb-ratings-{season}.json"
    data = build(int(season))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    ng = sum(1 for g in data["games"] if "line" in g)
    print(f"wrote {out}: {len(data['teams'])} teams through week {data['through_week']}, "
          f"{len(data['games'])} games ({ng} with lines), "
          f"{os.path.getsize(out) / 1024:.0f} KB", file=sys.stderr)

    # FCS file next to it: data/cfb-ratings-2026.json -> data/fcs-ratings-2026.json
    out2 = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        os.path.dirname(out) or ".", f"fcs-ratings-{season}.json")
    try:
        fdata = build_fcs(int(season))
        with open(out2, "w") as f:
            json.dump(fdata, f, separators=(",", ":"))
        n2 = sum(1 for g in fdata["games"] if "line" in g)
        print(f"wrote {out2}: {len(fdata['teams'])} FCS teams, {len(fdata['games'])} games "
              f"({n2} with lines), {os.path.getsize(out2) / 1024:.0f} KB", file=sys.stderr)
        if not fdata["full_fcs"]:
            print("  note: no FCS-vs-FCS games found -- is CFBD_KEY set?", file=sys.stderr)
    except Exception as e:
        print(f"FCS build failed: {type(e).__name__}: {e}", file=sys.stderr)
