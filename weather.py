#!/usr/bin/env python3
"""
Game-time weather for NFL and college games, from Open-Meteo (free, no key).

TWO JOBS, ONE DEFINITION OF "BAD WEATHER"
  forecast   Upcoming outdoor games: wind, rain/snow and temperature over
             the kickoff-to-final-whistle window, plus a WEATHER UNDER flag.
                 python3 weather.py forecast 2026 data/weather-2026.json
  backtest   Same flags on past games, from Open-Meteo's historical archive,
             checked against the closing total: how often did it go under?
                 python3 weather.py backtest data/weather-backtest.json

  The live flag uses exactly the thresholds the backtest graded, so what
  the app flags is what was tested.

WHY THIS ANGLE
  NFL 2016-2025, outdoor games with rain/snow or 15+ mph wind went under the
  closing total 190-135 (58.5%), landing 2.3 points under on average. Books
  shade totals about a point for weather; the real effect is ~3.5. (That
  test used the wind measured at kickoff; this module re-checks it with
  Open-Meteo, the same source the live forecast uses.)

THRESHOLDS
  WIND    max sustained wind (10 m) over the game window >= 15 mph
  PRECIP  backtest: >= 0.05 in of rain, or any snow, during the game window
          forecast: expected >= 0.05 in, or any snow, or >= 60% chance
  Domes and retractable roofs are never flagged (retractables close in
  bad weather). A game window is kickoff to kickoff + 3.5 hours.
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m"
UNITS = dict(wind_speed_unit="mph", temperature_unit="fahrenheit", precipitation_unit="inch",
             timezone="UTC")
WIND_MPH = 15.0
PRECIP_IN = 0.05
POP_PCT = 60
WINDOW_H = 3.5
UA = {"User-Agent": "callsheet-data (personal betting app)"}

# NFL venues: (lat, lon, outdoor). Retractable roofs count as indoor.
NFL_VENUES = {
    "ATL97": (33.7554, -84.4008, False), "BAL00": (39.2780, -76.6227, True),
    "BOS00": (42.0909, -71.2643, True), "BUF00": (42.7738, -78.7870, True),
    "BUF01": (42.7738, -78.7870, True),
    "CAR00": (35.2258, -80.8528, True), "CHI98": (41.8623, -87.6167, True),
    "CIN00": (39.0955, -84.5161, True), "CLE00": (41.5061, -81.6995, True),
    "DAL00": (32.7473, -97.0945, False), "DEN00": (39.7439, -105.0201, True),
    "DET00": (42.3400, -83.0456, False), "GNB00": (44.5013, -88.0622, True),
    "HOU00": (29.6847, -95.4107, False), "IND00": (39.7601, -86.1639, False),
    "JAX00": (30.3239, -81.6373, True), "KAN00": (39.0489, -94.4839, True),
    "LAX01": (33.9535, -118.3392, False), "LON00": (51.5560, -0.2795, True),
    "LON02": (51.6043, -0.0664, True), "MAD01": (40.4531, -3.6883, False),
    "MEL00": (-37.8200, 144.9834, True), "MEX00": (19.3029, -99.1505, True),
    "MIA00": (25.9580, -80.2389, True), "MIN01": (44.9737, -93.2577, False),
    "MUN01": (48.2188, 11.6247, True), "FRA00": (50.0686, 8.6455, True),
    "NAS00": (36.1665, -86.7713, True), "NOR00": (29.9511, -90.0812, False),
    "NYC01": (40.8135, -74.0745, True), "PAR00": (48.9245, 2.3602, True),
    "PHI00": (39.9008, -75.1675, True), "PHO00": (33.5276, -112.2626, False),
    "PIT00": (40.4468, -80.0158, True), "RIO00": (-22.9122, -43.2302, True),
    "SAO00": (-23.5453, -46.4742, True),
    "SEA00": (47.5952, -122.3316, True), "SFO01": (37.4030, -121.9700, True),
    "TAM00": (27.9759, -82.5033, True), "VEG00": (36.0909, -115.1833, False),
    "WAS00": (38.9076, -76.8645, True),
    # older homes that appear in 2016-2020 backtest seasons
    "LAX97": (34.0141, -118.2879, True), "LAX99": (33.8644, -118.2611, True),
    "SDG00": (32.7831, -117.1196, True), "OAK00": (37.7516, -122.2005, True),
    "STL00": (38.6328, -90.1885, False), "MIN98": (44.9765, -93.2242, True),
    "ATL00": (33.7577, -84.4008, False), "LON01": (51.4560, -0.3416, True),
    "GER00": (48.2188, 11.6247, True),
}

NFL_GAMES = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


# --------------------------------------------------------------------------
# Open-Meteo
# --------------------------------------------------------------------------

def _get(url, params, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=90)
            if r.status_code == 429:          # rate limited: back off hard
                print(f"open-meteo 429, waiting {60 * (k + 1)}s", file=sys.stderr)
                time.sleep(60 * (k + 1))
                continue
            r.raise_for_status()
            d = r.json()
            return d if isinstance(d, list) else [d]
        except Exception as e:
            if k == tries - 1:
                print(f"open-meteo failed: {e}", file=sys.stderr)
                return None
            time.sleep(5 * (k + 1))


def hourly_frames(points, url, extra):
    """points: list of (lat, lon). One request per batch of 25 locations.
    Returns list of DataFrames (UTC hourly), aligned with points."""
    out = []
    for i in range(0, len(points), 25):
        chunk = points[i:i + 25]
        p = dict(latitude=",".join(f"{a:.4f}" for a, _ in chunk),
                 longitude=",".join(f"{b:.4f}" for _, b in chunk),
                 hourly=extra.get("hourly", HOURLY), **UNITS)
        p.update({k: v for k, v in extra.items() if k != "hourly"})
        d = _get(url, p)
        if d is None:
            out += [None] * len(chunk)
            continue
        for loc in d:
            h = loc.get("hourly") or {}
            if not h:
                out.append(None)
                continue
            f = pd.DataFrame(h)
            f["time"] = pd.to_datetime(f.time, utc=True)
            out.append(f.set_index("time"))
        time.sleep(1)
    return out


def window(frame, kickoff):
    """Weather over one game window -> dict, or None if not covered."""
    if frame is None or pd.isna(kickoff):
        return None
    k = pd.Timestamp(kickoff)
    k = k.tz_localize("UTC") if k.tzinfo is None else k.tz_convert("UTC")
    w = frame[(frame.index >= k.floor("h")) & (frame.index <= k + pd.Timedelta(hours=WINDOW_H))]
    if not len(w) or w.wind_speed_10m.isna().all():
        return None
    out = dict(wind=round(float(w.wind_speed_10m.max()), 1),
               gust=round(float(w.wind_gusts_10m.max()), 1) if "wind_gusts_10m" in w else None,
               temp=round(float(w.temperature_2m.mean()), 0),
               precip=round(float(w.precipitation.sum()), 2),
               snow=round(float(w.snowfall.sum()), 2) if "snowfall" in w else 0.0)
    if "precipitation_probability" in w:
        out["pop"] = int(np.nanmax(w.precipitation_probability.values)) \
            if w.precipitation_probability.notna().any() else None
    return out


def flags(wx, forecast=False):
    if not wx:
        return []
    f = []
    if wx["wind"] >= WIND_MPH:
        f.append("wind")
    wet = wx["precip"] >= PRECIP_IN or (wx.get("snow") or 0) > 0
    if forecast and (wx.get("pop") or 0) >= POP_PCT:
        wet = True
    if wet:
        f.append("snow" if (wx.get("snow") or 0) > 0 else "rain")
    return f


# --------------------------------------------------------------------------
# venues
# --------------------------------------------------------------------------

def cfb_venues(season):
    """School -> (lat, lon, outdoor) for its home stadium."""
    import cfb_ratings as cr
    t = cr._read(cr.BASE + f"/team_info/parquet/cfb_team_info_{season}.parquet",
                 f"cfb_team_info_{season}.parquet")
    out = {}
    for _, r in t.iterrows():
        try:
            la, lo = float(r.latitude), float(r.longitude)
        except (TypeError, ValueError):
            continue
        if math.isnan(la) or math.isnan(lo):
            continue
        out[r.school] = (la, lo, not bool(r.dome) if not pd.isna(r.dome) else True)
    return out


def nfl_kickoff(r):
    """nflverse gameday + gametime are US Eastern."""
    try:
        t = pd.Timestamp(f"{r.gameday} {r.gametime}").tz_localize("America/New_York")
        return t.tz_convert("UTC")
    except Exception:
        return pd.NaT


# --------------------------------------------------------------------------
# forecast
# --------------------------------------------------------------------------

def forecast(season, days=8):
    now = pd.Timestamp.now(tz="UTC")
    horizon = now + pd.Timedelta(days=days)
    extra = dict(hourly=HOURLY + ",precipitation_probability", forecast_days=min(days + 2, 16))
    out = {"kind": "weather", "season": int(season),
           "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "thresholds": dict(wind_mph=WIND_MPH, precip_in=PRECIP_IN, pop_pct=POP_PCT),
           "nfl": {}, "cfb": {}}

    # NFL
    g = pd.read_csv(NFL_GAMES, low_memory=False)
    g = g[(g.season == int(season)) & g.home_score.isna()].copy()
    g["kick"] = g.apply(nfl_kickoff, axis=1)
    g = g[(g.kick >= now - pd.Timedelta(hours=4)) & (g.kick <= horizon)]
    rows = [(r, NFL_VENUES.get(r.stadium_id)) for _, r in g.iterrows()]
    todo = [(r, v) for r, v in rows if v and v[2]]
    frames = hourly_frames([(v[0], v[1]) for _, v in todo], FORECAST, extra)
    for (r, v), fr in zip(todo, frames):
        wx = window(fr, r.kick)
        if wx:
            wx["flags"] = flags(wx, forecast=True)
            wx["kick"] = r.kick.strftime("%Y-%m-%dT%H:%MZ")
            out["nfl"][r.game_id] = wx
    for r, v in rows:
        if r.game_id not in out["nfl"]:
            out["nfl"][r.game_id] = {"indoor": True} if (v and not v[2]) else {"unknown": True}

    # college: FBS + FCS home games at the home team's stadium
    import cfb_ratings as cr
    try:
        import export_cfb_ratings as ex
        extra_games = ex.cfbd_games(int(season))
        if len(extra_games):
            cr.EXTRA_GAMES[int(season)] = extra_games
    except Exception as e:
        print(f"FCS games unavailable: {e}", file=sys.stderr)
    s = cr.load_schedule(int(season), "di")
    s = s[~s.played].copy()
    s["kick"] = pd.to_datetime(s.start_date, utc=True, errors="coerce")
    s = s[(s.kick >= now - pd.Timedelta(hours=4)) & (s.kick <= horizon)]
    ven = cfb_venues(int(season))
    todo = []
    for _, r in s.iterrows():
        v = ven.get(r.home_team)
        gid = str(int(r.game_id))
        if r.neutral:
            out["cfb"][gid] = {"neutral": True}
        elif not v:
            out["cfb"][gid] = {"unknown": True}
        elif not v[2]:
            out["cfb"][gid] = {"indoor": True}
        else:
            todo.append((gid, r.kick, v))
    frames = hourly_frames([(v[0], v[1]) for _, _, v in todo], FORECAST, extra)
    for (gid, kick, v), fr in zip(todo, frames):
        wx = window(fr, kick)
        if wx:
            wx["flags"] = flags(wx, forecast=True)
            out["cfb"][gid] = wx
    nf = sum(1 for x in out["nfl"].values() if x.get("flags"))
    cf = sum(1 for x in out["cfb"].values() if x.get("flags"))
    print(f"forecast: NFL {len(out['nfl'])} games ({nf} flagged), "
          f"college {len(out['cfb'])} games ({cf} flagged)", file=sys.stderr)
    return out


# --------------------------------------------------------------------------
# backtest -- resumable
# --------------------------------------------------------------------------
# Open-Meteo's free tier counts a big historical request as several calls,
# and a full NFL + college backtest is ~1,400 venue-seasons. So the backtest
# keeps every game's weather it has fetched in data/weather-hist.json,
# stops cleanly when its time budget is up (or the API says slow down),
# grades whatever it has, and the next run picks up where it stopped.
# The report says how complete it is; rerun until it says 100%.

HIST = "data/weather-hist.json"
BUDGET_MIN = float(os.environ.get("WX_BUDGET_MIN", "40"))
_T0 = time.time()
_STOP = {"why": None}


def _over_budget():
    if _STOP["why"]:
        return True
    if (time.time() - _T0) / 60 > BUDGET_MIN:
        _STOP["why"] = f"time budget ({BUDGET_MIN:.0f} min) used"
        return True
    return False


# Historical weather comes from the Iowa Environmental Mesonet's archive of
# airport (ASOS) observations -- measured, hourly, free, no key. (Open-Meteo's
# historical server timed out on every request from GitHub's runners.) Each
# stadium uses the nearest airport station within 40 km. US games only;
# international NFL games are skipped in the backtest.
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_NET = "https://mesonet.agron.iastate.edu/geojson/network/{st}_ASOS.geojson"
STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
          "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()
_FAILS = {"n": 0}


def _http(url, params, timeout=45):
    """GET with fail-fast: 4 straight failures stops the run cleanly."""
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        if r.status_code == 429:
            print("rate limited -- waiting 65s", file=sys.stderr)
            time.sleep(65)
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
        r.raise_for_status()
        _FAILS["n"] = 0
        return r
    except Exception as e:
        _FAILS["n"] += 1
        print(f"request failed ({_FAILS['n']} in a row): {str(e)[:160]}", file=sys.stderr)
        if _FAILS["n"] >= 4:
            _STOP["why"] = f"weather source unreachable: {str(e)[:120]}"
        time.sleep(3)
        return None


def _stations(store):
    """All US ASOS stations (id, lat, lon), cached in the progress file."""
    if store.get("_stations"):
        return store["_stations"]
    out = []
    for st in STATES:
        r = _http(IEM_NET.format(st=st), None, timeout=30)
        if r is None:
            if _STOP["why"]:
                break
            continue
        for f in r.json().get("features", []):
            c = (f.get("geometry") or {}).get("coordinates") or []
            sid = (f.get("properties") or {}).get("sid") or f.get("id")
            if sid and len(c) == 2:
                out.append([sid, c[1], c[0]])
    print(f"ASOS stations: {len(out)}", file=sys.stderr)
    if len(out) > 500:
        store["_stations"] = out
    return out


def _nearest(stations, lat, lon, max_km=40.0):
    best, bd = None, 1e9
    for sid, la, lo in stations:
        d = 6371 * 2 * math.asin(math.sqrt(
            math.sin(math.radians(la - lat) / 2) ** 2 +
            math.cos(math.radians(lat)) * math.cos(math.radians(la)) *
            math.sin(math.radians(lo - lon) / 2) ** 2))
        if d < bd:
            best, bd = sid, d
    return best if bd <= max_km else None


def _asos_frame(sid, start, end):
    """Hourly observations for one station -> frame shaped like Open-Meteo's
    (wind_speed_10m mph, precipitation in, snowfall flag, temperature_2m F)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    params = [("station", sid), ("data", "sknt"), ("data", "p01i"), ("data", "tmpf"),
              ("data", "wxcodes"), ("year1", s.year), ("month1", s.month), ("day1", s.day),
              ("year2", e.year), ("month2", e.month), ("day2", e.day), ("tz", "Etc/UTC"),
              ("format", "onlycomma"), ("latlon", "no"), ("missing", "M"), ("trace", "0.0001"),
              ("direct", "no"), ("report_type", "3")]
    r = _http(IEM_ASOS, params, timeout=60)
    if r is None:
        return None
    from io import StringIO
    try:
        f = pd.read_csv(StringIO(r.text), na_values=["M"])
    except Exception:
        return None
    if not len(f) or "valid" not in f:
        return pd.DataFrame()
    f["time"] = pd.to_datetime(f.valid, utc=True, errors="coerce")
    f = f.dropna(subset=["time"]).set_index("time")
    out = pd.DataFrame(index=f.index)
    out["wind_speed_10m"] = pd.to_numeric(f.sknt, errors="coerce") * 1.15078
    out["precipitation"] = pd.to_numeric(f.p01i, errors="coerce").fillna(0.0)
    out["temperature_2m"] = pd.to_numeric(f.tmpf, errors="coerce")
    out["snowfall"] = f.wxcodes.astype(str).str.contains("SN").astype(float)
    return out


def _archive_for(games, coords_of, store, stations):
    """Fill store[game_id] = weather, one request per stadium per season,
    skipping games already in the store."""
    done = 0
    todo = games[~games.game_id.astype(str).isin(store)]
    for (key, _), gg in todo.groupby(["venue", "season"]):
        if _over_budget():
            break
        v = coords_of(key)
        if not v or not v[2]:
            for gid in gg.game_id.astype(str):
                store[gid] = None               # indoor: nothing to fetch
            continue
        sid = _nearest(stations, v[0], v[1])
        if not sid:
            for gid in gg.game_id.astype(str):
                store[gid] = None               # no airport nearby (or abroad)
            continue
        start = gg.kick.min().strftime("%Y-%m-%d")
        end = (gg.kick.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fr = _asos_frame(sid, start, end)
        if fr is None:
            if _STOP["why"]:
                break
            continue
        for _, r in gg.iterrows():
            store[str(r.game_id)] = window(fr if len(fr) else None, r.kick)
        done += 1
        time.sleep(0.3)
    return done


def _grade(df, label):
    if not len(df) or "res" not in df:
        return {"label": label + " (no weather data yet)"}
    d = df[df.res != 0]
    out = {"label": label}
    for name, m in (("all outdoor", d.res == d.res),
                    ("wind 15+", d.f_wind), ("rain/snow", d.f_wet),
                    ("either", d.f_wind | d.f_wet), ("both", d.f_wind & d.f_wet),
                    ("neither", ~(d.f_wind | d.f_wet))):
        x = d[m]
        u, o = int((x.res < 0).sum()), int((x.res > 0).sum())
        out[name] = dict(under=u, over=o, pct=round(u / (u + o), 3) if u + o else None,
                         avg=round(float(x.res.mean()), 2) if len(x) else None,
                         units=round(u * 0.909 - o, 1))
    by = {}
    for s, x in d[d.f_wind | d.f_wet].groupby("season"):
        by[int(s)] = f"{int((x.res < 0).sum())}-{int((x.res > 0).sum())}"
    out["either_by_season"] = by
    return out


def _flagged(df, store):
    df = df[df.game_id.astype(str).map(lambda g: store.get(g) is not None)].copy()
    if not len(df):
        return df
    wx = df.game_id.astype(str).map(store)
    df["f_wind"] = wx.map(lambda w: "wind" in flags(w))
    df["f_wet"] = wx.map(lambda w: any(f in ("rain", "snow") for f in flags(w)))
    return df


def backtest():
    try:
        store = json.load(open(HIST))
    except Exception:
        store = {}
    store.setdefault("nfl", {})
    store.setdefault("cfb", {})
    stations = _stations(store)
    if not stations:
        _STOP["why"] = _STOP["why"] or "could not load the airport station list"

    # NFL 2016-2025 first (smallest, and the angle we're re-checking)
    g = pd.read_csv(NFL_GAMES, low_memory=False)
    g = g[(g.game_type == "REG") & g.season.between(2016, 2025) & g.total.notna()
          & g.total_line.notna()].copy()
    g["kick"] = g.apply(nfl_kickoff, axis=1)
    g["venue"] = g.stadium_id
    g = g[g.venue.map(lambda k: bool(NFL_VENUES.get(k)) and NFL_VENUES[k][2])]
    _archive_for(g, NFL_VENUES.get, store["nfl"], stations)

    # College 2021-2025, newest season first, FBS-involved before FCS-only
    import cfb_ratings as cr
    lines = cr.load_lines()
    frames = []
    for s in range(2025, 2020, -1):
        sc = cr.load_schedule(s, "di")
        sc = sc[sc.played & ~sc.neutral].copy()
        sc["kick"] = pd.to_datetime(sc.start_date, utc=True, errors="coerce")
        sc["venue"] = sc.home_team
        sc["fbs2"] = (sc.home_division == "fbs") & (sc.away_division == "fbs")
        sc["fcs2"] = (sc.home_division == "fcs") & (sc.away_division == "fcs")
        sc = sc.merge(lines[["game_id", "line_total"]], on="game_id", how="inner")
        sc = sc[sc.line_total.notna()]
        frames.append((s, sc, cfb_venues(s)))
    for s, sc, ven in frames:
        _archive_for(sc[sc.home_division == "fbs"], ven.get, store["cfb"], stations)
    for s, sc, ven in frames:
        _archive_for(sc[sc.home_division != "fbs"], ven.get, store["cfb"], stations)

    os.makedirs(os.path.dirname(HIST) or ".", exist_ok=True)
    with open(HIST, "w") as f:
        json.dump(store, f, separators=(",", ":"))

    # grade whatever is in hand
    report = {"kind": "weather_backtest",
              "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
              "thresholds": dict(wind_mph=WIND_MPH, precip_in=PRECIP_IN),
              "stopped": _STOP["why"]}
    g["res"] = g.total - g.total_line
    gn = _flagged(g, store["nfl"])
    report["nfl"] = _grade(gn, f"NFL 2016-2025, {len(gn)} outdoor games")
    c = pd.concat([sc for _, sc, _ in frames])
    c["res"] = c.home_points + c.away_points - c.line_total
    cc = _flagged(c, store["cfb"])
    report["cfb_fbs"] = _grade(cc[cc.fbs2], f"FBS 2021-2025, {int(cc.fbs2.sum())} games")
    report["cfb_fcs"] = _grade(cc[cc.fcs2], f"FCS 2022-2025, {int(cc.fcs2.sum())} games")
    report["cfb_all"] = _grade(cc, f"all college 2021-2025, {len(cc)} games")
    nfl_have = g.game_id.astype(str).isin(store["nfl"]).sum()
    cfb_have = c.game_id.astype(str).isin(store["cfb"]).sum()
    report["coverage"] = dict(nfl=f"{nfl_have}/{len(g)}", cfb=f"{cfb_have}/{len(c)}",
                              pct=round(100 * (nfl_have + cfb_have) / max(1, len(g) + len(c)), 1))
    print(f"coverage: NFL {nfl_have}/{len(g)}, college {cfb_have}/{len(c)} "
          f"({report['coverage']['pct']}%)" + (f" -- stopped: {_STOP['why']}" if _STOP["why"] else ""),
          file=sys.stderr)
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    if cmd == "forecast":
        season = sys.argv[2] if len(sys.argv) > 2 else "2026"
        path = sys.argv[3] if len(sys.argv) > 3 else f"data/weather-{season}.json"
        data = forecast(int(season))
    else:
        path = sys.argv[2] if len(sys.argv) > 2 else "data/weather-backtest.json"
        data = backtest()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {path}", file=sys.stderr)
