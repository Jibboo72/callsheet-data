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
# backtest
# --------------------------------------------------------------------------

def _archive_for(games, coords_of):
    """games: DataFrame with game_id, kick (UTC), venue key. Fetch archive
    weather per venue over its own date span, return game_id -> wx."""
    res = {}
    # one request per venue per season: Open-Meteo weighs requests by how
    # much data they return, so a decade in one call costs more than ten
    # one-season calls and risks the free daily cap
    for (key, _), gg in games.groupby(["venue", "season"]):
        v = coords_of(key)
        if not v or not v[2]:
            continue
        start = gg.kick.min().strftime("%Y-%m-%d")
        end = (gg.kick.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fr = hourly_frames([(v[0], v[1])], ARCHIVE,
                           dict(start_date=start, end_date=end,
                                hourly="temperature_2m,precipitation,snowfall,wind_speed_10m"))[0]
        time.sleep(0.6)
        for _, r in gg.iterrows():
            wx = window(fr, r.kick)
            if wx:
                res[r.game_id] = wx
    return res


def _grade(df, label):
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


def backtest():
    report = {"kind": "weather_backtest",
              "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
              "thresholds": dict(wind_mph=WIND_MPH, precip_in=PRECIP_IN)}

    # NFL 2016-2025: re-check the angle with Open-Meteo instead of the
    # stadium reading, since the live flag will use Open-Meteo
    g = pd.read_csv(NFL_GAMES, low_memory=False)
    g = g[(g.game_type == "REG") & g.season.between(2016, 2025) & g.total.notna()
          & g.total_line.notna()].copy()
    g["kick"] = g.apply(nfl_kickoff, axis=1)
    g["venue"] = g.stadium_id
    g = g[g.venue.map(lambda k: bool(NFL_VENUES.get(k)) and NFL_VENUES[k][2])]
    missing = sorted(set(pd.read_csv(NFL_GAMES, low_memory=False).query(
        "season>=2016").stadium_id.dropna()) - set(NFL_VENUES))
    if missing:
        print(f"NFL venues without coordinates (skipped): {missing}", file=sys.stderr)
    wx = _archive_for(g, NFL_VENUES.get)
    g = g[g.game_id.isin(wx)].copy()
    g["res"] = g.total - g.total_line
    g["f_wind"] = g.game_id.map(lambda x: "wind" in flags(wx[x]))
    g["f_wet"] = g.game_id.map(lambda x: any(f in ("rain", "snow") for f in flags(wx[x])))
    report["nfl"] = _grade(g, f"NFL 2016-2025, {len(g)} outdoor games")
    print("NFL:", json.dumps(report["nfl"])[:600], file=sys.stderr)

    # College 2021-2025 (closing totals from the free cfbfastR file)
    import cfb_ratings as cr
    lines = cr.load_lines()
    rows = []
    for s in range(2021, 2026):
        sc = cr.load_schedule(s, "di")
        sc = sc[sc.played & ~sc.neutral].copy()
        sc["kick"] = pd.to_datetime(sc.start_date, utc=True, errors="coerce")
        sc["venue"] = sc.home_team
        sc["fbs2"] = (sc.home_division == "fbs") & (sc.away_division == "fbs")
        sc["fcs2"] = (sc.home_division == "fcs") & (sc.away_division == "fcs")
        ven = cfb_venues(s)
        sc = sc.merge(lines[["game_id", "line_total"]], on="game_id", how="inner")
        sc = sc[sc.line_total.notna()]
        w = _archive_for(sc, ven.get)
        sc = sc[sc.game_id.isin(w)].copy()
        sc["res"] = sc.home_points + sc.away_points - sc.line_total
        sc["f_wind"] = sc.game_id.map(lambda x: "wind" in flags(w[x]))
        sc["f_wet"] = sc.game_id.map(lambda x: any(f in ("rain", "snow") for f in flags(w[x])))
        rows.append(sc)
        print(f"college {s}: {len(sc)} games with weather", file=sys.stderr)
    c = pd.concat(rows)
    report["cfb_fbs"] = _grade(c[c.fbs2], f"FBS 2021-2025, {int(c.fbs2.sum())} games")
    report["cfb_fcs"] = _grade(c[c.fcs2], f"FCS 2022-2025, {int(c.fcs2.sum())} games")
    report["cfb_all"] = _grade(c, f"all college 2021-2025, {len(c)} games")
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
