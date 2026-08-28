#!/usr/bin/env python3
"""
Turn nflverse play-by-play into a compact weekly receiving file for the call sheet.

nflverse's ready-made weekly stats file has receptions, targets and yards, but
NOT longest reception. Longest has to come from play-by-play, where every catch
is its own row. That's the whole reason this exists.

Usage:  python build_receivers.py 2026 data/receivers-2026.json

MATCHUP CONTEXT
  If matchup.json (from export_matchup.py) sits next to the output file,
  each player also gets a "mu" block describing their NEXT opponent's pass
  defense and pace, and every log row gets the opponent it came against.

  Run export_matchup.py first in the same workflow. Without it this script
  behaves exactly as before, minus the new "opp" field on each log row.
"""
import sys, json, urllib.request, os, tempfile
import pandas as pd

PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv"
ROSTER = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"
SCHEDULE = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
COLS = ["season", "season_type", "week", "posteam", "defteam",
        "receiver_player_id", "receiver_player_name",
        "complete_pass", "pass_attempt", "yards_gained"]
ROSTER_COLS = ["gsis_id", "full_name", "position", "height", "weight",
               "college", "years_exp"]

# Only keep players with enough volume to ever be worth a prop.
MIN_SEASON_TARGETS = 20


def _get(url, path):
    if not os.path.exists(path):
        print(f"downloading {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return path


def load(season, cache_dir=None):
    d = cache_dir or tempfile.gettempdir()
    pbp = pd.read_csv(_get(PBP.format(season=season), os.path.join(d, f"pbp_{season}.csv")),
                      usecols=COLS, low_memory=False)
    ros = pd.read_csv(_get(ROSTER.format(season=season), os.path.join(d, f"roster_{season}.csv")),
                      usecols=ROSTER_COLS, low_memory=False)
    ros = ros.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
    return pbp, ros


def load_matchup(cache_dir):
    """Team ratings from export_matchup.py. Optional."""
    p = os.path.join(cache_dir or ".", "matchup.json")
    if not os.path.exists(p):
        print("no matchup.json found - skipping matchup context", file=sys.stderr)
        return None
    with open(p) as f:
        return json.load(f)


def next_opponents(season, played_through):
    """
    {team: (opponent, is_home)} for the first unplayed week.
    Uses the same schedule file as the ratings engine.
    """
    try:
        g = pd.read_csv(SCHEDULE)
    except Exception as e:
        print(f"schedule unavailable ({e}) - no next-opponent lookup", file=sys.stderr)
        return {}, None

    g = g[(g.season == int(season)) & (g.game_type == "REG")]
    up = g[g.week > played_through]
    if up.empty:
        return {}, None

    wk = int(up.week.min())
    out = {}
    for _, r in up[up.week == wk].iterrows():
        out[r.home_team] = (r.away_team, True)
        out[r.away_team] = (r.home_team, False)
    return out, wk


def _height(v):
    """Roster height may be inches (71) or already formatted (6-1)."""
    try:
        n = int(float(v))
        return f"{n // 12}-{n % 12}" if 60 <= n <= 90 else str(v)
    except (TypeError, ValueError):
        return "" if pd.isna(v) else str(v)


def _s(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def build(season, cache_dir=None):
    df, ros = load(season, cache_dir)
    df = df[(df.season_type == "REG") & df.receiver_player_id.notna()]
    df = df[df.pass_attempt == 1]

    df["rec"] = (df.complete_pass == 1).astype(int)
    df["yds"] = df.yards_gained.where(df.complete_pass == 1, 0)
    df["catch_len"] = df.yards_gained.where(df.complete_pass == 1)

    g = df.groupby(["receiver_player_id", "receiver_player_name", "week"], as_index=False).agg(
        tgt=("pass_attempt", "sum"),
        rec=("rec", "sum"),
        yds=("yds", "sum"),
        lng=("catch_len", "max"),
        team=("posteam", "first"),
        opp=("defteam", "first"),
    )
    g["lng"] = g["lng"].fillna(0)

    totals = g.groupby("receiver_player_id")["tgt"].sum()
    keep = set(totals[totals >= MIN_SEASON_TARGETS].index)
    g = g[g.receiver_player_id.isin(keep)]

    mu = load_matchup(cache_dir)
    teams = (mu or {}).get("teams", {})
    played_through = int(g.week.max()) if len(g) else 0
    nxt, next_week = next_opponents(season, played_through)

    players = {}
    for pid, chunk in g.groupby("receiver_player_id"):
        chunk = chunk.sort_values("week")
        bio = ros.loc[pid].to_dict() if pid in ros.index else {}
        name = _s(bio.get("full_name")) or _s(chunk.receiver_player_name.iloc[0])
        team = _s(chunk.team.iloc[-1])

        rec = {
            "name": name,
            "team": team,
            "pos": _s(bio.get("position")),
            "ht": _height(bio.get("height")),
            "wt": _s(bio.get("weight")).replace(".0", ""),
            "college": _s(bio.get("college")),
            "exp": _s(bio.get("years_exp")).replace(".0", ""),
            "log": [
                {"wk": str(int(r.week)), "tgt": int(r.tgt), "rec": int(r.rec),
                 "yds": int(r.yds), "lng": int(r.lng), "opp": _s(r.opp)}
                for r in chunk.itertuples()
            ],
        }

        # ---- next-opponent matchup block -----------------------------
        if team in nxt:
            opp, at_home = nxt[team]
            d = teams.get(opp)
            rec["mu"] = {
                "opp": opp,
                "home": at_home,
                # pass defense: rank 1 = toughest. 32 = softest.
                "d_rank": (d or {}).get("def_pass_rank"),
                "d_rating": (d or {}).get("def_pass"),
                # opponent pace drives target VOLUME independent of quality
                "plays": (d or {}).get("plays_pg"),
                "prate": (d or {}).get("pass_rate"),
            }

        players[pid] = rec

    return {
        "season": int(season),
        "count": len(players),
        "through_week": played_through,
        "next_week": next_week,
        "teams": teams,
        "players": players,
    }


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2025"
    out = sys.argv[2] if len(sys.argv) > 2 else f"receivers-{season}.json"
    data = build(season, cache_dir=os.path.dirname(out) or ".")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out}: {data['count']} receivers, {os.path.getsize(out)/1024:.0f} KB",
          file=sys.stderr)
