#!/usr/bin/env python3
"""
Turn nflverse play-by-play into a compact weekly receiving file for the call sheet.

nflverse's ready-made weekly stats file has receptions, targets and yards, but
NOT longest reception. Longest has to come from play-by-play, where every catch
is its own row. That's the whole reason this exists.

Usage:  python build_receivers.py 2026 data/receivers-2026.json
"""
import sys, json, urllib.request, os, tempfile
import pandas as pd

PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv"
ROSTER = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"
COLS = ["season", "season_type", "week", "posteam",
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
    )
    g["lng"] = g["lng"].fillna(0)

    totals = g.groupby("receiver_player_id")["tgt"].sum()
    keep = set(totals[totals >= MIN_SEASON_TARGETS].index)
    g = g[g.receiver_player_id.isin(keep)]

    players = {}
    for pid, chunk in g.groupby("receiver_player_id"):
        chunk = chunk.sort_values("week")
        bio = ros.loc[pid].to_dict() if pid in ros.index else {}
        name = _s(bio.get("full_name")) or _s(chunk.receiver_player_name.iloc[0])
        players[pid] = {
            "name": name,
            "team": _s(chunk.team.iloc[-1]),
            "pos": _s(bio.get("position")),
            "ht": _height(bio.get("height")),
            "wt": _s(bio.get("weight")).replace(".0", ""),
            "college": _s(bio.get("college")),
            "exp": _s(bio.get("years_exp")).replace(".0", ""),
            "log": [
                {"wk": str(int(r.week)), "tgt": int(r.tgt), "rec": int(r.rec),
                 "yds": int(r.yds), "lng": int(r.lng)}
                for r in chunk.itertuples()
            ],
        }
    return {"season": int(season), "count": len(players), "players": players}


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2025"
    out = sys.argv[2] if len(sys.argv) > 2 else f"receivers-{season}.json"
    data = build(season, cache_dir=os.path.dirname(out) or ".")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out}: {data['count']} receivers, {os.path.getsize(out)/1024:.0f} KB",
          file=sys.stderr)
