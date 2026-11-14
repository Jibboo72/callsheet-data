#!/usr/bin/env python3
"""
Current rosters for the call sheet — names, teams, bio, and availability.

    python3 export_roster.py 2026 data/roster-2026.json

WHY THIS IS SEPARATE FROM build_receivers.py
  Receiving logs come from play-by-play, which doesn't exist until games
  are played. Rosters are published months earlier. Between the last game
  of one season and the first of the next, every player's team in the
  receivers file is stale — after the 2025 season, 48 of 251 receivers
  had changed teams. A wrong team means a wrong next opponent and a wrong
  pass-defense read, so this keeps that current in the gap.

  The file deliberately carries NO "log" key. The app only overwrites a
  player's game log when the incoming file actually has one, so importing
  this updates team and bio while leaving every logged game intact.

AVAILABILITY (added so the candidate screener stops surfacing people who
can't play)
  Once a season is underway each player also gets, when the data exists:
    st    roster status this week: ACT, IR, INA (inactive), PS (practice
          squad), CUT, RET, or NONE (was on a roster earlier, isn't now)
    is    this week's injury-report designation: Out / Doubtful /
          Questionable (absent if not on the report)
    ib    the body part, if listed
    pr    practice participation this week when reduced: LP (limited) or
          DNP (did not practice)
    snap  average share of offensive snaps over his last 3 games played
          (0-1) -- a role measure; a 5% snap-share receiver is not a TD bet
    miss  how many of his team's most recent games in a row he had no
          offensive snaps in (0 = played last game)
  All free nflverse files (weekly rosters, injuries, snap counts). If any of
  them can't be fetched the roster still exports, just without these keys --
  the app treats missing availability as "unknown", never as "healthy".
"""
import json
import os
import sys

import pandas as pd

ROSTER = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "rosters/roster_{season}.csv")

# Pass catchers only. Nobody is betting receptions on a long snapper.
POSITIONS = {"WR", "TE", "RB", "FB", "QB"}

NV = "https://github.com/nflverse/nflverse-data/releases/download/"
WEEKLY = NV + "weekly_rosters/roster_weekly_{season}.parquet"
INJURIES = NV + "injuries/injuries_{season}.parquet"
SNAPS = NV + "snap_counts/snap_counts_{season}.parquet"
PLAYERS = NV + "players/players.parquet"

# nflverse roster status codes -> the short labels the app shows
STATUS = {"ACT": "ACT", "RES": "IR", "INA": "INA", "DEV": "PS", "CUT": "CUT",
          "RET": "RET", "EXE": "CUT", "TRD": "NONE", "TRT": "NONE"}


def _height(v):
    try:
        n = int(float(v))
        return f"{n // 12}-{n % 12}" if 60 <= n <= 90 else str(v)
    except (TypeError, ValueError):
        return "" if pd.isna(v) else str(v)


def _s(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def availability(season, players):
    """{gsis_id: {st, is, ib, snap, miss}} for the current week. Any piece
    whose source file is missing is simply left out. Never raises -- a
    roster without availability is still a roster."""
    ids = set(players)
    out = {g: {} for g in ids}

    # --- roster status, most recent week on file ---------------------
    try:
        wk = pd.read_parquet(WEEKLY.format(season=season))
        wk = wk.dropna(subset=["gsis_id", "week"])
        latest = int(wk.week.max())
        # a team on its bye has NO rows for the latest week, so "missing
        # from the latest week" only means released/moved when the player's
        # own team does have rows that week
        teams_now = set(wk[wk.week == latest].team)
        wk = wk[wk.gsis_id.isin(ids)].sort_values("week")
        last_row = wk.groupby("gsis_id").tail(1).set_index("gsis_id")
        for gid, row in last_row.iterrows():
            if int(row.week) < latest and players[gid].get("team") in teams_now:
                st = "NONE"          # on a roster earlier, not on this week's
            else:
                st = STATUS.get(_s(row.get("status")), "")
            if st:
                out[gid]["st"] = st
    except Exception as e:
        print(f"weekly roster status unavailable ({e})", file=sys.stderr)

    # --- this week's injury report ------------------------------------
    try:
        inj = pd.read_parquet(INJURIES.format(season=season))
        inj = inj[inj.gsis_id.isin(ids)].dropna(subset=["week"])
        if len(inj):
            inj = inj[inj.week == inj.week.max()]
            practice = {"Limited Participation in Practice": "LP",
                        "Did Not Participate In Practice": "DNP"}
            for row in inj.itertuples():
                stat = _s(row.report_status)
                if stat in ("Out", "Doubtful", "Questionable"):
                    out[row.gsis_id]["is"] = stat
                body = _s(row.report_primary_injury) or _s(row.practice_primary_injury)
                if body and (stat or practice.get(_s(row.practice_status))):
                    out[row.gsis_id]["ib"] = body
                pr = practice.get(_s(row.practice_status))
                if pr:
                    out[row.gsis_id]["pr"] = pr
    except Exception as e:
        print(f"injury report unavailable ({e})", file=sys.stderr)

    # --- role + recent absence from snap counts ------------------------
    try:
        sn = pd.read_parquet(SNAPS.format(season=season))
        pl = pd.read_parquet(PLAYERS)[["gsis_id", "pfr_id"]].dropna()
        sn = sn.merge(pl, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        sn = sn[sn.gsis_id.isin(ids)]
        # team-games actually completed, by team -- a bye or a game not yet
        # played doesn't count as a game missed
        full = pd.read_parquet(SNAPS.format(season=season), columns=["team", "week"])
        team_weeks = {t: sorted(set(g.week)) for t, g in full.groupby("team")}
        played = sn[sn.offense_snaps > 0]
        by_player = {gid: g.sort_values("week") for gid, g in played.groupby("gsis_id")}
        team_of = sn.sort_values("week").groupby("gsis_id").team.last().to_dict()
        for gid in ids:
            g = by_player.get(gid)
            if g is not None and len(g):
                out[gid]["snap"] = round(float(g.offense_pct.tail(3).mean()), 2)
                tw = team_weeks.get(team_of.get(gid), [])
                since = [w for w in tw if w > int(g.week.max())]
                out[gid]["miss"] = len(since)
            else:
                # on the roster but never taken an offensive snap: every
                # completed game of his team so far counts as missed
                tw = team_weeks.get(players[gid].get("team"), [])
                if tw:
                    out[gid]["miss"] = len(tw)
    except Exception as e:
        print(f"snap counts unavailable ({e})", file=sys.stderr)
    return out


# ESPN's live team rosters (2026-11-11). nflverse's roster files trail real
# moves by a day or more -- after the 2026 trade deadline they still had
# Kirk Cousins on the Raiders while he was already the Vikings' listed
# starter. A wrong team means a wrong game, opponent and pass defense for
# every prop on that player, so each player's team is checked against ESPN
# and moved when ESPN has him on exactly one other team.
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
ESPN_ROSTER = ESPN_TEAMS + "/{id}/roster"
ESPN_ABBR = {"WSH": "WAS", "LAR": "LA"}          # ESPN -> nflverse


def _key(name):
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    return " ".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))


def _get_json(url):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "callsheet-data (personal app)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def espn_teams_now():
    """{name key: {team, ...}} from ESPN's 32 live rosters, or {} if ESPN
    can't be reached or answers for fewer than 30 teams (a partial read must
    not move anyone)."""
    try:
        lg = _get_json(ESPN_TEAMS)["sports"][0]["leagues"][0]["teams"]
    except Exception as e:
        print(f"ESPN team list unavailable ({e}); teams from nflverse only", file=sys.stderr)
        return {}
    out, ok = {}, 0
    for t in lg:
        t = t.get("team") or {}
        ab = ESPN_ABBR.get(t.get("abbreviation"), t.get("abbreviation"))
        try:
            d = _get_json(ESPN_ROSTER.format(id=t.get("id")))
        except Exception as e:
            print(f"ESPN roster {ab} unavailable ({e})", file=sys.stderr)
            continue
        ok += 1
        for grp in d.get("athletes", []):
            for a in grp.get("items", []):
                if a.get("fullName"):
                    out.setdefault(_key(a["fullName"]), set()).add(ab)
    if ok < 30:
        print(f"ESPN rosters: only {ok} teams read; not using them", file=sys.stderr)
        return {}
    print(f"ESPN rosters: {ok} teams, {len(out)} players", file=sys.stderr)
    return out


def apply_moves(players, espn):
    """Move players whose ESPN team differs from nflverse's. Sets `team` to
    ESPN's and `from` to the old team. Returns the list of moves."""
    moves = []
    for pid, p in players.items():
        tms = espn.get(_key(p.get("name", "")))
        if not tms or len(tms) != 1:
            continue                                  # not found, or a name on two teams
        new = next(iter(tms))
        if p.get("team") and new != p["team"]:
            moves.append((p["name"], p["team"], new))
            p["from"], p["team"] = p["team"], new
    for nm, a, b in moves:
        print(f"  moved: {nm} {a} -> {b}", file=sys.stderr)
    return moves


def build(season):
    r = pd.read_csv(ROSTER.format(season=season), low_memory=False)
    r = r.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    r = r[r.position.isin(POSITIONS)]

    players = {}
    for _, row in r.iterrows():
        players[row.gsis_id] = {
            "name": _s(row.get("full_name")),
            "team": _s(row.get("team")),
            "pos": _s(row.get("position")),
            "ht": _height(row.get("height")),
            "wt": _s(row.get("weight")).replace(".0", ""),
            "college": _s(row.get("college")),
            "exp": _s(row.get("years_exp")).replace(".0", ""),
        }
    av = availability(season, players)
    for gid, fields in av.items():
        players[gid].update(fields)
    apply_moves(players, espn_teams_now())
    # (after ESPN's team moves, so a traded player matches his new team)
    # news overrides (data/overrides.json): Out / Doubtful / Questionable set
    # this week's designation, Active clears it, so props skip or keep players
    # the same way the game projections do
    from overrides import load as load_overrides, key as okey
    ov = {(o["team"], o["key"]): o["status"] for o in load_overrides()
          if o["status"] in ("Out", "Doubtful", "Questionable", "Active")}
    if ov:
        for p in players.values():
            st = ov.get((p.get("team", ""), okey(p.get("name", ""))))
            if st == "Active":
                p.pop("is", None)
            elif st:
                p["is"] = st
    # a player ESPN just moved isn't on nflverse's injury / snap files for his
    # new team yet; his old status (ACT / snaps) still describes him
    return {"season": int(season), "kind": "roster",
            "count": len(players), "players": players}


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2026"
    out = sys.argv[2] if len(sys.argv) > 2 else f"roster-{season}.json"
    data = build(season)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out}: {data['count']} players, "
          f"{os.path.getsize(out)/1024:.0f} KB", file=sys.stderr)
