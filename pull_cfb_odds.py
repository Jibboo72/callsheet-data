#!/usr/bin/env python3
"""
College football game lines from SportsGameOdds: spread, total, moneyline
and both team totals for every NCAAF game in the next few days, FBS and
whatever FCS games SGO carries.

    SGO_KEY=xxx python3 pull_cfb_odds.py 2026 data/cfb-odds-2026.json

COST
  SGO bills per EVENT (game), not per market. A college Saturday is roughly
  50-60 FBS games, plus any FCS games SGO lists. The window is kept to the
  next 4 days so one pull = one week's slate, never two. The workflow runs
  this twice a week (Friday evening, Saturday morning) -> roughly 450-900
  objects a month depending on FCS coverage, against 2,500 on the free plan
  shared with the NFL pull. The run log prints usage before and after.

WHY BOTH THIS AND CollegeFootballData
  CFBD is the record of what the line WAS (great for backtests and grading
  finished games) but it's slow to pick up FCS lines during the week. SGO
  is a live board. export_cfb_ratings.py uses SGO for upcoming games and
  CFBD for finished ones.

TEAM NAMES
  SGO's team names don't always match CollegeFootballData's ("Miami (FL)"
  vs "Miami"). Every name variant SGO sends is kept here; the matching to
  the schedule happens in export_cfb_ratings.py, which logs any game it
  couldn't match so the alias list can be fixed.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from pull_odds import call_either, usage
from sgo_lines import parse_game_lines

DAYS = int(os.environ.get("CFB_ODDS_DAYS", "4"))


def names(ev, side):
    t = (ev.get("teams") or {}).get(side) or {}
    n = t.get("names") or {}
    out = {k: n.get(k) for k in ("short", "medium", "long", "abbr", "location", "nickname")
           if n.get(k)}
    if t.get("teamID"):
        out["teamID"] = t.get("teamID")
    return out


def pull(key, days=DAYS):
    now = datetime.now(timezone.utc)
    params = {"leagueID": "NCAAF", "type": "match", "limit": 200,
              "startsAfter": now.strftime("%Y-%m-%d"),
              "startsBefore": (now + timedelta(days=days)).strftime("%Y-%m-%d")}
    events, cursor = [], None
    for _ in range(5):                      # page through, max 1,000 events
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        d = call_either("/events", key, p)
        batch = d.get("data", d if isinstance(d, list) else [])
        events += batch
        cursor = d.get("nextCursor") if isinstance(d, dict) else None
        if not cursor or not batch:
            break
    if events:
        print("--- one event's team block (for name matching) ---", file=sys.stderr)
        print(json.dumps((events[0].get("teams") or {}), indent=1)[:900], file=sys.stderr)

    out, no_lines = [], 0
    for ev in events:
        ln = parse_game_lines(ev)
        if not ln:
            no_lines += 1
            continue
        out.append({"id": ev.get("eventID"),
                    "start": (ev.get("status") or {}).get("startsAt"),
                    "home": names(ev, "home"), "away": names(ev, "away"),
                    "lines": ln})
    return out, len(events), no_lines


if __name__ == "__main__":
    key = os.environ.get("SGO_KEY")
    if not key:
        raise SystemExit("SGO_KEY not set — add it as a repository secret")
    season = sys.argv[1] if len(sys.argv) > 1 else "2026"
    path = sys.argv[2] if len(sys.argv) > 2 else f"cfb-odds-{season}.json"
    before = usage(key)
    games, n, nolines = pull(key)
    after = usage(key)
    doc = {"season": int(season), "kind": "cfb_odds",
           "pulled": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "events_pulled": n, "games": games, "usage": after or before}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    tt = sum(1 for g in games if "team_total" in g["lines"])
    tot = sum(1 for g in games if "total" in g["lines"])
    print(f"wrote {path}: {n} events, {len(games)} with game lines "
          f"({tot} with a total, {tt} with team totals), {nolines} without lines",
          file=sys.stderr)
    if before and after:
        print(f"objects used: {before.get('objectsUsed')} -> {after.get('objectsUsed')} "
              f"of {after.get('objectsLimit')}", file=sys.stderr)
