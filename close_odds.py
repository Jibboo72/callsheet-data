#!/usr/bin/env python3
"""
Closing-line capture: prices for games about to kick off.

    SGO_KEY=xxx python3 close_odds.py 2026 data/closing-2026.json

Runs many times on game days (close.yml). Each run looks at the games the
weekly pulls already know about (data/odds-2026.json for NFL,
data/cfb-odds-2026.json for college), picks the ones kicking off in the next
WINDOW_MIN minutes that haven't been captured yet, and asks SportsGameOdds for
just those events. Billing is per event, so each game costs one object once:
about 16 NFL + 60 college objects a week.

The app reads the file, matches each of your bets to its game's last pregame
price, and stores the result on the bet (closing line value). Games older than
KEEP_DAYS are pruned -- by then every bet on them has its number.

Not tested against the live API from here (no key in the dev box): the
eventIDs filter is the documented way to ask for specific events; if the API
ignores it, the start-time window below still keeps the spend to the games
that are about to start.
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta

from pull_odds import call_either, parse_event, usage
from pull_cfb_odds import names
from sgo_lines import parse_game_lines

WINDOW_MIN = 50      # capture games starting within this many minutes
KEEP_DAYS = 21
MAX_EVENTS = 40      # hard cap per run, whatever happens


def ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def wanted(now, season, have):
    """(event id, league, start) for games starting soon, not yet captured."""
    out = []
    for path, lg in ((f"data/odds-{season}.json", "NFL"), (f"data/cfb-odds-{season}.json", "NCAAF")):
        for g in load(path, {}).get("games", []):
            st = ts(g.get("start"))
            eid = g.get("id")
            if not eid or not st or eid in have:
                continue
            if now < st <= now + timedelta(minutes=WINDOW_MIN):
                out.append((eid, g.get("league") or lg, st))
    return out[:MAX_EVENTS]


def fetch(key, want, now):
    ids = [w[0] for w in want]
    events = []
    try:
        for i in range(0, len(ids), 25):
            d = call_either("/events", key, {"eventIDs": ",".join(ids[i:i + 25]), "limit": 50})
            events += d.get("data", d if isinstance(d, list) else [])
    except Exception as e:
        print(f"  eventIDs call failed ({e}); falling back to a start-time window", file=sys.stderr)
        for lg in sorted({w[1] for w in want}):
            d = call_either("/events", key, {
                "leagueID": lg, "type": "match", "limit": 60,
                "startsAfter": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "startsBefore": (now + timedelta(minutes=WINDOW_MIN + 10)).strftime("%Y-%m-%dT%H:%M:%SZ")})
            events += d.get("data", d if isinstance(d, list) else [])
    keep = set(ids)
    got = [e for e in events if e.get("eventID") in keep]
    if len(events) > 3 * max(1, len(ids)):
        print(f"  warning: asked for {len(ids)} events, got {len(events)} back", file=sys.stderr)
    return got


def main():
    key = os.environ.get("SGO_KEY")
    if not key:
        raise SystemExit("SGO_KEY not set")
    season = sys.argv[1] if len(sys.argv) > 1 else "2026"
    path = sys.argv[2] if len(sys.argv) > 2 else f"data/closing-{season}.json"
    now = datetime.now(timezone.utc)
    doc = load(path, {"season": int(season), "kind": "closing", "games": {}})
    games = doc.setdefault("games", {})

    # prune old games first
    cut = now - timedelta(days=KEEP_DAYS)
    for eid in [k for k, g in games.items() if (ts(g.get("start")) or now) < cut]:
        del games[eid]

    want = wanted(now, season, set(games))
    if not want:
        print("nothing kicking off in the window", file=sys.stderr)
        return
    lg_of = {w[0]: w[1] for w in want}
    print(f"capturing {len(want)} games: {[w[0] for w in want]}", file=sys.stderr)
    for ev in fetch(key, want, now):
        st = ts((ev.get("status") or {}).get("startsAt"))
        if st and st <= now:
            continue                                  # already live: not a pregame price
        eid = ev.get("eventID")
        lg = lg_of.get(eid, "NFL")
        rec = {"league": lg, "start": (ev.get("status") or {}).get("startsAt"),
               "captured": now.isoformat(timespec="seconds")}
        if lg == "NFL":
            x = parse_event(ev) or {}
            rec.update(home=x.get("home"), away=x.get("away"), lines=x.get("lines") or {},
                       props=x.get("props") or [])
        else:
            rec.update(home=names(ev, "home"), away=names(ev, "away"), lines=parse_game_lines(ev) or {})
        games[eid] = rec

    doc["updated"] = now.isoformat(timespec="seconds")
    doc["usage"] = usage(key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {path}: {len(games)} games held", file=sys.stderr)


if __name__ == "__main__":
    main()
