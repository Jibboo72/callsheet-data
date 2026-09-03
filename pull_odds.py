#!/usr/bin/env python3
"""
Weekly NFL prop pull from SportsGameOdds, plus quota tracking.

    SGO_KEY=xxx python3 pull_odds.py 2026 data/odds-2026.json

WHY ONCE A WEEK
  The free Amateur plan is 2,500 objects/month, shared across every sport on
  the account. Billing is per EVENT, not per market or bookmaker — one game
  with 50 prop markets across 20 books costs a single object. So a full NFL
  slate is ~16 objects and college FBS ~55. The quota is not the constraint
  football puts on you; baseball's daily slate is.

TIMING
  Set the cron in the workflow to whenever you want the snapshot. Before
  Sunday = prices you can still bet. After kickoff = closing lines, which is
  what you'd want for measuring closing line value. You cannot get both from
  one pull.

UNTESTED AGAINST THE LIVE API
  Written without a key, so the response parsing is defensive and the raw
  shape of the first event is dumped to stderr on every run. If the field
  names differ, that dump is what we need to fix it.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

BASE = "https://api.sportsgameodds.com/v2"
# Reception and longest-reception markets are the thin ones we care about;
# receiving yards is included because it's the liquid market next door.
WANT = ("receptions", "receiving_yards", "longest_reception",
        "receiving_longest", "targets")


# Cloudflare sits in front of this API and blocks the default
# "Python-urllib/3.x" agent outright — that is what a 403 with error code
# 1010 means. It is not an authentication failure. A normal browser-ish
# User-Agent is required.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json",
}


def call(path, key, params=None, query_auth=False):
    url = f"{BASE}{path}"
    p = dict(params or {})
    h = dict(HEADERS)
    # docs allow either the x-api-key header or an apiKey query param
    if query_auth:
        p["apiKey"] = key
    else:
        h["x-api-key"] = key
    if p:
        url += "?" + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode())


def call_either(path, key, params=None):
    """Header auth first, query param as fallback."""
    try:
        return call(path, key, params)
    except urllib.error.HTTPError as e:
        if e.code not in (401, 403):
            raise
        first = f"HTTP {e.code}: {e.read().decode()[:200]}"
        try:
            return call(path, key, params, query_auth=True)
        except urllib.error.HTTPError as e2:
            raise RuntimeError(
                f"both auth styles rejected.\n  header: {first}\n"
                f"  query : HTTP {e2.code}: {e2.read().decode()[:200]}")


# The account endpoint returns email, customerID and keyID alongside the
# quota. This repo is public — none of that gets written to disk. Only the
# rate-limit block is kept.
def usage(key):
    for path in ("/account/usage", "/account"):
        try:
            d = call_either(path, key)
            d = d.get("data", d)
            rl = (d or {}).get("rateLimits") or {}
            mon = rl.get("per-month") or {}
            return {
                "tier": d.get("tier"),
                "objectsUsed": mon.get("current-entities"),
                "objectsLimit": mon.get("max-entities"),
                "reqPerMin": (rl.get("per-minute") or {}).get("max-requests"),
            }
        except Exception as e:
            print(f"  usage via {path}: {e}", file=sys.stderr)
    return None


def pull(season, key, days=8):
    events = []
    try:
        # `season` alone came back with 2024 games, so bound it by date as
        # well and filter client-side. Also keeps the object spend down:
        # one NFL week is ~16 events, not 100.
        now = datetime.now(timezone.utc)
        params = {"leagueID": "NFL", "type": "match", "limit": 40,
                  "startsAfter": now.strftime("%Y-%m-%d"),
                  "startsBefore": (now + timedelta(days=days)).strftime("%Y-%m-%d")}
        d = call_either("/events", key, params)
        events = d.get("data", d if isinstance(d, list) else [])
        # belt and braces: drop anything already kicked off, whatever the
        # API did with the date params
        cutoff = now.isoformat()
        events = [e for e in events
                  if str((e.get("status") or {}).get("startsAt") or "") >= cutoff[:10]]
    except Exception as e:
        raise SystemExit(f"events call failed: {e}")

    if events:
        print("--- raw shape of first event (for fixing field names) ---",
              file=sys.stderr)
        print(json.dumps(events[0], indent=1)[:1500], file=sys.stderr)

    out = []
    for ev in events:
        odds = ev.get("odds") or {}
        props = []
        for oid, o in (odds.items() if isinstance(odds, dict) else []):
            name = str(o.get("statID") or o.get("marketName") or oid).lower()
            if not any(w in name for w in WANT):
                continue
            props.append({
                "id": oid,
                "market": o.get("statID") or o.get("marketName"),
                "player": o.get("playerID") or o.get("participantID"),
                "side": o.get("sideID"),
                "line": o.get("bookOverUnder") or o.get("fairOverUnder"),
                "odds": o.get("bookOdds") or o.get("fairOdds"),
            })
        if props:
            out.append({
                "id": ev.get("eventID"),
                "wk": (ev.get("info") or {}).get("week"),
                "home": (ev.get("teams") or {}).get("home", {}).get("names", {}).get("short"),
                "away": (ev.get("teams") or {}).get("away", {}).get("names", {}).get("short"),
                "start": (ev.get("status") or {}).get("startsAt"),
                "props": props,
            })
    return out, len(events)


if __name__ == "__main__":
    key = os.environ.get("SGO_KEY")
    if not key:
        raise SystemExit("SGO_KEY not set — add it as a repository secret")
    season = sys.argv[1] if len(sys.argv) > 1 else "2026"
    out = sys.argv[2] if len(sys.argv) > 2 else f"odds-{season}.json"

    before = usage(key)
    games, n_events = pull(season, key)
    after = usage(key)

    doc = {
        "season": int(season),
        "kind": "odds",
        "pulled": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "events_pulled": n_events,
        "games": games,
        "usage": after or before,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    u = doc["usage"] or {}
    print(f"wrote {out}: {len(games)} games with props, {n_events} events pulled",
          file=sys.stderr)
    print(f"usage: {json.dumps(u)[:200]}", file=sys.stderr)
