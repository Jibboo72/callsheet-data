#!/usr/bin/env python3
"""
Manual news overrides: injury and starting-QB news from X (or anywhere)
that lands before ESPN's or nflverse's feeds catch up.

    data/overrides.json
    {
      "entries": [
        {"team": "MIN", "player": "Kirk Cousins", "position": "QB",
         "status": "Starting", "source": "Adam Schefter",
         "posted": "2026-11-13T21:08:00Z", "expires": "2026-11-17"},
        {"team": "KC", "player": "Kenneth Walker", "position": "RB",
         "status": "Out", "source": "Ian Rapoport",
         "posted": "2026-11-15T16:05:00Z"}
      ]
    }

status
  Out / Doubtful / Questionable  treated exactly like an ESPN designation:
                                 replaces whatever ESPN says about him
  Active                         cleared: removes ESPN's designation
  Starting                       (QBs) this passer starts the team's next game

Only the adjustments the model already uses are touched -- the injury
discount, the starting quarterback, and the props' "skip players who are
Out" rule. Nothing here adds a new kind of input.

An entry stops counting after its `expires` date (UTC), or 4 days after
`posted` when there is no `expires`, so a stale override can never leak
into next week. A bad file is ignored, never fatal.
"""
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "overrides.json")
STATUSES = {"Out", "Doubtful", "Questionable", "Active", "Starting"}
DEFAULT_DAYS = 4


def key(name):
    """Same name key as export_games._nm: no accents, punctuation or suffixes."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())


def _live(e, today):
    exp = e.get("expires")
    if not exp and e.get("posted"):
        try:
            p = datetime.fromisoformat(str(e["posted"]).replace("Z", "+00:00"))
            exp = (p + timedelta(days=DEFAULT_DAYS)).date().isoformat()
        except ValueError:
            exp = None
    return exp is None or str(exp)[:10] >= today


def load(path=None):
    """Entries still in force, as dicts with team / player / position /
    status / key. [] when the file is missing or unreadable."""
    path = path or PATH
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"warning: overrides unreadable, ignoring ({e})", file=sys.stderr)
        return []
    today = datetime.now(timezone.utc).date().isoformat()
    out = []
    for e in doc.get("entries") or []:
        st = str(e.get("status", "")).strip().capitalize()
        team = str(e.get("team", "")).strip().upper()
        name = str(e.get("player", "")).strip()
        if st not in STATUSES or not team or not name or not _live(e, today):
            continue
        out.append(dict(team=team, player=name, key=key(name), status=st,
                        position=str(e.get("position", "")).strip().upper(),
                        source=e.get("source", "")))
    if out:
        print(f"overrides: {len(out)} in force -- " +
              "; ".join(f"{o['team']} {o['player']} {o['status']}" for o in out), file=sys.stderr)
    return out


def apply_to_injury_rows(rows, entries=None):
    """ESPN-style rows (team, key, name, position, status) with the overrides
    laid on top: an override replaces ESPN's row for that player, Active
    removes it, Starting is not an injury and leaves the rows alone."""
    entries = load() if entries is None else entries
    inj = [e for e in entries if e["status"] in ("Out", "Doubtful", "Questionable", "Active")]
    if not inj:
        return rows
    hit = {(e["team"], e["key"]) for e in inj}
    pos_of = {(r["team"], r["key"]): r.get("position", "") for r in rows}
    out = [r for r in rows if (r["team"], r["key"]) not in hit]
    for e in inj:
        if e["status"] == "Active":
            continue
        out.append(dict(team=e["team"], key=e["key"], name=e["player"],
                        position=e["position"] or pos_of.get((e["team"], e["key"]), ""),
                        status=e["status"], override=True))
    return out


def qb_starters(qb, entries=None):
    """{team: passer id} for 'Starting' overrides, matched by name to the
    passers the QB model knows (initial + last name, like 'K Cousins').
    A name it can't match is reported and skipped, never guessed."""
    entries = load() if entries is None else entries
    out = {}
    for e in entries:
        if e["status"] != "Starting":
            continue
        parts = e["key"].split()
        want = f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else e["key"]
        cands = [q for q, nm in qb.get("name", {}).items()
                 if nm and (lambda p: f"{p[0][0]} {p[-1]}" if len(p) >= 2 else " ".join(p))(key(str(nm).replace(".", " ")).split()) == want]
        depth = qb.get("depth", {}).get(e["team"], [])
        pick = next((q for q in depth if q in cands), cands[0] if len(cands) == 1 else None)
        if pick:
            out[e["team"]] = pick
            print(f"QB override: {e['team']} starts {e['player']}", file=sys.stderr)
        else:
            print(f"warning: QB override {e['team']} {e['player']} not matched to a passer; skipped",
                  file=sys.stderr)
    return out
