"""
Game lines from a SportsGameOdds event: spread, total, moneyline and both
team totals, as a consensus number plus each book's own price.

SGO names every market with an oddID:
    {statID}-{statEntityID}-{periodID}-{betTypeID}-{sideID}
    points-home-game-sp-home     home spread
    points-all-game-ou-over      game total, over
    points-home-game-ou-over     home TEAM total, over
    points-home-game-ml-home     home moneyline
Billing is per event, not per market, so pulling these alongside props (or
on their own) costs nothing extra for a game already being pulled.

Line convention matches the rest of the app: spread is the HOME team's
number, negative = home favored.
"""

PREFERRED = ("draftkings", "fanduel", "bet365", "espnbet", "caesars", "betmgm")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _book_line(b, kind):
    if kind == "sp":
        for k in ("spread", "bookSpread", "closeSpread"):
            v = _num(b.get(k))
            if v is not None:
                return v
    else:
        for k in ("overUnder", "bookOverUnder", "closeOverUnder"):
            v = _num(b.get(k))
            if v is not None:
                return v
    return None


def _side(odds, oid, kind):
    """One side of one market -> {line, odds, books{book: {line, odds}}}."""
    o = odds.get(oid)
    if not isinstance(o, dict):
        return None
    line = _num(o.get("bookSpread") if kind == "sp" else o.get("bookOverUnder"))
    if line is None:
        line = _num(o.get("fairSpread") if kind == "sp" else o.get("fairOverUnder"))
    books = {}
    for bid, b in (o.get("byBookmaker") or {}).items():
        if not isinstance(b, dict) or b.get("available") is False:
            continue
        px = b.get("odds")
        if px is None:
            continue
        books[bid] = {"odds": px}
        if kind != "ml":
            bl = _book_line(b, kind)
            if bl is not None:
                books[bid]["line"] = bl
    out = {"odds": o.get("bookOdds") or o.get("fairOdds"), "books": books or None}
    if kind != "ml":
        out["line"] = line
    return out


def _pick_book(side):
    """DraftKings if posted, else the next preferred book, else consensus."""
    if not side:
        return None, None
    bk = side.get("books") or {}
    for b in PREFERRED:
        if b in bk and (bk[b].get("line") is not None or "line" not in side):
            return b, bk[b]
    return "consensus", {"line": side.get("line"), "odds": side.get("odds")}


def parse_game_lines(ev):
    """Full-game spread / total / moneyline / team totals for one event.
    Returns None when the event has no game lines at all."""
    odds = ev.get("odds") or {}
    if not isinstance(odds, dict):
        return None
    sp_h = _side(odds, "points-home-game-sp-home", "sp")
    sp_a = _side(odds, "points-away-game-sp-away", "sp")
    to_o = _side(odds, "points-all-game-ou-over", "ou")
    to_u = _side(odds, "points-all-game-ou-under", "ou")
    ml_h = _side(odds, "points-home-game-ml-home", "ml")
    ml_a = _side(odds, "points-away-game-ml-away", "ml")
    tt = {}
    for team in ("home", "away"):
        o = _side(odds, f"points-{team}-game-ou-over", "ou")
        u = _side(odds, f"points-{team}-game-ou-under", "ou")
        if o or u:
            b, px = _pick_book(o or u)
            _, pu = _pick_book(u) if u else (None, None)
            tt[team] = {"line": (px or {}).get("line"), "over": (px or {}).get("odds"),
                        "under": (pu or {}).get("odds"), "book": b}

    out = {}
    if sp_h or sp_a:
        b, px = _pick_book(sp_h) if sp_h else _pick_book(sp_a)
        line = (px or {}).get("line")
        if sp_h is None and line is not None:      # only the away side came back
            line = -line
        _, pa = _pick_book(sp_a) if sp_a else (None, None)
        out["spread"] = {"home": line, "home_odds": (px or {}).get("odds"),
                         "away_odds": (pa or {}).get("odds"), "book": b}
    if to_o or to_u:
        b, po = _pick_book(to_o or to_u)
        _, pu = _pick_book(to_u) if to_u else (None, None)
        out["total"] = {"line": (po or {}).get("line"), "over": (po or {}).get("odds"),
                        "under": (pu or {}).get("odds"), "book": b}
    if ml_h or ml_a:
        _, ph = _pick_book(ml_h) if ml_h else (None, None)
        _, pa = _pick_book(ml_a) if ml_a else (None, None)
        out["ml"] = {"home": (ph or {}).get("odds"), "away": (pa or {}).get("odds")}
    if tt:
        out["team_total"] = tt
    return out or None
