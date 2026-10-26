"""Parse weekly pick files (wN.txt) into game docs and a bulk-picks doc.

File format, one statement per line ('#' comments allowed):
  game AWAY-HOME YYYY-MM-DD [awayScore homeScore]
  line GAME TEAM N              # a reference line quoted by a lines source (team's own spread)
  pick EXPERT GAME TEAM [N]     # ATS side; N = that team's spread as the expert quoted it
  grp GAME [TEAM N] | exp:TEAM exp:TEAM ...   # one table row, optional quoted line for TEAM
  score EXPERT GAME WINNER WS LS              # score prediction, converted vs consensus line
Consensus (standard) line per game = median of every home-perspective line quoted that week.
"""
import json, re, statistics, sys, os

def parse(path):
    games, quotes, raw, scores = {}, {}, [], []
    for ln in open(path):
        ln = ln.split('#')[0].strip()
        if not ln: continue
        t = ln.split()
        k = t[0]
        if k == 'game':
            a, h = t[1].split('-')
            g = {'away': a, 'home': h, 'date': t[2], 'order': len(games) + 1}
            if len(t) >= 5: g['awayScore'], g['homeScore'] = int(t[3]), int(t[4])
            games[t[1]] = g
        elif k == 'line':
            q(quotes, games, t[1], t[2], float(t[3]))
        elif k == 'pick':
            gk, team = t[2], t[3]
            n = float(t[4]) if len(t) > 4 else None
            chk(games, gk, team, ln)
            hl = q(quotes, games, gk, team, n) if n is not None else None
            raw.append((t[1], gk, team, hl))
        elif k == 'grp':
            head, tail = ln.split('|')
            h = head.split()
            gk = h[1]
            hl = None
            if len(h) >= 4:
                chk(games, gk, h[2], ln)
                hl = q(quotes, games, gk, h[2], float(h[3]))
            for tok in tail.split():
                e, team = tok.split(':')
                chk(games, gk, team, ln)
                raw.append((e, gk, team, hl))
        elif k == 'score':
            e, gk, w, ws, ls = t[1], t[2], t[3], float(t[4]), float(t[5])
            chk(games, gk, w, ln)
            scores.append((e, gk, w, ws, ls))
        else:
            sys.exit(f'bad line: {ln}')
    return games, quotes, raw, scores

def chk(games, gk, team, ln):
    if gk not in games: sys.exit(f'unknown game {gk}: {ln}')
    g = games[gk]
    if team not in (g['away'], g['home']): sys.exit(f'team {team} not in {gk}: {ln}')

def q(quotes, games, gk, team, n):
    hl = n if team == games[gk]['home'] else -n
    quotes.setdefault(gk, []).append(hl)
    return hl

def build(week, season=2026):
    path = f'w{week}.txt'
    games, quotes, raw, scores = parse(path)
    out_games, picks, seen = {}, [], set()
    for gk, g in games.items():
        qs = quotes.get(gk, [])
        if not qs: sys.exit(f'no line for {gk}')
        med = statistics.median(qs)
        med = round(med * 2) / 2  # snap to half point
        gid = f'{season}-w{week}-{g["away"]}-{g["home"]}'
        out_games[gk] = dict(id=gid, week=week, season=season, date=g['date'], order=g['order'],
                             away=g['away'], home=g['home'], homeLine=med, total=None,
                             awayScore=g.get('awayScore'), homeScore=g.get('homeScore'))
    for e, gk, team, hl in raw:
        g = out_games[gk]
        key = (e, gk)
        if key in seen: sys.exit(f'duplicate pick {key}')
        seen.add(key)
        p = {'e': e, 'g': g['id'], 's': 'home' if team == g['home'] else 'away'}
        if hl is not None: p['l'] = hl
        picks.append(p)
    for e, gk, w, ws, ls in scores:
        g = out_games[gk]
        key = (e, gk)
        if key in seen: sys.exit(f'duplicate pick {key}')
        seen.add(key)
        margin_home = (ws - ls) if w == g['home'] else (ls - ws)  # predicted home margin
        m = margin_home + g['homeLine']
        if m == 0:
            continue  # predicted exactly the number: no side
        picks.append({'e': e, 'g': g['id'], 's': 'home' if m > 0 else 'away', 'sc': f'{w} {ws:g}-{ls:g}'})
    os.makedirs('out', exist_ok=True)
    for g in out_games.values():
        json.dump({k: v for k, v in g.items() if k != 'id'}, open(f'out/{g["id"]}.json', 'w'))
    bulk = {'week': week, 'season': season, 'picks': picks}
    json.dump(bulk, open(f'out/bulk-{season}-w{week}.json', 'w'))
    return out_games, picks

if __name__ == '__main__':
    wk = int(sys.argv[1])
    gs, ps = build(wk)
    experts = sorted({p['e'] for p in ps})
    print(f'week {wk}: {len(gs)} games, {len(ps)} picks, {len(experts)} experts')
    for gk, g in gs.items():
        print(f'  {gk:9} line {g["homeLine"]:+5}  score {g["awayScore"]}-{g["homeScore"]}')
