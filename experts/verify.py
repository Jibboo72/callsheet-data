import json, sys, glob
from collections import defaultdict

def load(weeks):
    games, picks = {}, []
    for w in weeks:
        for f in glob.glob(f'out/2026-w{w}-*.json'):
            games[f.split('/')[-1][:-5]] = json.load(open(f))
        picks += json.load(open(f'out/bulk-2026-w{w}.json'))['picks']
    return games, picks

def grade(p, g, own):
    if g['homeScore'] is None: return None
    line = p['l'] if own and 'l' in p else g['homeLine']
    m = g['homeScore'] - g['awayScore'] + line
    if m == 0: return 'P'
    return 'W' if (p['s'] == 'home') == (m > 0) else 'L'

weeks = [int(x) for x in sys.argv[1].split(',')]
own = len(sys.argv) > 2 and sys.argv[2] == 'own'
filt = sys.argv[3].split(',') if len(sys.argv) > 3 else None
games, picks = load(weeks)
rec = defaultdict(lambda: [0, 0, 0])
for p in picks:
    if filt and p['e'] not in filt: continue
    r = grade(p, games[p['g']], own)
    if r: rec[p['e']]['WLP'.index(r)] += 1
for e, (w, l, pu) in sorted(rec.items(), key=lambda x: -(x[1][0] - x[1][1])):
    print(f'{e:20} {w}-{l}-{pu}')
