"""Build Call Sheet data/experts-2026.json from the ledger's weekly bulk files."""
import json, glob, sys, datetime
MODELS = {'sbd-model', 'espn-fpi'}          # computer models stay out of the human crowd
FIX = {'LAR': 'LA'}
cs = json.load(open(sys.argv[1]))            # games-2026.json from the Call Sheet repo
ids = {g['id'] for g in cs['games']}
out, missing = {}, []
for f in sorted(glob.glob('out/bulk-2026-w*.json')):
    b = json.load(open(f)); wk = b['week']
    for p in b['picks']:
        if p['e'] in MODELS: continue
        _, w, a, h = p['g'].split('-')
        a, h = FIX.get(a, a), FIX.get(h, h)
        gid = f"2026_{wk:02d}_{a}_{h}"
        if gid not in ids: missing.append(gid); continue
        x = out.setdefault(gid, {'wk': wk, 'home': 0, 'away': 0, 'n': 0})
        x[p['s']] += 1; x['n'] += 1
weeks = sorted({v['wk'] for v in out.values()})
experts = set()
for f in glob.glob('out/bulk-2026-w*.json'):
    experts |= {p['e'] for p in json.load(open(f))['picks'] if p['e'] not in MODELS}
doc = {'kind': 'experts', 'season': 2026,
       'generated': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
       'through_week': max(weeks), 'n_experts': len(experts),
       'source': 'Expert ATS Ledger — published TV/print/web picks (spread picks and score predictions)',
       'games': out}
json.dump(doc, open(sys.argv[2], 'w'), separators=(',', ':'))
print(f"{len(out)} games, weeks {weeks}, {len(experts)} experts, unmatched ids: {sorted(set(missing))}")
