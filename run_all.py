"""
Fetch SBR NFL odds archives and write one CSV, with a short diagnostic
log so failures are readable without opening the Actions log viewer.

    python3 run_all.py 2015 2021

Writes:
  nfl_open_close.csv   the parsed lines (only if at least one season worked)
  fetch_log.txt        one short line per season -- read this first
"""
import io
import sys
import time
import requests
import pandas as pd
from parse_sbr import parse_table, pick_table

URL = 'https://sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-{}-{}'
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36'),
    'Accept': ('text/html,application/xhtml+xml,application/xml;q=0.9,'
               'image/avif,image/webp,*/*;q=0.8'),
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://sportsbookreviewsonline.com/',
}

LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def fetch_season(season):
    """Return (DataFrame, dropped_list). Raises with a SHORT message."""
    url = URL.format(season, str(season + 1)[-2:])
    r = requests.get(url, headers=HEADERS, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f'HTTP {r.status_code} (likely IP blocked)')

    body = r.text
    if len(body) < 5000:
        raise RuntimeError(f'page only {len(body)} bytes - blocked or empty')

    try:
        tables = pd.read_html(io.StringIO(body))
    except ValueError:
        raise RuntimeError('no HTML tables on page - layout changed or blocked')

    if not tables:
        raise RuntimeError('zero tables parsed')

    tbl = pick_table(tables)
    if tbl is None:
        widths = sorted({len(t.columns) for t in tables})
        raise RuntimeError(f'no odds table found (table widths seen: {widths})')
    if len(tbl) < 100:
        raise RuntimeError(f'odds table only {len(tbl)} rows - wrong table')

    return parse_table(tbl.to_dict('records'), season)


def main(start, end):
    frames, ok = [], 0

    for s in range(start, end + 1):
        try:
            games, dropped = fetch_season(s)
        except Exception as e:
            # keep it to ONE short line -- no tracebacks, no page dumps
            log(f'{s}: FAILED  {type(e).__name__}: {str(e)[:120]}')
            time.sleep(2)
            continue

        frames.append(games)
        ok += 1
        total = len(games) + len(dropped)
        pct = 100 * len(dropped) / max(1, total)
        log(f'{s}: {len(games):3d} games, {len(dropped):2d} dropped ({pct:.1f}%)')
        time.sleep(2)

    log(f'\n{ok} of {end - start + 1} seasons fetched')

    if not frames:
        log('NOTHING FETCHED.')
        log('If every season shows HTTP 403 or a tiny page, the site is')
        log('blocking datacenter IPs. GitHub runners live in Azure ranges,')
        log('so this workflow cannot reach it. Fall back to saving the')
        log('pages from a browser and committing the HTML to the repo.')
        return 1

    out = pd.concat(frames, ignore_index=True)
    out.to_csv('nfl_open_close.csv', index=False)
    mv = out.close_spread - out.open_spread
    log(f'\nwrote nfl_open_close.csv ({len(out)} games)')
    log(f'  mean abs line move : {mv.abs().mean():.2f} pts')
    log(f'  no move            : {(mv == 0).mean():.1%} of games')
    return 0


if __name__ == '__main__':
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 2021
    try:
        code = main(a, b)
    finally:
        with open('fetch_log.txt', 'w') as f:
            f.write('\n'.join(LOG) + '\n')
    sys.exit(code)
