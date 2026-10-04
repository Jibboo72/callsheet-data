"""
Parse Sportsbook Reviews Online NFL odds archives into a clean
one-row-per-game table with opening and closing spreads and totals.

Source pages (free, 2007-08 through 2021-22):
  https://sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-YYYY-YY

Format notes:
  - Two rows per game, V (visitor) then H (home). Neutral games use N/N.
  - Of the two rows, ONE carries the point spread and the OTHER carries
    the game total, for both the Open and Close columns. Which row has
    which is not fixed -- normally the favorite's row carries the spread.
  - 'pk' means pick'em (spread 0).
  - The source contains real errors: values swapped between the paired
    rows, negative spreads, and totals in the spread slot. This parser
    detects and drops those games rather than silently ingesting them.

Output columns:
  season, date, away, home, away_score, home_score,
  open_spread, close_spread   (home perspective, negative = home favored)
  open_total, close_total
"""
import io
import re
import sys
import pandas as pd

# ------------------------------------------------------------------ teams
TEAMS = {
    'Arizona': 'ARI', 'Atlanta': 'ATL', 'Baltimore': 'BAL', 'Buffalo': 'BUF',
    'Carolina': 'CAR', 'Chicago': 'CHI', 'Cincinnati': 'CIN', 'Cleveland': 'CLE',
    'Dallas': 'DAL', 'Denver': 'DEN', 'Detroit': 'DET', 'GreenBay': 'GB',
    'Houston': 'HOU', 'Indianapolis': 'IND', 'Jacksonville': 'JAX',
    'KansasCity': 'KC', 'LasVegas': 'LV', 'Oakland': 'OAK',
    'LAChargers': 'LAC', 'SanDiego': 'SD', 'LARams': 'LA', 'StLouis': 'STL',
    'Miami': 'MIA', 'Minnesota': 'MIN', 'NewEngland': 'NE', 'NewOrleans': 'NO',
    'NYGiants': 'NYG', 'NYJets': 'NYJ', 'Philadelphia': 'PHI',
    'Pittsburgh': 'PIT', 'SanFrancisco': 'SF', 'Seattle': 'SEA',
    'TampaBay': 'TB', 'Tennessee': 'TEN', 'Washington': 'WAS',
}

# plausible ranges -- anything outside these is treated as corrupt
SPREAD_MAX = 28.0
TOTAL_MIN, TOTAL_MAX = 33.0, 75.0   # NFL totals never open near 30


# ------------------------------------------------------- table normalizing
SBR_COLS = ['Date', 'Rot', 'VH', 'Team', '1st', '2nd', '3rd', '4th',
            'Final', 'Open', 'Close', 'ML', '2H']
NEEDED = {'Date', 'VH', 'Team', 'Open', 'Close'}


def normalize_table(tbl):
    """
    SBR pages don't mark the header row with <th>, so pandas often reads
    it as data and names the columns 0,1,2... Recover the real header:
      1. already correct -> use as is
      2. a row containing 'VH' and 'Open' -> promote it to the header
      3. exactly 13 columns -> assign the known SBR layout positionally
    Returns a DataFrame with usable column names, or None.
    """
    tbl = tbl.copy()
    tbl.columns = [str(c).strip() for c in tbl.columns]

    if NEEDED <= set(tbl.columns):
        return tbl

    # look for the header row hiding in the data
    for i in range(min(5, len(tbl))):
        vals = [str(v).strip() for v in tbl.iloc[i].tolist()]
        if 'VH' in vals and 'Open' in vals:
            out = tbl.iloc[i + 1:].copy()
            out.columns = vals
            if NEEDED <= set(out.columns):
                return out.reset_index(drop=True)

    # fall back to the known fixed layout
    if len(tbl.columns) == len(SBR_COLS):
        tbl.columns = SBR_COLS
        return tbl

    return None


def pick_table(tables):
    """Choose the odds table: the biggest one that normalizes cleanly."""
    best = None
    for t in sorted(tables, key=len, reverse=True):
        n = normalize_table(t)
        if n is not None and len(n) >= 100:
            return n
        if n is not None and best is None:
            best = n
    return best


def _num(v):
    """'pk' -> 0.0, 'NL'/blank -> None, else float."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ('pk', 'p', 'pick'):
        return 0.0
    if s in ('', 'nl', 'na', '-'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _classify(a, b):
    """
    Given the two paired values from one column (visitor row, home row),
    return (spread_magnitude, total, favorite) where favorite is 'V', 'H',
    or None if the pair is unusable.

    The row carrying the SPREAD is the favorite. The other carries the TOTAL.
    """
    if a is None or b is None:
        return None
    # negative values in this source are corruption, not real spreads
    if a < 0 or b < 0:
        return None

    a_spread = a <= SPREAD_MAX
    b_spread = b <= SPREAD_MAX
    a_total = TOTAL_MIN <= a <= TOTAL_MAX
    b_total = TOTAL_MIN <= b <= TOTAL_MAX

    # exactly one looks like a spread and the other like a total
    if a_spread and b_total and not a_total:
        return (a, b, 'V')
    if b_spread and a_total and not b_total:
        return (b, a, 'H')
    return None          # both spreads, both totals, or out of range


def parse_table(rows, season):
    """rows: list of dicts with keys Date, VH, Team, Final, Open, Close."""
    out, dropped = [], []

    i = 0
    while i < len(rows) - 1:
        r1, r2 = rows[i], rows[i + 1]
        i += 2

        vh1, vh2 = r1['VH'].strip().upper(), r2['VH'].strip().upper()
        if {vh1, vh2} not in ({'V', 'H'}, {'N'}):
            dropped.append((r1['Team'], r2['Team'], 'unpaired rows'))
            continue

        # first row of a pair is the visitor (true for V/H and N/N)
        away_row, home_row = r1, r2

        away = TEAMS.get(away_row['Team'].strip())
        home = TEAMS.get(home_row['Team'].strip())
        if not away or not home:
            dropped.append((away_row['Team'], home_row['Team'], 'unknown team'))
            continue

        o = _classify(_num(away_row['Open']), _num(home_row['Open']))
        c = _classify(_num(away_row['Close']), _num(home_row['Close']))
        if o is None or c is None:
            dropped.append((away, home, 'ambiguous or corrupt open/close'))
            continue

        o_mag, o_total, o_fav = o
        c_mag, c_total, c_fav = c

        # express spread from the HOME perspective: negative = home favored
        open_spread = -o_mag if o_fav == 'H' else o_mag
        open_spread = open_spread + 0.0
        close_spread = -c_mag if c_fav == 'H' else c_mag

        # sanity: the favorite shouldn't flip between open and close on a
        # non-trivial number -- that pattern is the known row-swap bug
        if o_fav != c_fav and min(o_mag, c_mag) >= 2.0:
            dropped.append((away, home, 'favorite flips open->close'))
            continue

        md = str(away_row['Date']).zfill(3)
        month, day = int(md[:-2]), int(md[-2:])
        year = season if month >= 8 else season + 1

        out.append({
            'season': season,
            'date': f'{year}-{month:02d}-{day:02d}',
            'away': away, 'home': home, 'neutral': vh1 == 'N',
            'away_score': _num(away_row['Final']),
            'home_score': _num(home_row['Final']),
            'open_spread': open_spread, 'close_spread': close_spread,
            'open_total': o_total, 'close_total': c_total,
        })

    return pd.DataFrame(out), dropped


def read_pipe_file(path, season):
    """Read the sample/pipe-delimited form used for testing."""
    df = pd.read_csv(path, sep='|', dtype=str)
    return parse_table(df.to_dict('records'), season)


def read_html(path_or_html, season):
    """Read a saved SBR season page and parse its odds table.
    Accepts a file path or a raw HTML string."""
    if '<' in str(path_or_html)[:200]:
        src = io.StringIO(path_or_html)
    else:
        src = path_or_html
    tbl = pick_table(pd.read_html(src))
    if tbl is None:
        raise RuntimeError('no recognizable odds table')
    return parse_table(tbl.to_dict('records'), season)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'sample.txt'
    season = int(sys.argv[2]) if len(sys.argv) > 2 else 2021
    reader = read_html if path.endswith(('.html', '.htm')) else read_pipe_file
    games, dropped = reader(path, season)

    print(games.to_string(index=False))
    print(f'\nparsed {len(games)} games, dropped {len(dropped)}')
    for d in dropped:
        print('  DROPPED', d)
