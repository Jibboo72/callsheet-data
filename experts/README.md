# Expert ATS Ledger — weekly load

Published NFL expert picks, graded against the spread. Two places use them:

1. **Expert ATS Ledger** — Claude artifact `https://claude.ai/artifact/9Kcoigd7nydaf1Vrtg8Ab6`
   (database collections: `experts`, `games`, `bulk`, `picks`).
2. **Call Sheet** — `data/experts-2026.json` in this repo (crowd line on NFL Sides cards,
   model-vs-crowd tracker). Loaded by "Refresh all data".

Picks come from articles, so this is done by hand (or by the Friday scheduled task), not by a workflow.

## Files

- `wN.txt` — everything known about week N: games, final scores, and every expert's picks.
- `build.py N` — parses `wN.txt` into `out/2026-wN-AWAY-HOME.json` (game docs) and
  `out/bulk-2026-wN.json` (all picks for the week). Standard line = median of every spread
  quoted that week (home perspective, snapped to the half point).
- `verify.py WEEKS [own|std] [ids]` — season records; `own` grades each pick on the line the
  expert quoted. Use it to check against records outlets publish.
- `emit.py N` — prints ArtifactData batch `writes` for week N (add `if_version` for docs that exist).
- `export_cs.py ../data/games-2026.json ../data/experts-2026.json` — builds the Call Sheet file
  (human experts only; `sbd-model` and `espn-fpi` excluded; LAR→LA).

Run everything from this folder (`out/` is scratch and not committed).

## wN.txt format

```
game AWAY-HOME YYYY-MM-DD [awayScore homeScore]
line GAME TEAM N                         # reference line only
pick EXPERT GAME TEAM [N]                # ATS side; N = that team's spread as quoted
grp GAME [TEAM N] | exp:TEAM exp:TEAM    # one table row from a grid
score EXPERT GAME WINNER WS LS           # score prediction → side vs the standard line
```
Team codes: ledger style (LAR for the Rams, WAS, JAX). Neutral-site games: use the designated home team.
Leave a quoted line off (`pick e GAME TEAM`) when the source's sign looks wrong.

## Sources and expert ids (find each with WebSearch "extended"; constructed URLs get refused)

| Source | Expert ids | Format |
|---|---|---|
| CBS grid `cbssports.com/nfl/picks/experts/against-the-spread/N/` | pete-prisco, jared-dubin, ryan-wilson, john-breech, tyler-sullivan, dave-richard, jamey-eisenberg | ATS grid with lines; page also lists week/overall records — verify against them |
| Bleacher Report "Bleacher Report's Expert Week N 2026 NFL Picks" | moe-moton, gary-davenport, brad-gagnon, ian-hanford, kris-knox, brent-sobleski, wes-odonnell | ATS grid |
| Yahoo, Frank Schwab "Week N NFL picks against the spread, predictions for every game" | frank-schwab | ATS list |
| Athlon "Week N NFL Picks: Expert Picks Against the Spread" | cory-bonini, luke-easterling, craig-ellenport, doug-farrar | ATS grid |
| RotoWire "Beating the Book" (Nick Whalen) | nick-whalen | score predictions + lines |
| RotoWire "NFL Staff Picks: Week N" | jeff-erickson, scott-pianowski, jim-coventry, ian-hartitz | ATS grid (BetMGM) with season records |
| Covers "NFL Picks Against the Spread for Every Week N Game" (Chris Vasile) | chris-vasile | ATS list |
| Fantasy Alarm "2026 NFL Week N Picks Against The Spread" (Bryan Derr) | bryan-derr | ATS list |
| Sports Betting Dime AI model (Sascha Paruk) | sbd-model | ATS list (model) |
| PFT "PFT's Week N 2026 NFL picks" (Mike Florio) | mike-florio | score predictions |
| NFL.com "NFL Week N picks: Upset and score predictions" | bobby-kownack, brooke-cersosimo, dan-parr, gennaro-filice, jeremy-bergman | score predictions |
| SI "NFL Week N ATS Picks for Every Game" (Iain MacMillan only — skip weeks by other writers) | iain-macmillan | ATS list |
| FanSided "NFL Week N picks and predictions, straight-up and ATS" (Cody Williams' column) | cody-williams | ATS list |
| Sportsbook Review "NFL ATS Picks Week N" (Gary Pearson) | gary-pearson | ATS list |
| ESPN "Our guide to every NFL Week N game" (abc7news.com syndicated copy if espn.com reads empty) | pamela-maldonado, eric-moody, seth-walder, espn-fpi | score predictions; FPI margin as `score espn-fpi GAME TEAM margin 0` |
| PhillyVoice "Week N NFL straight up, against the spread, and survivor pool picks" (Jimmy Kempski) | jimmy-kempski | selective ATS |
| The Ringer best bets (Anthony Dabbundo) | anthony-dabbundo | selective ATS, when findable |

Scores: `mcp__widgets__fetch_sports_data` (scores, nfl) or footballdb.com week pages.
