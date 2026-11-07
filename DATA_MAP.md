# Data map — all of Josh's betting apps

This is the master map. Each of the other repos has a short `DATA_MAP.md` that
points back here. **Update this file in the same commit** whenever a data file,
workflow schedule, or cross-app link changes.

Last checked: 2026-11-06.

## The apps

| App | Sport | Repo | Where it runs | Data it publishes |
|---|---|---|---|---|
| **Call Sheet** | NFL + college football | `Jibboo72/callsheet-data` | GitHub Pages: `jibboo72.github.io/callsheet-data/` | `data/*.json` (this repo) |
| **SportsMF** | All (scores, bets, fantasy) | `Jibboo72/sportsmf` | GitHub Pages: `jibboo72.github.io/sportsmf/sportsmf.html` | `data/mfl.json` |
| **Buckets** | NBA | `Jibboo72/Hoops-Props` | GitHub Pages from `docs/`: `jibboo72.github.io/Hoops-Props/` | `docs/data/*.json` (board read by SportsMF; also `doubles.json` (incl. `parlay` = nightly 1-10 round-robin rating of the Best 6), `doubles_log.json`; the page floors chances at 1%; `backtest_live.json` = this-season-only backtest; `top6_live.json` = top-6 replay (most likely and biggest-bump picks), read by the Doubles page, incl. 2- and 3-team parlay counts) |
| **Scoresheet** | NHL | `Jibboo72/scoresheet` | GitHub Pages from `docs/`: `jibboo72.github.io/scoresheet/` | `docs/data/*.json` |
| **Vault** | MLB home runs | `Jibboo72/vault` (private) | Netlify (app + two key-holding functions) | nothing shared — browser storage only |
| Expert ATS Ledger | NFL experts | Claude artifact `claude.ai/artifact/9Kcoigd7nydaf1Vrtg8Ab6` | claude.ai | feeds `data/experts-2026.json` here (see `experts/README.md`) |
| *(old)* NFL-Power-Rank | NFL | `Jibboo72/NFL-Power-Rank` (private) | — | superseded by the Power tab in Call Sheet |

## How they connect

```
 nflverse / ESPN / CFBD / SportsGameOdds / Open-Meteo
                 │  (GitHub workflows in callsheet-data)
                 ▼
      callsheet-data/data/*.json ─────────────► Call Sheet (NFL + CFB)
                 │   ▲                                  │
                 │   └──── data/sync.json ◄─────────────┘  your bets/picks, pushed with a GitHub token
                 ▼
             SportsMF ◄──── scoresheet/docs/data/board.json  (NHL board)
                 ▲   ◄──── Hoops-Props/docs/data/board.json (NBA board)
                 ▲
                 └──── sportsmf/data/mfl.json  (MyFantasyLeague, pulled by sportsmf's own workflow)

 Vault (MLB) stands alone — nothing else reads it yet.
```

**Read pattern:** apps on `jibboo72.github.io` try the same-origin path first
(`/callsheet-data/data/…`, `/scoresheet/data/…`) and fall back to
`raw.githubusercontent.com/Jibboo72/<repo>/main/…`. Keep file names stable —
renaming a file breaks every app that reads it.

## callsheet-data `data/` files

| File | What's in it | Written by | When (UTC cron) | Read by |
|---|---|---|---|---|
| `games-2026.json` | NFL schedule, spreads/totals, model projection (`proj`, `ptot`), injuries, finals | `export_games.py` (build.yml) | Tue 10:17, Thu 18:37, Sat 09:23 + 12:41, Sun 09:17 + 12:47 | Call Sheet, SportsMF |
| `ratings-2026.json` | NFL power ratings | `export_ratings.py` (build.yml) | build.yml | Call Sheet, SportsMF |
| `matchup.json` | NFL matchup grades (pass/rush D) | `export_matchup.py` (build.yml) | build.yml | Call Sheet |
| `props-2026.json` | Player prop distributions (ATD with `tdu` TD-usage block, pass TDs, carries, longest rush, rush+rec yds), backtest calib, recent results | `export_props.py` (build.yml) — also pulls nflverse depth charts and ffopportunity expected yards (both free, optional) | build.yml | Call Sheet |
| `receivers-2026.json`, `receivers-2025.json` | Receiver game logs (incl. per-game `xtd`) | `build_receivers.py` (build.yml) | build.yml | Call Sheet |
| `roster-2026.json` | Roster, status, injury report, snaps | `export_roster.py` (build.yml) | build.yml | Call Sheet |
| `def_history.json` | Weekly defense ranks | `export_def_history.py` (build.yml) | build.yml | `build_receivers.py` only |
| `odds-2026.json` | NFL odds + player props from SportsGameOdds (`lines`, `props`, `start`) | `pull_odds.py` (odds.yml) | Wed 10:13, Fri 19:53 | Call Sheet, SportsMF |
| `closing-2026.json` | Last pregame prices per game (lines, team totals, NFL props), captured 0–50 min before kickoff; pruned after 21 days | `close_odds.py` (close.yml) | every 20 min in game windows (see Gotchas) | Call Sheet (stores closing line value on each bet as `clvp`) |
| `cfb-2026.json` | College schedule + finals (`kick` is **UTC**) | `export_cfb.py` (build.yml) | build.yml | Call Sheet |
| `cfb-ratings-2026.json` | College ratings, team colors/abbr/logos, game projections (`kick` is **Eastern**) | `export_cfb_ratings.py` (build.yml, cfb-odds.yml) | build.yml + Fri 19:43, Sat 10:07 | Call Sheet, SportsMF |
| `fcs-ratings-2026.json` | FCS ratings + games | `export_cfb_ratings.py` | same | Call Sheet, SportsMF |
| `cfb-odds-2026.json` | College odds from SportsGameOdds | `pull_cfb_odds.py` (cfb-odds.yml) | Fri 19:43, Sat 10:07 | `export_cfb_ratings.py` only |
| `weather-2026.json` | Game-day forecasts + "weather under" flags | `weather.py forecast` (build.yml, cfb-odds.yml) | same | Call Sheet, SportsMF |
| `weather-hist.json`, `weather-backtest.json` | Historical weather + the backtest behind the flags | `weather.py backtest` (weather-backtest.yml, manual) | manual | research only |
| `experts-2026.json` | Expert crowd per NFL game (home/away counts) | `experts/export_cs.py` — by hand or the Friday Claude scheduled task | Fridays | Call Sheet, SportsMF |
| `cfb-backtest.json`, `fcs-backtest.json` | Rating backtest summaries; `cfb-backtest.json` `market` = how much of the model-vs-line gap held up (k ≈ 0) | `backtest_cfb.py` (manual), copied into the ratings file by `export_cfb_ratings.py` | manual | Call Sheet (college "model lean" note) |
| `games-2025.json`, `roster-2025.json` | Last season | build scripts | frozen | research / prior-season fits |
| `sync.json` | **Your bets, picks, bankroll and settings** | the Call Sheet app itself, via the GitHub API with your token (commits named "Call Sheet sync") | seconds after each change | SportsMF, Call Sheet on other devices |
| `shared.json` | Read-only snapshot for friends | Call Sheet › Settings › Share (you upload it) | by hand | `share.html` |

Other folders: `experts/` (expert-pick pipeline, see its README), `research/cfbd/`
(raw CFBD pulls, cfbd-research.yml, manual).

## Outside data the workflows pull (all free, no keys unless noted)

| Source | Files | Used by |
|---|---|---|
| nflverse (`github.com/nflverse/nflverse-data` releases) | play-by-play, schedules, players, rosters, weekly rosters, snap counts, injuries, depth charts | NFL ratings, props, rosters, weather backtest |
| ffopportunity (`github.com/ffverse/ffopportunity`, `latest-data`) | `ep_weekly_{season}.parquet` (expected yards) | props (carries, rush+rec yards) |
| ESPN | NFL injury report | `export_games.py` |
| SportsGameOdds (`SGO_KEY`, credits) | odds + props | `pull_odds.py`, `pull_cfb_odds.py`, `close_odds.py` (one object per game, ~16 NFL + ~60 college a week) |
| CollegeFootballData (`CFBD_KEY`) | college games, ratings inputs | `export_cfb*.py` |
| Open-Meteo | forecasts | `weather.py` |

Tested and not used: FTN charting (box counts, blitz, play action, screens) added nothing to any prop in the 2025 and 2026 backtests.

## Live feeds the apps call from your phone (not from GitHub)

| App | Feed |
|---|---|
| Call Sheet | ESPN scoreboard (NFL + college finals), ESPN NFL summary, ESPN college rankings (AP/coaches/CFP for the game strip), GitHub Actions API (run history for Settings › Data schedule — keep the job list in `index.html` `JOBS` in step with the workflow crons), Anthropic Messages API (bet slip reader: 📷 buttons on Bank and each card send screenshots to Claude with your own key) |
| SportsMF | ESPN site + standings APIs, MLB Stats API, ESPN logos |
| Vault | MLB Stats API, Baseball Savant, Open-Meteo, plus Ballpark Pal and SportsGameOdds through its Netlify functions |

## Secrets (names only — values live in each repo's GitHub/Netlify settings)

| Repo | Secrets |
|---|---|
| callsheet-data | `SGO_KEY` (SportsGameOdds), `CFBD_KEY` (CollegeFootballData) |
| sportsmf | `MFL_APIKEY` |
| Hoops-Props | `ODDS_API_KEY` (The Odds API), used only when the repo variable `USE_ODDS` is `true`. Its balance (`docs/data/credits.json`) is separate from Scoresheet's key |
| scoresheet | `ODDS_API_KEY` (The Odds API), used only when the repo variable `USE_ODDS` is `true` (off since 2026-11-06) |
| vault | Ballpark Pal + SportsGameOdds keys in Netlify function env |
| Call Sheet / SportsMF on your phone | a GitHub fine-grained token (Contents read/write on callsheet-data; sportsmf for the MFL refresh), stored on the device only |
| Call Sheet on your phone | an Anthropic API key for the bet slip reader (Settings › Bet slip reader), in its own localStorage slot `cs-anthropic-key` — never in `S`, so never in backups, `sync.json` or `shared.json` |

Never put a key in an `index.html` — anything there is public.

## Known gaps and gotchas

- **Schedules (since 2026-11-03):** crons are UTC set for Eastern *standard* time
  (in summer they land an hour later on the ET clock), off the top of the hour
  because GitHub started top-of-the-hour runs 3-6 hours late, with backup build
  runs Saturday and Sunday mornings. Odds jobs get no backups (they spend SGO
  credits). Every job re-syncs before pushing because the app commits
  `sync.json` often. Settings › Data schedule shows planned vs actual runs;
  its `JOBS` list in `index.html` must match these crons.
- **Closing prices (since 2026-11-06):** `close.yml` runs at :07/:27/:47 UTC in
  hours 0–1 daily, 16–23 Sat/Sun, 2–3 Sun, 13–14 Sun. Each run asks SGO only for
  games in `odds-2026.json` / `cfb-odds-2026.json` starting in the next 50 min that
  aren't captured yet, so a quiet run spends nothing. Settings › Data schedule shows
  it as a "loose" job (last runs, no slot table).
- **Two kickoff clocks for college:** `cfb-2026.json` `kick` is UTC,
  `cfb-ratings-2026.json` `kick` is Eastern. The game strip uses the ratings one.
- **`sportsmf/index.html` is an old copy of Call Sheet**, not SportsMF. The real
  app is `sportsmf.html` (the manifest and service worker start there).
- **SportsMF reads both prop boards** (NBA from Hoops-Props, NHL from Scoresheet) — keep
  their `board.json` paths and top-level keys stable.
- **Odds moved from Scoresheet to Buckets (2026-11-06):** Buckets prices only double-doubles (`player_double_double`,
  first priority) and threes (`player_threes`) when `USE_ODDS` is `true`, with `NBA_SHARE` = 1.0. Threes prices go on
  `board.json` (`thr_*`); `reb_*` / `ast_*` stay fair-only; double-double prices go on `doubles.json` as each player's
  `odds` block, not on `board.json`, so SportsMF's NBA Props tab keeps its shape. With `USE_ODDS` off, `board.json` has
  model-only fair prices (no `price`, `book`, `ev`, `likely` on markets).
- **Scoresheet odds are off (since 2026-11-06):** its `tonight.yml` passes the key only when its own `USE_ODDS` is `true`.
  `board.json` is model-only (`assist` and `sog1.5` with `p`/`fair`, no prices or game totals); SportsMF's NHL Props
  tab shows model chance and fair odds with "no price".
  Players ruled Out on ESPN's injury report are dropped from `players[]`; others carry an optional `status`.
- **Buckets round-robin rating (since 2026-11-06):** `parlay.py` (run by `board.py`) adds a `parlay` block to
  `doubles.json`: the Best 6 double-doubles as a $30 round robin, 2-pick (15 x $2) and 3-pick (20 x $1.50), each rated
  1-10 (5 = break-even) with the single book used, expected profit, chance to profit, chance to lose all, and `better`.
  No extra odds credits; it reuses the board's pull. The channel's daily post reads it; final numbers come from the
  first Tonight run of the day when odds are on (15:17 UTC). SportsMF doesn't read it; `board.json` is unchanged.
- **Buckets schedule (since 2026-11-05):** `tonight.yml` runs at 15:17, 18:43 and 21:47 UTC (the old 16/19/22:00 runs were
  skipped or hours late). With odds off every run rebuilds the board; with odds on only the first run of the day does.
- **Vault keeps everything in the browser** — no shared file, so its picks can't
  show in SportsMF until it publishes or syncs something.
