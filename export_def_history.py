"""
Point-in-time pass-defense ratings, one snapshot per week.

    python3 export_def_history.py 2025 [def_history.json]

Why point-in-time and not end-of-season: to judge what a receiver did in
Week 12, you want the defense as it looked ENTERING Week 12. build_ratings
uses only data strictly before the week you ask for, so that game — and the
receiver's own production in it — is excluded from the rating he's being
graded against. End-of-season ratings would let a player's own big day
inflate the very number used to discount it.

Writes {"season": 2025, "weeks": {"12": {"MIN": 7.31, ...}, ...}}
where the value is def_pass in points per game, higher = tougher defense,
league average = 0.
"""
import json
import sys

from power_ratings import load_pbp, build_ratings, CONFIG

# Ridge shrinks everything toward league average early, so Weeks 1-3 carry
# almost no signal. Grade those games against a neutral opponent instead of
# a rating the data hasn't earned.
FIRST_RATED_WEEK = 4


def build(season):
    pbp = load_pbp([season])
    reg = pbp[pbp.season_type == "REG"]
    last = int(reg.week.max())

    weeks = {}
    for w in range(1, last + 1):
        if w < FIRST_RATED_WEEK:
            continue
        r = build_ratings(pbp, season, w, CONFIG)
        weeks[str(w)] = {t: round(float(v), 3) for t, v in r["def_pass"].items()}
        print(f"  week {w:2d}: {len(weeks[str(w)])} teams", file=sys.stderr)

    return {"season": int(season), "first_rated_week": FIRST_RATED_WEEK,
            "last_week": last, "weeks": weeks}


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    out = sys.argv[2] if len(sys.argv) > 2 else "def_history.json"
    data = build(season)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out}: {len(data['weeks'])} weekly snapshots", file=sys.stderr)
