#!/usr/bin/env python3
"""
One-off pull of CollegeFootballData team context for preseason-prior research
(returning production, 247 team talent, recruiting class points, transfer
portal), saved as raw JSON under research/cfbd/ so the modeling can be done
offline. Needs CFBD_KEY. Roughly 40 API calls.

    python3 fetch_cfbd_research.py 2017 2026
"""
import json
import os
import sys
import time

import requests

API = "https://api.collegefootballdata.com"
ENDPOINTS = {
    "returning": "/player/returning",
    "talent": "/talent",
    "recruiting": "/recruiting/teams",
    "portal": "/player/portal",
}


def main(a, b):
    key = os.environ.get("CFBD_KEY", "").strip()
    if not key:
        raise SystemExit("CFBD_KEY not set")
    os.makedirs("research/cfbd", exist_ok=True)
    log = []
    for y in range(a, b + 1):
        for name, path in ENDPOINTS.items():
            if name == "portal" and y < 2021:
                continue                      # portal data starts 2021
            try:
                r = requests.get(API + path, params={"year": y},
                                 headers={"Authorization": f"Bearer {key}"}, timeout=60)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                log.append(f"{name} {y}: FAILED {e}")
                continue
            with open(f"research/cfbd/{name}_{y}.json", "w") as f:
                json.dump(data, f, separators=(",", ":"))
            n = len(data) if isinstance(data, list) else 1
            log.append(f"{name} {y}: {n} rows")
            time.sleep(0.5)
    with open("research/cfbd/fetch_log.txt", "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 2017
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    main(a, b)
