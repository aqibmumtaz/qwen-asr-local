#!/usr/bin/env python3
"""Run scenario files against the bot. Usage: python scripts/run_scenarios.py [--tag smoke] [--id sw_a_to_b] [-v]"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.scenarios import load_scenarios, run_many  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--id")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    scs = load_scenarios()
    if a.tag:
        scs = [s for s in scs if a.tag in s.get("tags", [])]
    if a.id:
        scs = [s for s in scs if s["id"] == a.id]
    res = await run_many(scs)
    bad = 0
    for r in res:
        print(("PASS " if r.ok else "FAIL ") + r.id + "  " + r.title)
        if not r.ok:
            bad += 1
            for f in r.failures:
                print("     -", f)
        if a.v or not r.ok:
            for s in r.steps:
                print(f"       U: {s.say}\n       B: {s.reply}   [{s.route['decision']}/{s.route.get('workflow')}]")
    print(f"\n{len(res) - bad}/{len(res)} scenarios passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
