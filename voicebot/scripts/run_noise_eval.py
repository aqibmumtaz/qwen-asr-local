#!/usr/bin/env python3
"""Robustness to ASR errors: replay scenarios with noise injected into the caller lines.

For every scenario that ends in a booking (final.booking) it compares each noisy run with the clean run:
  completed      the booking was created
  wrong_submit   a booking was created but with different tests/date/phone/area/age than the clean run
  safe_stop      no booking, but the call ended safely (transfer / asked to repeat) rather than submitting wrong data
Usage: python scripts/run_noise_eval.py [--levels 0.1 0.2 0.3] [--seeds 8]
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.nlp.noise import add_noise  # noqa: E402
from app.scenarios import load_scenarios, make_bot, new_scenario_session  # noqa: E402

KEYS = ("tests", "date", "phone", "age")


async def run_lines(sc, lines):
    bot = make_bot()
    s = new_scenario_session(bot, sc)
    for ln in lines:
        r = await bot.turn(s, ln)
        if s.ended:
            break
    created = [b for b in s.scratch["backend"].bookings.values() if b["ref"] != "HS-1000"]
    return (created[-1] if created else None), s


def sig(b):
    if b is None:
        return None
    out = {}
    for k in KEYS:
        v = b.get(k)
        out[k] = [t["name"] for t in v] if k == "tests" else v
    out["area"] = (b.get("address") or {}).get("area")
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    scs = [s for s in load_scenarios() if (s.get("final") or {}).get("booking") and not (s.get("setup") or {}).get("blocked_slots")]
    print(f"{len(scs)} booking scenarios x {a.seeds} seeds per noise level\n")
    print("%-8s %10s %13s %10s" % ("noise", "completed", "wrong_submit", "safe_stop"))
    rc = 0
    for lvl in a.levels:
        comp = wrong = safe = n = 0
        for sc in scs:
            lines = [st["say"] for st in sc["steps"]]
            clean, _ = await run_lines(sc, lines)
            for seed in range(a.seeds):
                noisy = [add_noise(l, lvl, seed * 1000 + i) for i, l in enumerate(lines)]
                b, sess = await run_lines(sc, noisy)
                n += 1
                if b is None:
                    safe += 1
                    if a.v:
                        print("   no booking:", sc["id"], seed, noisy[:3])
                elif sig(b) == sig(clean):
                    comp += 1
                else:
                    wrong += 1
                    print("   WRONG", sc["id"], "seed", seed, "\n     clean:", sig(clean), "\n     noisy:", sig(b))
        print("%-8s %9.0f%% %12.1f%% %9.0f%%" % (lvl, 100 * comp / n, 100 * wrong / n, 100 * safe / n))
        if wrong:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
