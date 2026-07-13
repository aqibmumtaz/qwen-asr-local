#!/usr/bin/env python3
"""
END-TO-END A/B: old lexicon vs lexicons_v2, on all 183 gold turns.

For every turn we take the RAW Devanagari the ASR produced (model_output_hindi),
run the FULL transliterate() pipeline under each lexicon, and score the Roman
Urdu against the human gold (roman_urdu_reference).

This is the real test — not "how many entries did we keep" but "does the output
actually get closer to what a human wrote".

Metrics:
  WER accuracy  = 1 - edit_distance(hyp, ref) / len(ref)       (word level)
  exact-word    = fraction of gold words reproduced exactly
  turns better / worse / same

Usage:
    python3 compare_lexicons.py
    python3 compare_lexicons.py --examples 15
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import openpyxl

SCRIPT_DIR = Path(__file__).resolve().parent
XLSX = SCRIPT_DIR / "data" / "CLL analysis" / "turnwise_results_eval_full.xlsx"


def load_transliterate(version: str):
    """Re-import hindi_to_roman_urdu with LEXICON=<version>."""
    os.environ["LEXICON"] = version
    sys.path.insert(0, str(SCRIPT_DIR))
    import hindi_to_roman_urdu as H
    importlib.reload(H)
    return H.transliterate, len(H.WORD_MAP), len(H.PHRASE_MAP)


def wer_acc(hyp: str, ref: str) -> float:
    h, r = hyp.lower().split(), ref.lower().split()
    if not r:
        return 1.0
    dp = list(range(len(h) + 1))
    for rw in r:
        nd = [dp[0] + 1] + [0] * len(h)
        for j, hw in enumerate(h):
            nd[j + 1] = min(dp[j] + (0 if hw == rw else 1), dp[j + 1] + 1, nd[j] + 1)
        dp = nd
    return max(0.0, 1.0 - dp[len(h)] / len(r))


def exact_words(hyp: str, ref: str) -> tuple[int, int]:
    """How many gold words appear in the hypothesis (multiset intersection)."""
    from collections import Counter
    h, r = Counter(hyp.lower().split()), Counter(ref.lower().split())
    return sum((h & r).values()), sum(r.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    wb = openpyxl.load_workbook(str(XLSX), data_only=True)
    rows = list(wb["asr_results"].iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}

    turns = []
    for r in rows[1:]:
        hindi = r[idx["model_output_hindi"]]
        ref = r[idx["roman_urdu_reference"]]
        if isinstance(hindi, str) and isinstance(ref, str) and hindi.strip() and ref.strip():
            turns.append((r[idx["turn"]], r[idx["speaker"]], hindi, ref))

    print("=" * 78)
    print("  END-TO-END COMPARISON — old lexicon vs lexicons_v2")
    print("=" * 78)
    print(f"  turns: {len(turns)}   (raw Devanagari -> transliterate() -> Roman Urdu -> vs gold)")
    print()

    results = {}
    for ver in ("old", "v2"):
        tr, nw, npz = load_transliterate(ver)
        accs, ex_hit, ex_tot, outs = [], 0, 0, []
        for turn, spk, hindi, ref in turns:
            hyp = tr(hindi)
            accs.append(wer_acc(hyp, ref))
            h, t = exact_words(hyp, ref)
            ex_hit += h
            ex_tot += t
            outs.append(hyp)
        results[ver] = {
            "acc": sum(accs) / len(accs),
            "accs": accs,
            "exact": ex_hit / ex_tot,
            "ex_hit": ex_hit,
            "ex_tot": ex_tot,
            "outs": outs,
            "entries": nw + npz,
        }
        print(f"  [{ver:<3}] lexicon entries: {nw:>6} words + {npz:>3} phrases")

    o, v = results["old"], results["v2"]
    print()
    print("=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print()
    print(f"  {'metric':<26} {'OLD':>12} {'v2':>12} {'delta':>12}")
    print(f"  {'-'*26} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'WER accuracy (mean)':<26} {o['acc']*100:>11.2f}% {v['acc']*100:>11.2f}% "
          f"{(v['acc']-o['acc'])*100:>+11.2f}%")
    print(f"  {'gold words reproduced':<26} {o['exact']*100:>11.2f}% {v['exact']*100:>11.2f}% "
          f"{(v['exact']-o['exact'])*100:>+11.2f}%")
    print(f"  {'  (count)':<26} {o['ex_hit']:>12} {v['ex_hit']:>12} "
          f"{v['ex_hit']-o['ex_hit']:>+12}")

    better = sum(1 for a, b in zip(o["accs"], v["accs"]) if b > a + 1e-9)
    worse = sum(1 for a, b in zip(o["accs"], v["accs"]) if b < a - 1e-9)
    same = len(turns) - better - worse
    print()
    print(f"  turns IMPROVED by v2 : {better:>4}")
    print(f"  turns WORSE with v2  : {worse:>4}")
    print(f"  turns unchanged      : {same:>4}")

    # biggest wins / losses
    deltas = sorted(
        ((v["accs"][i] - o["accs"][i], i) for i in range(len(turns))),
        reverse=True,
    )
    print()
    print("=" * 78)
    print(f"  BIGGEST IMPROVEMENTS (top {args.examples})")
    print("=" * 78)
    for d, i in deltas[: args.examples]:
        if d <= 0:
            break
        turn, spk, hindi, ref = turns[i]
        print(f"\n  turn {turn} ({spk})   {o['accs'][i]*100:.0f}% -> {v['accs'][i]*100:.0f}%  ({d*100:+.0f})")
        print(f"    gold : {ref}")
        print(f"    old  : {o['outs'][i]}")
        print(f"    v2   : {v['outs'][i]}")

    regress = [(d, i) for d, i in deltas if d < 0]
    if regress:
        print()
        print("=" * 78)
        print(f"  REGRESSIONS ({len(regress)})")
        print("=" * 78)
        for d, i in regress[-args.examples:]:
            turn, spk, hindi, ref = turns[i]
            print(f"\n  turn {turn} ({spk})   {o['accs'][i]*100:.0f}% -> {v['accs'][i]*100:.0f}%  ({d*100:+.0f})")
            print(f"    gold : {ref}")
            print(f"    old  : {o['outs'][i]}")
            print(f"    v2   : {v['outs'][i]}")
    print()


if __name__ == "__main__":
    main()
