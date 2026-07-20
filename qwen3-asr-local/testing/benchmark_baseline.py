#!/usr/bin/env python3
"""
Baseline benchmark on the 80-call lab-test set, CALL-LEVEL.

Ground truth is per-call (benchmark_roman_urdu, on chunk_index==0). model_output_*
are per-chunk. So for each call we concatenate the per-chunk column in chunk order,
produce Roman Urdu, and score against that call's benchmark.

Metric: the dev's testing/test_accuracy.py :: diff_words   (fuzzy word RECALL, >=0.70)
Also reported: classic edit-distance WER accuracy (1 - edit/ref_words), same tokeniser.

Configs:
  C-prev     previous model's own Roman  (model_output_roman_urdu)
  C0         our transliterate() + v2 exact lexicon, RESOLVER OFF
  C0+res     our transliterate() + v2 exact lexicon, RESOLVER ON

Writes per-call columns into a COPY of the sheet; original is untouched.
"""
import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))   # test_accuracy.py
sys.path.insert(0, str(ROOT))   # hindi_to_roman_urdu.py

from test_accuracy import diff_words, normalize_tokens

XLSX = HERE / "lab_test_80_calls_urdu_roman_urdu.xlsx"
OUT = HERE / "lab_test_80_calls_urdu_roman_urdu_benchmarked.xlsx"


def wer_acc(benchmark: str, hypothesis: str) -> tuple[int, int]:
    """Classic word edit-distance. Returns (edits, ref_len) using the dev tokeniser."""
    ref = normalize_tokens(benchmark)
    hyp = normalize_tokens(hypothesis)
    if not ref:
        return (0, 0)
    dp = list(range(len(hyp) + 1))
    for rw in ref:
        nd = [dp[0] + 1] + [0] * len(hyp)
        for j, hw in enumerate(hyp):
            nd[j + 1] = min(dp[j] + (0 if hw == rw else 1), dp[j + 1] + 1, nd[j] + 1)
        dp = nd
    return (dp[len(hyp)], len(ref))


def load_calls():
    wb = openpyxl.load_workbook(str(XLSX))
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    calls = defaultdict(list)
    for r in rows[1:]:
        calls[r[idx["call_id"]]].append(r)
    for cid in calls:
        calls[cid].sort(key=lambda r: (r[idx["chunk_index"]] if r[idx["chunk_index"]] is not None else 0))
    return wb, ws, idx, calls, rows


def concat(rows, idx, col):
    return " ".join(
        str(r[idx[col]]).strip()
        for r in rows
        if isinstance(r[idx[col]], str) and r[idx[col]].strip()
    )


def transliterate_for(resolver_on: bool):
    os.environ["LEXICON"] = "v2"
    os.environ["RESOLVER"] = "1" if resolver_on else "0"
    import hindi_to_roman_urdu as H
    importlib.reload(H)
    return H.transliterate


def main():
    wb, ws, idx, calls, rows = load_calls()

    # gather bench + concatenated Hindi per call
    per_call = {}
    for cid, crows in calls.items():
        bench = next((r[idx["benchmark_roman_urdu"]] for r in crows
                      if isinstance(r[idx["benchmark_roman_urdu"]], str)
                      and r[idx["benchmark_roman_urdu"]].strip()), None)
        if not bench:
            continue
        per_call[cid] = {"bench": bench,
                         "hindi": concat(crows, idx, "model_output_hindi")}

    # IMPORTANT: run each config in its OWN pass. transliterate() reads the module
    # global _RESOLVER at call time, and importlib.reload mutates the shared module
    # dict in place — so a function reference captured before a reload would still
    # see the NEW _RESOLVER. Compute all of one config, reload, then the next.
    for key, on in (("v0", False), ("v1", True)):
        tr = transliterate_for(on)
        for cid, d in per_call.items():
            d[key] = tr(d["hindi"])

    for cid, d in per_call.items():
        d["acc"] = {k: diff_words(d["bench"], d[k]) for k in ("v0", "v1")}
        d["wer"] = {k: wer_acc(d["bench"], d[k]) for k in ("v0", "v1")}

    # ---- aggregate ----
    def agg(cfg):
        m = sum(d["acc"][cfg].matched for d in per_call.values())
        t = sum(d["acc"][cfg].total for d in per_call.values())
        mean = sum(d["acc"][cfg].accuracy for d in per_call.values()) / len(per_call)
        we = sum(d["wer"][cfg][0] for d in per_call.values())
        wt = sum(d["wer"][cfg][1] for d in per_call.values())
        return dict(corpus=100 * m / t, mean=mean, matched=m, total=t,
                    wer=100 * max(0.0, 1 - we / wt))

    A = {c: agg(c) for c in ("v0", "v1")}
    labels = {"v0": "C0      (v2, resolver OFF)",
              "v1": "C0+res  (v2, resolver ON)"}

    print("=" * 78)
    print(f"  BASELINE BENCHMARK — 80-call set, call-level   ({len(per_call)} calls)")
    print("=" * 78)
    print(f"  metric: diff_words (fuzzy word recall >=0.70)  +  edit-distance WER acc")
    print()
    print(f"  {'config':<28} {'diff_words':>11} {'mean/call':>10} {'WER acc':>9} {'vs C0':>7}")
    print(f"  {'-'*28} {'-'*11} {'-'*10} {'-'*9} {'-'*7}")
    for c in ("v0", "v1"):
        d = A[c]
        vs = d["corpus"] - A["v0"]["corpus"]
        print(f"  {labels[c]:<28} {d['corpus']:>10.2f}% {d['mean']:>9.2f}% "
              f"{d['wer']:>8.2f}% {vs:>+6.2f}")
    print()
    print(f"  resolver effect (C0+res - C0): {A['v1']['corpus']-A['v0']['corpus']:+.2f} "
          f"pts diff_words, {A['v1']['wer']-A['v0']['wer']:+.2f} pts WER")
    print()

    # ---- write copy ----
    new_cols = ["our_roman_v2_res0", "our_roman_v2_res1",
                "acc_v2_res0", "acc_v2_res1",
                "wer_v2_res0", "wer_v2_res1"]
    base = len(rows[0])
    for j, name in enumerate(new_cols):
        ws.cell(row=1, column=base + 1 + j, value=name)
    # map call_id -> the chunk_index==0 excel row number
    header = base
    for ri, r in enumerate(ws.iter_rows(values_only=True), start=1):
        if ri == 1:
            continue
        cid = r[idx["call_id"]]
        ci = r[idx["chunk_index"]]
        if cid in per_call and ci == 0:
            d = per_call[cid]
            def werp(k):
                e, n = d["wer"][k]
                return round(100 * max(0.0, 1 - e / n), 2) if n else None
            vals = [d["v0"], d["v1"],
                    d["acc"]["v0"].accuracy, d["acc"]["v1"].accuracy,
                    werp("v0"), werp("v1")]
            for j, v in enumerate(vals):
                ws.cell(row=ri, column=base + 1 + j, value=v)
    wb.save(str(OUT))
    print(f"  written: {OUT.name}")
    print()


if __name__ == "__main__":
    main()
