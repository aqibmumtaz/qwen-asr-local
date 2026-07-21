"""
Sample end-to-end test: run the FULL two-pass pipeline on one call's audio.

For every chunk of the chosen call:  pass1 (no ctx) -> retrieve names -> pass2 (biased).
Then concatenate, transliterate() -> Roman Urdu, and score vs the call's benchmark, so we
can see baseline (pass1) vs two-pass (pass2) vs gold.

  python -m acoustic_contextual_biasing.sample_test            # first call with audio
  python -m acoustic_contextual_biasing.sample_test --backend local
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark"))

import openpyxl
from test_accuracy import diff_words

from .two_pass import TwoPass

XLSX = ROOT / "benchmark" / "lab_test_80_calls_urdu_roman_urdu.xlsx"
AUDIO = ROOT / "benchmark" / "lab_test_80_audios_chunks_25s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="remote", choices=["remote", "local"])
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    os.environ["LEXICON"] = "v2"; os.environ["RESOLVER"] = "0"; os.environ["PHONETIC"] = ""
    import hindi_to_roman_urdu as H
    importlib.reload(H)

    wb = openpyxl.load_workbook(str(XLSX)); ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True)); idx = {h: i for i, h in enumerate(rows[0])}
    calls = defaultdict(list)
    for r in rows[1:]:
        calls[r[idx["call_id"]]].append(r)
    for cid in calls:
        calls[cid].sort(key=lambda x: x[idx["chunk_index"]] or 0)

    # first call that has audio + a benchmark
    cid = bench = None
    for c, cr in calls.items():
        b = next((x[idx["benchmark_roman_urdu"]] for x in cr
                  if isinstance(x[idx["benchmark_roman_urdu"]], str)
                  and x[idx["benchmark_roman_urdu"]].strip()), None)
        if b and (AUDIO / str(c)).exists():
            cid, bench = c, b
            break
    chunks = sorted((AUDIO / str(cid)).glob("chunk_*.wav"))
    print(f"call {cid[-12:]}  ({len(chunks)} chunks)\n", flush=True)

    tp = TwoPass(backend=args.backend, k=args.k)
    p1_all, p2_all = [], []
    for ch in chunks:
        t0 = time.time()
        r = tp.transcribe(ch)
        p1_all.append(r["pass1"]); p2_all.append(r["pass2"])
        print(f"  {ch.name}  {time.time()-t0:.0f}s", flush=True)
        print(f"    pass1     : {r['pass1'][:100]}", flush=True)
        print(f"    retrieved : {r['names'][:10]}", flush=True)
        print(f"    pass2     : {r['pass2'][:100]}\n", flush=True)

    r1 = H.transliterate(" ".join(p1_all))
    r2 = H.transliterate(" ".join(p2_all))
    a1 = diff_words(bench, r1).accuracy
    a2 = diff_words(bench, r2).accuracy
    print("=" * 70)
    print("  SAMPLE — full two-pass, end to end")
    print("=" * 70)
    print(f"  pass1 (baseline)  {a1:5.1f}%   {r1[:110]}")
    print(f"  pass2 (two-pass)  {a2:5.1f}%   {r2[:110]}")
    print(f"  gold                     {bench[:110]}")
    print()


if __name__ == "__main__":
    main()
