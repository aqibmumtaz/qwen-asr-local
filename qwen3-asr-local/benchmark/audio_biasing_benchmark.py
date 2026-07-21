"""
Audio contextual-biasing benchmark — does biasing fix mishearings on the 8kHz set?

Re-transcribes call chunks with the BASE Qwen3-ASR under two conditions:
  A) no context          -> baseline re-ASR
  B) context = the call's gold NAMES  -> upper bound of biasing (if we knew the names)

Then transliterate() -> Roman Urdu, score vs benchmark with diff_words. If (B) beats
(A), biasing recovers in-gazetteer names and the 8kHz audio has the detail; if not, the
ceiling is the audio itself.

Scoped to a subset of calls (CPU is slow ~ tens of s/chunk).

  python testing/audio_biasing_benchmark.py --calls 4
  python testing/audio_biasing_benchmark.py --calls 4 --limit-chunks 3
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))

import openpyxl
import torch
from test_accuracy import diff_words

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
XLSX = HERE / "lab_test_80_calls_urdu_roman_urdu.xlsx"
AUDIO = HERE / "lab_test_80_audios_chunks_25s"
CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")   # crude proper-noun grabber from gold


def load_rows():
    wb = openpyxl.load_workbook(str(XLSX)); ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True)); idx = {h: i for i, h in enumerate(rows[0])}
    from collections import defaultdict
    calls = defaultdict(list)
    for r in rows[1:]:
        calls[r[idx["call_id"]]].append(r)
    for cid in calls:
        calls[cid].sort(key=lambda x: x[idx["chunk_index"]] or 0)
    return calls, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--limit-chunks", type=int, default=0, help="0 = all chunks per call")
    ap.add_argument("--language", default="Hindi")
    args = ap.parse_args()

    os.environ["LEXICON"] = "v2"; os.environ["RESOLVER"] = "0"; os.environ["PHONETIC"] = ""
    import hindi_to_roman_urdu as H
    importlib.reload(H)

    calls, idx = load_rows()
    # pick calls that HAVE names in the gold and whose audio exists
    picked = []
    for cid, cr in calls.items():
        bench = next((x[idx["benchmark_roman_urdu"]] for x in cr
                      if isinstance(x[idx["benchmark_roman_urdu"]], str)
                      and x[idx["benchmark_roman_urdu"]].strip()), None)
        if not bench:
            continue
        names = sorted(set(CAP.findall(bench)))
        d = AUDIO / str(cid)
        if names and d.exists():
            picked.append((cid, cr, bench, names))
    picked.sort(key=lambda t: -len(t[3]))       # most names first
    picked = picked[: args.calls]
    print(f"selected {len(picked)} calls (most gold names first)")

    print(f"loading {MODEL_ID} on CPU ...", flush=True)
    t0 = time.time()
    model = __import__("qwen_asr").Qwen3ASRModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cpu", max_new_tokens=256)
    print(f"loaded in {time.time()-t0:.0f}s\n", flush=True)

    def transcribe(path, ctx):
        out = model.transcribe(audio=[str(path)], context=[ctx], language=[args.language])
        return out[0].text if out else ""

    base_m = base_t = bias_m = bias_t = 0
    for cid, cr, bench, names in picked:
        ctx = ", ".join(names)
        chunks = sorted((AUDIO / str(cid)).glob("chunk_*.wav"))
        if args.limit_chunks:
            chunks = chunks[: args.limit_chunks]
        hi_base, hi_bias = [], []
        for ch in chunks:
            t0 = time.time()
            hi_base.append(transcribe(ch, ""))
            hi_bias.append(transcribe(ch, ctx))
            print(f"  {cid[-8:]} {ch.name}: {time.time()-t0:.0f}s", flush=True)
        rb = H.transliterate(" ".join(hi_base))
        rx = H.transliterate(" ".join(hi_bias))
        db = diff_words(bench, rb); dx = diff_words(bench, rx)
        base_m += db.matched; base_t += db.total
        bias_m += dx.matched; bias_t += dx.total
        print(f"\n  call {cid[-8:]}  names biased: {ctx}")
        print(f"    baseline: {db.accuracy:5.1f}%   {rb[:90]}")
        print(f"    biased  : {dx.accuracy:5.1f}%   {rx[:90]}")
        print(f"    gold    :         {bench[:90]}\n")

    print("=" * 70)
    print("  AUDIO CONTEXTUAL BIASING — result")
    print("=" * 70)
    print(f"  re-ASR, NO context : {100*base_m/base_t:.2f}%")
    print(f"  re-ASR, + names    : {100*bias_m/bias_t:.2f}%   ({100*bias_m/bias_t-100*base_m/base_t:+.2f})")
    print()


if __name__ == "__main__":
    main()
