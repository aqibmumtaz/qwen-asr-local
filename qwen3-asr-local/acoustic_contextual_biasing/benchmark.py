"""
Thorough benchmark of acoustic contextual biasing on the 80-call 8kHz set.

Per chunk, up to three conditions (each = one Qwen3-ASR decode):
  A  baseline      no context
  B  two-pass      retrieve names from A's hypothesis, re-decode biased  (the real system)
  C  oracle        context = the call's GOLD names                       (upper bound)

Then concatenate per call, transliterate() -> Roman Urdu, and score vs benchmark with
diff_words. Also reports NAME RECOVERY: of the gold names in a call, how many the output
actually contains (fuzzy), per condition — the number that really matters.

CPU is slow (~1-6 min/decode); scope with --calls / --limit-chunks. --device mps is faster.

  python -m acoustic_contextual_biasing.benchmark --calls 6 --device cpu
  python -m acoustic_contextual_biasing.benchmark --calls 6 --limit-chunks 3 --no-oracle
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark"))

import openpyxl
from test_accuracy import diff_words

from .asr import BiasedASR
from .retriever import NameRetriever

XLSX = ROOT / "benchmark" / "lab_test_80_calls_urdu_roman_urdu.xlsx"
AUDIO = ROOT / "benchmark" / "lab_test_80_audios_chunks_25s"
CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")


def fuzzy_in(name: str, text_words: list[str], thr: float = 0.75) -> bool:
    n = name.lower()
    return any(SequenceMatcher(a=n, b=w, autojunk=False).ratio() >= thr for w in text_words)


def load_calls():
    wb = openpyxl.load_workbook(str(XLSX)); ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True)); idx = {h: i for i, h in enumerate(rows[0])}
    calls = defaultdict(list)
    for r in rows[1:]:
        calls[r[idx["call_id"]]].append(r)
    for cid in calls:
        calls[cid].sort(key=lambda x: x[idx["chunk_index"]] or 0)
    return calls, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--k", type=int, default=15, help="retrieved names to bias with")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--language", default="Hindi")
    args = ap.parse_args()

    os.environ["LEXICON"] = "v2"; os.environ["RESOLVER"] = "0"; os.environ["PHONETIC"] = ""
    import hindi_to_roman_urdu as H
    importlib.reload(H)

    calls, idx = load_calls()
    picked = []
    for cid, cr in calls.items():
        bench = next((x[idx["benchmark_roman_urdu"]] for x in cr
                      if isinstance(x[idx["benchmark_roman_urdu"]], str)
                      and x[idx["benchmark_roman_urdu"]].strip()), None)
        if not bench:
            continue
        names = sorted(set(CAP.findall(bench)))
        if names and (AUDIO / str(cid)).exists():
            picked.append((cid, bench, names))
    picked.sort(key=lambda t: -len(t[2]))
    picked = picked[: args.calls]
    print(f"selected {len(picked)} name-heavy calls\n", flush=True)

    asr = BiasedASR(device=args.device)
    retr = NameRetriever(device=args.device)

    agg = {c: [0, 0] for c in ("A", "B", "C")}          # matched, total
    name_hit = {c: [0, 0] for c in ("A", "B", "C")}     # names found, names total

    for cid, bench, gnames in picked:
        chunks = sorted((AUDIO / str(cid)).glob("chunk_*.wav"))
        if args.limit_chunks:
            chunks = chunks[: args.limit_chunks]
        outA, outB, outC = [], [], []
        for ch in chunks:
            t0 = time.time()
            p1 = asr.transcribe(ch, context="", language=args.language)          # A
            names = retr.retrieve(p1, k=args.k)
            p2 = asr.transcribe(ch, context=", ".join(names), language=args.language)  # B
            outA.append(p1); outB.append(p2)
            if not args.no_oracle:
                outC.append(asr.transcribe(ch, context=", ".join(gnames), language=args.language))
            print(f"  {cid[-8:]} {ch.name}: {time.time()-t0:.0f}s  retrieved={names[:6]}", flush=True)

        for tag, out in (("A", outA), ("B", outB), ("C", outC)):
            if not out:
                continue
            roman = H.transliterate(" ".join(out))
            d = diff_words(bench, roman)
            agg[tag][0] += d.matched; agg[tag][1] += d.total
            words = roman.lower().split()
            name_hit[tag][0] += sum(1 for n in gnames if fuzzy_in(n, words))
            name_hit[tag][1] += len(gnames)
        print(f"  call {cid[-8:]}  gold names: {gnames}\n", flush=True)

    labels = {"A": "baseline (no context)", "B": f"two-pass (retrieved k={args.k})",
              "C": "oracle (gold names)"}
    print("=" * 68)
    print("  ACOUSTIC CONTEXTUAL BIASING — 8kHz set")
    print("=" * 68)
    print(f"  {'config':<30}{'diff_words':>11}{'name recovery':>15}")
    print("  " + "-" * 56)
    for c in ("A", "B", "C"):
        if agg[c][1] == 0:
            continue
        dw = 100 * agg[c][0] / agg[c][1]
        nr = 100 * name_hit[c][0] / max(name_hit[c][1], 1)
        print(f"  {labels[c]:<30}{dw:>10.2f}%{nr:>13.1f}%  ({name_hit[c][0]}/{name_hit[c][1]})")
    print()


if __name__ == "__main__":
    main()
