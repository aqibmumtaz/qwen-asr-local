"""
Abstain-threshold sweep for the Phonetic Contrastive Model.

The model generalises well (high held-out recall) but at a low threshold it fires
on real words and corrupts them. This finds the operating point by sweeping the
threshold and reporting, at each:

  RECALL   — held-out unseen variants -> correct canonical (want high)
  SAFETY   — real gold words (NOT canonicals) left UNCHANGED (want high)
  dw       — 80-call diff_words standalone on top of v2 (the production metric)

Everything is scored per-token ONCE and the threshold applied cheaply, so the
whole curve is fast.

  python -m phonetic_contrastive_model.sweep
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark"))

import openpyxl  # noqa: E402
from phonetic_contrastive_model.corrector import PhoneticContrastiveCorrector  # noqa: E402
from phonetic_contrastive_model.data import make_splits  # noqa: E402
from test_accuracy import diff_words  # noqa: E402

XLSX = ROOT / "benchmark" / "lab_test_80_calls_urdu_roman_urdu.xlsx"
TOKEN = re.compile(r"[A-Za-z]+")
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]


def score_tokens(corr, tokens):
    """token -> (best_canon, best_score). Batched, one pass."""
    uniq = sorted(set(tokens))
    out = {}
    B = 1024
    for i in range(0, len(uniq), B):
        chunk = uniq[i:i + B]
        emb = corr._encode(chunk)                       # (b, d)
        sims = emb @ corr.index.t()                     # (b, N)
        vals, idx = sims.max(dim=1)
        for w, v, j in zip(chunk, vals.tolist(), idx.tolist()):
            out[w] = (corr.canonicals[j], float(v))
    return out


def decide(word, cache, known, thr, min_len=3):
    lw = word.lower()
    if not word.isalpha() or lw in known or len(lw) < min_len:
        return word
    canon, score = cache.get(lw, (word, -1.0))
    if score < thr:
        return word
    if word[0].isupper() and canon == canon.lower():
        return canon[0].upper() + canon[1:]
    return canon


def main():
    sp = make_splits()
    heldout = sp["heldout_pairs"]
    canon_set = {c.lower() for c in sp["canonicals"]}
    corr = PhoneticContrastiveCorrector.load()
    known = corr._known

    # gold words that are NOT canonicals -> the abstain-safety population
    wb = openpyxl.load_workbook(str(XLSX)); ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True)); idx = {h: i for i, h in enumerate(rows[0])}
    gold_words = set()
    for r in rows[1:]:
        ref = r[idx["benchmark_roman_urdu"]]
        if isinstance(ref, str):
            for w in ref.split():
                w = w.strip(".,?!;:").lower()
                if w.isalpha() and len(w) >= 3 and w not in canon_set:
                    gold_words.add(w)
    gold_words = sorted(gold_words)

    # 80-call base transliterations (v2, resolver off)
    os.environ["LEXICON"] = "v2"; os.environ["RESOLVER"] = "0"
    import hindi_to_roman_urdu as H; importlib.reload(H)
    calls = defaultdict(list)
    for r in rows[1:]:
        calls[r[idx["call_id"]]].append(r)
    for cid in calls:
        calls[cid].sort(key=lambda x: x[idx["chunk_index"]] or 0)
    call_bench, call_base = {}, {}
    for cid, cr in calls.items():
        bench = next((x[idx["benchmark_roman_urdu"]] for x in cr
                      if isinstance(x[idx["benchmark_roman_urdu"]], str)
                      and x[idx["benchmark_roman_urdu"]].strip()), None)
        if not bench:
            continue
        hindi = " ".join(str(x[idx["model_output_hindi"]]).strip() for x in cr
                         if isinstance(x[idx["model_output_hindi"]], str)
                         and x[idx["model_output_hindi"]].strip())
        call_bench[cid] = bench
        call_base[cid] = H.transliterate(hindi)

    # score every token we will ever need, once
    all_tokens = [v for v, _ in heldout] + gold_words
    for base in call_base.values():
        all_tokens += TOKEN.findall(base)
    cache = score_tokens(corr, all_tokens)

    cidx = {c: i for i, c in enumerate(corr.canonicals)}
    base_m = base_t = 0
    for cid, base in call_base.items():
        d = diff_words(call_bench[cid], base); base_m += d.matched; base_t += d.total
    base_dw = 100 * base_m / base_t

    print("=" * 70)
    print("  ABSTAIN-THRESHOLD SWEEP  (v2 diff_words baseline = %.2f%%)" % base_dw)
    print("=" * 70)
    print(f"  {'thr':>5} {'recall':>8} {'safety':>8} {'dw':>8} {'dw vs v2':>9}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
    for thr in THRESHOLDS:
        rec = sum(1 for v, c in heldout
                  if decide(v, cache, known, thr).lower() == c.lower()) / len(heldout)
        safe = sum(1 for w in gold_words
                   if decide(w, cache, known, thr).lower() == w) / len(gold_words)
        cm = ct = 0
        for cid, base in call_base.items():
            fixed = TOKEN.sub(lambda m: decide(m.group(0), cache, known, thr), base)
            d = diff_words(call_bench[cid], fixed); cm += d.matched; ct += d.total
        dw = 100 * cm / ct
        print(f"  {thr:>5.2f} {100*rec:>7.1f}% {100*safe:>7.1f}% "
              f"{dw:>7.2f}% {dw-base_dw:>+8.2f}")
    print()


if __name__ == "__main__":
    main()
