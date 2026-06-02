#!/usr/bin/env python3
"""
Test Corrector on turnwise_results_eval_full.xlsx.

For each turn:
  - Input:     word_scores (roman + hindi + min_conf + geo_conf per word)
  - Corrector: Qwen LLM fixes low-conf words, leaves high-conf unchanged
  - Ground truth: roman_urdu_reference
  - Shows: before WER, after WER, what was fixed

Usage:
    python3 test_corrector.py                    # first 10 rows
    python3 test_corrector.py --rows 30          # first 30 rows
    python3 test_corrector.py --rows all         # all 183 rows
    python3 test_corrector.py --backend mt5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corrector import Corrector, WordEntry

XLSX = Path(__file__).resolve().parent / "data/CLL analysis/turnwise_results_eval_full.xlsx"


# ── WER accuracy (same formula as roman_urdu_accuracy_row) ────────────────────
def wer_accuracy(hypothesis: str, reference: str) -> float:
    """1 - edit_distance(hyp, ref) / len(ref_words)"""
    h = hypothesis.lower().split()
    r = reference.lower().split()
    if not r:
        return 1.0
    # Levenshtein distance on word sequences
    dp = list(range(len(h) + 1))
    for rw in r:
        ndp = [dp[0] + 1] + [0] * len(h)
        for j, hw in enumerate(h):
            ndp[j+1] = min(dp[j] + (0 if hw == rw else 1),
                           dp[j+1] + 1,
                           ndp[j] + 1)
        dp = ndp
    edits = dp[len(h)]
    return max(0.0, 1.0 - edits / len(r))


# ── Load xlsx rows ────────────────────────────────────────────────────────────
def load_rows(xlsx_path: Path, max_rows: int = 10):
    import openpyxl
    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    ws = wb["asr_results"]
    rows = list(ws.iter_rows(values_only=True))
    hdr  = rows[0]
    idx  = {h: i for i, h in enumerate(hdr)}
    data = rows[1:]
    if max_rows > 0:
        data = data[:max_rows]
    return data, idx


# ── Parse word_scores JSON → WordEntry list ───────────────────────────────────
def parse_words(word_scores_json: str) -> list[WordEntry]:
    if not isinstance(word_scores_json, str):
        return []
    try:
        arr = json.loads(word_scores_json)
    except Exception:
        return []
    return [
        WordEntry(
            roman=w["roman"].rstrip(".,;:"),
            hindi=w["hindi"],
            min_conf=w["min_conf"],
            geo_conf=w["geo_conf"],
        )
        for w in arr
    ]


# ── Main test loop ────────────────────────────────────────────────────────────
def run_test(backend: str, model_path: str | None, max_rows: int):
    print(f"\n{'═'*72}")
    print(f"  Corrector test — backend={backend}  rows={max_rows}")
    print(f"  xlsx: {XLSX.name}")
    print(f"{'═'*72}\n")

    corrector = Corrector(backend=backend, model_path=model_path)

    data, idx = load_rows(XLSX, max_rows)

    total_before = total_after = 0
    improved = unchanged = degraded = 0

    for row_i, row in enumerate(data, start=1):
        ref     = row[idx["roman_urdu_reference"]]
        model   = row[idx["roman_urdu_model"]]
        ws_json = row[idx["word_scores"]]
        turn    = row[idx["turn"]]
        speaker = row[idx["speaker"]]

        if not isinstance(ref, str) or not ref.strip():
            continue
        if not isinstance(ws_json, str):
            continue

        words = parse_words(ws_json)
        if not words:
            continue

        n_flagged = sum(1 for w in words if w.needs_fix)

        # Run corrector
        corrected = corrector.fix(words)

        # Score
        acc_before = wer_accuracy(model or "", ref)
        acc_after  = wer_accuracy(corrected, ref)
        total_before += acc_before
        total_after  += acc_after

        delta = acc_after - acc_before
        if delta > 0.01:
            improved += 1
        elif delta < -0.01:
            degraded += 1
        else:
            unchanged += 1

        # Per-turn display
        status = ("↑ improved" if delta > 0.01
                  else "↓ degraded" if delta < -0.01
                  else "= same")
        print(f"Turn {row_i:>3} ({speaker}, turn {turn})  "
              f"flagged={n_flagged}/{len(words)}  "
              f"before={acc_before:.2f}  after={acc_after:.2f}  {status}")
        print(f"  ref:       {ref}")
        print(f"  model:     {model}")
        print(f"  corrected: {corrected}")

        # Show what changed
        orig_words  = [w.roman for w in words]
        fixed_words = corrected.split()
        changes = []
        # align by position (best effort for same-length output)
        if len(orig_words) == len(fixed_words):
            for o, f in zip(orig_words, fixed_words):
                if o != f:
                    changes.append(f"'{o}' → '{f}'")
        if changes:
            print(f"  fixes:     {',  '.join(changes)}")
        print()

    n = row_i
    avg_before = total_before / n
    avg_after  = total_after  / n
    print(f"{'─'*72}")
    print(f"  Rows tested : {n}")
    print(f"  Avg WER acc before correction : {avg_before:.3f} ({avg_before*100:.1f}%)")
    print(f"  Avg WER acc after  correction : {avg_after:.3f} ({avg_after*100:.1f}%)")
    print(f"  Delta       : {avg_after - avg_before:+.3f}")
    print(f"  Improved    : {improved}  |  Same: {unchanged}  |  Degraded: {degraded}")
    print(f"{'─'*72}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Corrector on xlsx eval set")
    parser.add_argument("--backend",  choices=["qwen", "mt5"], default="qwen")
    parser.add_argument("--model",    default=None, help="Model path override")
    parser.add_argument("--rows",     default="10",
                        help="Number of rows to test (default 10, 'all' for all)")
    args = parser.parse_args()

    max_rows = 0 if args.rows == "all" else int(args.rows)
    run_test(args.backend, args.model, max_rows)
