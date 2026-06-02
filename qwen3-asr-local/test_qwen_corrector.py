#!/usr/bin/env python3
"""
Test Qwen corrector on turnwise_results_eval_full.xlsx.

Reads roman_urdu_model + word_scores, calls Corrector.fix(), compares
against roman_urdu_reference. All prompt/model/glossary logic lives in corrector.py.

Usage:
    python3 test_qwen_corrector.py            # first 5 rows
    python3 test_qwen_corrector.py --rows 10
    python3 test_qwen_corrector.py --rows all
"""

import argparse
import json
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrector import Corrector, WordEntry

XLSX     = Path("data/CLL analysis/turnwise_results_eval_full.xlsx")
CONF_THR = 0.65
GEO_THR  = 0.90


def wer_accuracy(hyp: str, ref: str) -> float:
    h, r = hyp.lower().split(), ref.lower().split()
    if not r:
        return 1.0
    dp = list(range(len(h) + 1))
    for rw in r:
        ndp = [dp[0] + 1] + [0] * len(h)
        for j, hw in enumerate(h):
            ndp[j+1] = min(dp[j] + (0 if hw == rw else 1), dp[j+1]+1, ndp[j]+1)
        dp = ndp
    return max(0.0, 1.0 - dp[len(h)] / len(r))


def parse_words(roman_urdu: str, word_scores_json: str) -> list[WordEntry]:
    """Build WordEntry list from roman_urdu_model text + word_scores JSON."""
    try:
        scores = json.loads(word_scores_json)
    except Exception:
        return []
    words = roman_urdu.split()
    return [
        WordEntry(
            roman=words[i].rstrip(".,;:") if i < len(words) else s["roman"],
            hindi=s["hindi"],
            min_conf=s["min_conf"],
            geo_conf=s["geo_conf"],
        )
        for i, s in enumerate(scores)
        if i < len(words)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="5",
                        help="Number of rows to test (default 5, 'all' for all)")
    args = parser.parse_args()
    max_rows = 0 if args.rows == "all" else int(args.rows)

    # Load xlsx
    wb   = openpyxl.load_workbook(str(XLSX), data_only=True)
    ws   = wb["asr_results"]
    rows = list(ws.iter_rows(values_only=True))
    hdr  = rows[0]
    idx  = {h: i for i, h in enumerate(hdr)}
    data = rows[1:] if max_rows == 0 else rows[1:max_rows+1]

    # Load corrector once — model stays in memory for all turns
    corrector = Corrector(backend="qwen")

    total_before = total_after = 0
    improved = same = degraded = 0

    for i, row in enumerate(data, 1):
        roman_model = row[idx["roman_urdu_model"]]
        word_scores = row[idx["word_scores"]]
        reference   = row[idx["roman_urdu_reference"]]
        speaker     = row[idx["speaker"]]
        turn        = row[idx["turn"]]

        if not isinstance(roman_model, str) or not isinstance(reference, str):
            continue

        words      = parse_words(roman_model, word_scores)
        n_flagged  = sum(1 for w in words
                        if w.min_conf < CONF_THR or w.geo_conf < GEO_THR)
        corrected  = corrector.fix(words) if words else roman_model

        acc_before = wer_accuracy(roman_model, reference)
        acc_after  = wer_accuracy(corrected,   reference)
        total_before += acc_before
        total_after  += acc_after

        delta  = acc_after - acc_before
        status = "↑ improved" if delta > 0.01 else ("↓ degraded" if delta < -0.01 else "= same")
        if delta > 0.01:    improved += 1
        elif delta < -0.01: degraded += 1
        else:               same += 1

        print(f"Turn {i:>2} ({speaker}, t{turn})  "
              f"flagged={n_flagged}/{len(words)}  "
              f"before={acc_before:.2f}  after={acc_after:.2f}  {status}")
        print(f"  ref:       {reference}")
        print(f"  model:     {roman_model}")
        print(f"  corrected: {corrected}")
        print()

    n = i
    print("─" * 70)
    print(f"Rows: {n}  |  avg WER before: {total_before/n:.3f}  "
          f"after: {total_after/n:.3f}  delta: {total_after/n - total_before/n:+.3f}")
    print(f"Improved: {improved}  Same: {same}  Degraded: {degraded}")


if __name__ == "__main__":
    main()
