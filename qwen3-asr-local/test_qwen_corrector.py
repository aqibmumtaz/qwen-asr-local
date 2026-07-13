#!/usr/bin/env python3
"""
Test Qwen corrector on turnwise_results_eval_full.xlsx.

Default: runs on Qwen3-0.6B only.
Use --all-models to compare 0.6B / 1.7B / 3B / 4B side-by-side.
Use --models to specify a custom list.

Usage:
    python3 test_qwen_corrector.py                        # 0.6B, first 5 rows
    python3 test_qwen_corrector.py --rows 1               # 0.6B, first row only
    python3 test_qwen_corrector.py --rows all             # 0.6B, all 183 rows
    python3 test_qwen_corrector.py --all-models --rows 5  # all 4 models, 5 rows
    python3 test_qwen_corrector.py --models Qwen/Qwen3-1.7B Qwen/Qwen3-4B --rows 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrector import Corrector, WordEntry

XLSX     = Path("data/CLL analysis/turnwise_results_eval_full.xlsx")
CONF_THR = 0.65
GEO_THR  = 0.90

# Available models in order of size
ALL_MODELS = [
    "Qwen/Qwen3-0.6B",   # ~1.2GB  — smallest, needs examples for proper nouns
    "Qwen/Qwen3-1.7B",   # ~3.5GB  — good balance of speed and quality
    "Qwen/Qwen3-4B",     # ~8GB    — reliable, handles proper nouns without examples
]
# Note: Qwen3 sizes are 0.6B, 1.7B, 4B, 8B, 14B, 32B — no 2B or 3B exists
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


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


def parse_words(roman_urdu: str, word_scores_json: str) -> list:
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


def load_data(max_rows: int):
    wb   = openpyxl.load_workbook(str(XLSX), data_only=True)
    ws   = wb["asr_results"]
    rows = list(ws.iter_rows(values_only=True))
    hdr  = rows[0]
    idx  = {h: i for i, h in enumerate(hdr)}
    data = rows[1:] if max_rows == 0 else rows[1:max_rows+1]
    return data, idx


def run_model(model_id: str, data, idx, use_guardrail: bool = True,
              verbose: bool = True):
    """Run corrector for one model across all rows. Returns list of result dicts."""
    short     = model_id.split("/")[-1]
    corrector = Corrector(backend="qwen", model_path=model_id,
                          use_guardrail=use_guardrail)
    results = []
    for row in data:
        roman_model = row[idx["roman_urdu_model"]]
        word_scores = row[idx["word_scores"]]
        reference   = row[idx["roman_urdu_reference"]]
        speaker     = row[idx["speaker"]]
        turn        = row[idx["turn"]]

        if not isinstance(roman_model, str) or not isinstance(reference, str):
            continue

        words     = parse_words(roman_model, word_scores)
        n_flagged = sum(1 for w in words
                        if w.min_conf < CONF_THR or w.geo_conf < GEO_THR)

        if verbose:
            print(f"  [{short}] Turn {turn} ({speaker}) — running...", flush=True)

        t0        = time.time()
        corrected = corrector.fix(words) if words else roman_model
        elapsed   = time.time() - t0

        acc_before = wer_accuracy(roman_model, reference)
        acc_after  = wer_accuracy(corrected,   reference)
        delta      = acc_after - acc_before
        status     = "✓" if acc_after >= 0.99 else ("↑" if delta > 0.01 else ("↓" if delta < -0.01 else "="))

        if verbose:
            print(f"  [{short}] Turn {turn} done — "
                  f"before={acc_before:.2f} after={acc_after:.2f} {status} "
                  f"({elapsed:.1f}s)", flush=True)

        results.append({
            "speaker":     speaker,
            "turn":        turn,
            "roman_model": roman_model,
            "reference":   reference,
            "corrected":   corrected,
            "n_flagged":   n_flagged,
            "n_words":     len(words),
            "elapsed_s":   elapsed,
            "acc_before":  wer_accuracy(roman_model, reference),
            "acc_after":   wer_accuracy(corrected,   reference),
        })
    return results


def print_single_model(model_id: str, results: list):
    """Print per-turn detail for a single model."""
    short = model_id.split("/")[-1]
    print(f"\n{'═'*70}")
    print(f"  Model: {model_id}")
    print(f"{'═'*70}\n")

    improved = same = degraded = 0
    total_before = total_after = 0

    for r in results:
        delta  = r["acc_after"] - r["acc_before"]
        status = "↑ improved" if delta > 0.01 else ("↓ degraded" if delta < -0.01 else "= same")
        if delta > 0.01:    improved += 1
        elif delta < -0.01: degraded += 1
        else:               same += 1
        total_before += r["acc_before"]
        total_after  += r["acc_after"]

        print(f"Turn {r['turn']:>2} ({r['speaker']})  "
              f"flagged={r['n_flagged']}/{r['n_words']}  "
              f"before={r['acc_before']:.2f}  after={r['acc_after']:.2f}  "
              f"{r['elapsed_s']:.1f}s  {status}")
        print(f"  ref:       {r['reference']}")
        print(f"  model:     {r['roman_model']}")
        print(f"  corrected: {r['corrected']}")
        print()

    n = len(results)
    avg_time = sum(r["elapsed_s"] for r in results) / n
    print(f"{'─'*70}")
    print(f"  [{short}]  Rows: {n}  |  "
          f"avg before: {total_before/n:.3f}  after: {total_after/n:.3f}  "
          f"delta: {total_after/n - total_before/n:+.3f}  "
          f"avg time: {avg_time:.1f}s/turn")
    print(f"  Improved: {improved}  Same: {same}  Degraded: {degraded}")


def print_comparative(model_ids: list, all_results: dict, data, idx):
    """Print side-by-side comparison across models."""
    rows_list = []
    for row in data:
        if isinstance(row[idx["roman_urdu_model"]], str):
            rows_list.append(row)

    print(f"\n{'═'*80}")
    print("  COMPARATIVE RESULTS")
    print(f"{'═'*80}\n")

    # Per-turn comparison
    for i, row in enumerate(rows_list):
        ref      = row[idx["roman_urdu_reference"]]
        model_in = row[idx["roman_urdu_model"]]
        turn     = row[idx["turn"]]
        speaker  = row[idx["speaker"]]

        acc_before = wer_accuracy(model_in, ref)
        w = 90
        print(f"{'─'*w}")
        print(f"  Turn {turn} ({speaker})  "
              f"flagged={all_results[model_ids[0]][i]['n_flagged']}/{all_results[model_ids[0]][i]['n_words']}  "
              f"WER before={acc_before:.2f}")
        print(f"{'─'*w}")
        label_w = 18
        print(f"  {'INPUT':<{label_w}} {model_in}")
        print(f"  {'REFERENCE':<{label_w}} {ref}")
        print()
        for mid in model_ids:
            short = mid.split("/")[-1]
            if i < len(all_results[mid]):
                r      = all_results[mid][i]
                delta  = r["acc_after"] - acc_before
                marker = "✓" if r["acc_after"] >= 0.99 else ("↑" if delta > 0.01 else ("↓" if delta < -0.01 else "="))
                label  = f"{short} ({r['acc_after']:.2f}{marker} {r['elapsed_s']:.1f}s)"
                print(f"  {label:<{label_w}} {r['corrected']}")
        print()

    # Summary table
    print(f"{'─'*90}")
    print(f"  {'Model':<22} {'Avg Before':>11} {'Avg After':>10} {'Delta':>7} "
          f"{'Avg Time':>9} {'Improved':>9} {'Same':>6} {'Degraded':>9}")
    print(f"  {'-'*22} {'-'*11} {'-'*10} {'-'*7} {'-'*9} {'-'*9} {'-'*6} {'-'*9}")

    for mid in model_ids:
        res = all_results[mid]
        if not res:
            continue
        n        = len(res)
        avg_before = sum(r["acc_before"]  for r in res) / n
        avg_after  = sum(r["acc_after"]   for r in res) / n
        avg_time   = sum(r["elapsed_s"]   for r in res) / n
        improved   = sum(1 for r in res if r["acc_after"] - r["acc_before"] > 0.01)
        same       = sum(1 for r in res if abs(r["acc_after"] - r["acc_before"]) <= 0.01)
        degraded   = sum(1 for r in res if r["acc_after"] - r["acc_before"] < -0.01)
        short      = mid.split("/")[-1]
        print(f"  {short:<22} {avg_before:>11.3f} {avg_after:>10.3f} "
              f"{avg_after - avg_before:>+7.3f} {avg_time:>8.1f}s "
              f"{improved:>9} {same:>6} {degraded:>9}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen corrector on xlsx eval set"
    )
    parser.add_argument(
        "--rows", default="5",
        help="Rows to test: number or 'all' (default: 5)"
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help=f"Test all models: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Custom list of HuggingFace model IDs to test"
    )
    parser.add_argument(
        "--no-guardrail", action="store_true",
        help="Disable the high-conf word reinsertion guardrail"
    )
    args = parser.parse_args()

    max_rows   = 0 if args.rows == "all" else int(args.rows)
    guardrail  = not args.no_guardrail

    # Determine which models to run
    if args.models:
        model_ids = args.models
    elif args.all_models:
        model_ids = ALL_MODELS
    else:
        model_ids = [DEFAULT_MODEL]

    data, idx = load_data(max_rows)

    # Single model — show per-turn detail
    if len(model_ids) == 1:
        results = run_model(model_ids[0], data, idx, use_guardrail=guardrail)
        print_single_model(model_ids[0], results)

    # Multiple models — show comparative table
    else:
        all_results = {}
        for mid in model_ids:
            print(f"\n{'─'*60}")
            print(f"  Loading {mid} ...", flush=True)
            print(f"{'─'*60}")
            all_results[mid] = run_model(mid, data, idx, use_guardrail=guardrail,
                                         verbose=True)
        print_comparative(model_ids, all_results, data, idx)


if __name__ == "__main__":
    main()
