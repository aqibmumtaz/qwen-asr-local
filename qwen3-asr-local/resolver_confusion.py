#!/usr/bin/env python3
"""
CONFUSION MATRIX for the resolver — can it replace the enumerated variant list
with ZERO false matches?

TWO POPULATIONS. Testing only one of them is how you fool yourself.

  P1  MISSPELLINGS  — the 14,319 v2 variants. Each SHOULD be rewritten to its
                      canonical. A "positive" here is a correct rewrite.

  P2  CORRECT WORDS — words a human actually wrote (the gold references) that
                      are NOT canonicals in the lexicon. Each MUST be left
                      untouched. `assessment` lives here. ANY rewrite is a
                      false match.

P2 is the population that decides the question. If you delete variants from the
exact map you must turn the resolver ON, and the moment it is on it is also
loose on P2 — where every fire is a corruption. That risk exists whether or not
you delete a single entry.

Usage:
    python3 resolver_confusion.py
    python3 resolver_confusion.py --sweep
"""

import argparse
import json
from pathlib import Path

import openpyxl

from resolver import Resolver
from resolver_coverage import fuzzy_only

SCRIPT_DIR = Path(__file__).resolve().parent
V2 = SCRIPT_DIR / "data" / "lexicons_v2.json"
XLSX = SCRIPT_DIR / "data" / "CLL analysis" / "turnwise_results_eval_full.xlsx"


def gold_words() -> set:
    wb = openpyxl.load_workbook(str(XLSX), data_only=True)
    rows = list(wb["asr_results"].iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    out = set()
    for r in rows[1:]:
        ref = r[idx["roman_urdu_reference"]]
        if isinstance(ref, str):
            for w in ref.split():
                w = w.strip(".,?!;:").lower()
                if w.isalpha():
                    out.add(w)
    return out


def evaluate(min_len: int):
    raw = json.loads(V2.read_text(encoding="utf-8"))["lexicons"]
    r = Resolver(min_len=min_len)

    # P1 — misspellings that SHOULD be rewritten
    pairs = [(v, c) for c, vs in raw["lexicon"].items() for v in vs]
    p1 = {"correct": 0, "wrong": 0, "unchanged": 0}
    wrong_ex = []
    for variant, canon in pairs:
        kind, out = fuzzy_only(r, variant)
        if kind in ("protected", "short", "no_match", "ambiguous"):
            p1["unchanged"] += 1
        elif out.lower() == canon.lower():
            p1["correct"] += 1
        else:
            p1["wrong"] += 1
            if len(wrong_ex) < 8:
                wrong_ex.append((variant, canon, out))

    # P2 — correct words that MUST NOT be touched.
    # Anything a human wrote that the lexicon does not already know as a canonical.
    canon_set = {c.lower() for c in raw["lexicon"]}
    p2_words = sorted(gold_words() - canon_set)
    p2 = {"unchanged": 0, "corrupted": 0}
    corrupt_ex = []
    for w in p2_words:
        kind, out = fuzzy_only(r, w)
        if kind in ("protected", "short", "no_match", "ambiguous") or out.lower() == w:
            p2["unchanged"] += 1
        else:
            p2["corrupted"] += 1
            if len(corrupt_ex) < 12:
                corrupt_ex.append((w, out))

    return r, len(pairs), p1, p2, len(p2_words), wrong_ex, corrupt_ex


def show(min_len: int):
    r, n1, p1, p2, n2, wrong_ex, corrupt_ex = evaluate(min_len)

    print("=" * 78)
    print(f"  CONFUSION MATRIX — resolver fuzzy path only (G3 min length = {min_len})")
    print("=" * 78)
    print()
    print(f"  {'':<34} {'-> correct':>12} {'-> WRONG word':>14} {'-> unchanged':>13}")
    print(f"  {'-'*34} {'-'*12} {'-'*14} {'-'*13}")
    print(f"  {'P1 misspellings (should fix)':<34} {p1['correct']:>12} "
          f"{p1['wrong']:>14} {p1['unchanged']:>13}")
    print(f"  {'   n = '+str(n1):<34} {'TRUE POS':>12} {'FALSE MATCH':>14} {'missed':>13}")
    print()
    print(f"  {'P2 correct words (must NOT fix)':<34} {'—':>12} "
          f"{p2['corrupted']:>14} {p2['unchanged']:>13}")
    print(f"  {'   n = '+str(n2):<34} {'':>12} {'FALSE MATCH':>14} {'TRUE NEG':>13}")
    print()

    fm = p1["wrong"] + p2["corrupted"]
    print("  " + "-" * 74)
    print(f"  TOTAL FALSE MATCHES : {fm}"
          f"    ({p1['wrong']} in P1  +  {p2['corrupted']} in P2)")
    print(f"  redundant entries   : {p1['correct']}  "
          f"({100*p1['correct']/n1:.1f}% of the variant list)")
    print(f"  ZERO false matches? : {'YES' if fm == 0 else 'NO'}")
    print()

    if corrupt_ex:
        print("  P2 CORRUPTIONS — correct words the resolver would destroy:")
        for w, o in corrupt_ex:
            print(f"    {w:<18} -> {o}")
        print()
    if wrong_ex:
        print("  P1 FALSE MATCHES — variant sent to the wrong canonical:")
        for v, c, o in wrong_ex:
            print(f"    {v:<18} should be {c:<16} got {o}")
        print()


def sweep():
    print("=" * 78)
    print("  SWEEP — is there ANY threshold with zero false matches?")
    print("=" * 78)
    print()
    print(f"  {'min_len':>7} {'redundant':>10} {'%list':>7} "
          f"{'FM in P1':>9} {'FM in P2':>9} {'TOTAL FM':>9}  {'zero?':>6}")
    print(f"  {'-'*7} {'-'*10} {'-'*7} {'-'*9} {'-'*9} {'-'*9}  {'-'*6}")
    for L in (4, 5, 6, 7, 8, 9, 10, 12):
        _, n1, p1, p2, _, _, _ = evaluate(L)
        fm = p1["wrong"] + p2["corrupted"]
        print(f"  {L:>7} {p1['correct']:>10} {100*p1['correct']/n1:>6.1f}% "
              f"{p1['wrong']:>9} {p2['corrupted']:>9} {fm:>9}  "
              f"{'YES' if fm == 0 else 'no':>6}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    show(a.min_len)
    if a.sweep:
        sweep()
