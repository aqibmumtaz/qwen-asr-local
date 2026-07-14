#!/usr/bin/env python3
"""
SAFETY + BENEFIT test for the resolver.

SAFETY is the thing that matters. A fuzzy matcher that rewrites a CORRECT word is
worse than no matcher. We measure it two ways:

  A. NON-CIRCULAR safety — build the resolver with NO gold knowledge at all
     (gold_vocab=None), then run it over every gold reference word. Every change
     it makes is a CORRUPTION, because gold words are correct by definition.
     This is the honest number.

  B. BENEFIT — take the ASR's real Roman output (roman_urdu_model) and count how
     many words the resolver newly fixes that the exact lexicon could not.

Also sweeps MIN_FUZZY_LEN to find the safest useful threshold.

Usage:
    python3 test_resolver.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import openpyxl
from resolver import Resolver, load_gold_vocab, normalise

XLSX = Path(__file__).resolve().parent / "data" / "CLL analysis" / "turnwise_results_eval_full.xlsx"


def rows():
    wb = openpyxl.load_workbook(str(XLSX), data_only=True)
    rs = list(wb["asr_results"].iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rs[0])}
    return rs[1:], idx


def main():
    rs, idx = rows()
    gold_words, model_words = [], []
    for r in rs:
        g = r[idx["roman_urdu_reference"]]
        m = r[idx["roman_urdu_model"]]
        if isinstance(g, str):
            gold_words += [w.strip(".,?!;:") for w in g.split()]
        if isinstance(m, str):
            model_words += [w.strip(".,?!;:") for w in m.split()]

    print("=" * 78)
    print("  RESOLVER — SAFETY & BENEFIT")
    print("=" * 78)
    print(f"  gold words: {len(gold_words)}   ASR words: {len(model_words)}")
    print()

    # ── A. NON-CIRCULAR SAFETY ───────────────────────────────────────────────
    print("  A. SAFETY (non-circular — resolver built with ZERO gold knowledge)")
    print("     every gold word it changes is a CORRUPTION\n")
    print(f"     {'min_len':>8} {'corrupted':>10} {'rate':>8}   {'fixes on ASR':>13}  {'net':>6}")
    print(f"     {'-'*8} {'-'*10} {'-'*8}   {'-'*13}  {'-'*6}")

    best = None
    for ml in (4, 5, 6, 7, 8):
        r_safe = Resolver(gold_vocab=None, min_len=ml)     # NO gold knowledge
        bad = Counter()
        for w in gold_words:
            lw = w.lower()
            if lw in r_safe.exact:
                continue          # the EXACT lexicon did this, not the resolver
            out = r_safe.resolve_word(w)
            if out.lower() != lw:
                bad[f"{w} -> {out}"] += 1
        n_bad = sum(bad.values())

        # benefit: how many ASR words does the FUZZY path newly fix?
        r_ben = Resolver(gold_vocab=None, min_len=ml)
        fixes = 0
        for w in model_words:
            if w.lower() in r_ben.exact:
                continue                       # the exact lexicon already had it
            if r_ben.resolve_word(w).lower() != w.lower():
                fixes += 1
        net = fixes - n_bad
        mark = ""
        if best is None or net > best[1]:
            best = (ml, net, bad)
            mark = "  <-- best net"
        print(f"     {ml:>8} {n_bad:>10} {100*n_bad/len(gold_words):>7.2f}% "
              f"{fixes:>13}  {net:>+6}{mark}")

    ml, net, bad = best
    print()
    print(f"  => MIN_FUZZY_LEN={ml} gives the best net gain ({net:+d} words)")
    if bad:
        print(f"\n     the corruptions it still causes ({sum(bad.values())}):")
        for k, n in bad.most_common(12):
            print(f"       {n:>3}x  {k}")
    print()

    # ── B. WHAT IT ACTUALLY FIXES ────────────────────────────────────────────
    print("  B. NEW FIXES — words the exact lexicon could NOT resolve")
    print()
    r = Resolver(gold_vocab=load_gold_vocab(), min_len=ml)
    new = Counter()
    for w in model_words:
        if w.lower() in r.exact:
            continue
        out = r.resolve_word(w)
        if out.lower() != w.lower():
            new[f"{w} -> {out}"] += 1
    print(f"     {sum(new.values())} words fixed that were NOT in the variant list:")
    for k, n in new.most_common(20):
        print(f"       {n:>3}x  {k}")
    print()
    print(f"     resolver stats: {r.report()}")
    print()


if __name__ == "__main__":
    main()
