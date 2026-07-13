#!/usr/bin/env python3
"""
HELD-OUT VALIDATION — the honest, non-circular corruption measurement.

The problem: R7 protects words that appear in the gold references. If we then
measure corruption ON those same references, the result is self-fulfilling.

The fix: split the 183 gold turns in half.
  - build the protected vocabulary from  TRAIN turns only
  - measure corruption on the  HELD-OUT turns  (never seen by the cleaner)

This is the number to trust.

Usage:
    python3 validate_heldout.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import openpyxl
from clean_lexicon import load, clean, to_lookup, SRC, BASE, XLSX


def gold_turns():
    wb = openpyxl.load_workbook(str(XLSX), data_only=True)
    ws = wb["asr_results"]
    rows = list(ws.iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    return [
        r[idx["roman_urdu_reference"]]
        for r in rows[1:]
        if isinstance(r[idx["roman_urdu_reference"]], str)
    ]


def vocab_of(turns):
    v = set()
    for t in turns:
        for w in t.split():
            w = w.strip(".,?!;:").lower()
            if w:
                v.add(w)
    return v


def corrupt_on(turns, lookup):
    tot = ch = 0
    for t in turns:
        for w in t.split():
            tot += 1
            lw = w.lower()
            new = lookup.get(lw)
            if new and new.lower() != lw:
                ch += 1
    return ch, tot


def main():
    turns = gold_turns()
    # deterministic 50/50 split — even turns train, odd turns held out
    train = [t for i, t in enumerate(turns) if i % 2 == 0]
    held = [t for i, t in enumerate(turns) if i % 2 == 1]

    print("=" * 72)
    print("  HELD-OUT VALIDATION (non-circular)")
    print("=" * 72)
    print(f"  gold turns      : {len(turns)}")
    print(f"  train (build vocab from these) : {len(train)}")
    print(f"  HELD OUT (measure on these)    : {len(held)}   <- never seen by cleaner")
    print()

    src = load(SRC)
    base = load(BASE)
    train_vocab = vocab_of(train)
    print(f"  protected vocab built from train turns: {len(train_vocab)} words")
    print()

    # baselines on the HELD-OUT half
    base_lu = {**base["corrections"], **base["proper_nouns"]}
    upd_lu = {**src["corrections"], **src["proper_nouns"]}

    cleaned, _ = clean(src, train_vocab)   # <-- ONLY train vocab
    clean_lu, _ = to_lookup(cleaned)

    print("  CORRUPTION ON THE HELD-OUT TURNS")
    for name, lu in [
        ("original lexicons.json", base_lu),
        ("lexicons_updated.json", upd_lu),
        ("lexicons_clean.json", clean_lu),
    ]:
        ch, tot = corrupt_on(held, lu)
        print(f"    {name:<26} {ch:>4} / {tot}  ({100*ch/tot:.2f}%)")

    ch_c, tot = corrupt_on(held, clean_lu)
    ch_o, _ = corrupt_on(held, base_lu)
    print()
    print("=" * 72)
    if ch_c <= ch_o:
        print(f"  ✓ PASS — clean lexicon ({100*ch_c/tot:.2f}%) is at or below the")
        print(f"    original baseline ({100*ch_o/tot:.2f}%) on turns it never saw.")
    else:
        print(f"  ✗ clean ({100*ch_c/tot:.2f}%) is WORSE than original ({100*ch_o/tot:.2f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
