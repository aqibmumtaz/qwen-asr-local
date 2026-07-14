#!/usr/bin/env python3
"""
THE REAL TEST for the resolver — does it handle spellings it has NEVER SEEN?

WHY THE PREVIOUS TEST UNDERSTATES IT
    lexicons_v2's 14,575 variants were derived FROM this same eval data. So on
    this data the exact lexicon already covers nearly everything, and the resolver
    looks pointless (+6 words). That is survivorship bias — it is grading the
    lexicon on its own training set.

    On a NEW call the ASR produces spellings that are not in the list, and the
    exact lexicon is helpless. That is the case the resolver exists for.

HOW WE SIMULATE A NEW CALL
    Hold out a random X% of the variants. Build the resolver with only the
    remaining ones. Then feed it the HELD-OUT variants — spellings it has
    literally never seen — and ask: does it still resolve them to the right word?

    exact-only    : cannot possibly resolve them (0% by construction)
    with resolver : how many does it recover?

Usage:
    python3 test_resolver_heldout.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolver import Resolver, normalise, load_gold_vocab

V2 = Path(__file__).resolve().parent / "data" / "lexicons_v2.json"


def build_holdout(frac: float, seed: int):
    """Split v2's variants into train / heldout. Returns (train_v2_dict, heldout pairs)."""
    raw = json.loads(V2.read_text(encoding="utf-8"))["lexicons"]
    rng = random.Random(seed)
    train = {"lexicon": {}, "phrases": raw["phrases"]}
    heldout = []
    for canon, variants in raw["lexicon"].items():
        kept = []
        for v in variants:
            if rng.random() < frac and len(v) >= 6:
                heldout.append((v, canon))     # pretend we have never seen it
            else:
                kept.append(v)
        train["lexicon"][canon] = kept
    return train, heldout


def main():
    print("=" * 78)
    print("  RESOLVER on UNSEEN SPELLINGS  (the case it actually exists for)")
    print("=" * 78)
    print()
    print(f"  {'held out':>9} {'unseen':>8} {'exact-only':>12} {'+ resolver':>12} {'recovered':>11}")
    print(f"  {'-'*9} {'-'*8} {'-'*12} {'-'*12} {'-'*11}")

    gold = load_gold_vocab()
    for frac in (0.10, 0.25, 0.50, 0.75):
        train, heldout = build_holdout(frac, seed=42)

        tmp = Path("/tmp/_v2_train.json")
        tmp.write_text(json.dumps({"lexicons": train}), encoding="utf-8")

        r = Resolver(v2_path=tmp, gold_vocab=gold, min_len=8)

        exact_hits = 0
        resolver_hits = 0
        for variant, canon in heldout:
            if variant.lower() in r.exact:      # should be 0 — we removed them
                exact_hits += 1
                continue
            if r.resolve_word(variant).lower() == canon.lower():
                resolver_hits += 1

        n = len(heldout)
        if not n:
            continue
        print(f"  {frac*100:>8.0f}% {n:>8} {exact_hits:>12} "
              f"{resolver_hits:>12} {100*resolver_hits/n:>10.1f}%")

    # concrete examples at 50%
    print()
    print("=" * 78)
    print("  EXAMPLES — spellings the lexicon has NEVER seen")
    print("=" * 78)
    train, heldout = build_holdout(0.50, seed=42)
    tmp = Path("/tmp/_v2_train.json")
    tmp.write_text(json.dumps({"lexicons": train}), encoding="utf-8")
    r = Resolver(v2_path=tmp, gold_vocab=gold, min_len=8)

    shown_ok, shown_bad = [], []
    for variant, canon in heldout:
        if variant.lower() in r.exact:
            continue
        out = r.resolve_word(variant)
        if out.lower() == canon.lower():
            if len(shown_ok) < 10:
                shown_ok.append((variant, out))
        elif out.lower() != variant.lower():
            if len(shown_bad) < 6:
                shown_bad.append((variant, out, canon))

    print("\n  RECOVERED (exact lexicon could not — resolver computed it):")
    for v, o in shown_ok:
        print(f"     {v:<18} -> {o}")
    if shown_bad:
        print("\n  WRONG (resolved, but to the wrong word):")
        for v, o, c in shown_bad:
            print(f"     {v:<18} -> {o:<16} (should be {c})")
    print()


if __name__ == "__main__":
    main()
