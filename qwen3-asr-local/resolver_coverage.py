#!/usr/bin/env python3
"""
Could the RESOLVER replace the enumerated variant list?

THE QUESTION
    lexicons_v2 lists 14,486 misspellings for 2,045 canonical words. If the
    resolver can COMPUTE a variant's canonical from the canonical alone, then
    listing that variant explicitly earns nothing and it could be deleted.

THE METHOD
    For every (variant -> canonical) pair in v2:
      - bypass G1 (the exact map). Otherwise every variant "passes" trivially,
        because the exact map is exactly the thing we are trying to make redundant.
      - run the remaining guards + the fuzzy path, as production would for a word
        it has never seen.
      - compare the result with the canonical the lexicon says is right.

    The fuzzy index holds CANONICALS ONLY, so nothing leaks: the resolver has to
    reconstruct the mapping phonetically, never look it up.

VERDICTS
    RECOVERED    fuzzy returned the correct canonical  -> the entry is redundant
    WRONG        fuzzy returned a DIFFERENT canonical  -> deleting it would CORRUPT
    MISSED       fuzzy declined (too short / no match / ambiguous) -> entry needed
    PROTECTED    the variant is itself a canonical (G2) -> entry needed

Usage:
    python3 resolver_coverage.py
    python3 resolver_coverage.py --min-len 6
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from resolver import Resolver, normalise, max_distance, _edit

SCRIPT_DIR = Path(__file__).resolve().parent
V2 = SCRIPT_DIR / "data" / "lexicons_v2.json"


def fuzzy_only(r: Resolver, word: str) -> tuple[str, str]:
    """
    The resolver's path WITHOUT the exact-map shortcut (G1).
    Returns (verdict_kind, result) where kind is 'protected' | 'short' |
    'no_match' | 'ambiguous' | 'fuzzy'.
    """
    lw = word.lower()

    # G2 — the variant is itself a correct word; production would never touch it
    if lw in r.known_correct:
        return "protected", word

    # G3 — too short to fuzzy-match safely
    if len(lw) < r.min_len or not lw.isalpha():
        return "short", word

    nk = normalise(lw)
    if not nk:
        return "no_match", word

    # fast path — normalised skeletons match exactly
    cands = r.norm_exact.get(nk)
    if cands:
        if len(cands) == 1:
            return "fuzzy", cands[0]
        return "ambiguous", word

    cap = max_distance(len(nk), len(nk))
    if cap == 0:
        return "no_match", word

    best, best_d, runner_up = None, cap + 1, cap + 1
    for L in range(len(nk) - cap, len(nk) + cap + 1):
        for cnk, canon in r.by_len.get(L, ()):
            d = _edit(nk, cnk, max_distance(len(nk), len(cnk)))
            if d < best_d:
                best, runner_up, best_d = canon, best_d, d
            elif d < runner_up:
                runner_up = d

    if best is None or best_d > cap:
        return "no_match", word
    if runner_up == best_d:
        return "ambiguous", word
    return "fuzzy", best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=None,
                    help="override the resolver's G3 length guard")
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()

    raw = json.loads(V2.read_text(encoding="utf-8"))["lexicons"]
    r = Resolver(min_len=args.min_len) if args.min_len else Resolver()

    pairs = [(v, c) for c, vs in raw["lexicon"].items() for v in vs]

    verdict = Counter()
    wrong_examples, missed_reason = [], Counter()
    per_canon_recovered = defaultdict(int)
    per_canon_total = defaultdict(int)

    for variant, canon in pairs:
        kind, out = fuzzy_only(r, variant)
        per_canon_total[canon] += 1
        if kind == "protected":
            verdict["PROTECTED (variant is itself a real word)"] += 1
        elif kind in ("short", "no_match", "ambiguous"):
            verdict["MISSED (resolver declined)"] += 1
            missed_reason[kind] += 1
        elif out.lower() == canon.lower():
            verdict["RECOVERED (entry is redundant)"] += 1
            per_canon_recovered[canon] += 1
        else:
            verdict["WRONG (deleting it would CORRUPT)"] += 1
            if len(wrong_examples) < 400:
                wrong_examples.append((variant, canon, out))

    total = len(pairs)
    print("=" * 78)
    print("  CAN THE RESOLVER REPLACE THE VARIANT LIST?")
    print("=" * 78)
    print(f"  v2 single-word variants tested : {total}")
    print(f"  canonicals in the fuzzy index  : {len(r.canonicals)}")
    print(f"  G3 min length                  : {r.min_len}")
    print()
    for k in ["RECOVERED (entry is redundant)",
              "WRONG (deleting it would CORRUPT)",
              "MISSED (resolver declined)",
              "PROTECTED (variant is itself a real word)"]:
        n = verdict[k]
        print(f"    {n:>6}  {100*n/total:>5.1f}%   {k}")
    print()
    print("  why the resolver declined:")
    for k, n in missed_reason.most_common():
        label = {"short": f"variant shorter than {r.min_len} chars",
                 "no_match": "no canonical within the edit budget",
                 "ambiguous": "two canonicals tied — refused to guess"}[k]
        print(f"    {n:>6}  {label}")
    print()

    rec = verdict["RECOVERED (entry is redundant)"]
    wrong = verdict["WRONG (deleting it would CORRUPT)"]
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  Safe to delete   : {rec:>6}  ({100*rec/total:.1f}%)")
    print(f"  Unsafe to delete : {total-rec:>6}  ({100*(total-rec)/total:.1f}%)")
    print(f"  Of those, entries whose deletion would ACTIVELY CORRUPT: {wrong}")
    print()
    print(f"  EXAMPLES — deleting these would send the word to the WRONG canonical:")
    for v, c, o in wrong_examples[: args.examples]:
        print(f"    {v:<20} should be {c:<18} resolver says {o}")
    print()


if __name__ == "__main__":
    main()
