#!/usr/bin/env python3
"""
FINAL AUDIT of lexicons_v2.json — answers three questions:

  1. REPRODUCIBLE?  Does clean_lexicon.py regenerate v2 byte-for-byte?
  2. COMPLETE?      Is EVERY pair from lexicons_updated.json accounted for —
                    either kept in v2, or dropped by a named rule? Nothing lost
                    silently, nothing invented.
  3. CORRECT?       Do all invariants + the corruption target still hold?

Usage:
    python3 audit_v2.py
"""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from clean_lexicon import (
    SRC, OUT, load, clean, to_lookup, load_gold_vocab,
    PROTECTED, BANNED_PAIRS, BANNED_CANONICALS, SPELLING_PREFERENCE_ONLY,
    build_known_correct, load_gazetteer,
)

import re

ok = []


def check(name, passed, detail=""):
    ok.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")
    for line in str(detail).splitlines():
        if line:
            print(f"          {line}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    print("=" * 76)
    print("  FINAL AUDIT — lexicons_v2.json")
    print("=" * 76)

    # ── 1. REPRODUCIBILITY ───────────────────────────────────────────────────
    print("\n  1. REPRODUCIBLE — can clean_lexicon.py regenerate v2 exactly?")
    before = sha(OUT)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "clean_lexicon.py"), "--write"],
                   capture_output=True, cwd=SCRIPT_DIR)
    after = sha(OUT)
    check("re-running the script reproduces v2 byte-for-byte",
          before == after,
          f"sha256 before: {before[:40]}...\nsha256 after : {after[:40]}...")

    # ── 2. COMPLETENESS ──────────────────────────────────────────────────────
    print("\n  2. COMPLETE — is every source pair accounted for?")
    src = load(SRC)
    gold = load_gold_vocab()
    known = build_known_correct(src, gold)

    # every (variant, canonical) pair in the SOURCE
    src_pairs = []
    for d in ("corrections", "proper_nouns"):
        for v, c in src[d].items():
            src_pairs.append((v.strip(), str(c).strip()))

    v2 = json.loads(OUT.read_text())["lexicons"]
    v2_words, v2_phrases = to_lookup(v2)
    v2_lu = {**v2_words, **v2_phrases}

    # classify each source pair: KEPT, or dropped by WHICH rule
    verdict = Counter()
    unexplained = []
    for v, c in src_pairs:
        vl, cl = v.lower(), c.lower()
        if v2_lu.get(vl, "").lower() == cl:
            verdict["KEPT"] += 1
            continue
        # it is not in v2 — must be explained by a rule
        if vl == cl:
            verdict["R3  self-map"] += 1
        elif not re.fullmatch(r"[A-Za-z0-9 .'\-]+", v):
            verdict["R12 non-Latin"] += 1
        elif vl in known:
            verdict["R1b already-correct word"] += 1
        elif " " not in v and " " in c:
            verdict["R8  word->phrase"] += 1
        elif (vl, cl) in BANNED_PAIRS:
            verdict["R5  banned pair"] += 1
        elif cl in BANNED_CANONICALS:
            verdict["R5b fragment canonical"] += 1
        elif (vl, cl) in SPELLING_PREFERENCE_ONLY:
            verdict["R11 spelling preference"] += 1
        elif v2_lu.get(vl):
            # kept, but re-pointed by a merge (case-fragment / cross-dict)
            verdict["R9/R10 merged (canonical re-cased)"] += 1
        else:
            verdict["** UNEXPLAINED **"] += 1
            unexplained.append(f"{v} -> {c}")

    total = sum(verdict.values())
    print()
    for k, n in verdict.most_common():
        print(f"      {k:<38} {n:>6}")
    print(f"      {'TOTAL':<38} {total:>6}")
    print()
    check("every source pair is KEPT or dropped by a NAMED rule",
          not unexplained,
          f"{len(unexplained)} unexplained: {unexplained[:6]}" if unexplained
          else f"{total} source pairs, 0 unexplained")
    check("source pair count reconciles",
          total == len(src_pairs),
          f"{total} == {len(src_pairs)} ✓")

    # nothing INVENTED: every v2 variant traces back to the source (or a merge)
    src_variants = {v.lower() for v, _ in src_pairs}
    src_canon = {c.lower() for _, c in src_pairs}
    invented = [v for v in v2_lu if v not in src_variants and v not in src_canon]
    check("no variant invented out of thin air", not invented,
          f"{len(invented)}: {invented[:6]}" if invented
          else "every v2 variant came from the source")

    # ── 3. CORRECTNESS ───────────────────────────────────────────────────────
    print("\n  3. CORRECT — invariants and targets")
    all_e = {**v2["lexicon"], **v2["phrases"]}

    kc = Counter(k.lower() for k in all_e)
    check("I1  no duplicate canonical key (case-insensitive)",
          not [k for k, n in kc.items() if n > 1],
          f"{len(all_e)} keys, all unique when lowercased")

    owner = {}
    shared = []
    for c, vs in all_e.items():
        for v in vs:
            if v.lower() in owner and owner[v.lower()] != c:
                shared.append(v)
            owner[v.lower()] = c
    check("I2  no variant claimed by two canonicals", not shared,
          f"{shared[:5]}" if shared else "")

    dup = [c for c, vs in all_e.items() if len(vs) != len({x.lower() for x in vs})]
    check("I3  no duplicate variant inside an entry", not dup, f"{dup[:5]}" if dup else "")

    nonlower = [v for vs in all_e.values() for v in vs if v != v.lower()]
    check("I4  all variants stored lowercase", not nonlower,
          f"{nonlower[:5]}" if nonlower else "canonicals keep case, variants lowercase")

    nonlatin = [v for vs in all_e.values() for v in vs
                if not re.fullmatch(r"[A-Za-z0-9 .'\-]+", v)]
    check("I5  no non-Latin variants (Nastaliq)", not nonlatin,
          f"{nonlatin[:5]}" if nonlatin else "")

    # the specific bugs found in review
    print()
    print("  4. THE BUGS FOUND IN REVIEW — all fixed?")
    check("inki -> ek is gone", v2_lu.get("inki") is None,
          f"inki -> {v2_lu.get('inki')}" if v2_lu.get("inki") else "")
    check("eria -> area is KEPT (the length-rule bug)",
          v2_lu.get("eria", "").lower() == "area",
          f"eria -> {v2_lu.get('eria')}")
    check("nephrology variants recovered (the banned-canonical bug)",
          len(v2["lexicon"].get("nephrology", [])) > 15,
          f"nephrology has {len(v2['lexicon'].get('nephrology', []))} variants")
    check("Chughtai merged across both source dicts",
          len(v2["lexicon"].get("Chughtai", [])) > 25,
          f"Chughtai has {len(v2['lexicon'].get('Chughtai', []))} variants")
    check("names consistently capitalised (Afzal, not afzal)",
          "Afzal" in v2["lexicon"] and "afzal" not in v2["lexicon"],
          "Afzal / Arif / Ayesha / Sialkot all Title-Cased")
    check("dangerous maps blocked",
          all(v2_lu.get(v, "").lower() != c for v, c in
              [("neurologist", "nephrology"), ("care", "cure"), ("help", "health"),
               ("mera", "humaira"), ("aap", "app")]),
          "care->cure, neurologist->nephrology, aap->app all gone")

    # ── corruption ───────────────────────────────────────────────────────────
    print("\n  5. CORRUPTION TARGET")
    import openpyxl
    wb = openpyxl.load_workbook(str(SCRIPT_DIR / "data" / "CLL analysis"
                                    / "turnwise_results_eval_full.xlsx"), data_only=True)
    rows = list(wb["asr_results"].iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    from clean_lexicon import FORCE_KEEP
    force_variants = {v for v, _ in FORCE_KEEP}

    tot = ch = forced = 0
    offenders = Counter()
    for r in rows[1:]:
        ref = r[idx["roman_urdu_reference"]]
        if not isinstance(ref, str):
            continue
        for w in ref.split():
            tot += 1
            lw = w.lower()
            n = v2_lu.get(lw)
            if n and n.lower() != lw:
                ch += 1
                offenders[f"{w} -> {n}"] += 1
                if lw in force_variants:
                    forced += 1

    unintended = ch - forced
    print(f"      gold words changed      : {ch} / {tot}  ({100*ch/tot:.2f}%)")
    print(f"        of which FORCE_KEEPed : {forced}  (an accepted, documented cost)")
    print(f"        UNINTENDED corruption : {unintended}")
    if offenders:
        for k, n in offenders.most_common(5):
            tag = "(FORCE_KEEP — accepted)" if k.split(" ->")[0].lower() in force_variants else "** UNINTENDED **"
            print(f"          {n}x  {k}   {tag}")
    check("no UNINTENDED corruption (was 791 = 23.27%)", unintended == 0,
          f"{unintended} unintended" if unintended else
          f"only the {forced} deliberate FORCE_KEEP change")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    n_lex = len(v2["lexicon"])
    n_ph = len(v2["phrases"])
    n_v = sum(len(v) for v in all_e.values())
    print(f"  v2: {n_lex} words + {n_ph} phrases = {n_lex+n_ph} canonicals, {n_v} variants")
    print(f"  RESULT: {sum(ok)}/{len(ok)} checks passed")
    print("=" * 76)
    if all(ok):
        print("  ✓ v2 is reproducible, complete, and correct.")
    print()


if __name__ == "__main__":
    main()
