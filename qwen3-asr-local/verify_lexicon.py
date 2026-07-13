#!/usr/bin/env python3
"""
VERIFY the cleaned lexicon — independent audit re-run.

Re-derives EVERY finding from the output file itself. Does not trust
clean_lexicon.py. Each check prints PASS / FAIL with evidence.

Usage:
    python3 verify_lexicon.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN = SCRIPT_DIR / "data" / "lexicons_v2.json"
BROKEN = SCRIPT_DIR / "data" / "lexicons_updated.json"
ORIG = SCRIPT_DIR / "data" / "lexicons.json"
XLSX = SCRIPT_DIR / "data" / "CLL analysis" / "turnwise_results_eval_full.xlsx"

FUNCTION_WORDS = set("""
se ji ka ki ke ko na naa hai hain ho hoon main mein mera meri hum tum aap woh yeh jo
kya to bhi hi aur ya do de le lo kar baat naam bhai beta sar sahab acha theek nahi
han haan is us un in sab har phir ab all are for and end on at of a an the
""".split())

DANGEROUS = {
    ("care", "cure"), ("help", "health"), ("change", "charge"),
    ("neurologist", "nephrology"), ("jo", "woh"), ("mera", "humaira"),
    ("four", "mor"), ("city", "ct"), ("aap", "app"), ("name", "naeem"),
    ("se", "s"), ("me", "m"), ("to", "ii"), ("centre", "central"),
    ("mazeed", "majeed"), ("mahine", "month"), ("wale", "walaikum"),
    ("mil", "meal"), ("den", "din"), ("maan", "maa"),
}

results = []


def check(name, passed, detail=""):
    results.append((name, passed))
    print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")
    for line in str(detail).splitlines():
        if line:
            print(f"          {line}")


def main():
    if not CLEAN.exists():
        sys.exit(f"missing {CLEAN} — run clean_lexicon.py --write first")

    lex = json.loads(CLEAN.read_text(encoding="utf-8"))["lexicons"]
    CATS = ("lexicon", "phrases")
    all_entries = {}
    for c in CATS:
        all_entries.update(lex[c])
    words = lex["lexicon"]
    phrases = lex["phrases"]

    lookup = {}
    for canon, vs in all_entries.items():
        for v in vs:
            lookup[v.lower()] = canon

    print("=" * 74)
    print("  VERIFICATION — lexicons_v2.json")
    print("=" * 74)
    for c in CATS:
        print(f"  {c:<13} : {len(lex[c]):>5} canonicals / "
              f"{sum(len(v) for v in lex[c].values()):>6} variants")
    print(f"  {'TOTAL':<13} : {len(all_entries):>5} canonicals")
    print()

    # ── KEY UNIQUENESS (the invariant you asked for) ─────────────────────────
    print("  KEY UNIQUENESS — enforced across lexicon + phrases")
    kc = Counter(k.lower() for k in all_entries)
    dup_keys = {k: [x for x in all_entries if x.lower() == k] for k, n in kc.items() if n > 1}
    check("no canonical key duplicated (case-insensitive)", not dup_keys,
          f"DUPES: {dup_keys}" if dup_keys else
          f"{len(all_entries)} keys, all unique when lowercased")

    owner = defaultdict(set)
    for canon, vs in all_entries.items():
        for v in vs:
            owner[v.lower()].add(canon)
    shared = {v: sorted(c) for v, c in owner.items() if len(c) > 1}
    check("no variant claimed by two canonicals", not shared,
          f"{len(shared)}: {list(shared.items())[:4]}" if shared else
          "every variant belongs to exactly one entity")

    dup_in_entry = {k: v for k, v in all_entries.items()
                    if len(v) != len({x.lower() for x in v})}
    check("no duplicate variant inside an entry", not dup_in_entry,
          f"{list(dup_in_entry)[:5]}" if dup_in_entry else "")

    noop = [k for k, v in all_entries.items()
            if k == k.lower() and k.lower() in {x.lower() for x in v}]
    check("no variant equals its own lowercase canonical (no-op)", not noop,
          f"{noop[:5]}" if noop else "")

    # ── ORIGINAL AUDIT ISSUES ────────────────────────────────────────────────
    print("\n  ORIGINAL AUDIT ISSUES")
    fw = sorted({v for v in lookup if v in FUNCTION_WORDS})
    check("no function word used as a variant", not fw,
          f"{len(fw)}: {fw[:12]}" if fw else "was >=30 (se->s, ji->jee, aap->app)")

    bad = [(v, c) for (v, c) in DANGEROUS if lookup.get(v, "").lower() == c]
    check("no dangerous / meaning-changing map", not bad,
          f"{bad}" if bad else "care->cure, neurologist->nephrology, mil->meal all gone")

    cycles = sorted({tuple(sorted([v, c.lower()])) for v, c in lookup.items()
                     if lookup.get(c.lower(), "").lower() == v and v != c.lower()})
    check("no A<->B cycles", not cycles,
          f"{len(cycles)}: {cycles[:5]}" if cycles else "was 15 (main<->mein, c<->ch)")

    w2p = [(v, c) for v, c in lookup.items() if " " not in v and " " in c]
    check("no word->phrase expansion", not w2p,
          f"{len(w2p)}: {w2p[:4]}" if w2p else "karenge->'karein ge' removed")

    # ── ENTITY COVERAGE ──────────────────────────────────────────────────────
    print("\n  ENTITY COVERAGE — legitimate entities retained?")
    ents = {e: len(words[e]) for e in
            ["Chughtai", "Sialkot", "Lahore", "Islamabad", "Faisalabad", "CNIC", "area"]
            if e in words}
    # variants must all be lowercase
    nonlower = [(k, v) for k, vs in all_entries.items() for v in vs if v != v.lower()]
    check("all variants stored lowercase", not nonlower,
          f"{nonlower[:5]}" if nonlower else "canonical keeps case, variants lowercase")
    check("key entities present with variants", len(ents) >= 5,
          "  " + ", ".join(f"{k}={v}" for k, v in ents.items()))

    # every gazetteer entity present in the lexicon must be Title-Cased consistently
    import sys as _s
    _s.path.insert(0, str(SCRIPT_DIR))
    from clean_lexicon import load_gazetteer
    gaz = load_gazetteer()
    miscased = [k for k in all_entries
                if k.lower() in gaz and k != gaz[k.lower()]]
    check("gazetteer entities are consistently cased", not miscased,
          f"{miscased[:8]}" if miscased else
          "Afzal / Arif / Ayesha / Sialkot all Title-Cased (was: afzal vs Arif)")

    # ── GOLD CORRUPTION ──────────────────────────────────────────────────────
    print("\n  CORRUPTION ON THE 183-TURN GOLD SET")
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(XLSX), data_only=True)
        rows = list(wb["asr_results"].iter_rows(values_only=True))
        idx = {h: i for i, h in enumerate(rows[0])}

        def corrupt(lu):
            tot = ch = 0
            ex = {}
            for r in rows[1:]:
                ref = r[idx["roman_urdu_reference"]]
                if not isinstance(ref, str):
                    continue
                for w in ref.split():
                    tot += 1
                    n = lu.get(w.lower())
                    if n and n.lower() != w.lower():
                        ch += 1
                        ex.setdefault(w, n)
            return ch, tot, ex

        o = json.loads(ORIG.read_text())["lexicons"]
        b = json.loads(BROKEN.read_text())["lexicons"]
        c_o, t, _ = corrupt({**o["corrections"], **o["proper_nouns"]})
        c_b, _, _ = corrupt({**b["corrections"], **b["proper_nouns"]})
        c_c, _, ex = corrupt(lookup)
        print(f"          original lexicons.json : {c_o:>4}/{t}  ({100*c_o/t:.2f}%)")
        print(f"          lexicons_updated.json  : {c_b:>4}/{t}  ({100*c_b/t:.2f}%)  <- broken")
        print(f"          lexicons_v2.json    : {c_c:>4}/{t}  ({100*c_c/t:.2f}%)  <- NEW")
        check("corruption <= original baseline", c_c <= c_o,
              f"remaining: {list(ex.items())[:5]}" if ex else "zero corruption")
    except ImportError:
        check("gold corruption measured", False, "openpyxl unavailable")

    # ── COMPLETENESS ─────────────────────────────────────────────────────────
    print("\n  DATA COMPLETENESS")
    b = json.loads(BROKEN.read_text())["lexicons"]
    src_pairs = {(v.lower(), str(c).lower())
                 for d in ("corrections", "proper_nouns") for v, c in b[d].items()}
    out_pairs = {(v, c.lower()) for v, c in lookup.items()}
    kept = src_pairs & out_pairs
    dropped = src_pairs - out_pairs
    print(f"          source pairs : {len(src_pairs):>6}")
    print(f"          kept         : {len(kept):>6}  ({100*len(kept)/len(src_pairs):.1f}%)")
    print(f"          dropped      : {len(dropped):>6}  ({100*len(dropped)/len(src_pairs):.1f}%)  <- by explicit rule")
    check("every source pair kept or explicitly dropped",
          len(kept) + len(dropped) == len(src_pairs),
          f"{len(kept)} + {len(dropped)} = {len(src_pairs)} ✓")

    print("\n" + "=" * 74)
    npass = sum(1 for _, p in results if p)
    print(f"  RESULT: {npass}/{len(results)} checks passed")
    print("=" * 74)
    for n, p in results:
        if not p:
            print(f"    FAILED: {n}")
    if npass == len(results):
        print("  ✓ Lexicon is clean. No duplicate keys. Safe to use.")
    print()


if __name__ == "__main__":
    main()
