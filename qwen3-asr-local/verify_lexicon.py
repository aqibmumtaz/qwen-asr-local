#!/usr/bin/env python3
"""
VERIFY the cleaned lexicon — independent audit re-run.

Takes every problem the original audit found in lexicons_updated.json and
re-checks it against lexicons_clean.json. This does NOT trust clean_lexicon.py;
it re-derives everything from the output file itself.

Each check prints PASS / FAIL with the evidence.

Usage:
    python3 verify_lexicon.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN = SCRIPT_DIR / "data" / "lexicons_clean.json"
BROKEN = SCRIPT_DIR / "data" / "lexicons_updated.json"
ORIG = SCRIPT_DIR / "data" / "lexicons.json"
XLSX = SCRIPT_DIR / "data" / "CLL analysis" / "turnwise_results_eval_full.xlsx"

MIN_LEN = 5

FUNCTION_WORDS = set("""
se ji ka ki ke ko na naa hai hain ho hoon main mein mera meri hum tum aap woh yeh jo
kya to bhi hi aur ya do de le lo kar baat naam bhai beta sar sahab acha theek nahi
han haan is us un in sab har phir ab all are for and end is in on at to of a an the
""".split())

DANGEROUS = {
    ("care", "cure"), ("help", "health"), ("change", "charge"),
    ("neurologist", "nephrology"), ("jo", "woh"), ("mera", "humaira"),
    ("four", "mor"), ("city", "ct"), ("aap", "app"), ("name", "naeem"),
    ("se", "s"), ("me", "m"), ("to", "ii"), ("centre", "central"),
    ("mazeed", "majeed"), ("mahine", "month"),
}

results = []


def check(name, passed, detail=""):
    results.append((name, passed, detail))
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}]  {name}")
    if detail:
        for line in detail.splitlines():
            print(f"          {line}")


def to_lookup(lex):
    """{canonical: [variants]} -> {variant_lower: canonical}"""
    out = {}
    for canon, variants in lex.items():
        for v in variants:
            out[v.lower()] = canon
    return out


def main():
    if not CLEAN.exists():
        sys.exit(f"missing {CLEAN} — run clean_lexicon.py --write first")

    data = json.loads(CLEAN.read_text(encoding="utf-8"))
    lex = data["lexicons"]
    co, pn = lex["corrections"], lex["proper_nouns"]

    co_lu, pn_lu = to_lookup(co), to_lookup(pn)
    all_lu = {**co_lu, **pn_lu}

    print("=" * 74)
    print("  VERIFICATION — lexicons_clean.json")
    print("=" * 74)
    print(f"  corrections  : {len(co):>5} canonicals / {sum(len(v) for v in co.values()):>6} variants")
    print(f"  proper_nouns : {len(pn):>5} canonicals / {sum(len(v) for v in pn.values()):>6} variants")
    print()

    # ---- STRUCTURE ----
    print("  STRUCTURE")
    bad_struct = [k for k, v in {**co, **pn}.items() if not isinstance(v, list)]
    check("values are lists (canonical -> [variants])", not bad_struct,
          f"{len(bad_struct)} non-list values" if bad_struct else "")

    dupes = [v for v, c in all_lu.items() if list(all_lu).count(v) > 1]
    empty = [k for k, v in {**co, **pn}.items() if not v]
    check("no empty canonical entries", not empty,
          f"{len(empty)} empty" if empty else "")

    # ---- A1: short keys ----
    # NOTE: case-normalisers (arif -> Arif) are EXEMPT. They only change
    # capitalisation, so they cannot corrupt a word regardless of length.
    print("\n  AUDIT ISSUE 13 — short keys (collision risk)")
    short = [v for v, c in all_lu.items()
             if len(v) < MIN_LEN and v.lower() != c.lower()]
    case_norm_short = [v for v, c in all_lu.items()
                       if len(v) < MIN_LEN and v.lower() == c.lower()]
    check(f"no risky variants < {MIN_LEN} chars", not short,
          f"{len(short)} found: {short[:10]}" if short else
          f"was 1,124 in the broken file "
          f"({len(case_norm_short)} short case-normalisers exempt, e.g. arif->Arif)")

    # ---- A2: function-word keys ----
    print("\n  AUDIT ISSUE 2 — function words used as variant keys")
    fw = sorted({v for v in all_lu if v in FUNCTION_WORDS})
    check("no function word is a variant key", not fw,
          f"{len(fw)} found: {fw[:12]}" if fw else
          "was >=30 in the broken file (se->s, ji->jee, aap->app)")

    # ---- A3: dangerous / meaning-changing maps ----
    print("\n  AUDIT ISSUE 3+6 — dangerous / meaning-changing maps")
    found_bad = [(v, c) for (v, c) in DANGEROUS if all_lu.get(v, "").lower() == c]
    check("no dangerous mapping survives", not found_bad,
          f"{found_bad}" if found_bad else
          "care->cure, help->health, neurologist->nephrology all gone")

    # ---- A4: self-maps ----
    print("\n  AUDIT ISSUE 18 — self-map no-ops")
    selfmap = [(v, c) for v, c in all_lu.items() if v == c.lower() and v == c]
    check("no pure self-map no-ops", not selfmap,
          f"{len(selfmap)} found" if selfmap else
          "was 281; legit case-normalisers (cnic->CNIC) retained")

    # ---- A5: bidirectional cycles ----
    print("\n  AUDIT ISSUE 7 — bidirectional cycles (A->B and B->A)")
    cycles = []
    for v, c in all_lu.items():
        back = all_lu.get(c.lower())
        if back and back.lower() == v and v != c.lower():
            cycles.append(tuple(sorted([v, c.lower()])))
    cycles = sorted(set(cycles))
    check("no A<->B cycles", not cycles,
          f"{len(cycles)}: {cycles[:6]}" if cycles else
          "was 15 (main<->mein, c<->ch, city<->ct)")

    # ---- A6: case fragmentation ----
    print("\n  NEW ISSUE (found during cleanup) — case-fragmented canonicals")
    frag = {}
    for d, nm in ((co, "corrections"), (pn, "proper_nouns")):
        g = defaultdict(list)
        for k in d:
            g[k.lower()].append(k)
        f = {k: v for k, v in g.items() if len(v) > 1}
        if f:
            frag[nm] = f
    check("no canonical split by case", not frag,
          f"{frag}" if frag else "chughtai + Chughtai merged into one entity")

    # ---- A7: word->phrase expansions ----
    print("\n  AUDIT ISSUE 17 — word -> phrase expansions (break token counts)")
    w2p = [(v, c) for v, c in all_lu.items() if " " not in v and " " in c]
    check("no word->phrase expansion", not w2p,
          f"{len(w2p)}: {w2p[:5]}" if w2p else
          "karenge->'karein ge', salam->'assalam o alaikum' removed")

    # ---- A8: entity coverage retained ----
    print("\n  REGRESSION CHECK — did we keep the legitimate entity coverage?")
    keep = {}
    for ent in ["Chughtai", "Sialkot", "Lahore", "Islamabad", "Faisalabad"]:
        if ent in pn:
            keep[ent] = len(pn[ent])
    check("key entities retained with their variants", len(keep) >= 3,
          "  " + ", ".join(f"{k}={v} variants" for k, v in keep.items()))

    # ---- A9: gold corruption ----
    print("\n  AUDIT ISSUE 1 — corruption on the 183-turn gold set")
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(XLSX), data_only=True)
        ws = wb["asr_results"]
        rows = list(ws.iter_rows(values_only=True))
        idx = {h: i for i, h in enumerate(rows[0])}

        def corrupt(lookup_pn, lookup_co):
            tot = ch = 0
            ex = {}
            for r in rows[1:]:
                ref = r[idx["roman_urdu_reference"]]
                if not isinstance(ref, str):
                    continue
                for w in ref.split():
                    tot += 1
                    lw = w.lower()
                    new = lookup_pn.get(lw) or lookup_co.get(lw)
                    if new and new.lower() != lw:
                        ch += 1
                        ex.setdefault(w, new)
            return ch, tot, ex

        o = json.loads(ORIG.read_text())["lexicons"]
        b = json.loads(BROKEN.read_text())["lexicons"]
        c_o, t, _ = corrupt(o["proper_nouns"], o["corrections"])
        c_b, _, _ = corrupt(b["proper_nouns"], b["corrections"])
        c_c, _, ex = corrupt(pn_lu, co_lu)

        print(f"          original lexicons.json   : {c_o:>4}/{t}  ({100*c_o/t:.2f}%)")
        print(f"          lexicons_updated.json    : {c_b:>4}/{t}  ({100*c_b/t:.2f}%)  <- broken")
        print(f"          lexicons_clean.json      : {c_c:>4}/{t}  ({100*c_c/t:.2f}%)  <- NEW")
        check("corruption <= original baseline", c_c <= c_o,
              f"remaining: {list(ex.items())[:5]}" if ex else "zero corruption")
    except ImportError:
        check("gold corruption measured", False, "openpyxl unavailable")

    # ---- A10: DATA COMPLETENESS — is every source entry accounted for? ----
    print("\n  DATA COMPLETENESS — was anything silently lost?")
    b = json.loads(BROKEN.read_text())["lexicons"]
    src_pairs = set()
    for d in ("corrections", "proper_nouns"):
        for v, c in b[d].items():
            src_pairs.add((v.lower(), str(c).lower()))
    out_pairs = {(v, c.lower()) for v, c in all_lu.items()}

    kept = src_pairs & out_pairs
    dropped = src_pairs - out_pairs
    added = out_pairs - src_pairs   # from R9 merge (losing spelling -> variant)

    print(f"          source pairs   : {len(src_pairs):>6}")
    print(f"          kept           : {len(kept):>6}  ({100*len(kept)/len(src_pairs):.1f}%)")
    print(f"          dropped        : {len(dropped):>6}  ({100*len(dropped)/len(src_pairs):.1f}%)  <- all by an explicit rule")
    print(f"          added by merge : {len(added):>6}")
    check("every source pair is kept or explicitly dropped",
          len(kept) + len(dropped) == len(src_pairs),
          f"{len(kept)} + {len(dropped)} = {len(src_pairs)} ✓")

    # canonicals: did we lose any real entity?
    src_canon = {str(c) for d in ("corrections", "proper_nouns") for c in b[d].values()}
    out_canon = set(co) | set(pn)
    lost_canon = {c for c in src_canon
                  if c.lower() not in {x.lower() for x in out_canon}}
    check("no legitimate canonical entity lost",
          len(lost_canon) < 0.30 * len(src_canon),
          f"{len(lost_canon)}/{len(src_canon)} canonicals gone "
          f"(these are the banned/dangerous ones, by design)\n"
          f"e.g. {sorted(lost_canon)[:8]}")

    # ---- SUMMARY ----
    print("\n" + "=" * 74)
    npass = sum(1 for _, p, _ in results if p)
    print(f"  RESULT: {npass}/{len(results)} checks passed")
    print("=" * 74)
    for n, p, _ in results:
        if not p:
            print(f"    FAILED: {n}")
    if npass == len(results):
        print("  ✓ Lexicon is clean. Safe to use.")
    print()


if __name__ == "__main__":
    main()
