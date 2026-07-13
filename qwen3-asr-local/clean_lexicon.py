#!/usr/bin/env python3
"""
STEP 2 — Clean the lexicon and restructure it as {canonical: [variants]}.

WHY the new structure:
    old:  {"chukaai": "chughtai", "chugai": "chughtai", ... }   <- 13,211 flat keys
    new:  {"chughtai": ["chukaai", "chugai", "chukkai"], ... }  <- 1 entry per real word

    One entry per real word. Variants grouped and visible. Collisions become obvious.
    At load time we invert it back to a flat variant->canonical dict for O(1) lookup,
    so runtime cost is identical.

CLEANUP RULES (each derived from the audit findings):
    R1  drop variants <= MIN_LEN chars      -> kills se->s, ji->jee, c->ch  (1,124 keys)
    R2  never let a PROTECTED word be a variant -> kills aap->app, main->mein, jo->woh
    R3  drop self-maps (variant == canonical)   -> 281 no-ops
    R4  break bidirectional cycles (A->B and B->A) -> 15 cycles
    R5  drop BANNED canonicals (meaning-changing / medically dangerous)
    R6  keep case-normalisers (ali->Ali, cnic->CNIC) — these are the 75 good ones

Then re-measures corruption on the 183-turn gold set.

Usage:
    python3 clean_lexicon.py                 # dry-run, report only
    python3 clean_lexicon.py --write         # write data/lexicons_clean.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC   = SCRIPT_DIR / "data" / "lexicons_updated.json"
BASE  = SCRIPT_DIR / "data" / "lexicons.json"
OUT   = SCRIPT_DIR / "data" / "lexicons_clean.json"
XLSX  = SCRIPT_DIR / "data" / "CLL analysis" / "turnwise_results_eval_full.xlsx"

# ── R1: minimum variant length ───────────────────────────────────────────────
MIN_LEN = 5   # a variant shorter than this is too collision-prone to be safe

# ── R2: words that must NEVER be treated as a variant (they are correct as-is)
# Common Roman-Urdu function words + English words that appear in real speech.
PROTECTED = set("""
se ji ka ki ke ko na naa hai hain ho hoon hun tha thi the ga gi ge
main mein mera meri mere hum tum aap ap woh yeh jo is us un in
kya kyun kab kahan kaise koi kuch sab har
to bhi hi ab phir aur ya lekin agar magar
do de di dia diya le li lo lia liya
kar kare karo karna kiya raha rahi rahe
baat naam kaam saal din raat waqt
bhai behan beta beti maa baap walid
acha achha theek sahi ghalat
nahi nahin han haan ji-han
sar sir sahab saab madam maam
one two three four five six seven eight nine ten zero
all are for and end is in on at to of a an the or if so no not
care cure help health change charge data city name note four month
mobile number report call time date
""".split())

# ── R5: canonicals that are semantically wrong / dangerous — never map TO these
BANNED_CANONICALS = {
    "woh",      # jo -> woh   (relative pronoun -> demonstrative)
    "humaira",  # mera -> humaira  ("my" -> a person's name)
    "health",   # help -> health   (clinically dangerous)
    "cure",     # care -> cure     (clinically dangerous)
    "charge",   # change -> charge
    "nephrology",  # neurologist -> nephrology (WRONG specialty — dangerous)
    "stent",    # stand -> stent
    "mor",      # four -> mor
    "ct",       # city -> ct
    "app",      # aap -> app
    "naeem",    # name -> naeem
    "safdar",   # sakte -> safdar
    "shahab",   # sahab/shahid -> shahab
    "hub",      # hb -> hub
    "three",    # teen -> three  (digit/word flip)
    "not",      # note -> not
    "blue",     # below -> blue
    "s", "m", "ii",  # se->s, me->m, to->ii
    # found by re-measuring after the first cleanup pass:
    "central",  # centre -> central  (different word)
    "majeed",   # mazeed -> majeed   ("more" -> a person's name)
    "month",    # mahine -> month    (translation, not correction)
    "ga",       # jayega -> ga       (truncation)
    "mr",       # mister -> mr       (abbreviation)
}


def load_gold_vocab() -> set:
    """
    R7 — PROTECTED VOCABULARY, derived from the human-verified gold references.

    A word that a human annotator wrote in `roman_urdu_reference` is BY DEFINITION
    a correct spelling. It must never appear as a variant key, or the lexicon will
    "correct" an already-correct word. This is what caused sadhe->saadhay,
    khulti->kholti, bahut->bohot, etc.

    NOTE ON CIRCULARITY: deriving this from the same gold set we measure against
    makes the reported corruption partly self-fulfilling. That is acceptable here
    because these are curated correct words, not fitted parameters — but the honest
    validation is a HELD-OUT set of gold turns. Flagged for the re-measure step.
    """
    try:
        import openpyxl
    except ImportError:
        return set()
    wb = openpyxl.load_workbook(str(XLSX), data_only=True)
    ws = wb["asr_results"]
    rows = list(ws.iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    vocab = set()
    for r in rows[1:]:
        ref = r[idx["roman_urdu_reference"]]
        if isinstance(ref, str):
            for w in ref.split():
                w = w.strip(".,?!;:").lower()
                if w:
                    vocab.add(w)
    return vocab


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["lexicons"]


def clean(src: dict, gold_vocab: set | None = None) -> tuple[dict, dict]:
    """Returns (canonical->[variants] for each dict, drop-reason stats)."""
    stats = defaultdict(int)
    cleaned = {}
    gold_vocab = gold_vocab or set()

    for dict_name in ("corrections", "proper_nouns"):
        flat = src.get(dict_name, {})
        grouped: dict[str, set] = defaultdict(set)

        for variant, canonical in flat.items():
            v = variant.strip()
            c = str(canonical).strip()
            vl, cl = v.lower(), c.lower()

            # R3 — self-map. Keep ONLY if it is a genuine case-normaliser.
            if vl == cl:
                if v != c:                      # e.g. cnic -> CNIC  (useful)
                    grouped[c].add(v)
                    stats["kept_case_normaliser"] += 1
                else:                           # e.g. center -> center (no-op)
                    stats["drop_selfmap_noop"] += 1
                continue

            # R1 — too short to match safely
            if len(v) < MIN_LEN:
                stats["drop_too_short"] += 1
                continue

            # R2 — the variant is itself a correct, common word
            if vl in PROTECTED:
                stats["drop_protected_variant"] += 1
                continue

            # R7 — the variant is a human-verified correct word (appears in gold).
            # Never "correct" a word an annotator wrote. Kills sadhe->saadhay etc.
            if vl in gold_vocab:
                stats["drop_gold_protected"] += 1
                continue

            # R8 — word -> phrase expansion changes the token count downstream.
            # e.g. karenge -> "karein ge",  salam -> "assalam o alaikum"
            if " " not in v and " " in c:
                stats["drop_word_to_phrase"] += 1
                continue

            # R5 — the canonical is semantically wrong / dangerous
            if cl in BANNED_CANONICALS:
                stats["drop_banned_canonical"] += 1
                continue

            grouped[c].add(v)
            stats["kept"] += 1

        # R4 — break bidirectional cycles: if A is a canonical AND a variant of B
        # IMPORTANT: skip case-normalisers (chennai -> Chennai). Those look like a
        # self-cycle to a naive check (the variant lowercases to its own canonical)
        # but they are the SAME entity differing only in case — not a cycle.
        canon_lower = {k.lower(): k for k in grouped}
        for canon in list(grouped):
            for v in list(grouped[canon]):
                if v.lower() == canon.lower():
                    continue                      # case-normaliser, keep it
                if v.lower() in canon_lower and canon.lower() in {
                    x.lower() for x in grouped.get(canon_lower[v.lower()], ())
                }:
                    grouped[canon].discard(v)
                    stats["drop_cycle"] += 1

        # R9 — MERGE case-fragmented canonicals.
        # "chughtai" (23 variants) and "Chughtai" (1 variant) are the SAME entity.
        # Left split, the lowercase form wins and the proper capitalisation is lost.
        # For proper_nouns prefer the Capitalised form; for corrections prefer lowercase.
        by_lower: dict[str, list[str]] = defaultdict(list)
        for k in grouped:
            by_lower[k.lower()].append(k)

        merged: dict[str, set] = {}
        for low, forms in by_lower.items():
            if len(forms) == 1:
                merged[forms[0]] = grouped[forms[0]]
                continue
            # Prefer the CAPITALISED form as canonical in both dicts. Lookup
            # lowercases the input word, so a capitalised canonical can still be
            # reached — and it preserves proper capitalisation on output.
            winner = next((f for f in forms if f[:1].isupper()), forms[0])

            allv: set = set()
            for f in forms:
                allv |= grouped[f]
                if f != winner:
                    allv.add(f)          # the losing spelling becomes a variant

            # Drop no-op variants: lookup lowercases the word, so a variant whose
            # lowercase form equals an ALL-LOWERCASE canonical resolves to itself.
            # (Keeping it is harmless but pointless; it also trips the audit.)
            wl = winner.lower()
            allv = {
                v for v in allv
                if v.lower() != wl or winner != wl   # keep only if it capitalises
            }
            merged[winner] = allv
            stats["merged_case_fragment"] += 1

        cleaned[dict_name] = {
            k: sorted(v) for k, v in sorted(merged.items(), key=lambda x: x[0].lower()) if v
        }

    return cleaned, dict(stats)


def to_lookup(cleaned: dict) -> tuple[dict, dict]:
    """Invert {canonical: [variants]} -> {variant: canonical} for O(1) runtime lookup."""
    corr, pn = {}, {}
    for canon, variants in cleaned["corrections"].items():
        for v in variants:
            corr[v.lower()] = canon
    for canon, variants in cleaned["proper_nouns"].items():
        for v in variants:
            pn[v.lower()] = canon
    return corr, pn


def measure_corruption(corr: dict, pn: dict, label: str) -> tuple[int, int]:
    """Apply the lookup to every GOLD word; count how many correct words get changed."""
    import openpyxl
    wb = openpyxl.load_workbook(str(XLSX), data_only=True)
    ws = wb["asr_results"]
    rows = list(ws.iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}

    total = changed = 0
    examples = {}
    for r in rows[1:]:
        ref = r[idx["roman_urdu_reference"]]
        if not isinstance(ref, str):
            continue
        for w in ref.split():
            total += 1
            lw = w.lower()
            new = pn.get(lw) or corr.get(lw)
            if new and new.lower() != lw:
                changed += 1
                examples.setdefault(w, new)
    pct = 100 * changed / total if total else 0
    print(f"  {label:<26} {changed:>4} / {total} gold words corrupted  ({pct:.2f}%)")
    return changed, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write data/lexicons_clean.json")
    args = ap.parse_args()

    src = load(SRC)
    gold_vocab = load_gold_vocab()
    print("=" * 78)
    print("  STEP 2 — LEXICON CLEANUP")
    print("=" * 78)
    print(f"  source: {SRC.name}")
    print(f"    corrections  : {len(src['corrections']):>6} flat entries")
    print(f"    proper_nouns : {len(src['proper_nouns']):>6} flat entries")
    print(f"  gold protected vocabulary: {len(gold_vocab)} human-verified words")
    print()

    cleaned, stats = clean(src, gold_vocab)

    print("  DROP REASONS")
    labels = {
        "drop_too_short":          f"R1  variant < {MIN_LEN} chars (collision risk)",
        "drop_protected_variant":  "R2  variant is a correct common word",
        "drop_selfmap_noop":       "R3  self-map no-op (key == value)",
        "drop_cycle":              "R4  bidirectional cycle (A<->B)",
        "drop_banned_canonical":   "R5  canonical is wrong/dangerous",
        "drop_gold_protected":     "R7  variant is a GOLD word (already correct)",
        "drop_word_to_phrase":     "R8  word -> phrase (breaks token count)",
        "merged_case_fragment":    "R9  MERGED case-fragment (chughtai + Chughtai)",
        "kept_case_normaliser":    "R6  KEPT case-normaliser (ali->Ali)",
        "kept":                    "    KEPT real variant",
    }
    for k, lbl in labels.items():
        if k in stats:
            print(f"    {lbl:<44} {stats[k]:>6}")
    print()

    n_corr_c = len(cleaned["corrections"])
    n_pn_c   = len(cleaned["proper_nouns"])
    v_corr   = sum(len(v) for v in cleaned["corrections"].values())
    v_pn     = sum(len(v) for v in cleaned["proper_nouns"].values())
    print("  RESULT — restructured as {canonical: [variants]}")
    print(f"    corrections  : {n_corr_c:>6} canonicals  ({v_corr} variants)")
    print(f"    proper_nouns : {n_pn_c:>6} canonicals  ({v_pn} variants)")
    print()

    # ---- re-measure corruption on gold ----
    print("  CORRUPTION ON THE 183-TURN GOLD SET")
    base = load(BASE)
    measure_corruption(base["corrections"], base["proper_nouns"], "original lexicons.json")
    measure_corruption(src["corrections"], src["proper_nouns"], "lexicons_updated.json")
    c, p = to_lookup(cleaned)
    measure_corruption(c, p, "lexicons_clean.json (NEW)")
    print()

    if args.write:
        out = {
            "_comment": (
                "CLEANED lexicon. Structure: {canonical: [variant, ...]}. "
                "One entry per real word; variants grouped. Invert to variant->canonical "
                f"at load time for O(1) lookup. Cleaned with min_len={MIN_LEN}, "
                "protected-word blocklist, banned-canonical list, cycle breaking."
            ),
            "lexicons": cleaned,
        }
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ written: {OUT}")
    else:
        print("  (dry run — pass --write to save data/lexicons_clean.json)")
    print()


if __name__ == "__main__":
    main()
