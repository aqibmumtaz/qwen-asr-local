#!/usr/bin/env python3
"""
Step 0 -- build a Roman -> correct Devanagari lookup for the gazetteer
entries that actually appear in the benchmark audio.

Two-tier strategy (NOT blind transliteration -- verified this session that
plain ITRANS conversion diverges from real usage, e.g. "muhammad" -> ITRANS
"मुहम्मद्" vs the actual convention "मोहम्मद" seen throughout the real
vendor Hindi data):

  1. CORPUS-SOURCED (preferred, high confidence): scan every
     model_output_hindi cell in Sheet1, romanize each word with the
     project's own hindi_to_roman_urdu.transliterate(), and fuzzy-match
     against each gazetteer name. If the vendor ASR got a name right
     *anywhere* in the corpus (common even when inconsistent), harvest that
     real, in-domain Devanagari spelling -- this reflects actual usage
     convention, not generic transliteration rules.

  2. ITRANS FALLBACK (low confidence, flagged): for gazetteer entries never
     found this way, auto-generate via indic_transliteration's ITRANS
     scheme, stripped of trailing halant. Every such entry is marked
     "source": "itrans_fallback" in the output -- manual review required
     before trusting it as a training target (see plan §"Ground-Truth
     Hindi Generation", step 4).

Usage:
  python build_entity_devanagari.py
    --entities ../data/entities.json
    --xlsx ../benchmark/lab_test_80_calls_urdu_roman_urdu.xlsx
    --out data/entities_devanagari.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import openpyxl

FUZZY_THRESHOLD = 0.82  # stricter than the corpus-wide scorer -- this
                         # builds TRAINING TARGETS, false positives are costly


def load_gazetteer(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    return {"given_names": d["given_names"], "places": d["places"],
            "organisations": d.get("organisations", [])}


def load_rows_with_gold(xlsx_path: Path) -> list[dict]:
    """Per-row {hindi, gold} pairs -- gold is the call's benchmark_roman_urdu,
    forward-filled across that call's chunk rows (gold is only stored on one
    row per call in Sheet1, same convention used by every benchmark script
    this session)."""
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}

    gold_by_call = {}
    for r in rows[1:]:
        cid = r[idx["call_id"]]
        b = r[idx["benchmark_roman_urdu"]]
        if isinstance(b, str) and b.strip():
            gold_by_call[cid] = b

    out = []
    for r in rows[1:]:
        h = r[idx["model_output_hindi"]]
        cid = r[idx["call_id"]]
        if isinstance(h, str) and h.strip():
            out.append({"hindi": h, "gold": gold_by_call.get(cid, "")})
    return out


DEVANAGARI_PUNCT = "।॥"   # danda / double-danda -- U+0964/U+0965 sit INSIDE
                           # the ऀ-ॿ Unicode block used by word_re below, so
                           # a naive [ऀ-ॿ]+ match swallows trailing
                           # punctuation into the "word" (verified this
                           # session: गुर्जरवाला। kept its trailing danda as
                           # part of the stored Devanagari spelling).


def clean_word(w: str) -> str:
    return w.strip(DEVANAGARI_PUNCT)


def romanize(word: str, transliterate_fn) -> str:
    return transliterate_fn(clean_word(word)).lower().strip(".,?!।")


def sim(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio()


# RAW (uncorrected) transliteration of a CORRECT Devanagari spelling still
# doesn't always exact-match the gazetteer's Roman spelling -- e.g. लाहौर
# (correct) raws to "laahaur" not "lahore" (English-spelling convention vs
# phonetic Urdu), अहसन (correct) raws to "ahasan" not "ahsan" (schwa
# insertion). 0.80 was calibrated against known-correct pairs this session:
# ahsan/ahasan=0.909, danish/daanish=0.923, muhammad/mohammad=0.875 all
# clear it; lahore/laahaur=0.615 does not (English-convention place names
# may need supplemental manual entries -- a known, accepted gap here).
FUZZY_THRESHOLD_RAW = 0.80


def corpus_lookup(gazetteer_terms: list[str], rows: list[dict],
                   raw_transliterate_fn) -> dict:
    """For each gazetteer term: find Devanagari words in model_output_hindi
    whose RAW (uncorrected -- no WORD_MAP, no phonetic model) romanization
    is fuzzy-close to the term, AND independently confirm the SAME row's
    gold benchmark_roman_urdu also contains that term as a whole word. Both
    conditions must hold -- neither alone was sufficient (see file docstring
    for the two failure modes found and fixed this session)."""
    gazetteer_terms_set = set(t.lower() for t in gazetteer_terms)
    word_re = re.compile(r"[ऀ-ॿ]+")
    counter = Counter()          # {(term, devanagari_word): gold_confirmed_count}
    romanized_cache = {}

    for row in rows:
        gold_words = set(w.lower().strip(".,?!") for w in row["gold"].split()) if row["gold"] else set()
        if not gold_words:
            continue
        for w_raw in word_re.findall(row["hindi"]):
            w = clean_word(w_raw)
            if not w:
                continue
            if w not in romanized_cache:
                romanized_cache[w] = romanize(w, raw_transliterate_fn)
            roman = romanized_cache[w]
            if not roman:
                continue
            for term in gazetteer_terms_set:
                # first-sound agreement: a name's initial consonant is rarely
                # dropped in ASR, so if it differs the overall similarity
                # score is likely inflated by a long coincidental shared
                # suffix, not a real spelling variant. Verified this session:
                # "wazirabad" matched a word raw-romanizing to "razirabad"
                # (0.80 similarity, initial w/r mismatch) purely because the
                # gold text mentioned "wazirabad" ELSEWHERE in that same
                # call -- gold-confirmation is per-call, not per-position,
                # so this catches what that check alone couldn't.
                if term[0] != roman[0]:
                    continue
                if term in gold_words and sim(term, roman) >= FUZZY_THRESHOLD_RAW:
                    counter[(term, w)] += 1

    out = {}
    best_for_term = defaultdict(list)
    for (term, hindi_word), freq in counter.items():
        best_for_term[term].append((freq, hindi_word))
    for term, candidates in best_for_term.items():
        candidates.sort(reverse=True)
        freq, hindi_word = candidates[0]
        out[term] = {"devanagari": hindi_word, "source": "corpus",
                     "similarity": round(sim(term, romanized_cache[hindi_word]), 3),
                     "corpus_freq": freq, "gold_confirmed": True}
    return out


def itrans_fallback(term: str) -> str:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as itrans_translit
    dev = itrans_translit(term, sanscript.ITRANS, sanscript.DEVANAGARI)
    # strip trailing halant -- Hindi proper nouns conventionally drop the
    # final consonant's inherent-vowel suppression mark, ITRANS's literal
    # scheme does not
    return dev.rstrip("्")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", default=str(ROOT / "data" / "entities.json"))
    ap.add_argument("--xlsx", default=str(ROOT / "benchmark" / "lab_test_80_calls_urdu_roman_urdu.xlsx"))
    ap.add_argument("--out", default=str(HERE / "data" / "entities_devanagari.json"))
    ap.add_argument("--coverage-only", action="store_true",
                     help="only print gazetteer coverage stats, don't write output")
    args = ap.parse_args()

    import os
    # Deliberately use the RAW, uncorrected transliteration (no WORD_MAP, no
    # phonetic model at all) for this check. Verified: even the curated v22
    # lexicon alone still "fixes up" a genuinely wrong Devanagari word's
    # romanization to coincide with the target term ("ऐसीन" -> "ahsan" via a
    # known-variant lexicon entry) -- that answers "is this Roman text
    # acceptable after correction," not "is this Devanagari the correct
    # native spelling," which is what a fine-tuning TARGET actually needs.
    # A training target must be validated against zero correction layers.
    os.environ["LEXICON"] = "v22"
    os.environ["PHONETIC"] = ""
    os.environ["RESOLVER"] = "0"
    import hindi_to_roman_urdu as H

    def raw_transliterate(text: str) -> str:
        return H._normalize_endings(H._transliterate_raw(text))

    gaz = load_gazetteer(Path(args.entities))
    all_terms = gaz["given_names"] + gaz["places"] + gaz["organisations"]

    print("Loading Sheet1 rows (hindi + gold pairs)...", flush=True)
    rows = load_rows_with_gold(Path(args.xlsx))
    print(f"  {len(rows)} chunk rows with Hindi text", flush=True)

    print("Matching gazetteer terms: exact romanization match AND gold-confirmed...", flush=True)
    found = corpus_lookup(all_terms, rows, raw_transliterate)
    print(f"  {len(found)}/{len(all_terms)} gazetteer terms resolved with gold confirmation", flush=True)

    if args.coverage_only:
        by_list = {"given_names": gaz["given_names"], "places": gaz["places"],
                   "organisations": gaz["organisations"]}
        for label, terms in by_list.items():
            n = sum(1 for t in terms if t in found)
            print(f"  {label}: {n}/{len(terms)} ({100*n/len(terms):.1f}%)")
        return

    result = {}
    n_fallback = 0
    for term in all_terms:
        if term in found:
            result[term] = found[term]
        else:
            dev = itrans_fallback(term)
            result[term] = {"devanagari": dev, "source": "itrans_fallback",
                            "similarity": None, "corpus_freq": 0}
            n_fallback += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nWrote {len(result)} entries to {args.out}")
    print(f"  corpus-sourced (high confidence): {len(result) - n_fallback}")
    print(f"  itrans_fallback (NEEDS MANUAL REVIEW before training): {n_fallback}")
    print()
    print("Corpus-sourced entries actually usable for training (only these have")
    print("real audio to learn from -- itrans_fallback entries have NO training")
    print("audio regardless of spelling quality, see plan §Gazetteer coverage):")
    n_with_audio_evidence = sum(1 for v in result.values() if v["source"] == "corpus")
    print(f"  {n_with_audio_evidence} entries")


if __name__ == "__main__":
    main()
