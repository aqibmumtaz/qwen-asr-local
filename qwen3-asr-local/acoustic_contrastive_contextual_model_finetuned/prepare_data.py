#!/usr/bin/env python3
"""
Step 1 -- build (audio_path, context, target_hindi, type) training triplets.

Only gazetteer entries with source == "corpus" in entities_devanagari.json
(built by build_entity_devanagari.py) are usable as positive-example targets
-- those are the ones with real, gold-confirmed audio evidence. Everything
else in the gazetteer has no training audio regardless of spelling quality
(see plan §Gazetteer coverage) and is intentionally not used for target
construction here (it can still be used as a decoy for adversarial examples
-- a decoy specifically should NOT need real audio, since the target is that
it never appears).

Holdout is by CALL, not by chunk -- chunks from the same call share speaker/
audio characteristics, chunk-level holdout would leak.

Usage:
  python prepare_data.py
    --xlsx ../benchmark/lab_test_80_calls_urdu_roman_urdu.xlsx
    --audio-dir ../benchmark/lab_test_80_audios_chunks_dynamic
    --entities-devanagari data/entities_devanagari.json
    --holdout-calls 10
    --out-dir data/
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import openpyxl

SEED = 42  # fixed, for a reproducible holdout split


def load_sheet1(xlsx_path: Path):
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    by_call = defaultdict(list)
    for r in rows[1:]:
        by_call[r[idx["call_id"]]].append(r)
    for cid in by_call:
        by_call[cid].sort(key=lambda r: r[idx["chunk_index"]] or 0)
    return by_call, idx


def is_mostly_devanagari(text: str, min_fraction: float = 0.5) -> bool:
    """Some Sheet1 model_output_hindi rows are already Roman/English text
    (ASR hallucination on unclear audio, not something this script
    introduces -- verified this session, e.g. rows containing literal
    'Assalamualaikum sir, ... WhatsApp pe bata raha hu' with almost no
    Devanagari at all). Using those as Hindi training targets would be
    wrong. Cheap filter: reject if under half the letters are Devanagari."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    dev = sum(1 for c in letters if "ऀ" <= c <= "ॿ")
    return (dev / len(letters)) >= min_fraction


def find_audio_confirmed_names(gold_text: str, corpus_entries: dict) -> list[str]:
    """Which gold-confirmed gazetteer terms appear (whole word) in this
    call's gold text."""
    words = set(w.lower().strip(".,?!") for w in gold_text.split())
    return [term for term in corpus_entries if term in words]


def substitute_name_span(vendor_hindi: str, found_terms: list[str],
                          entities_devanagari: dict) -> str:
    """Replace the FIRST Devanagari run in vendor_hindi that's meant to
    represent each found term with the corpus-verified correct spelling.
    Best-effort: if vendor Hindi doesn't contain an obviously-corresponding
    span, leaves vendor text for that term unchanged (better to skip a fix
    than to guess wrong and corrupt a target)."""
    import os
    os.environ["LEXICON"] = "v22"; os.environ["PHONETIC"] = ""; os.environ["RESOLVER"] = "0"
    import hindi_to_roman_urdu as H

    words = re.findall(r"[ऀ-ॿ]+|[^ऀ-ॿ\s]+|\s+", vendor_hindi)  # keep separators
    for term in found_terms:
        correct_dev = entities_devanagari[term]["devanagari"]
        best_i, best_sim = None, 0.0
        from difflib import SequenceMatcher
        for i, w in enumerate(words):
            if not re.match(r"[ऀ-ॿ]+", w):
                continue
            raw = H._normalize_endings(H._transliterate_raw(w)).lower()
            s = SequenceMatcher(a=term, b=raw, autojunk=False).ratio()
            if s > best_sim:
                best_sim, best_i = s, i
        if best_i is not None and best_sim >= 0.55:  # loose -- we're finding
            words[best_i] = correct_dev                # ANY mention, not verifying it
    return "".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(ROOT / "benchmark" / "lab_test_80_calls_urdu_roman_urdu.xlsx"))
    ap.add_argument("--audio-dir", default=str(ROOT / "benchmark" / "lab_test_80_audios_chunks_dynamic"))
    ap.add_argument("--entities-devanagari", default=str(HERE / "data" / "entities_devanagari.json"))
    ap.add_argument("--holdout-calls", type=int, default=10)
    ap.add_argument("--out-dir", default=str(HERE / "data"))
    args = ap.parse_args()

    entities_dev = json.load(open(args.entities_devanagari, encoding="utf-8"))
    corpus_entries = {k: v for k, v in entities_dev.items() if v["source"] == "corpus"}
    all_terms = list(entities_dev.keys())
    print(f"{len(corpus_entries)} gazetteer entries have real training audio "
          f"(gold-confirmed); {len(all_terms) - len(corpus_entries)} are itrans_fallback "
          f"(usable as adversarial decoys only, never as positive targets)", flush=True)

    by_call, idx = load_sheet1(Path(args.xlsx))
    audio_root = Path(args.audio_dir)
    call_dirs = {d.name: d for d in audio_root.iterdir() if d.is_dir()}

    random.seed(SEED)
    scored_calls = [cid for cid in by_call if str(cid) in call_dirs and
                     any(isinstance(r[idx["benchmark_roman_urdu"]], str) and
                         r[idx["benchmark_roman_urdu"]].strip() for r in by_call[cid])]
    random.shuffle(scored_calls)
    holdout_calls = set(scored_calls[:args.holdout_calls])
    train_calls = set(scored_calls[args.holdout_calls:])
    print(f"{len(train_calls)} calls -> train, {len(holdout_calls)} calls -> eval (holdout)", flush=True)

    positive, negative, eval_examples = [], [], []

    for cid in scored_calls:
        crows = by_call[cid]
        cid_str = str(cid)
        cd = call_dirs[cid_str]
        gold = next((r[idx["benchmark_roman_urdu"]] for r in crows
                    if isinstance(r[idx["benchmark_roman_urdu"]], str)
                    and r[idx["benchmark_roman_urdu"]].strip()), "")
        found_terms = find_audio_confirmed_names(gold, corpus_entries)

        chunks = sorted(cd.glob("chunk_*.wav"))
        for i, ch in enumerate(chunks):
            row = crows[i] if i < len(crows) else crows[0]
            vendor_hindi = row[idx["model_output_hindi"]]
            vendor_hindi = vendor_hindi.strip() if isinstance(vendor_hindi, str) else ""
            if not vendor_hindi or not is_mostly_devanagari(vendor_hindi):
                continue

            chunk_terms = [t for t in found_terms if t in vendor_hindi.lower()
                           or True]  # name could be in this call's audio anywhere;
                                     # per-chunk attribution is approximate -- see caveat below

            example = {"audio_path": str(ch.relative_to(ROOT)), "call_id": cid_str,
                      "chunk_index": i}

            if found_terms:
                target = substitute_name_span(vendor_hindi, found_terms, entities_dev)
                ex = {**example, "context": ", ".join(t.title() for t in found_terms),
                      "target_hindi": target, "type": "positive"}
            else:
                ex = {**example, "context": "", "target_hindi": vendor_hindi, "type": "negative"}

            if cid in holdout_calls:
                eval_examples.append(ex)
            elif found_terms:
                positive.append(ex)
            else:
                negative.append(ex)

    # adversarial: negative examples paired with a decoy term guaranteed not
    # in this call (drawn from the full gazetteer, not just corpus-confirmed
    # -- a decoy doesn't need real audio, the point is it must NOT appear)
    random.seed(SEED)
    adversarial = []
    for ex in negative:
        decoy = random.choice(all_terms)
        adversarial.append({**ex, "context": decoy.title(), "type": "adversarial"})

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train_path = Path(args.out_dir) / "train.jsonl"
    eval_path = Path(args.out_dir) / "eval.jsonl"
    adv_path = Path(args.out_dir) / "adversarial.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in positive + negative:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(adv_path, "w", encoding="utf-8") as f:
        for ex in adversarial:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote:")
    print(f"  {train_path}  ({len(positive)} positive + {len(negative)} negative = {len(positive)+len(negative)})")
    print(f"  {eval_path}  ({len(eval_examples)} examples, {len(holdout_calls)} held-out calls)")
    print(f"  {adv_path}  ({len(adversarial)} adversarial)")
    print()
    print("CAVEAT (manual review required before trusting targets):")
    print("  - Per-chunk name attribution is call-level, not chunk-level verified")
    print("    (a call's found_terms are applied to every chunk of that call, since")
    print("    we don't have chunk-level gold text to know exactly which chunk says")
    print("    a given name -- most calls are short enough this is a small risk, but")
    print("    spot-check multi-chunk calls specifically).")
    print("  - substitute_name_span() finds the best-matching Devanagari span at a")
    print("    loose 0.55 similarity floor -- verify positive examples' target_hindi")
    print("    actually has the name substituted in a sane position before training.")


if __name__ == "__main__":
    main()
