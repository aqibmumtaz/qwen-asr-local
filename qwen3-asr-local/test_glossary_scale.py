#!/usr/bin/env python3
"""
SCALE TEST — does a BIG glossary still bias correctly, or does it dilute?

The 3-term glossary worked (flipped 'app' -> 'lab'). Before scaling to the full
entity list we must answer two questions:

  1. DILUTION — does the real entity still get recognised when it is buried
     among 179 other terms? (soft biasing spreads probability mass)
  2. LEAKAGE  — does a big glossary make the model start parroting terms that
     were never spoken?

Conditions (same audio each time):
  A  none      : context=""                    -> control
  B  tiny      : 3 real entities               -> known-good from earlier test
  C  full      : 179 entities (Tier1+Tier2)    -> the proposed scale-up
  D  full+decoy: 179 + 5 never-spoken decoys   -> leakage check at scale

Usage:
    python3 test_glossary_scale.py
"""

import json
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from qwen_asr import Qwen3ASRModel

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
LEXICON = SCRIPT_DIR / "data" / "lexicons_updated.json"

AUDIO = (
    SCRIPT_DIR / "data" / "CLL analysis"
    / "in-4234500300-+393928520852-20260503-134723-1777798043.1006725"
    / "turn_001_agent.wav"
)
REFERENCE = "ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon"

TINY = "Chughtai Lab, Danish Ali, assalam o alaikum"

# Never spoken in this audio — if these appear, the model is parroting.
DECOYS = "Peshawar, Zubaida Khan, radiology, Faisalabad, Tariq Mehmood"
DECOY_TOKENS = ["peshawar", "zubaida", "radiology", "faisalabad", "tariq", "mehmood",
                "पेशावर", "ज़ुबैदा", "रेडियोलॉजी", "फैसलाबाद", "तारिक"]


def build_full_glossary() -> list[str]:
    """Tier 1 (capitalised proper nouns) + Tier 2 (acronyms) from the lexicon."""
    d = json.loads(LEXICON.read_text())
    canon = set(d["lexicons"]["proper_nouns"].values()) | set(d["lexicons"]["corrections"].values())
    proper  = {t for t in canon if t and t[0].isupper() and len(t) > 2}
    acronym = {t for t in canon if t and t.isupper() and 2 <= len(t) <= 6}
    return sorted(proper | acronym, key=str.lower)


def leaked(text: str) -> list[str]:
    low = text.lower()
    return [d for d in DECOY_TOKENS if d in low]


def main():
    full = build_full_glossary()
    # make sure the entities we care about are present
    for must in ["Chughtai", "Danish", "Ali"]:
        if not any(must.lower() in t.lower() for t in full):
            full.append(must)
    FULL = ", ".join(full)

    conditions = [
        ("A  none",       ""),
        ("B  tiny (3)",   TINY),
        (f"C  full ({len(full)})",  FULL),
        (f"D  full+decoy", f"{FULL}, {DECOYS}"),
    ]

    print("=" * 80)
    print("  GLOSSARY SCALE TEST — does a big list still bias, or dilute?")
    print("=" * 80)
    print(f"  audio     : {AUDIO.name}")
    print(f"  reference : {REFERENCE}")
    print(f"  full glossary: {len(full)} terms | {len(FULL):,} chars | ~{len(FULL)//4:,} tokens")
    print(f"  decoys       : {DECOYS}  <-- must NOT appear\n")

    print(f"  loading {MODEL_ID} ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cpu", max_new_tokens=256,
    )
    print("  loaded.\n", flush=True)

    from hindi_to_roman_urdu import transliterate

    results = {}
    for label, ctx in conditions:
        print(f"  [{label}] running ...", flush=True)
        t0 = time.time()
        out = model.transcribe(audio=[str(AUDIO)], context=[ctx], language=["Hindi"])
        text = out[0].text if out else ""
        el = time.time() - t0
        lk = leaked(text)
        results[label] = (text, transliterate(text), lk)
        warn = f"   ⚠ LEAK {lk}" if lk else ""
        print(f"  [{label}] {el:.0f}s -> {text}{warn}\n", flush=True)

    print("=" * 80)
    print("  ROMAN OUTPUT COMPARISON")
    print("=" * 80)
    for label, (dv, rom, lk) in results.items():
        mark = "⚠" if lk else " "
        print(f" {mark} {label:<16} {rom}")
    print(f"   {'REFERENCE':<16} {REFERENCE}")

    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)

    tiny_rom = results["B  tiny (3)"][1]
    full_key = [k for k in results if k.startswith("C  full")][0]
    full_rom = results[full_key][1]
    none_rom = results["A  none"][1]
    any_leak = any(r[2] for r in results.values())

    tiny_hit = "lab" in tiny_rom.split()
    full_hit = "lab" in full_rom.split()

    print(f"  'lab' recognised —  none: {'lab' in none_rom.split()}   "
          f"tiny: {tiny_hit}   full: {full_hit}")
    print(f"  decoys leaked at scale: {'YES ⚠' if any_leak else 'NO ✓'}")
    print()
    if any_leak:
        print("  ✗ LEAKAGE AT SCALE — big glossary causes parroting. Do NOT scale up.")
    elif full_hit:
        print("  ✓ SCALES CLEANLY — 179-term glossary still biases correctly, no leakage.")
        print("    => Safe to scale the entity list to the full lexicon canonicals.")
    elif tiny_hit and not full_hit:
        print("  ~ DILUTION — the tiny glossary worked but the full one lost the entity.")
        print("    => Soft biasing spreads too thin. Use a SMALL, per-call scoped glossary.")
    else:
        print("  ~ INCONCLUSIVE on this sample — neither size recovered the entity.")
    print()


if __name__ == "__main__":
    main()
