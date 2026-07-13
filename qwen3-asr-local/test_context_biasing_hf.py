#!/usr/bin/env python3
"""
LAYER-1 TEST (BASE MODEL) — does ASR context biasing genuinely work?

Uses the OFFICIAL qwen_asr API with the real `context=` argument — NOT the
hand-rolled llama.cpp system slot (which we proved leaks the prompt into output).

Four conditions on the SAME audio:
  A) baseline    context=""                         -> control
  B) biased      context=real entities              -> does it help?
  C) decoy-only  context=entities NOT in the audio  -> do they leak in?
  D) mixed       context=real + decoys              -> the honest test

DECOY LOGIC — this is the whole point:
  If decoy terms (never spoken) appear in the transcript, the model is PARROTING
  the glossary, not biasing recognition. Genuine biasing = decoys stay absent
  while real entities get recognised.

Usage:
    python3 test_context_biasing_hf.py
    python3 test_context_biasing_hf.py --language Hindi
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from qwen_asr import Qwen3ASRModel

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

DEFAULT_AUDIO = (
    SCRIPT_DIR / "data" / "CLL analysis"
    / "in-4234500300-+393928520852-20260503-134723-1777798043.1006725"
    / "turn_001_agent.wav"
)

REFERENCE = "ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon"

# REAL entities — actually spoken in this audio
REAL = "Chughtai Lab, Danish Ali, assalam o alaikum"

# DECOYS — deliberately NOT spoken anywhere in this audio.
# If these appear in the output, the model is parroting the glossary.
DECOYS = "Peshawar, Zubaida Khan, radiology, Faisalabad, Tariq Mehmood"

CONDITIONS = [
    ("A  baseline",   ""),
    ("B  real only",  REAL),
    ("C  decoy only", DECOYS),
    ("D  real+decoy", f"{REAL}, {DECOYS}"),
]

# Individual decoy tokens to scan the output for
DECOY_TOKENS = [
    "peshawar", "zubaida", "radiology", "faisalabad", "tariq", "mehmood",
    "पेशावर", "ज़ुबैदा", "रेडियोलॉजी", "फैसलाबाद", "तारिक",
]


def contains_decoy(text: str) -> list[str]:
    """Return which decoy tokens leaked into the transcript."""
    low = text.lower()
    return [d for d in DECOY_TOKENS if d in low]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(DEFAULT_AUDIO))
    # NOTE: Qwen3-ASR does NOT support "Urdu". Supported list includes Hindi,
    # Persian, Arabic — but not Urdu. This is why the pipeline transcribes to
    # Hindi (Devanagari) and transliterates to Roman Urdu downstream.
    ap.add_argument("--language", default="Hindi")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        sys.exit(f"audio not found: {audio}")

    print("=" * 78)
    print("  LAYER-1 TEST — context biasing on the BASE model (official qwen_asr API)")
    print("=" * 78)
    print(f"  model     : {MODEL_ID}")
    print(f"  audio     : {audio.name}")
    print(f"  language  : {args.language}")
    print(f"  reference : {REFERENCE}")
    print(f"\n  REAL   entities : {REAL}")
    print(f"  DECOY  entities : {DECOYS}   <-- NOT spoken; must NOT appear")

    print(f"\n  loading {MODEL_ID} on CPU ...", flush=True)
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map="cpu",
        max_new_tokens=256,
    )
    print(f"  loaded in {time.time()-t0:.0f}s\n", flush=True)

    results = {}
    for label, ctx in CONDITIONS:
        print(f"  [{label}] running ...", flush=True)
        t0 = time.time()
        out = model.transcribe(
            audio=[str(audio)],
            context=[ctx],
            language=[args.language],
        )
        text = out[0].text if out else ""
        elapsed = time.time() - t0
        leaked = contains_decoy(text)
        results[label] = (ctx, text, leaked)
        flag = f"  ⚠ LEAKED: {leaked}" if leaked else ""
        print(f"  [{label}] {elapsed:.0f}s -> {text}{flag}\n", flush=True)

    # ---- verdict ----
    print("=" * 78)
    print("  RESULTS")
    print("=" * 78)
    for label, (ctx, text, leaked) in results.items():
        print(f"\n  {label}")
        print(f"    context : {ctx or '(empty)'}")
        print(f"    output  : {text or '(empty)'}")
        if leaked:
            print(f"    ⚠ DECOYS LEAKED: {leaked}")

    print(f"\n    REFERENCE: {REFERENCE}")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)

    leak_c = bool(results["C  decoy only"][2])
    leak_d = bool(results["D  real+decoy"][2])

    if leak_c or leak_d:
        print("  ✗ PARROTING — decoy terms that were never spoken appeared in the")
        print("    transcript. The model is copying the glossary, not biasing on it.")
        print("    => Layer 1 is UNSAFE. Do not build the entity path on this.")
    else:
        base = results["A  baseline"][1]
        real = results["B  real only"][1]
        if base.strip() != real.strip():
            print("  ✓ GENUINE BIASING — no decoys leaked, and the real glossary")
            print("    changed the output. Context biasing is working as intended.")
            print("    => Layer 1 is VIABLE. Build the entity path on it.")
        else:
            print("  ~ NO EFFECT — no decoys leaked, but the real glossary also")
            print("    changed nothing. Biasing is safe but not helping here.")
            print("    => Layer 1 inert on this sample; rely on Layers 2+3.")
    print()


if __name__ == "__main__":
    main()
