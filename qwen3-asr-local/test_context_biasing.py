#!/usr/bin/env python3
"""
LAYER-1 TEST — does ASR context biasing work on the local Qwen3-ASR?

Runs the SAME audio through llama.cpp twice:
  A) baseline  — system prompt EMPTY (what transcribe.sh does today)
  B) biased    — system prompt = Roman entity glossary

The only variable is the system slot. We then check the three outcomes:
  A) model emits Latin  "Chughtai"     -> transliterator passes it through  = BEST
  B) model emits cleaner Devanagari    -> fuzzy-match layer resolves it     = OK
  C) no change                         -> cross-script biasing didn't take  = fallback

Usage:
    python3 test_context_biasing.py
    python3 test_context_biasing.py --audio path/to.wav --language Urdu
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LLAMA_MTMD = SCRIPT_DIR / "llama.cpp" / "build" / "bin" / "llama-mtmd-cli"
ASR_MODEL  = SCRIPT_DIR / "models" / "Qwen3-ASR-1.7B-Q8_0-new.gguf"
MMPROJ     = SCRIPT_DIR / "models" / "mmproj-Qwen3-ASR-1.7B-bf16-new.gguf"

DEFAULT_AUDIO = (
    SCRIPT_DIR / "data" / "CLL analysis"
    / "in-4234500300-+393928520852-20260503-134723-1777798043.1006725"
    / "turn_001_agent.wav"
)

# Gold reference for turn_001_agent.wav
REFERENCE = "ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon"

# The entity glossary — CANONICAL forms only, derived from the call data.
# Deliberately small and targeted (soft biasing dilutes as the list grows).
GLOSSARY = (
    "Chughtai Lab, Danish Ali, Muhammad Ahsan, Abdurrahman, Ehtisham, Asim, Ammar, "
    "Sialkot, Lahore, Karachi, "
    "assalam o alaikum, wa alaikum salam, "
    "appointment, report, CNIC, NADRA, nephrology, cardiology, ultrasound"
)


def build_prompt(system: str, language: str) -> str:
    """Exact Qwen3-ASR chat format. `system` is the context/biasing slot."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\nlanguage {language}<asr_text>"
    )


def run_asr(audio: Path, system: str, language: str) -> tuple[str, float]:
    """Run llama-mtmd-cli, return (transcript, elapsed_s)."""
    prompt = build_prompt(system, language)
    cmd = [
        str(LLAMA_MTMD),
        "-m", str(ASR_MODEL),
        "--mmproj", str(MMPROJ),
        "--image", str(audio),
        "-p", prompt,
        "-n", "256",
        "--no-warmup",
    ]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    # transcript is on the line containing <asr_text>
    text = ""
    for line in out.stdout.splitlines():
        if "<asr_text>" in line:
            text = line.split("<asr_text>")[-1].strip()
            break
    return text, elapsed


def analyse(baseline: str, biased: str) -> None:
    """Classify the result into outcome A / B / C."""
    def has_latin(s):   return bool(re.search(r"[A-Za-z]{3,}", s))
    def has_deva(s):    return bool(re.search(r"[ऀ-ॿ]", s))

    print(f"\n{'='*74}")
    print("  OUTCOME ANALYSIS")
    print(f"{'='*74}")

    changed = baseline.strip() != biased.strip()
    latin_new = has_latin(biased) and not has_latin(baseline)

    print(f"  Output changed at all?     {'YES' if changed else 'NO'}")
    print(f"  Baseline has Latin words?  {'YES' if has_latin(baseline) else 'NO'}")
    print(f"  Biased   has Latin words?  {'YES' if has_latin(biased) else 'NO'}")
    print(f"  Biased   has Devanagari?   {'YES' if has_deva(biased) else 'NO'}")
    print()

    if latin_new:
        print("  ==> OUTCOME A  — model now emits LATIN for biased entities.")
        print("      Transliterator passes Latin through unchanged => entities land correct.")
        print("      ACTION: build the entity path on context biasing. Lexicon entity")
        print("              enumeration can be deleted entirely.")
    elif changed:
        print("  ==> OUTCOME B  — output changed, but still Devanagari.")
        print("      Biasing IS taking effect on token probabilities, just within script.")
        print("      ACTION: keep biasing (it reduces variance) + add the fuzzy-match")
        print("              resolver to map the cleaner Devanagari to canonicals.")
    else:
        print("  ==> OUTCOME C  — no change from the Roman glossary.")
        print("      Cross-script biasing did not take.")
        print("      ACTION: retry with a DEVANAGARI glossary (bias the script the model")
        print("              actually emits), else rely on Layers 2+3.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(DEFAULT_AUDIO))
    ap.add_argument("--language", default="Urdu")
    ap.add_argument("--glossary", default=GLOSSARY)
    args = ap.parse_args()

    audio = Path(args.audio)
    for p, name in [(LLAMA_MTMD, "llama-mtmd-cli"), (ASR_MODEL, "ASR model"),
                    (MMPROJ, "mmproj"), (audio, "audio")]:
        if not p.exists():
            sys.exit(f"ERROR: {name} not found: {p}")

    print(f"{'='*74}")
    print("  LAYER-1 TEST — ASR context biasing (with vs without glossary)")
    print(f"{'='*74}")
    print(f"  audio     : {audio.name}")
    print(f"  language  : {args.language}")
    print(f"  reference : {REFERENCE}")
    print(f"\n  glossary  : {args.glossary}")

    print(f"\n{'-'*74}")
    print("  [A] BASELINE — empty system prompt")
    print(f"{'-'*74}")
    baseline, t1 = run_asr(audio, "", args.language)
    print(f"  output ({t1:.1f}s): {baseline or '(empty)'}")

    print(f"\n{'-'*74}")
    print("  [B] BIASED — glossary in system prompt")
    print(f"{'-'*74}")
    biased, t2 = run_asr(audio, args.glossary, args.language)
    print(f"  output ({t2:.1f}s): {biased or '(empty)'}")

    analyse(baseline, biased)

    # Show what each becomes after transliteration
    try:
        from hindi_to_roman_urdu import transliterate
        print(f"{'-'*74}")
        print("  AFTER transliterate()")
        print(f"{'-'*74}")
        print(f"  baseline  -> {transliterate(baseline)}")
        print(f"  biased    -> {transliterate(biased)}")
        print(f"  REFERENCE -> {REFERENCE}")
        print()
    except Exception as e:
        print(f"  (transliterate unavailable: {e})")


if __name__ == "__main__":
    main()
