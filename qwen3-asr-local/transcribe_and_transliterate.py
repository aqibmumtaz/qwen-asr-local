#!/usr/bin/env python3
"""
ASR + transliteration pipeline (Qwen3-ASR → Hindi → Urdu Nastaliq + Roman Urdu)
Replaces the 5–8B LLM translation step with deterministic rule-based converters.

Outputs three forms from a single ASR pass:
  1. Hindi Devanagari  — raw ASR output
  2. Urdu Nastaliq     — GokulNC rule-based (Hindi → Arabic script)
  3. Roman Urdu        — hindi_to_roman_urdu module (Hindi → Latin script, direct)

Usage:
    python3 transcribe_and_transliterate.py [audio.wav]
    python3 transcribe_and_transliterate.py        # batch: all samples/
"""

import subprocess
import sys
import time
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path
from indo_arabic_transliteration.hindustani import HindustaniTransliterator
from hindi_to_roman_urdu import transliterate as to_roman_urdu

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_MODEL  = SCRIPT_DIR / "models/Qwen3-ASR-1.7B-Q8_0-new.gguf"
MMPROJ     = SCRIPT_DIR / "models/mmproj-Qwen3-ASR-1.7B-bf16-new.gguf"
LLAMA_MTMD = SCRIPT_DIR / "llama.cpp/build/bin/llama-mtmd-cli"
SAMPLES    = SCRIPT_DIR / "samples"
OUT_DIR    = SCRIPT_DIR / "transcriptions"
OUT_DIR.mkdir(exist_ok=True)

# ── ASR language config ──────────────────────────────────────────────────────
# Qwen3-ASR's official language-forcing convention: append 'language <Language>'
# + literal '<asr_text>' tag right after the assistant prompt. Equivalent to
# `model.transcribe(..., language="Hindi")` in the qwen_asr Python API.
# Source: https://github.com/QwenLM/Qwen3-ASR — qwen_asr/inference/utils.py
#
# Supported languages (case-sensitive): Chinese, English, Cantonese, Arabic,
# German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian,
# Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish,
# Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian,
# Macedonian. (Urdu NOT supported — use 'Hindi' to bias toward Devanagari.)

# ISO code → full language name (matches app.py / qwen_asr conventions)
LANG_MAP = {
    "en": "English",  "hi": "Hindi",     "ar": "Arabic",   "fa": "Persian",
    "zh": "Chinese",  "yue": "Cantonese","ja": "Japanese", "ko": "Korean",
    "de": "German",   "fr": "French",    "es": "Spanish",  "pt": "Portuguese",
    "id": "Indonesian","ms": "Malay",    "th": "Thai",     "vi": "Vietnamese",
    "tr": "Turkish",  "ru": "Russian",   "it": "Italian",  "nl": "Dutch",
    "sv": "Swedish",  "da": "Danish",    "fi": "Finnish",  "pl": "Polish",
    "cs": "Czech",    "fil": "Filipino", "el": "Greek",    "ro": "Romanian",
    "hu": "Hungarian","mk": "Macedonian",
}

# Default to Hindi (best for Urdu speech given our Devanagari→Roman pipeline).
# Override via env: ASR_LANGUAGE=Hindi  or  ASR_LANGUAGE=hi
_lang_raw = os.getenv("ASR_LANGUAGE", "Hindi")
LANGUAGE = LANG_MAP.get(_lang_raw.lower(), _lang_raw)


def build_asr_prompt(language: str = LANGUAGE) -> str:
    """Build Qwen3-ASR prompt with language forcing."""
    return (
        "<|im_start|>system\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
        f"language {language}<asr_text>"
    )


ASR_PROMPT = build_asr_prompt(LANGUAGE)

_nastaliq = HindustaniTransliterator()


def run_asr(audio_path: Path) -> tuple[str, float]:
    t0 = time.time()
    result = subprocess.run(
        [
            str(LLAMA_MTMD),
            "-m", str(ASR_MODEL),
            "--mmproj", str(MMPROJ),
            "--image", str(audio_path),
            "-p", ASR_PROMPT,
            "-n", "256",
            "--no-warmup",
        ],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    transcript = ""
    for line in result.stdout.splitlines():
        if "<asr_text>" in line:
            transcript = line.split("<asr_text>")[-1].strip()
            break
    return transcript, elapsed


def to_nastaliq(hindi_text: str) -> tuple[str, float]:
    t0 = time.time()
    urdu = _nastaliq.transliterate_from_hindi_to_urdu(hindi_text)
    return urdu, time.time() - t0


def to_roman(hindi_text: str) -> tuple[str, float]:
    t0 = time.time()
    roman = to_roman_urdu(hindi_text)
    return roman, time.time() - t0


def process_file(audio_path: Path):
    print(f"\n{'─'*60}")
    print(f"File        : {audio_path.name}")

    transcript, asr_time = run_asr(audio_path)
    if not transcript:
        print("ERROR       : ASR returned empty transcript")
        return

    nastaliq, nastaliq_ms = to_nastaliq(transcript)
    roman, roman_ms = to_roman(transcript)
    total = asr_time + nastaliq_ms + roman_ms

    print(f"Hindi       : {transcript}")
    print(f"Urdu        : {nastaliq}")
    print(f"Roman Urdu  : {roman}")
    print(
        f"Timing      : ASR={asr_time:.1f}s  "
        f"Nastaliq={nastaliq_ms*1000:.1f}ms  "
        f"Roman={roman_ms*1000:.1f}ms  "
        f"Total={total:.1f}s"
    )

    out_file = OUT_DIR / f"{audio_path.stem}_post_processor.txt"
    out_file.write_text(
        f"Hindi (ASR)          : {transcript}\n"
        f"Urdu Nastaliq        : {nastaliq}\n"
        f"Roman Urdu           : {roman}\n",
        encoding="utf-8"
    )
    print(f"Saved       : {out_file}")


def main():
    global ASR_PROMPT
    args = sys.argv[1:]

    # Optional --language / -l override (CLI > env var > default 'Hindi')
    if args and args[0] in ('--language', '-l') and len(args) >= 2:
        override = LANG_MAP.get(args[1].lower(), args[1])
        ASR_PROMPT = build_asr_prompt(override)
        print(f"ASR language: {override}")
        args = args[2:]
    else:
        print(f"ASR language: {LANGUAGE} (override via --language <name|iso> or ASR_LANGUAGE env)")

    if args:
        audio = Path(args[0])
        if not audio.exists():
            print(f"Error: file not found: {audio}")
            sys.exit(1)
        process_file(audio)
    else:
        audio_files = sorted(
            f for f in SAMPLES.iterdir()
            if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        )
        if not audio_files:
            print(f"No audio files found in {SAMPLES}")
            sys.exit(1)
        for f in audio_files:
            process_file(f)

    print(f"\n{'─'*60}")
    print("Done. Output saved to transcriptions/")


if __name__ == "__main__":
    main()
