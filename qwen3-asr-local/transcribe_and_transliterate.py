#!/usr/bin/env python3
"""
ASR + transliteration pipeline (Qwen3-ASR → Hindi → Urdu Nastaliq + Roman Urdu)

Calls transcribe.sh for the ASR step (shared with transcribe_and_translate.sh).
Adds two transliterations:
  1. Urdu Nastaliq — GokulNC indo_arabic_transliteration
  2. Roman Urdu   — hindi_to_roman_urdu module

Usage:
    python3 transcribe_and_transliterate.py [--language <name|iso>] [audio.wav]
    python3 transcribe_and_transliterate.py                  # batch: all samples/
    python3 transcribe_and_transliterate.py --language hi audio.wav
"""

import os
import sys
import time
import subprocess
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path
from indo_arabic_transliteration.hindustani import HindustaniTransliterator

SCRIPT_DIR             = Path(__file__).resolve().parent
TRANSCRIBE_SH          = SCRIPT_DIR / "transcribe.sh"
HINDI_TO_ROMAN_URDU_SH = SCRIPT_DIR / "hindi_to_roman_urdu.sh"
SAMPLES                = SCRIPT_DIR / "samples"
OUT_DIR                = SCRIPT_DIR / "transcriptions"
OUT_DIR.mkdir(exist_ok=True)

# Default language (matches transcribe.sh default and app.py env var convention)
DEFAULT_LANGUAGE = os.getenv("ASR_LANGUAGE", "English")

_nastaliq = HindustaniTransliterator()


def run_asr(audio_path: Path, language: str = DEFAULT_LANGUAGE) -> tuple[str, float]:
    """Call transcribe.sh and return (transcript, elapsed_seconds)."""
    t0 = time.time()
    result = subprocess.run(
        ["bash", str(TRANSCRIBE_SH), "--language", language, str(audio_path)],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    transcript = result.stdout.strip()
    return transcript, elapsed


def to_nastaliq(hindi_text: str) -> tuple[str, float]:
    t0 = time.time()
    urdu = _nastaliq.transliterate_from_hindi_to_urdu(hindi_text)
    return urdu, time.time() - t0


def to_roman(hindi_text: str) -> tuple[str, float]:
    """Call hindi_to_roman_urdu.sh and return (roman_urdu, elapsed_seconds)."""
    t0 = time.time()
    result = subprocess.run(
        ["bash", str(HINDI_TO_ROMAN_URDU_SH), hindi_text],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.rstrip('\n'), time.time() - t0


def process_file(audio_path: Path, language: str = DEFAULT_LANGUAGE):
    print(f"\n{'─'*60}")
    print(f"File        : {audio_path.name}")

    transcript, asr_time = run_asr(audio_path, language=language)
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
    args = sys.argv[1:]
    language = DEFAULT_LANGUAGE

    # Optional --language / -l override
    if args and args[0] in ('--language', '-l') and len(args) >= 2:
        language = args[1]
        args = args[2:]

    print(f"ASR language: {language}")

    if args:
        audio = Path(args[0])
        if not audio.exists():
            print(f"Error: file not found: {audio}")
            sys.exit(1)
        process_file(audio, language=language)
    else:
        audio_files = sorted(
            f for f in SAMPLES.iterdir()
            if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        )
        if not audio_files:
            print(f"No audio files found in {SAMPLES}")
            sys.exit(1)
        for f in audio_files:
            process_file(f, language=language)

    print(f"\n{'─'*60}")
    print("Done. Output saved to transcriptions/")


if __name__ == "__main__":
    main()
