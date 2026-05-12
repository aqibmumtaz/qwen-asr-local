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

# Force ASR to transcribe in Hindi Devanagari script.
# Qwen3-ASR's official language-forcing convention is to append
# 'language <Language>' + the literal '<asr_text>' tag right after the
# assistant prompt. The model then emits the transcription directly.
# Supported languages incl: Chinese, English, Hindi, Arabic, Persian, etc.
# Source: https://github.com/QwenLM/Qwen3-ASR — qwen_asr/inference/utils.py
ASR_PROMPT = (
    "<|im_start|>system\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
    "language Hindi<asr_text>"
)

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
    if len(sys.argv) > 1:
        audio = Path(sys.argv[1])
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
