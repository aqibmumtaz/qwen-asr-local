#!/usr/bin/env python3
"""
Test script: ASR + post-processor transliteration (Hindi→Urdu via GokulNC rules)
Replaces the 5–8B LLM translation step with a deterministic rule-based converter.

Usage:
    python3 test_post_processor.py [audio.wav]
    python3 test_post_processor.py                  # runs all samples/
"""

import subprocess
import sys
import time
import os
import warnings

# Suppress TensorFlow noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path
from indo_arabic_transliteration.hindustani import HindustaniTransliterator

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_MODEL  = SCRIPT_DIR / "models/Qwen3-ASR-1.7B-Q8_0-new.gguf"
MMPROJ     = SCRIPT_DIR / "models/mmproj-Qwen3-ASR-1.7B-bf16-new.gguf"
LLAMA_MTMD = SCRIPT_DIR / "llama.cpp/build/bin/llama-mtmd-cli"
SAMPLES    = SCRIPT_DIR / "samples"
OUT_DIR    = SCRIPT_DIR / "transcriptions"
OUT_DIR.mkdir(exist_ok=True)

ASR_PROMPT = (
    "<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)

transliterator = HindustaniTransliterator()


def run_asr(audio_path: Path) -> tuple[str, float]:
    """Run Qwen3-ASR and return (transcript, seconds)."""
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


def transliterate(hindi_text: str) -> tuple[str, float]:
    """Convert Hindi Devanagari → Urdu Nastaliq. Returns (urdu, seconds)."""
    t0 = time.time()
    urdu = transliterator.transliterate_from_hindi_to_urdu(hindi_text)
    elapsed = time.time() - t0
    return urdu, elapsed


def process_file(audio_path: Path):
    print(f"\n{'─'*60}")
    print(f"File   : {audio_path.name}")

    transcript, asr_time = run_asr(audio_path)
    if not transcript:
        print("ERROR  : ASR returned empty transcript")
        return

    urdu, trans_time = transliterate(transcript)
    total = asr_time + trans_time

    print(f"Hindi  : {transcript}")
    print(f"Urdu   : {urdu}")
    print(f"Timing : ASR={asr_time:.1f}s  Transliteration={trans_time*1000:.1f}ms  Total={total:.1f}s")

    out_file = OUT_DIR / f"{audio_path.stem}_post_processor.txt"
    out_file.write_text(
        f"Hindi (ASR): {transcript}\nUrdu (transliterated): {urdu}\n",
        encoding="utf-8"
    )
    print(f"Saved  : {out_file}")


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
