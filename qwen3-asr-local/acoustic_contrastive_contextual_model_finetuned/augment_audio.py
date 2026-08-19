#!/usr/bin/env python3
"""
Step 2 -- augment POSITIVE (name-bearing) training examples to counter the
small dataset. Negatives are already plentiful and left mostly unaugmented.

Implemented with numpy + soundfile only (both already in this repo's
environment) rather than the audiomentations dependency listed in
requirements.txt -- simpler, one fewer thing to install on the GPU box.
requirements.txt keeps audiomentations as optional/commented if a richer
augmentation set is wanted later.

Augmentations, each applied independently (so one positive example can
produce up to 3 x 3 x 3 = 27 variants if all are combined -- default here
is a smaller, fixed set of realistic combinations, not the full cross
product, to avoid over-multiplying a small dataset into near-duplicates):
  - speed: 0.9x, 1.0x (original), 1.1x  (linear resample, same technique
    already used elsewhere in this repo for 24kHz resampling)
  - noise: none, light office/line noise at ~20dB SNR (synthetic white
    noise -- swap in real recorded noise clips if available on the GPU box,
    synthetic is a placeholder, not claimed to be realistic)
  - gain: -3dB, 0dB, +3dB

Usage:
  python augment_audio.py --in data/train.jsonl --out data/train_augmented.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 42


def load_wav(path: Path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return audio
    n_new = int(round(len(audio) / factor))
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, n_new)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return audio + noise


def apply_gain(audio: np.ndarray, db: float) -> np.ndarray:
    if db == 0.0:
        return audio
    factor = 10 ** (db / 20)
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


# fixed, realistic combinations -- not the full cross product (would
# over-multiply a small dataset into near-duplicate variants)
VARIANTS = [
    {"speed": 1.0, "noise_db": None, "gain_db": 0.0, "suffix": "orig"},
    {"speed": 0.9, "noise_db": None, "gain_db": 0.0, "suffix": "speed09"},
    {"speed": 1.1, "noise_db": None, "gain_db": 0.0, "suffix": "speed11"},
    {"speed": 1.0, "noise_db": 20.0, "gain_db": 0.0, "suffix": "noise20db"},
    {"speed": 1.0, "noise_db": None, "gain_db": -3.0, "suffix": "gainm3"},
    {"speed": 1.0, "noise_db": None, "gain_db": 3.0, "suffix": "gainp3"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(HERE / "data" / "train.jsonl"))
    ap.add_argument("--out", default=str(HERE / "data" / "train_augmented.jsonl"))
    ap.add_argument("--audio-out-dir", default=str(HERE / "data" / "augmented_audio"))
    ap.add_argument("--augment-negatives", action="store_true",
                     help="also augment negative examples (default: positives only)")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    out_audio_dir = Path(args.audio_out_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    examples = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(examples)} examples from {args.inp}", flush=True)

    out_examples = []
    n_augmented = 0
    for ex in examples:
        do_augment = (ex["type"] == "positive") or args.augment_negatives
        variants = VARIANTS if do_augment else VARIANTS[:1]  # negatives: original only

        src_path = ROOT / ex["audio_path"] if not Path(ex["audio_path"]).is_absolute() else Path(ex["audio_path"])
        try:
            audio, sr = load_wav(src_path)
        except Exception as e:
            print(f"  WARN: failed to load {src_path}: {e}", flush=True)
            continue

        for v in variants:
            if v["suffix"] == "orig":
                out_examples.append(ex)
                continue
            a = speed_perturb(audio, v["speed"])
            if v["noise_db"] is not None:
                a = add_noise(a, v["noise_db"], rng)
            a = apply_gain(a, v["gain_db"])

            stem = Path(ex["audio_path"]).stem
            call_dir = out_audio_dir / ex["call_id"]
            call_dir.mkdir(parents=True, exist_ok=True)
            out_path = call_dir / f"{stem}_{v['suffix']}.wav"
            sf.write(str(out_path), a, sr)

            new_ex = dict(ex)
            new_ex["audio_path"] = str(out_path.relative_to(ROOT))
            new_ex["augmentation"] = v["suffix"]
            out_examples.append(new_ex)
            n_augmented += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in out_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_examples)} examples to {args.out}")
    print(f"  ({len(examples)} original + {n_augmented} augmented variants)")


if __name__ == "__main__":
    main()
