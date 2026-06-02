#!/usr/bin/env python3
"""
compare_upsample.py — Does WavePad 16 kHz upsampling improve Qwen3-ASR?

Runs ASR on both:
  A) 8 kHz original  → pipeline resamples to 16 kHz internally
  B) 16 kHz WavePad-upsampled → fed directly to ASR

Outputs:
  - Per-chunk transcript comparison
  - Aggregate confidence stats
  - Word-level diff (insertions / deletions / common)
  - results/upsample_comparison.json
"""

import os, sys, json, time, difflib
from pathlib import Path
from pydub import AudioSegment

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from asr_transcribe_and_transliterate import (
    get_asr_model,
    hf_asr_with_confidence,
    to_nastaliq,
    to_roman_urdu,
    LANGUAGE,
)

CHUNK_S = int(os.getenv("CHUNK_S", "12"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "0"))  # 0 = no limit; set >0 for quick test
USE_LAST20 = os.getenv("USE_LAST20", "1") == "1"  # use pre-extracted last-20s clips
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

AUDIO_DIR = PARENT_DIR / "data" / "audio"
STEM = "in-4234500300-+923000308246-20260502-195955-1777733995.999525"

FILES = {
    "8kHz_original": AUDIO_DIR / f"{STEM}.WAV",
    "16kHz_wavepad": AUDIO_DIR / f"{STEM}-16khz.WAV.wav",
}

# Pre-extracted last-20s clips (already 16kHz mono)
LAST20_FILES = {
    "8kHz_original": RESULTS_DIR / f"{STEM}.8kHz_original.last20s.wav",
    "16kHz_wavepad": RESULTS_DIR / f"{STEM}.16kHz_wavepad.last20s.wav",
}


# ─── helpers ─────────────────────────────────────────────────────────────


def prep_16k(src: Path, tag: str) -> Path:
    """Ensure audio is 16 kHz mono PCM-16; cache result."""
    dst = RESULTS_DIR / f"{STEM}.{tag}.16k.wav"
    if not dst.exists():
        seg = AudioSegment.from_file(src)
        seg.set_frame_rate(16000).set_channels(1).set_sample_width(2).export(
            dst, format="wav"
        )
        print(f"  Converted {src.name} → {dst.name}")
    else:
        print(f"  Reusing cached {dst.name}")
    return dst


def run_asr(wav_path: Path, tag: str) -> dict:
    seg = (
        AudioSegment.from_wav(wav_path)
        .set_frame_rate(16000)
        .set_channels(1)
        .set_sample_width(2)
    )
    duration_s = len(seg) / 1000.0
    n_chunks = max(1, (len(seg) + CHUNK_S * 1000 - 1) // (CHUNK_S * 1000))
    if MAX_CHUNKS > 0:
        n_chunks = min(n_chunks, MAX_CHUNKS)
    chunks_dir = RESULTS_DIR / f"{STEM}.{tag}.chunks"
    chunks_dir.mkdir(exist_ok=True)

    hindi_parts, word_confs, asr_s = [], [], 0.0
    for i in range(n_chunks):
        start_ms = i * CHUNK_S * 1000
        chunk_path = chunks_dir / f"chunk_{i:02d}.wav"
        seg[start_ms : start_ms + CHUNK_S * 1000].export(chunk_path, format="wav")
        text_i, t_i, wc_i = hf_asr_with_confidence(chunk_path)
        confs = [wc.min_conf for wc in wc_i]
        avg_c = sum(confs) / len(confs) if confs else 0.0
        print(
            f"    chunk {i+1:02d}/{n_chunks}  {t_i:5.1f}s  {len(wc_i):3d}w  avg_conf={avg_c:.3f}  →  {text_i[:70]}"
        )
        hindi_parts.append(text_i)
        word_confs.extend(wc_i)
        asr_s += t_i

    hyp_hindi = " ".join(p for p in hindi_parts if p)
    all_confs = [wc.min_conf for wc in word_confs]
    n_flagged = sum(1 for wc in word_confs if wc.is_low)
    return {
        "tag": tag,
        "duration_s": duration_s,
        "asr_seconds": asr_s,
        "n_words": len(word_confs),
        "n_flagged": n_flagged,
        "avg_conf": sum(all_confs) / len(all_confs) if all_confs else 0.0,
        "min_conf": min(all_confs) if all_confs else 0.0,
        "hyp_hindi": hyp_hindi,
        "hyp_nastaliq": to_nastaliq(hyp_hindi),
        "hyp_roman": to_roman_urdu(hyp_hindi),
    }


def word_diff(a: str, b: str) -> dict:
    aw, bw = a.lower().split(), b.lower().split()
    sm = difflib.SequenceMatcher(None, aw, bw)
    added, removed, common = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            common.extend(aw[i1:i2])
        elif op == "insert":
            added.extend(bw[j1:j2])
        elif op == "delete":
            removed.extend(aw[i1:i2])
        elif op == "replace":
            removed.extend(aw[i1:i2])
            added.extend(bw[j1:j2])
    return {
        "common": len(common),
        "added_in_16k": len(added),
        "removed_in_16k": len(removed),
        "similarity": sm.ratio(),
    }


# ─── main ────────────────────────────────────────────────────────────────


def main():
    print(f"\n{'='*65}")
    print("Upsample Comparison: 8 kHz original vs 16 kHz WavePad")
    print(f"{'='*65}")

    print("\n[Prep] Converting / caching audio...")
    if USE_LAST20:
        print("  Using pre-extracted last-20s clips.")
        wavs = LAST20_FILES
    else:
        wavs = {tag: prep_16k(src, tag) for tag, src in FILES.items()}

    print(f"\n[ASR] Loading model (language='{LANGUAGE}')...")
    get_asr_model()

    results = {}
    for tag, wav_path in wavs.items():
        print(f"\n── Running ASR: {tag} ──")
        t0 = time.time()
        results[tag] = run_asr(wav_path, tag)
        elapsed = time.time() - t0
        r = results[tag]
        print(
            f"  Done in {elapsed:.1f}s | words={r['n_words']} flagged={r['n_flagged']} "
            f"avg_conf={r['avg_conf']:.3f} min_conf={r['min_conf']:.3f}"
        )

    # ── stats comparison ─────────────────────────────────────────────────
    a, b = results["8kHz_original"], results["16kHz_wavepad"]
    diff = word_diff(a["hyp_hindi"], b["hyp_hindi"])

    print(f"\n{'='*65}")
    print("COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"{'Metric':<28} {'8kHz_orig':>14} {'16kHz_WavePad':>14}  {'Δ':>8}")
    print("-" * 68)
    metrics = [
        ("Words transcribed", a["n_words"], b["n_words"], ""),
        ("Low-conf words", a["n_flagged"], b["n_flagged"], "lower=better"),
        ("Avg confidence", a["avg_conf"], b["avg_conf"], "higher=better"),
        ("Min confidence", a["min_conf"], b["min_conf"], "higher=better"),
        ("ASR wall-time (s)", a["asr_seconds"], b["asr_seconds"], ""),
    ]
    for label, va, vb, note in metrics:
        if isinstance(va, float):
            delta = vb - va
            print(f"{label:<28} {va:>14.3f} {vb:>14.3f}  {delta:>+8.3f}  {note}")
        else:
            delta = vb - va
            print(f"{label:<28} {va:>14d} {vb:>14d}  {delta:>+8d}  {note}")

    print(f"\n{'─'*68}")
    print(f"Word-level diff (Hindi tokens):")
    print(f"  Common words        : {diff['common']}")
    print(f"  Added in 16k        : {diff['added_in_16k']}")
    print(f"  Removed in 16k      : {diff['removed_in_16k']}")
    print(f"  Sequence similarity : {diff['similarity']*100:.1f}%")

    print(f"\n{'─'*68}")
    print("Transcripts:")
    for tag, r in results.items():
        print(f"\n  [{tag}]")
        print(f"  Hindi    : {r['hyp_hindi'][:200]}...")
        print(f"  Nastaliq : {r['hyp_nastaliq'][:200]}...")
        print(f"  Roman    : {r['hyp_roman'][:200]}...")

    # ── persist ──────────────────────────────────────────────────────────
    out = RESULTS_DIR / f"{STEM}.upsample_comparison.json"
    out.write_text(
        json.dumps(
            {
                "stem": STEM,
                "language": LANGUAGE,
                "chunk_s": CHUNK_S,
                "results": results,
                "diff": diff,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out.relative_to(PARENT_DIR)}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
