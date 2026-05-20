#!/usr/bin/env python3
"""
Stage-1 language-hint sweep — re-runs the Stage-0 harness once per
language hint (Hindi, Arabic, Persian, auto) on the same call audio,
using one shared model load. Appends each run to scoreboard.csv and
prints a side-by-side comparison at the end.

Run after stage0_baseline.py so the chunks are already cached.
"""
import os, sys, time, csv, json, warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import pandas as pd
import jiwer
from pydub import AudioSegment

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from asr_transcribe_and_transliterate import (   # noqa: E402
    get_asr_model,
    hf_asr_with_confidence,
    to_nastaliq,
    to_roman_urdu,
)
from stage0_baseline import (  # noqa: E402
    load_ground_truth, prep_audio, normalize, score, AUDIO_DIR, RESULTS_DIR,
)

CHUNK_S = int(os.getenv("CHUNK_S", "12"))
LANGUAGES = ["Hindi", "Arabic", "Persian", None]   # None = auto-detect


def run_one(language, prep_wav: Path, chunks_dir: Path, n_chunks: int, seg):
    label = language or "auto"
    print(f"\n── language = {label!r} ──")
    hindi_parts, word_confs = [], []
    asr_s = 0.0
    for i in range(n_chunks):
        chunk_path = chunks_dir / f"chunk_{i:02d}.wav"
        if not chunk_path.exists():
            start_ms = i * CHUNK_S * 1000
            seg[start_ms : start_ms + CHUNK_S*1000].export(chunk_path, format="wav")
        text_i, t_i, wc_i = hf_asr_with_confidence(chunk_path, language=language)
        print(f"  chunk {i+1}/{n_chunks}  {t_i:5.1f}s  {len(wc_i):2d} words  →  {text_i[:90]}")
        hindi_parts.append(text_i); word_confs.extend(wc_i); asr_s += t_i

    hyp_hindi    = " ".join(p for p in hindi_parts if p)
    hyp_nastaliq = to_nastaliq(hyp_hindi)
    hyp_roman    = to_roman_urdu(hyp_hindi)
    return {
        "language":      label,
        "asr_s":         asr_s,
        "n_words":       len(word_confs),
        "n_flagged":     sum(1 for w in word_confs if w.is_low),
        "hyp_hindi":     hyp_hindi,
        "hyp_nastaliq":  hyp_nastaliq,
        "hyp_roman":     hyp_roman,
    }


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else \
           "in-4234500300-+923000754715-20260501-221336-1777655616.988465"
    src_wav  = AUDIO_DIR / f"{stem}.wav"
    xlsx     = AUDIO_DIR / f"{stem}.xlsx"
    prep_wav = RESULTS_DIR / f"{stem}.16k.wav"
    chunks_dir = RESULTS_DIR / f"{stem}.chunks"; chunks_dir.mkdir(exist_ok=True)

    gt = load_ground_truth(xlsx)
    audio_info = prep_audio(src_wav, prep_wav)
    seg = AudioSegment.from_wav(prep_wav).set_frame_rate(16000).set_channels(1).set_sample_width(2)
    n_chunks = max(1, (len(seg) + CHUNK_S*1000 - 1) // (CHUNK_S*1000))
    print(f"Audio: {audio_info['duration_s']:.1f}s, chunks: {n_chunks} × ~{CHUNK_S}s")
    print(f"GT turns: {len(gt['turns'])}")

    print("\nLoading ASR model (HF, fp32 CPU) once...")
    get_asr_model()

    results = []
    sb = RESULTS_DIR / "scoreboard.csv"
    for lang in LANGUAGES:
        r = run_one(lang, prep_wav, chunks_dir, n_chunks, seg)
        sc = {
            "nast_cs": score(r["hyp_nastaliq"], gt["gt_cs_joined"]),
            "nast_pu": score(r["hyp_nastaliq"], gt["gt_pu_joined"]),
            "hin_cs":  score(r["hyp_hindi"],    gt["gt_cs_joined"]),
            "hin_pu":  score(r["hyp_hindi"],    gt["gt_pu_joined"]),
        }
        r["scores"] = sc
        results.append(r)

        # append to scoreboard
        new = not sb.exists()
        with sb.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp","audio_stem","language","asr_s",
                            "n_words_asr","n_flagged",
                            "wer_nast_cs","cer_nast_cs",
                            "wer_nast_pu","cer_nast_pu",
                            "wer_hindi_cs","wer_hindi_pu"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), stem, r["language"],
                        f"{r['asr_s']:.1f}", r["n_words"], r["n_flagged"],
                        f"{sc['nast_cs']['wer']*100:.2f}",
                        f"{sc['nast_cs']['cer']*100:.2f}",
                        f"{sc['nast_pu']['wer']*100:.2f}",
                        f"{sc['nast_pu']['cer']*100:.2f}",
                        f"{sc['hin_cs']['wer']*100:.2f}",
                        f"{sc['hin_pu']['wer']*100:.2f}"])

    # Persist per-language json dumps
    sweep_out = RESULTS_DIR / f"{stem}.sweep.json"
    sweep_out.write_text(json.dumps(
        {"audio_stem": stem, "results": [
            {"language": r["language"], "asr_s": r["asr_s"],
             "n_words": r["n_words"], "n_flagged": r["n_flagged"],
             "hyp_hindi": r["hyp_hindi"], "hyp_nastaliq": r["hyp_nastaliq"],
             "hyp_roman": r["hyp_roman"],
             "scores": {k: v for k,v in r["scores"].items()}}
            for r in results]},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # ── side-by-side summary ────────────────────────────────────────────
    print("\n\n══════════════ SUMMARY ══════════════")
    hdr = f"{'language':10s} {'asr_s':>6s} {'words':>5s} {'flag':>5s} "\
          f"{'WER nast/cs':>12s} {'WER nast/pu':>12s} {'CER nast/cs':>12s} {'CER nast/pu':>12s}"
    print(hdr); print("─" * len(hdr))
    for r in results:
        sc = r["scores"]
        print(f"{r['language']:10s} {r['asr_s']:6.1f} {r['n_words']:5d} {r['n_flagged']:5d} "
              f"{sc['nast_cs']['wer']*100:11.2f}% {sc['nast_pu']['wer']*100:11.2f}% "
              f"{sc['nast_cs']['cer']*100:11.2f}% {sc['nast_pu']['cer']*100:11.2f}%")
    print(f"\nJSON dump: {sweep_out.relative_to(PARENT_DIR)}")
    print(f"Scoreboard: {sb.relative_to(PARENT_DIR)}")


if __name__ == "__main__":
    main()
