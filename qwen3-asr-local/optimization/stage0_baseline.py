#!/usr/bin/env python3
"""
Stage-0 measurement harness — baseline ASR accuracy on real call audio.

Pipeline:
  1. Load ground-truth from xlsx (col 1 = code-switched, col 2 = pure Urdu).
  2. Resample call wav to 16kHz mono.
  3. Chunk into ~CHUNK_S-second segments (HF backend OOMs on Mac CPU
     for 90s+ whole-file audio at float32).
  4. Run Qwen3-ASR (HF, with confidence) on each chunk; concat
     transcripts and WordConf lists.
  5. Transliterate Hindi → Urdu Nastaliq + Roman Urdu.
  6. Score WER + CER against both GT variants.
  7. Persist a JSON dump + append a row to scoreboard.csv so successive
     runs are directly comparable.

This script is the scoreboard. Every later optimization (preprocessing,
language hints, lexicon, fine-tuning) re-runs this and the numbers move.
"""

import os
import sys
import time
import csv
import json
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import pandas as pd
import jiwer
from pydub import AudioSegment

# Make sibling module importable
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from asr_transcribe_and_transliterate import (   # noqa: E402
    get_asr_model,
    hf_asr_with_confidence,
    to_nastaliq,
    to_roman_urdu,
    LANGUAGE,
)

CHUNK_S = int(os.getenv("CHUNK_S", "12"))   # chunk duration in seconds

AUDIO_DIR    = PARENT_DIR / "data" / "audio"
RESULTS_DIR  = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── ground-truth loading ────────────────────────────────────────────────

def load_ground_truth(xlsx_path: Path) -> dict:
    """
    Return dict with keys:
      - turns        : list of (turn_no, speaker, gt_codeswitched, gt_pure_urdu)
      - gt_cs_joined : full code-switched reference as one string
      - gt_pu_joined : full pure-Urdu reference as one string
    """
    df = pd.read_excel(xlsx_path, header=None)
    # Skip first row (filename header). Columns observed: 0 nan, 1 code-switched,
    # 2 pure-urdu, 3 speaker, 4 turn-no.
    turns = []
    for _, row in df.iloc[1:].iterrows():
        cs = row[1]; pu = row[2]; sp = row[3]; tn = row[4]
        if pd.isna(cs) or pd.isna(pu):
            continue
        turns.append((int(tn) if not pd.isna(tn) else len(turns)+1,
                      str(sp) if not pd.isna(sp) else "",
                      str(cs).strip(),
                      str(pu).strip()))
    return {
        "turns":        turns,
        "gt_cs_joined": " ".join(t[2] for t in turns),
        "gt_pu_joined": " ".join(t[3] for t in turns),
    }


# ─── audio prep ──────────────────────────────────────────────────────────

def prep_audio(src_wav: Path, dst_wav: Path) -> dict:
    """Resample to 16kHz mono PCM-16 (what HF ASR expects). Idempotent."""
    seg = AudioSegment.from_wav(src_wav)
    info = {
        "src_rate":     seg.frame_rate,
        "src_channels": seg.channels,
        "duration_s":   len(seg) / 1000.0,
    }
    if not dst_wav.exists():
        seg.set_frame_rate(16000).set_channels(1).set_sample_width(2).export(dst_wav, format="wav")
    return info


# ─── scoring ─────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    # Light normalization that is fair to both variants: lowercase Latin,
    # collapse whitespace, strip common Urdu/Latin punctuation.
    s = s.lower()
    for ch in "،۔.,?!؟:;\"'()[]—–-":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def score(hyp: str, ref: str) -> dict:
    hyp_n, ref_n = normalize(hyp), normalize(ref)
    return {
        "wer": jiwer.wer(ref_n, hyp_n),
        "cer": jiwer.cer(ref_n, hyp_n),
        "hyp_words": len(hyp_n.split()),
        "ref_words": len(ref_n.split()),
    }


# ─── main run ────────────────────────────────────────────────────────────

def run(audio_stem: str):
    src_wav  = AUDIO_DIR / f"{audio_stem}.wav"
    xlsx     = AUDIO_DIR / f"{audio_stem}.xlsx"
    prep_wav = RESULTS_DIR / f"{audio_stem}.16k.wav"

    print(f"\n── Stage 0 baseline ──")
    print(f"Audio  : {src_wav.name}")
    print(f"GT     : {xlsx.name}")

    gt = load_ground_truth(xlsx)
    print(f"Turns  : {len(gt['turns'])}")

    audio_info = prep_audio(src_wav, prep_wav)
    print(f"Audio  : {audio_info['duration_s']:.1f}s, "
          f"{audio_info['src_rate']}Hz→16000Hz, "
          f"{audio_info['src_channels']}ch→1ch")

    print(f"Loading ASR model (language='{LANGUAGE}', backend=HF)...")
    get_asr_model()

    # Chunk into CHUNK_S-second pieces (avoid OOM on long files at fp32 CPU).
    seg = AudioSegment.from_wav(prep_wav).set_frame_rate(16000).set_channels(1).set_sample_width(2)
    n_chunks = max(1, (len(seg) + CHUNK_S*1000 - 1) // (CHUNK_S*1000))
    chunks_dir = RESULTS_DIR / f"{audio_stem}.chunks"; chunks_dir.mkdir(exist_ok=True)
    print(f"Chunks : {n_chunks} × ~{CHUNK_S}s")

    hindi_parts: list[str] = []
    word_confs: list = []
    asr_s = 0.0
    for i in range(n_chunks):
        start_ms = i * CHUNK_S * 1000
        chunk_path = chunks_dir / f"chunk_{i:02d}.wav"
        seg[start_ms : start_ms + CHUNK_S*1000].export(chunk_path, format="wav")
        text_i, t_i, wc_i = hf_asr_with_confidence(chunk_path)
        print(f"  chunk {i+1}/{n_chunks}  {t_i:5.1f}s  {len(wc_i):2d} words  →  {text_i[:80]}")
        hindi_parts.append(text_i)
        word_confs.extend(wc_i)
        asr_s += t_i

    hyp_hindi    = " ".join(p for p in hindi_parts if p)
    hyp_roman    = to_roman_urdu(hyp_hindi)
    hyp_nastaliq = to_nastaliq(hyp_hindi)
    n_flagged    = sum(1 for wc in word_confs if wc.is_low)
    print(f"ASR    : {asr_s:.1f}s total  ({len(word_confs)} words, {n_flagged} flagged)")

    # Score against both GT variants. Hyp options:
    #   - hyp_hindi    : raw Devanagari from ASR
    #   - hyp_nastaliq : transliterated to Urdu script (matches col 1/col 2 script)
    #   - hyp_roman    : Roman Urdu (no GT today, just for inspection)
    scores = {
        "nastaliq_vs_codeswitched": score(hyp_nastaliq, gt["gt_cs_joined"]),
        "nastaliq_vs_pure_urdu":    score(hyp_nastaliq, gt["gt_pu_joined"]),
        "hindi_vs_codeswitched":    score(hyp_hindi,    gt["gt_cs_joined"]),
        "hindi_vs_pure_urdu":       score(hyp_hindi,    gt["gt_pu_joined"]),
    }

    # ── print ───────────────────────────────────────────────────────────
    print("\n── References ──")
    print(f"GT (code-switched) : {gt['gt_cs_joined'][:160]}...")
    print(f"GT (pure Urdu)     : {gt['gt_pu_joined'][:160]}...")
    print("\n── Hypotheses ──")
    print(f"Hindi (ASR)        : {hyp_hindi[:160]}...")
    print(f"Nastaliq           : {hyp_nastaliq[:160]}...")
    print(f"Roman Urdu         : {hyp_roman[:160]}...")
    print("\n── Scores ──")
    header = f"{'comparison':32s} {'WER':>8s} {'CER':>8s} {'hyp_w':>8s} {'ref_w':>8s}"
    print(header); print("─" * len(header))
    for k, v in scores.items():
        print(f"{k:32s} {v['wer']*100:7.2f}% {v['cer']*100:7.2f}% {v['hyp_words']:8d} {v['ref_words']:8d}")
    print(f"\nLow-conf words flagged: {n_flagged}/{len(word_confs)} "
          f"(threshold = {os.getenv('LOW_CONF_THRESHOLD', '0.65')})")

    # ── persist ─────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / f"{audio_stem}.baseline.json"
    out_json.write_text(json.dumps({
        "audio_stem":     audio_stem,
        "language":       LANGUAGE,
        "asr_seconds":    asr_s,
        "audio_info":     audio_info,
        "n_turns_gt":     len(gt["turns"]),
        "n_words_asr":    len(word_confs),
        "n_words_flagged": n_flagged,
        "chunks":         n_chunks,
        "chunk_seconds":  CHUNK_S,
        "hyp_hindi":      hyp_hindi,
        "hyp_nastaliq":   hyp_nastaliq,
        "hyp_roman":      hyp_roman,
        "gt_codeswitched": gt["gt_cs_joined"],
        "gt_pure_urdu":   gt["gt_pu_joined"],
        "scores":         scores,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved  : {out_json.relative_to(PARENT_DIR)}")

    # Append a one-line scoreboard row (so successive runs are comparable).
    sb = RESULTS_DIR / "scoreboard.csv"
    new = not sb.exists()
    with sb.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "audio_stem", "language",
                        "asr_s", "n_words_asr", "n_flagged",
                        "wer_nast_cs", "cer_nast_cs",
                        "wer_nast_pu", "cer_nast_pu",
                        "wer_hindi_cs", "wer_hindi_pu"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                    audio_stem, LANGUAGE, f"{asr_s:.1f}",
                    len(word_confs), n_flagged,
                    f"{scores['nastaliq_vs_codeswitched']['wer']*100:.2f}",
                    f"{scores['nastaliq_vs_codeswitched']['cer']*100:.2f}",
                    f"{scores['nastaliq_vs_pure_urdu']['wer']*100:.2f}",
                    f"{scores['nastaliq_vs_pure_urdu']['cer']*100:.2f}",
                    f"{scores['hindi_vs_codeswitched']['wer']*100:.2f}",
                    f"{scores['hindi_vs_pure_urdu']['wer']*100:.2f}"])
    print(f"Scoreboard appended: {sb.relative_to(PARENT_DIR)}")


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else \
           "in-4234500300-+923000754715-20260501-221336-1777655616.988465"
    run(stem)
