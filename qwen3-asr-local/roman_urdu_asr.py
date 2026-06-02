#!/usr/bin/env python3
"""
RomanUrduASR — unified single-call pipeline:

    Audio
      → Qwen3-ASR 1.7B        (Hindi text + per-word confidence)
      → transliterate()        (deterministic Hindi → Roman Urdu)
      → Corrector              (ONLY for low-confidence words)
          'qwen'  — Qwen3 LLM via llama.cpp, zero training needed
          'mt5'   — fine-tuned mT5-small (requires trained weights)
           None   — skip correction, return raw transliteration
      → Corrected Roman Urdu

High-confidence words bypass the corrector entirely — they are never
sent to the LLM, keeping latency low and leaving correct words unchanged.

Detection gate (either condition → flagged):
    min_conf < CONF_THRESHOLD (default 0.65)   — acoustically uncertain
    geo_conf < GEO_THRESHOLD  (default 0.90)   — linguistically risky

Usage:
    from roman_urdu_asr import RomanUrduASR

    pipe = RomanUrduASR(corrector='qwen')
    result = pipe.transcribe('call.wav')
    pipe.print_result(result)
    print(result.roman_corrected)
"""

import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
MODELS_DIR  = SCRIPT_DIR / "models"
LLAMA_CLI   = SCRIPT_DIR / "llama.cpp" / "build" / "bin" / "llama-cli"

DEFAULT_QWEN_MODEL = MODELS_DIR / "Qwen3-4B-Q4_K_M.gguf"
DEFAULT_MT5_MODEL  = "google/mt5-small"

# Confidence gate thresholds (env-overridable)
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.65"))
GEO_THRESHOLD  = float(os.getenv("GEO_THRESHOLD",  "0.90"))

log = logging.getLogger("roman-urdu-asr")


# ── Result types ──────────────────────────────────────────────────────────────
@dataclass
class WordResult:
    """Per-word result after the full pipeline."""
    hindi:      str
    roman_raw:  str     # transliterate(hindi) — deterministic, pre-correction
    roman:      str     # final roman (= roman_raw if not corrected)
    min_conf:   float
    geo_conf:   float
    n_tokens:   int
    flagged:    bool    # sent to corrector (low confidence)
    corrected:  bool    # corrector changed the word


@dataclass
class ASRResult:
    """Full pipeline output for one audio file."""
    audio_file:      str
    hindi:           str
    roman_raw:       str       # full turn, pre-correction
    roman_corrected: str       # full turn, post-correction
    words:           List[WordResult]
    elapsed_asr:     float
    elapsed_correct: float
    n_flagged:       int
    n_corrected:     int


# ── Qwen LLM corrector (llama.cpp) ────────────────────────────────────────────
class QwenCorrector:
    """
    Corrects a single flagged Roman Urdu word using Qwen3 LLM via llama.cpp.

    Only called for words that fail the confidence gate. High-confidence
    words never reach this — they pass through unchanged.
    """

    _SYSTEM = (
        "/no_think\n"
        "You are an Urdu ASR post-processor. "
        "A speech recognition model produced a garbled Roman Urdu word. "
        "Your job: output ONLY the corrected Roman Urdu word or words. "
        "No explanation. No punctuation. Nothing else."
    )

    def __init__(
        self,
        model_path: Path = DEFAULT_QWEN_MODEL,
        n_predict:  int  = 20,
        n_ctx:      int  = 512,
    ):
        self.model_path = Path(model_path)
        self.n_predict  = n_predict
        self.n_ctx      = n_ctx

        if not self.model_path.exists():
            raise FileNotFoundError(f"Qwen model not found: {self.model_path}")
        if not LLAMA_CLI.exists():
            raise FileNotFoundError(f"llama-cli not found: {LLAMA_CLI}")

        log.info(f"QwenCorrector ready: {self.model_path.name}")

    def _build_prompt(
        self,
        roman_word: str,
        hindi_word: str,
        ctx_before: str,
        ctx_after:  str,
        min_conf:   float,
        geo_conf:   float,
    ) -> str:
        user = (
            f"Roman Urdu context: ...{ctx_before} [[[{roman_word}]]] {ctx_after}...\n"
            f"Hindi ASR output for this word: {hindi_word}\n"
            f"Confidence scores: min={min_conf:.2f}  geo={geo_conf:.2f}\n\n"
            f"The word inside [[[ ]]] is garbled. "
            f"What is the correct Roman Urdu word (or words) for it?"
        )
        return (
            f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def correct(
        self,
        roman_word: str,
        hindi_word: str,
        ctx_before: str,
        ctx_after:  str,
        min_conf:   float,
        geo_conf:   float,
    ) -> str:
        prompt = self._build_prompt(
            roman_word, hindi_word, ctx_before, ctx_after, min_conf, geo_conf
        )
        cmd = [
            str(LLAMA_CLI),
            "-m", str(self.model_path),
            "--ctx-size", str(self.n_ctx),
            "-n", str(self.n_predict),
            "--temp", "0.0",
            "--no-warmup",
            "-p", prompt,
            "--log-disable",
        ]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            raw = out.stdout.strip()
            # Strip any stray <think>...</think> blocks
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # Strip echoed prompt if llama-cli printed it
            if "<|im_start|>assistant" in raw:
                raw = raw.split("<|im_start|>assistant")[-1].strip()
            # Take first non-empty line only
            for line in raw.splitlines():
                line = line.strip().rstrip(".,;:")
                if line:
                    return line
            return roman_word
        except subprocess.TimeoutExpired:
            log.warning(f"QwenCorrector timed out on '{roman_word}'")
            return roman_word
        except Exception as exc:
            log.warning(f"QwenCorrector error on '{roman_word}': {exc}")
            return roman_word


# ── mT5 corrector (HuggingFace) ───────────────────────────────────────────────
class MT5Corrector:
    """
    mT5-small fine-tuned on (corrupted Roman Urdu → clean Roman Urdu) pairs.

    Pass model_path to a trained local checkpoint, or use the default
    'google/mt5-small' as a placeholder (untrained — no real corrections).

    Training data: (roman_urdu_model → roman_urdu_reference) from the xlsx,
    augmented with error-injected synthetic pairs.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MT5_MODEL,
        device:     str = "cpu",
    ):
        self.model_path = model_path
        self.device     = device
        self._tok   = None
        self._model = None
        self._load()

    def _load(self):
        try:
            from transformers import MT5ForConditionalGeneration, AutoTokenizer
            import torch
            log.info(f"Loading mT5 corrector from '{self.model_path}' ...")
            self._tok   = AutoTokenizer.from_pretrained(self.model_path)
            self._model = MT5ForConditionalGeneration.from_pretrained(
                self.model_path
            ).to(self.device)
            self._model.eval()
            log.info("mT5 corrector loaded.")
        except Exception as exc:
            log.warning(f"mT5 corrector unavailable: {exc}  (words will pass through)")

    def correct(
        self,
        roman_word: str,
        hindi_word: str,
        ctx_before: str,
        ctx_after:  str,
        min_conf:   float,
        geo_conf:   float,
    ) -> str:
        if self._model is None:
            return roman_word
        import torch
        inp = f"fix: {ctx_before} [{roman_word}] {ctx_after}"
        enc = self._tok(
            inp, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=20, num_beams=4)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()


# ── Main pipeline class ───────────────────────────────────────────────────────
class RomanUrduASR:
    """
    Unified ASR pipeline. See module docstring for full architecture.

    Args:
        corrector:       'qwen' | 'mt5' | None
        model_path:      path to corrector model
                         Qwen → .gguf file  (default: Qwen3-4B-Q4_K_M.gguf)
                         mT5  → HF id or local dir
        conf_threshold:  min_conf below which word is flagged (default 0.65)
        geo_threshold:   geo_conf below which word is flagged (default 0.90)
        language:        ASR language hint passed to Qwen3-ASR (default 'Urdu')
    """

    def __init__(
        self,
        corrector:      Optional[str] = "qwen",
        model_path:     Optional[str] = None,
        conf_threshold: float         = CONF_THRESHOLD,
        geo_threshold:  float         = GEO_THRESHOLD,
        language:       str           = "Urdu",
    ):
        self.conf_threshold = conf_threshold
        self.geo_threshold  = geo_threshold
        self.language       = language
        self.corrector_type = corrector
        self._asr_fn        = None   # lazy-loaded on first call

        if corrector == "qwen":
            mp = Path(model_path) if model_path else DEFAULT_QWEN_MODEL
            self._corrector = QwenCorrector(model_path=mp)
        elif corrector == "mt5":
            mp = model_path or DEFAULT_MT5_MODEL
            self._corrector = MT5Corrector(model_path=mp)
        else:
            self._corrector = None
            log.info("No corrector — pipeline returns raw transliteration.")

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _load_asr(self):
        if self._asr_fn is None:
            from asr_transcribe_and_transliterate import hf_asr_with_confidence
            self._asr_fn = hf_asr_with_confidence
        return self._asr_fn

    def _flagged(self, min_conf: float, geo_conf: float) -> bool:
        return min_conf < self.conf_threshold or geo_conf < self.geo_threshold

    # ── Public API ────────────────────────────────────────────────────────────
    def transcribe(self, audio_path) -> ASRResult:
        """
        Run the full pipeline on one audio file.
        Returns ASRResult with both raw and corrected Roman Urdu.
        High-confidence words are never sent to the corrector.
        """
        audio_path = Path(audio_path)
        from hindi_to_roman_urdu import transliterate

        # ── 1. ASR → Hindi + per-word confidence ─────────────────────────────
        asr_fn = self._load_asr()
        hindi, elapsed_asr, word_confs = asr_fn(audio_path, language=self.language)

        if not hindi:
            return ASRResult(
                audio_file=str(audio_path), hindi="", roman_raw="",
                roman_corrected="", words=[], elapsed_asr=elapsed_asr,
                elapsed_correct=0.0, n_flagged=0, n_corrected=0,
            )

        # ── 2. Transliterate every word (deterministic, fast) ─────────────────
        words: List[WordResult] = []
        for wc in word_confs:
            roman_raw = transliterate(wc.text)
            words.append(WordResult(
                hindi=wc.text,
                roman_raw=roman_raw,
                roman=roman_raw,
                min_conf=wc.min_conf,
                geo_conf=wc.geo_conf,
                n_tokens=wc.n_tokens,
                flagged=self._flagged(wc.min_conf, wc.geo_conf),
                corrected=False,
            ))

        raw_roman = " ".join(w.roman_raw for w in words) or transliterate(hindi)

        # ── 3. Correct flagged words only ─────────────────────────────────────
        t0 = time.time()
        n_flagged = n_corrected = 0

        if self._corrector and words:
            for i, w in enumerate(words):
                if not w.flagged:
                    continue   # HIGH CONF — skip corrector entirely
                n_flagged += 1

                ctx_before = " ".join(wj.roman for wj in words[max(0, i-3):i])
                ctx_after  = " ".join(wj.roman for wj in words[i+1:i+4])

                fixed = self._corrector.correct(
                    roman_word=w.roman_raw,
                    hindi_word=w.hindi,
                    ctx_before=ctx_before,
                    ctx_after=ctx_after,
                    min_conf=w.min_conf,
                    geo_conf=w.geo_conf,
                )
                if fixed and fixed != w.roman_raw:
                    w.roman     = fixed
                    w.corrected = True
                    n_corrected += 1
                    log.debug(f"  '{w.roman_raw}' → '{fixed}'  (min={w.min_conf:.2f})")

        elapsed_correct = time.time() - t0

        return ASRResult(
            audio_file=str(audio_path),
            hindi=hindi,
            roman_raw=raw_roman,
            roman_corrected=" ".join(w.roman for w in words),
            words=words,
            elapsed_asr=elapsed_asr,
            elapsed_correct=elapsed_correct,
            n_flagged=n_flagged,
            n_corrected=n_corrected,
        )

    def print_result(self, result: ASRResult):
        """Pretty-print the pipeline result to stdout."""
        print(f"\n{'═'*72}")
        print(f"  {Path(result.audio_file).name}")
        print(f"{'═'*72}")
        print(f"  Hindi        │ {result.hindi}")
        print(f"  Roman raw    │ {result.roman_raw}")
        print(f"  Roman fixed  │ {result.roman_corrected}")
        print(f"  ASR time     │ {result.elapsed_asr:.1f}s")
        if self._corrector:
            print(
                f"  Correction   │ {result.n_flagged} flagged, "
                f"{result.n_corrected} changed  ({result.elapsed_correct:.2f}s)"
            )
        if result.words:
            print()
            hdr = f"  {'Hindi':<15} {'Raw Roman':<18} {'Fixed':<18} {'min':>6} {'geo':>6}  Flag"
            print(hdr)
            print(f"  {'-'*15} {'-'*18} {'-'*18} {'-'*6} {'-'*6}  ----")
            for w in result.words:
                flag  = "✓ fixed" if w.corrected else ("⚠ flagged" if w.flagged else "")
                fixed = w.roman if w.corrected else "—"
                print(
                    f"  {w.hindi[:15]:<15} {w.roman_raw[:18]:<18} "
                    f"{fixed[:18]:<18} {w.min_conf:>6.3f} {w.geo_conf:>6.3f}  {flag}"
                )
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="RomanUrduASR — ASR + transliteration + LLM correction"
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--corrector", choices=["qwen", "mt5", "none"], default="qwen",
        help="Corrector backend (default: qwen)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Corrector model path: .gguf for qwen, HF id/dir for mt5"
    )
    parser.add_argument("--language", default="Urdu")
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help=f"min_conf flag threshold (default {CONF_THRESHOLD})"
    )
    parser.add_argument(
        "--geo", type=float, default=GEO_THRESHOLD,
        help=f"geo_conf flag threshold (default {GEO_THRESHOLD})"
    )
    args = parser.parse_args()

    pipe = RomanUrduASR(
        corrector=None if args.corrector == "none" else args.corrector,
        model_path=args.model,
        conf_threshold=args.conf,
        geo_threshold=args.geo,
        language=args.language,
    )
    result = pipe.transcribe(args.audio)
    pipe.print_result(result)
