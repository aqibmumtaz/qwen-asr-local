#!/usr/bin/env python3
"""
Corrector — post-correction stage for the Roman Urdu ASR pipeline.

    transliterate()  →  raw Roman Urdu words + confidence scores
                     →  Corrector.fix(words)
                     →  corrected Roman Urdu sentence

Design:
  - High-confidence words are passed through unchanged (never sent to LLM).
  - Low-confidence words are marked and sent to the selected backend.
  - Qwen backend: ONE LLM call for the full turn — LLM sees all context.
  - mT5  backend: one model call per flagged word with ±3 word context.

Usage:
    from corrector import Corrector
    from asr_transcribe_and_transliterate import WordConf

    corrector = Corrector(backend='qwen')
    corrected_sentence = corrector.fix(word_conf_list)
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("corrector")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

DEFAULT_QWEN_MODEL = MODELS_DIR / "Qwen3-4B-Q4_K_M.gguf"
DEFAULT_MT5_MODEL  = "google/mt5-small"

# Confidence gate thresholds
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.65"))  # min_conf
GEO_THRESHOLD  = float(os.getenv("GEO_THRESHOLD",  "0.90"))  # geo_conf


# ── Word input type (matches WordConf from asr_transcribe_and_transliterate) ──
@dataclass
class WordEntry:
    """
    One word's data coming out of transliterate() + ASR confidence.
    Compatible with WordConf — pass WordConf objects directly or build these.
    """
    roman:    str    # Roman Urdu from transliterate()
    hindi:    str    # original Hindi (Devanagari) from ASR
    min_conf: float  # minimum sub-token confidence
    geo_conf: float  # geometric mean confidence

    @property
    def needs_fix(self) -> bool:
        return self.min_conf < CONF_THRESHOLD or self.geo_conf < GEO_THRESHOLD


def _from_word_conf(wc) -> "WordEntry":
    """Convert a WordConf (from asr_transcribe_and_transliterate) to WordEntry."""
    from hindi_to_roman_urdu import transliterate
    return WordEntry(
        roman=transliterate(wc.text),
        hindi=wc.text,
        min_conf=wc.min_conf,
        geo_conf=wc.geo_conf,
    )


# ── Domain glossary — injected into every Qwen prompt ────────────────────────
# Built from incorrect_words column across all 183 eval turns.
# Each entry maps a garbled ASR word/phrase → correct Roman Urdu.
# Extend this as more calls are annotated.
DOMAIN_GLOSSARY = """
Known domain entities and common ASR corrections for this call-center:
  - es + romalikoom = assalam o alaikum   (Islamic greeting, often split/garbled)
  - assalam / assalaam = assalam o alaikum
  - walekum / waalekum = walaikum
  - daanishli / danishli = danish ali     (person name — TWO words: first + last name)
  - chukaai / chugai / chukkai = chughtai (LAB name — ONE word, always before "lab"; NOT a person name)
  - karun / karun. = kar raha hoon        (verb: "am talking/doing")
  - kontoon = kya                         (question word)
  - aisan / aise men = aise mein
  - baalak = walid                        (father)
  - kaaiting = waiting
  - lipting = slip                        (report slip)
  - sinpal = sample
  - bradar / baradar = brother
  - pahle = pehle                         (before/earlier)
  - aapka / aapke = aap ka / aap ke
"""


# ── Qwen backend — HuggingFace transformers, model loaded once ────────────────
class QwenBackend:
    """
    Runs Qwen3 LLM via HuggingFace transformers.
    Model loads once, stays in memory for all turns.
    One LLM call per turn — sees full annotated sentence with [FIX:] markers.
    High-confidence words are never sent to the LLM.
    """

    MODEL_ID = "Qwen/Qwen3-0.6B"  # testing — swap to Qwen3-4B for production

    # System prompt: role + domain glossary + output format rules
    _SYSTEM = (
        "/no_think\n"
        "You are an Urdu ASR post-corrector for a medical call-center transcription system.\n"
        "Words marked [FIX:word] are low-confidence and likely garbled or wrong.\n"
        "Unmarked words are correct — copy them exactly as-is.\n\n"
        f"{DOMAIN_GLOSSARY}\n"
        "Rules:\n"
        "  1. Fix [FIX:...] words using the glossary above and your Urdu knowledge.\n"
        "  2. Multiple consecutive [FIX:] words may together form one phrase "
        "(e.g. [FIX:es] [FIX:romalikoom] = assalam o alaikum).\n"
        "  3. A single garbled word may expand to multiple words.\n"
        "  4. Output ONLY the corrected Roman Urdu sentence. Nothing else."
    )

    def __init__(
        self,
        model_id:     str          = MODEL_ID,
        max_tokens:   int          = 512,
        device:       Optional[str] = None,
        use_guardrail: bool         = True,   # re-inserts dropped high-conf words
    ):
        self.model_id      = model_id
        self.max_tokens    = max_tokens  # 512 needed — 0.6B uses think block before answering
        self.use_guardrail = use_guardrail
        self._tok          = None
        self._model        = None
        self._device       = device or self._pick_device()
        self._load()

    @staticmethod
    def _pick_device() -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # MPS has a 2GB per-buffer cap — 4B float16 (~8GB) and 1.7B float16 (~3.2GB)
        # both exceed it. CPU float32 is the stable choice on Apple Silicon for >1B.
        return "cpu"

    def _load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        log.info(f"Loading {self.model_id} on {self._device} ...")
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype, device_map=self._device,
        )
        self._model.eval()
        log.info(f"QwenBackend ready — {self.model_id} on {self._device}.")

    def _annotate(self, words: List[WordEntry]) -> str:
        """Build annotated sentence: low-conf → [FIX:word], high-conf → plain."""
        return " ".join(
            f"[FIX:{w.roman}]" if w.needs_fix else w.roman
            for w in words
        )

    # Few-shot examples — show the model exactly how to apply glossary entries.
    # Required for small models (0.6B) to disambiguate similar-sounding proper nouns.
    _EXAMPLES = [
        (
            "ji [FIX:es] [FIX:romalikoom] [FIX:chukaai] lab se [FIX:daanishli] baat [FIX:karun]",
            "ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon"
        ),
        (
            "[FIX:kontoon] [FIX:maine] Muhammad [FIX:aisan] baat kar raha hoon",
            "kya assalam o alaikum Muhammad aise mein baat kar raha hoon"
        ),
    ]

    def _build_prompt(self, annotated: str) -> str:
        few_shot = ""
        for ex_in, ex_out in self._EXAMPLES:
            few_shot += (
                f"<|im_start|>user\n"
                f"ASR text: {ex_in}\n"
                f"Corrected:<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{ex_out}<|im_end|>\n"
            )
        return (
            f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
            f"{few_shot}"
            f"<|im_start|>user\n"
            f"ASR text: {annotated}\n"
            f"Corrected:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _parse_output(self, raw: str) -> str:
        """Strip think block, prompt echoes, non-Latin script. Return first clean line."""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*",          "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^corrected:\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"^output:\s*",    "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\[FIX:([^\]]+)\]", r"\1", raw)
        lines = [
            l.strip() for l in raw.splitlines()
            if l.strip() and all(ord(c) < 0x0600 or c.isspace() for c in l)
        ]
        return lines[0] if lines else ""

    def _reinsert_dropped(self, corrected: str, words: List[WordEntry]) -> str:
        """
        Guardrail: re-insert any high-conf word the LLM dropped.

        Uses the original word order as a guide:
        - If a dropped word had no high-conf word before it (i.e. it was first
          or only preceded by flagged words) → prepend it.
        - Otherwise → insert it immediately after the nearest preceding
          high-conf word that IS present in the output.

        This preserves the original relative order of clean words without
        touching the LLM's corrections.
        """
        out_tokens = corrected.split()
        out_lower  = [t.lower() for t in out_tokens]

        # Build ordered list of (original_index, roman) for high-conf words only
        clean_indexed = [
            (i, w.roman) for i, w in enumerate(words) if not w.needs_fix
        ]

        for idx, word in clean_indexed:
            if word.lower() in out_lower:
                continue  # already present — nothing to do

            # Find the closest preceding high-conf word that IS in the output
            prev_anchor = None
            for _, prev_word in reversed(
                [(j, w) for j, w in clean_indexed if j < idx]
            ):
                if prev_word.lower() in out_lower:
                    prev_anchor = prev_word
                    break

            if prev_anchor is None:
                # No preceding anchor → this word belongs at the very start
                out_tokens.insert(0, word)
            else:
                # Insert immediately AFTER the last occurrence of prev_anchor
                anchor_pos = max(
                    i for i, t in enumerate(out_tokens)
                    if t.lower() == prev_anchor.lower()
                )
                out_tokens.insert(anchor_pos + 1, word)

            out_lower = [t.lower() for t in out_tokens]  # refresh

        return " ".join(out_tokens)

    def correct(self, words: List[WordEntry]) -> str:
        if not any(w.needs_fix for w in words):
            return " ".join(w.roman for w in words)

        import torch
        annotated = self._annotate(words)
        prompt    = self._build_prompt(annotated)
        inputs    = self._tok(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self._tok.eos_token_id,
                eos_token_id=self._tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        raw       = self._tok.decode(new_tokens, skip_special_tokens=True).strip()
        corrected = self._parse_output(raw)

        if not corrected:
            return " ".join(w.roman for w in words)   # full fallback

        if self.use_guardrail:
            corrected = self._reinsert_dropped(corrected, words)

        return corrected


# ── mT5 backend — one call per flagged word ───────────────────────────────────
class MT5Backend:
    """
    Fine-tuned mT5-small for word-level correction.

    High-confidence words are skipped entirely. For each low-confidence word,
    mT5 receives a context window and outputs the corrected word(s).
    Returns the full sentence with fixes applied.
    """

    def __init__(self, model_path: str = DEFAULT_MT5_MODEL, device: str = "cpu"):
        self.model_path = model_path
        self.device     = device
        self._tok   = None
        self._model = None
        self._load()

    def _load(self):
        try:
            from transformers import MT5ForConditionalGeneration, AutoTokenizer
            log.info(f"Loading mT5 from '{self.model_path}' ...")
            self._tok   = AutoTokenizer.from_pretrained(self.model_path)
            self._model = MT5ForConditionalGeneration.from_pretrained(
                self.model_path
            ).to(self.device)
            self._model.eval()
            log.info("mT5 backend ready.")
        except Exception as exc:
            log.warning(f"mT5 unavailable ({exc}) — words will pass through unchanged")

    def _fix_word(self, word: WordEntry, ctx_before: str, ctx_after: str) -> str:
        if self._model is None:
            return word.roman
        import torch
        inp = f"fix: {ctx_before} [{word.roman}] {ctx_after}"
        enc = self._tok(inp, return_tensors="pt",
                        truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=20, num_beams=4)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()

    def correct(self, words: List[WordEntry]) -> str:
        result = []
        for i, w in enumerate(words):
            if not w.needs_fix:
                # HIGH CONFIDENCE — leave exactly as-is
                result.append(w.roman)
                continue
            # low confidence — fix with context window ±3 words
            ctx_before = " ".join(words[j].roman for j in range(max(0, i-3), i))
            ctx_after  = " ".join(words[j].roman for j in range(i+1, min(len(words), i+4)))
            fixed = self._fix_word(w, ctx_before, ctx_after)
            result.append(fixed if fixed else w.roman)
        return " ".join(result)


# ── Main Corrector class ──────────────────────────────────────────────────────
class Corrector:
    """
    Post-correction stage. Call fix() after transliterate().

    Args:
        backend:    'qwen' — Qwen3 LLM via llama.cpp (one call, full turn)
                    'mt5'  — mT5-small fine-tuned (one call per flagged word)
        model_path: .gguf path for qwen, HF id or local dir for mt5
    """

    def __init__(self, backend: str = "qwen", model_path: Optional[str] = None,
                 use_guardrail: bool = True):
        self.backend_name = backend
        if backend == "qwen":
            mid = model_path if model_path else QwenBackend.MODEL_ID
            self._backend = QwenBackend(model_id=mid, use_guardrail=use_guardrail)
        elif backend == "mt5":
            mp = model_path or DEFAULT_MT5_MODEL
            self._backend = MT5Backend(model_path=mp)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'qwen' or 'mt5'.")

    def fix(self, words) -> str:
        """
        Fix low-confidence words and return corrected Roman Urdu sentence.

        Args:
            words: List of WordEntry or WordConf (from asr_transcribe_and_transliterate)

        Returns:
            Corrected Roman Urdu sentence. High-confidence words are unchanged.
        """
        # Accept WordConf objects from asr_transcribe_and_transliterate
        if words and not isinstance(words[0], WordEntry):
            words = [_from_word_conf(w) for w in words]

        if not words:
            return ""

        # Short-circuit: if nothing is flagged, skip the LLM entirely
        if not any(w.needs_fix for w in words):
            return " ".join(w.roman for w in words)

        return self._backend.correct(words)

    def fix_with_transliterate(self, word_confs) -> str:
        """
        Convenience: transliterate + fix in one call.
        Pass the raw WordConf list from hf_asr_with_confidence().
        Returns corrected Roman Urdu sentence.
        """
        entries = [_from_word_conf(wc) for wc in word_confs]
        return self.fix(entries)
