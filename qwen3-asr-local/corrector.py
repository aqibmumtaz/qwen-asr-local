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
  - daanishli / danishli = danish ali     (agent name)
  - chukaai / chugai / chukkai = chughtai (Chughtai Lab — medical lab)
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
        "You are an Urdu ASR post-corrector for a medical call-center transcription system.\n\n"
        f"{DOMAIN_GLOSSARY}\n"
        "You will receive garbled word groups labeled GROUP_1, GROUP_2, etc.\n"
        "For each group output the correct Roman Urdu replacement — pipe-separated.\n"
        "Rules:\n"
        "  1. Use the glossary above and your Urdu knowledge.\n"
        "  2. A group may be one garbled word or multiple consecutive garbled words "
        "that together form one phrase (e.g. 'es romalikoom' = 'assalam o alaikum').\n"
        "  3. A single garbled word may expand to multiple words.\n"
        "  4. Output ONLY the pipe-separated corrections in order. Nothing else.\n"
        "  5. If a group cannot be corrected, repeat it unchanged."
    )

    def __init__(
        self,
        model_id:   str          = MODEL_ID,
        max_tokens: int          = 512,
        device:     Optional[str] = None,
    ):
        self.model_id   = model_id
        self.max_tokens = max_tokens  # 512 needed — 0.6B uses think block before answering
        self._tok       = None
        self._model     = None
        self._device    = device or self._pick_device()
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

    def _build_groups(self, words: List[WordEntry]):
        """
        Split words into alternating high-conf and flagged segments.
        Returns list of (is_flagged, [words]) tuples.
        High-conf words are code-controlled — never touched by LLM.
        Consecutive flagged words form one group (one LLM correction).
        """
        segments = []
        i = 0
        while i < len(words):
            if not words[i].needs_fix:
                # collect consecutive high-conf words
                run = []
                while i < len(words) and not words[i].needs_fix:
                    run.append(words[i])
                    i += 1
                segments.append((False, run))
            else:
                # collect consecutive flagged words as one group
                run = []
                while i < len(words) and words[i].needs_fix:
                    run.append(words[i])
                    i += 1
                segments.append((True, run))
        return segments

    def _build_prompt(self, segments) -> str:
        """
        Ask LLM to output only pipe-separated corrections for each flagged group.
        High-conf words are not shown to LLM as output targets.
        """
        flagged_groups = [grp for is_fix, grp in segments if is_fix]
        if not flagged_groups:
            return ""

        lines = []
        for i, grp in enumerate(flagged_groups, 1):
            garbled = " ".join(w.roman for w in grp)
            lines.append(f"GROUP_{i}: {garbled}")

        groups_text = "\n".join(lines)
        expected    = " | ".join(f"correction_{i}" for i in range(1, len(flagged_groups)+1))

        return (
            f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Correct these garbled Roman Urdu word groups:\n"
            f"{groups_text}\n\n"
            f"Output format: {expected}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _parse_fixes(self, raw: str, segments) -> List[str]:
        """
        Parse pipe-separated LLM output into a list of n_groups corrections.
        Falls back to original garbled words if parsing fails.
        """
        # Strip think block
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*",          "", raw, flags=re.DOTALL).strip()
        # Strip Nastaliq/Devanagari lines
        lines = [
            l.strip() for l in raw.splitlines()
            if l.strip() and all(ord(c) < 0x0600 or c.isspace() for c in l)
        ]
        raw = lines[0] if lines else ""

        fixes = [f.strip() for f in raw.split("|")]

        # Fallback: if wrong number, use original garbled words per group
        flagged_groups = [grp for is_fix, grp in segments if is_fix]
        result = []
        for i, grp in enumerate(flagged_groups):
            original = " ".join(w.roman for w in grp)
            result.append(fixes[i] if i < len(fixes) and fixes[i] else original)
        return result

    def _reconstruct(self, segments, fixes: List[str]) -> str:
        """
        Assemble final sentence — code-controlled, not model-controlled:
          - High-conf segments → ALWAYS original words verbatim (LLM output ignored)
          - Flagged segments   → LLM correction from fixes list (with fallback)

        This guarantees no unmarked word can ever be dropped, moved, or altered
        regardless of what the LLM outputs.
        """
        result = []
        fix_idx = 0
        for is_fix, grp in segments:
            if not is_fix:
                # HIGH CONF — code writes these, not the LLM
                result.extend(w.roman for w in grp)
            else:
                # FLAGGED — use LLM fix; fall back to original if empty/invalid
                fix = fixes[fix_idx] if fix_idx < len(fixes) else ""
                if not fix or not fix.strip():
                    fix = " ".join(w.roman for w in grp)  # safe fallback
                result.append(fix.strip())
                fix_idx += 1
        return " ".join(result)

    def correct(self, words: List[WordEntry]) -> str:
        if not any(w.needs_fix for w in words):
            return " ".join(w.roman for w in words)

        import torch
        segments = self._build_groups(words)
        prompt   = self._build_prompt(segments)

        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
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
        raw   = self._tok.decode(new_tokens, skip_special_tokens=True).strip()
        fixes = self._parse_fixes(raw, segments)
        return self._reconstruct(segments, fixes)


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

    def __init__(self, backend: str = "qwen", model_path: Optional[str] = None):
        self.backend_name = backend
        if backend == "qwen":
            mid = model_path if model_path else QwenBackend.MODEL_ID
            self._backend = QwenBackend(model_id=mid)
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
