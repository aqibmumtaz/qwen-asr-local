#!/usr/bin/env python3
"""
Step 5 -- runtime inference: pass1 (base) -> flag spans -> pass2 (LoRA) ->
align -> splice. This is what actually gets used to transcribe a chunk, NOT
a raw adapter-attached transcribe() call -- see plan §"Runtime Design" for
why a raw swap was rejected (LoRA's effect is whole-sequence via attention,
not token-local; a raw swap can't structurally guarantee non-entity words
are unaffected, only this splice design can).

*** NOT RUN -- needs a trained adapter (GPU) to exercise end-to-end. The
alignment/splice logic (align_words, splice) is pure Python/no ML and could
be unit-tested locally, but wasn't in this session (no adapter exists yet
to generate a real pass-2 output to test against). ***

Imports NameRetriever from acoustic_contextual_biasing/ rather than
duplicating gazetteer-similarity logic -- that directory stays the reusable
library, this project only adds what's new (the splice orchestration and
the LoRA-attached second pass).

Usage (as a library, not typically run standalone):
  from splice_inference import SpliceASR
  asr = SpliceASR(adapter_path="adapters/run1/phase3")
  text = asr.transcribe(audio_chunk_path)
"""
from __future__ import annotations

import sys
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from train_lora import build_model, build_prompt, resume_lora

CONFIDENCE_THRESHOLD = 0.7   # per-token generation confidence below this ->
                              # flag for pass-2 re-check. NOT verified this
                              # is a good threshold -- qwen_asr's exact
                              # confidence/logprob exposure needs checking
                              # on the GPU machine, see _flag_low_confidence
                              # below for the fallback if it isn't exposed.


class SpliceASR:
    def __init__(self, adapter_path: str, gazetteer_path: str = None):
        from acoustic_contextual_biasing.retriever import NameRetriever

        self.base_wrapper, self.device = build_model()
        # separate model instance for the adapter path -- NOT toggling
        # enable/disable_adapter_layers() on one shared instance, because
        # that requires wrapper.model to already BE a PeftModel at load
        # time for both branches; two instances is simpler and correct at
        # the cost of 2x memory, revisit if VRAM is tight on deployment.
        lora_wrapper, _ = build_model()
        self.lora_wrapper = resume_lora(lora_wrapper, adapter_path)
        self.lora_wrapper.model.eval()
        self.base_wrapper.model.eval()

        self.retriever = NameRetriever(device="cpu")

    def _pass1(self, audio_path: Path) -> tuple[list[str], list[float]]:
        """Base model, no context. Returns (words, per_word_confidence).
        Confidence source: qwen_asr's generate() output_scores if exposed
        (verify on GPU machine -- may need
        model.generate(..., output_scores=True, return_dict_in_generate=True)
        and a softmax-over-logits-at-argmax computation per token, then
        aggregate sub-word tokens back to whole words). If confidence isn't
        cleanly exposed, _flag_low_confidence falls back to phonetic-
        similarity-only flagging (still functional, just less precise)."""
        import soundfile as sf

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        prompt = build_prompt(self.base_wrapper, "")
        inputs = self.base_wrapper.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.base_wrapper.model.generate(
            **inputs, max_new_tokens=256, output_scores=True, return_dict_in_generate=True)
        text = self.base_wrapper.processor.batch_decode(
            out.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

        words = text.split()
        # NOTE: per-word confidence from out.scores requires mapping
        # sub-word token positions back to whitespace-split words -- not
        # implemented here (needs the actual tokenizer's word-boundary
        # behavior, inspect on GPU machine). Placeholder: uniform 1.0
        # (never flags on confidence alone; retriever-based flagging below
        # still works independently).
        confidences = [1.0] * len(words)
        return words, confidences

    def _flag_spans(self, words: list[str], confidences: list[float]) -> list[int]:
        flagged = []
        for i, (w, conf) in enumerate(zip(words, confidences)):
            if conf < CONFIDENCE_THRESHOLD:
                flagged.append(i)
                continue
            # phonetically near a gazetteer entry -- reuse the retriever's
            # own similarity scoring rather than duplicating it
            candidates = self.retriever.retrieve(w, k=1)
            if candidates:
                flagged.append(i)
        return sorted(set(flagged))

    def _pass2(self, audio_path: Path, candidate_names: list[str]) -> list[str]:
        import soundfile as sf

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        context = ", ".join(candidate_names)
        prompt = build_prompt(self.lora_wrapper, context)
        inputs = self.lora_wrapper.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out_ids = self.lora_wrapper.model.generate(**inputs, max_new_tokens=256)
        text = self.lora_wrapper.processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return text.split()

    @staticmethod
    def _align_words(pass1_words: list[str], pass2_words: list[str]) -> list[tuple[int, int]]:
        """Word-level alignment, same technique as test_accuracy.diff_words
        (SequenceMatcher over token sequences) -- LoRA pass-2 can insert/
        drop/reorder words, this is NOT a positional zip. Returns list of
        (pass1_index, pass2_index) pairs for positions considered equal/
        aligned; unmatched pass1 positions are simply absent from the list."""
        sm = SequenceMatcher(a=pass1_words, b=pass2_words, autojunk=False)
        pairs = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                pairs.extend(zip(range(i1, i2), range(j1, j2)))
            elif tag == "replace":
                # best-effort positional pairing within the replaced span --
                # imperfect but keeps flagged-position splicing meaningful
                for k in range(min(i2 - i1, j2 - j1)):
                    pairs.append((i1 + k, j1 + k))
        return pairs

    def transcribe(self, audio_path: str) -> str:
        audio_path = Path(audio_path)
        pass1_words, confidences = self._pass1(audio_path)

        flagged = self._flag_spans(pass1_words, confidences)
        if not flagged:
            return " ".join(pass1_words)   # LoRA never runs -- zero cost, zero risk

        candidate_names = self.retriever.retrieve(" ".join(pass1_words), k=5)
        pass2_words = self._pass2(audio_path, candidate_names)
        alignment = self._align_words(pass1_words, pass2_words)
        align_map = dict(alignment)

        result = list(pass1_words)
        for i in flagged:
            if i in align_map:
                result[i] = pass2_words[align_map[i]]
            # if flagged position has no aligned pass-2 counterpart, leave
            # pass-1's word rather than guessing -- never insert an
            # unaligned pass-2 word into an arbitrary position
        return " ".join(result)
