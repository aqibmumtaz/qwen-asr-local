"""
Qwen3-ASR wrapper with contextual biasing.

Loads the BASE (non-quantized) Qwen3-ASR once and exposes transcribe(audio, context).
`context` is Qwen3-ASR's built-in biasing hook — a string of terms that nudges
recognition toward them without mandating output.

Model load is lazy (first transcribe) and cached, so importing this module is cheap.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"     # base model, per project decision (not quantized)


class BiasedASR:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu",
                 max_new_tokens: int = 256, verbose: bool = True):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from qwen_asr import Qwen3ASRModel
        if self.verbose:
            print(f"[asr] loading {self.model_id} on {self.device} ...", flush=True)
        t0 = time.time()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_id, dtype=dtype, device_map=self.device,
            max_new_tokens=self.max_new_tokens)
        if self.verbose:
            print(f"[asr] loaded in {time.time()-t0:.0f}s", flush=True)

    def transcribe(self, audio: str | Path, context: str = "",
                   language: str = "Hindi") -> str:
        """One clip -> transcript string. context = biasing terms (may be empty)."""
        self._ensure()
        out = self._model.transcribe(
            audio=[str(audio)], context=[context], language=[language])
        return out[0].text if out else ""

    def transcribe_many(self, audios, contexts, language: str = "Hindi") -> list[str]:
        """Batched: audios[i] transcribed with contexts[i]."""
        self._ensure()
        out = self._model.transcribe(
            audio=[str(a) for a in audios],
            context=list(contexts),
            language=[language] * len(audios))
        return [o.text for o in out]
