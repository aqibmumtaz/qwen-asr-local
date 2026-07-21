"""
Two-pass biased transcription.

  Pass 1 : transcribe with NO context           -> a rough hypothesis
  Retrieve: pull the relevant gazetteer names from that hypothesis (retriever.py)
  Pass 2 : re-transcribe the SAME audio, context = the retrieved names

This is the training-free approximation of BR-ASR: retrieval keyed on the first-pass
text (not a learned audio embedding), so it costs two decodes but no training. The
retrieved list is small (~k names), which avoids the dilution of biasing with the
whole gazetteer.

  from acoustic_contextual_biasing.two_pass import TwoPass
  tp = TwoPass()
  r = tp.transcribe("chunk_000.wav")     # -> {pass1, pass2, context}
"""
from __future__ import annotations

from pathlib import Path

from .asr import make_asr
from .retriever import NameRetriever


class TwoPass:
    def __init__(self, asr=None, retriever: NameRetriever | None = None,
                 backend: str = "remote", k: int = 15, language: str = "Hindi"):
        self.asr = asr or make_asr(backend=backend)     # remote GPU by default
        self.retriever = retriever or NameRetriever()
        self.k = k
        self.language = language

    def transcribe(self, audio: str | Path) -> dict:
        pass1 = self.asr.transcribe(audio, context="", language=self.language)
        names = self.retriever.retrieve(pass1, k=self.k)
        ctx = ", ".join(names)
        pass2 = self.asr.transcribe(audio, context=ctx, language=self.language)
        return {"pass1": pass1, "pass2": pass2, "context": ctx, "names": names}
