"""
Inference for the Phonetic Contrastive Model.

Runtime-robust:
  - loads weights + the pre-computed canonical index from one checkpoint
  - eval() + no_grad, deterministic; no training deps needed to run
  - single forward + one matmul against the cached index -> fast
  - ABSTAIN threshold: if the nearest canonical's cosine similarity is below
    `threshold`, the word is left UNCHANGED (the safety guard that stops the
    resolver-style corruption of real words)
  - guards: already-a-canonical -> returned as-is; non-alpha -> untouched

Usage:
    from phonetic_contrastive_model.corrector import PhoneticContrastiveCorrector
    c = PhoneticContrastiveCorrector.load()          # default checkpoint
    c.resolve_word("chugataai")        -> "Chughtai"
    c.resolve_word("apareshan")        -> "apareshan"   (abstained, below threshold)
    c.resolve_text("... chugataai lab ...")
    c.add_canonical("XYZlab")          # onboard a NEW canonical, no retrain
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import Vocab, pad_batch
from .model import CharEncoder

CKPT = Path(__file__).resolve().parent / "models" / "phonetic_contrastive_v1.pt"
_TOKEN = re.compile(r"[A-Za-z]+")


class PhoneticContrastiveCorrector:
    def __init__(self, model, vocab, canonicals, canon_emb,
                 threshold: float = 0.90, min_len: int = 3, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocab = vocab
        self.canonicals = list(canonicals)
        self.index = F.normalize(canon_emb.to(self.device), dim=-1)   # (N, d)
        self.threshold = threshold
        self.min_len = min_len
        self._known = {c.lower() for c in self.canonicals}
        self.stats = {"exact": 0, "matched": 0, "abstain_short": 0,
                      "abstain_low": 0, "already": 0}

    # ---- loading ---------------------------------------------------------
    @classmethod
    def load(cls, path: Path = CKPT, threshold: float = 0.90, device: str = "cpu"):
        ck = torch.load(path, map_location="cpu")
        vocab = Vocab(itos=ck["itos"])
        cfg = ck["config"]
        model = CharEncoder(len(vocab), cfg["emb"], cfg["hidden"], cfg["out"],
                            cfg["layers"], vocab.pad_idx)
        model.load_state_dict(ck["state_dict"])
        return cls(model, vocab, ck["canonicals"], ck["canonical_embeddings"],
                   threshold=threshold, device=device)

    # ---- encoding --------------------------------------------------------
    @torch.no_grad()
    def _encode(self, words: list[str]) -> torch.Tensor:
        ids = pad_batch([self.vocab.encode(w) for w in words],
                        self.vocab.pad_idx).to(self.device)
        return self.model(ids)

    # ---- onboarding a NEW canonical without retraining -------------------
    @torch.no_grad()
    def add_canonical(self, canon: str):
        if canon.lower() in self._known:
            return
        emb = self._encode([canon])                       # (1, d)
        self.index = torch.cat([self.index, F.normalize(emb, dim=-1)], dim=0)
        self.canonicals.append(canon)
        self._known.add(canon.lower())

    # ---- the one method that matters -------------------------------------
    @torch.no_grad()
    def resolve_word(self, word: str) -> str:
        lw = word.lower()
        if not word.isalpha():
            return word
        if lw in self._known:                             # already correct
            self.stats["already"] += 1
            return word
        if len(lw) < self.min_len:                        # too short to trust
            self.stats["abstain_short"] += 1
            return word
        q = self._encode([word])                          # (1, d)
        sims = (q @ self.index.t()).squeeze(0)            # (N,)
        best = int(sims.argmax())
        score = float(sims[best])
        if score < self.threshold:                        # ABSTAIN
            self.stats["abstain_low"] += 1
            return word
        self.stats["matched"] += 1
        canon = self.canonicals[best]
        # mirror an incoming capital only when canonical is all-lowercase
        if word[0].isupper() and canon == canon.lower():
            return canon[0].upper() + canon[1:]
        return canon

    def resolve_text(self, text: str) -> str:
        return _TOKEN.sub(lambda m: self.resolve_word(m.group(0)), text)

    # ---- inspection helper ----------------------------------------------
    @torch.no_grad()
    def topk(self, word: str, k: int = 5):
        q = self._encode([word])
        sims = (q @ self.index.t()).squeeze(0)
        vals, idx = sims.topk(min(k, len(self.canonicals)))
        return [(self.canonicals[int(i)], round(float(v), 3)) for v, i in zip(vals, idx)]
