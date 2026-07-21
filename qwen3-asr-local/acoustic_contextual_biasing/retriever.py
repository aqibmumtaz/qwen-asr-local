"""
Per-utterance name retriever — the BR-ASR-lite retrieval step.

A full gazetteer passed as context DILUTES biasing (each term's nudge shrinks as the
list grows). So instead of biasing with all names, we retrieve only the handful
RELEVANT to what was actually said.

Mechanism (text-keyed, no training): reuse the Phonetic Contrastive Model's encoder.
Embed every gazetteer name once; at query time embed each content word of the
first-pass hypothesis and pull the nearest names above a loose threshold. Those are
the candidates the second pass biases toward.

  from acoustic_contextual_biasing.retriever import NameRetriever
  r = NameRetriever()
  r.retrieve("shaher baat karun chughtai lab", k=15)  -> ["Shahid", "Chughtai", ...]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENTITIES = DATA / "entities.json"
TOKEN = re.compile(r"[A-Za-z]+")


def load_gazetteer() -> list[str]:
    """All proper-noun names/places/orgs from entities.json (+ v2.2 entity canonicals)."""
    names: set[str] = set()
    if ENTITIES.exists():
        d = json.loads(ENTITIES.read_text(encoding="utf-8"))
        for k, v in d.items():
            if k != "_comment" and isinstance(v, list):
                names |= {str(x) for x in v}
    # fold in the entity canonicals added to v2.2 (capitalised, single word)
    v22 = DATA / "lexicons_v22.json"
    if v22.exists():
        lex = json.loads(v22.read_text(encoding="utf-8"))["lexicons"]["lexicon"]
        for c in lex:
            if " " not in c and c[:1].isupper():
                names.add(c)
    return sorted(names)


class NameRetriever:
    def __init__(self, threshold: float = 0.55, device: str = "cpu"):
        from phonetic_contrastive_model.corrector import PhoneticContrastiveCorrector
        self.enc = PhoneticContrastiveCorrector.load(device=device)
        self.threshold = threshold
        self.names = load_gazetteer()
        # encode the gazetteer once
        embs = []
        B = 1024
        for i in range(0, len(self.names), B):
            embs.append(self.enc._encode(self.names[i:i + B]))
        self.index = F.normalize(torch.cat(embs, dim=0), dim=-1)   # (N, d)

    @torch.no_grad()
    def retrieve(self, hypothesis: str, k: int = 15) -> list[str]:
        """Return up to k gazetteer names most relevant to the hypothesis words."""
        words = [w for w in TOKEN.findall(hypothesis) if len(w) >= 3]
        if not words:
            return []
        q = self.enc._encode(words)                 # (W, d)
        sims = q @ self.index.t()                    # (W, N)
        best_per_name = sims.max(dim=0).values       # (N,) best match to any word
        vals, idx = best_per_name.topk(min(k * 3, len(self.names)))
        out = []
        for v, j in zip(vals.tolist(), idx.tolist()):
            if v >= self.threshold:
                out.append(self.names[j])
            if len(out) >= k:
                break
        return out
