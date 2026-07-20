"""
Data for the Phonetic Contrastive Model.

Source of truth: data/lexicons_v2.json  ({canonical: [variants]}).
Entity typing (for the exact-name test): data/entities.json.

Splits (seeded, reproducible):
  - TRAIN            : most variants; every canonical keeps >=1 variant so the
                       canonical stays "known" while we test UNSEEN spellings.
  - HELDOUT          : variants hidden from training -> generalisation test.
  - HELDOUT (names)  : the subset of HELDOUT whose canonical is an entity name
                       -> the decisive Siddique/Siddiqui test.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent.parent / "data"
V2 = DATA / "lexicons_v2.json"
ENTITIES = DATA / "entities.json"

PAD, UNK = "<pad>", "<unk>"


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, strings: list[str]) -> "Vocab":
        chars = sorted({ch for s in strings for ch in s.lower()})
        return cls(itos=[PAD, UNK] + chars)

    def encode(self, s: str, max_len: int = 40) -> list[int]:
        unk = self.stoi[UNK]
        ids = [self.stoi.get(ch, unk) for ch in s.lower()[:max_len]]
        return ids or [unk]                       # never empty


def pad_batch(seqs: list[list[int]], pad_idx: int) -> torch.Tensor:
    m = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_idx] * (m - len(s)) for s in seqs], dtype=torch.long)


def load_entity_canonicals() -> set[str]:
    if not ENTITIES.exists():
        return set()
    d = json.loads(ENTITIES.read_text(encoding="utf-8"))
    out: set[str] = set()
    for k, v in d.items():
        if isinstance(v, list) and k != "_comment":
            out |= {str(x).lower() for x in v}
    return out


def load_pairs() -> tuple[list[tuple[str, str]], list[str]]:
    """Returns (pairs, canonicals) for single-word canonicals only."""
    lex = json.loads(V2.read_text(encoding="utf-8"))["lexicons"]["lexicon"]
    canonicals = [c for c in lex if " " not in c]
    pairs = [(v, c) for c in canonicals for v in lex[c] if " " not in v]
    return pairs, canonicals


def make_splits(seed: int = 13, heldout_frac: float = 0.20, min_keep: int = 1):
    """
    Returns dict with: canonicals, train_pairs, heldout_pairs, entity_canonicals,
    and the vocab (built over ALL strings so held-out chars are never unknown).
    """
    lex = json.loads(V2.read_text(encoding="utf-8"))["lexicons"]["lexicon"]
    canonicals = [c for c in lex if " " not in c]
    rng = random.Random(seed)

    train, heldout = [], []
    for c in canonicals:
        variants = [v for v in lex[c] if " " not in v]
        rng.shuffle(variants)
        # keep at least min_keep in train; hold out a fraction of the rest
        n_hold = int(round(len(variants) * heldout_frac))
        n_hold = min(n_hold, max(0, len(variants) - min_keep))
        held = set(variants[:n_hold])
        for v in variants:
            (heldout if v in held else train).append((v, c))

    all_strings = [s for p in (train + heldout) for s in p] + canonicals
    vocab = Vocab.build(all_strings)

    return {
        "canonicals": canonicals,
        "train_pairs": train,
        "heldout_pairs": heldout,
        "entity_canonicals": load_entity_canonicals(),
        "vocab": vocab,
    }
