"""
Phonetic Contrastive Model — the char-level bi-encoder.

A Siamese encoder: the SAME network embeds both a variant spelling and a
canonical into a shared L2-normalised space, trained so a variant sits next to
its canonical. Character-level, because the signal is phonetic/spelling
(z<->j, aa<->a, ee<->i), not semantic.

Robust by design for inference:
  - masked mean+max pooling  -> length-invariant, no reliance on a final state
  - LayerNorm on the projection -> stable embeddings across inputs
  - deterministic in eval() (dropout off), pure forward, no side effects
  - handles any length; unseen chars map to <unk>; empty -> zero-safe
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 96, hidden: int = 128,
                 out_dim: int = 128, num_layers: int = 2, pad_idx: int = 0,
                 dropout: float = 0.2):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            emb_dim, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        pooled_dim = hidden * 2 * 2  # bi-directional (x2) * (mean ++ max) (x2)
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) long. Returns L2-normalised (B, out_dim)."""
        mask = (ids != self.pad_idx).unsqueeze(-1)           # (B, T, 1)
        x = self.emb(ids)                                    # (B, T, E)
        out, _ = self.gru(x)                                 # (B, T, 2H)

        out = out * mask                                     # zero the pad steps
        summ = out.sum(dim=1)                                # (B, 2H)
        cnt = mask.sum(dim=1).clamp(min=1)                   # (B, 1)
        mean = summ / cnt

        very_neg = torch.finfo(out.dtype).min
        mx = out.masked_fill(~mask, very_neg).max(dim=1).values
        mx = torch.nan_to_num(mx, neginf=0.0)                # all-pad row -> 0

        pooled = torch.cat([mean, mx], dim=-1)               # (B, 4H)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)


def info_nce(var_emb: torch.Tensor, canon_emb: torch.Tensor,
             target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    In-batch InfoNCE.
      var_emb   : (B, d)  variant embeddings (anchors)
      canon_emb : (U, d)  embeddings of the UNIQUE canonicals in the batch
      target    : (B,)    index into U of each variant's true canonical
    Using unique canonicals (not the raw B) avoids false negatives when two
    variants in a batch share a canonical.
    """
    logits = var_emb @ canon_emb.t() / temperature          # (B, U)
    return F.cross_entropy(logits, target)
