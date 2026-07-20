"""
Training pipeline for the Phonetic Contrastive Model.

  python -m phonetic_contrastive_model.train                 # default config
  python -m phonetic_contrastive_model.train --epochs 40 --device mps

InfoNCE with in-batch negatives over the UNIQUE canonicals in each batch.
After training, embeds ALL canonicals once and saves the index with the weights,
so inference is a single matmul (no re-encoding of the gazetteer at runtime).

Checkpoint (models/phonetic_contrastive_v1.pt) contains:
  state_dict, vocab.itos, model config, canonicals, canonical_embeddings, meta.
Everything needed to reconstruct the corrector — no external state.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import make_splits, pad_batch
from .model import CharEncoder, info_nce

CKPT = Path(__file__).resolve().parent / "models" / "phonetic_contrastive_v1.pt"


class PairDS(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs, self.vocab = pairs, vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        v, c = self.pairs[i]
        return self.vocab.encode(v), c


def make_collate(vocab):
    """Returns padded variant ids + the list of their true canonical strings.
    Negatives (in-batch + hard-mined + random) are assembled in the train loop."""
    pad = vocab.pad_idx

    def collate(batch):
        var_ids = [b[0] for b in batch]
        canons = [b[1] for b in batch]
        return pad_batch(var_ids, pad), canons

    return collate


def encode_strings(model, strings, vocab, device):
    ids = pad_batch([vocab.encode(s) for s in strings], vocab.pad_idx).to(device)
    return model(ids)


@torch.no_grad()
def hard_neighbor_index(model, canonicals, vocab, device, k=8):
    """Per-canonical top-k nearest OTHER canonicals — the hard negatives.
    Recomputed each epoch as the embedding space moves."""
    bank = embed_all(model, canonicals, vocab, device).to(device)      # (N, d)
    sims = bank @ bank.t()
    sims.fill_diagonal_(-2.0)                                          # exclude self
    return sims.topk(k, dim=1).indices.cpu().tolist()                 # (N, k)


def embed_all(model, strings, vocab, device, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(strings), batch):
            chunk = strings[i:i + batch]
            ids = pad_batch([vocab.encode(s) for s in chunk], vocab.pad_idx).to(device)
            outs.append(model(ids).cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def val_recall(model, val_pairs, canonicals, vocab, device):
    """Top-1 nearest-canonical accuracy on held-out-from-train val pairs."""
    if not val_pairs:
        return 0.0
    cidx = {c: i for i, c in enumerate(canonicals)}
    canon_emb = embed_all(model, canonicals, vocab, device).to(device)
    var_emb = embed_all(model, [v for v, _ in val_pairs], vocab, device).to(device)
    pred = (var_emb @ canon_emb.t()).argmax(dim=1).cpu().tolist()
    hits = sum(1 for p, (_, c) in zip(pred, val_pairs) if p == cidx[c])
    return hits / len(val_pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100,
                    help="max epochs (training runs up to this unless early-stopped)")
    ap.add_argument("--patience", type=int, default=5,
                    help="early stop after N EPOCHS without val-recall improvement")
    ap.add_argument("--val-every", type=int, default=1,
                    help="val-recall check cadence in epochs (1 = every epoch, so "
                         "patience is measured in real epochs)")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--hard-k", type=int, default=8,
                    help="hard negatives mined per canonical (0 = in-batch only)")
    ap.add_argument("--neg-sample", type=int, default=256,
                    help="fresh random canonical negatives encoded per batch")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    sp = make_splits(seed=args.seed)
    vocab = sp["vocab"]
    canonicals = sp["canonicals"]

    # carve a validation slice out of TRAIN (held-out set stays untouched for eval)
    all_train = list(sp["train_pairs"])
    random.Random(args.seed).shuffle(all_train)
    n_val = int(len(all_train) * args.val_frac)
    val_pairs = all_train[:n_val]
    train_pairs = all_train[n_val:]
    print(f"vocab={len(vocab)}  canonicals={len(canonicals)}  "
          f"train={len(train_pairs)}  val={len(val_pairs)}  "
          f"heldout={len(sp['heldout_pairs'])}")

    ds = PairDS(train_pairs, vocab)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=make_collate(vocab), drop_last=True)

    model = CharEncoder(len(vocab), args.emb, args.hidden, args.out,
                        args.layers, vocab.pad_idx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    cidx = {c: i for i, c in enumerate(canonicals)}
    N = len(canonicals)
    rng = random.Random(args.seed)

    best_val, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for var, canons in dl:
            var = var.to(device)
            # Candidates = the batch's true canonicals (positives) + a fresh sample of
            # OTHER canonicals as negatives. ALL encoded with the CURRENT model, so
            # positives and negatives are consistent (no stale-bank collapse).
            uniq_true = list(dict.fromkeys(canons))
            true_ids = {cidx[c] for c in uniq_true}
            negs = [i for i in rng.sample(range(N), min(args.neg_sample + len(true_ids), N))
                    if i not in true_ids][:args.neg_sample]
            cand_strings = uniq_true + [canonicals[i] for i in negs]   # true first
            pos_local = {c: i for i, c in enumerate(uniq_true)}
            target = torch.tensor([pos_local[c] for c in canons],
                                   dtype=torch.long, device=device)

            ve = model(var)                                    # (B, d)
            ce = encode_strings(model, cand_strings, vocab, device)  # (U+neg, d) all fresh
            loss = info_nce(ve, ce, target, args.temp)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sched.step()

        do_val = (ep % args.val_every == 0) or ep == 1 or ep == args.epochs
        if do_val:
            vr = val_recall(model, val_pairs, canonicals, vocab, device)
            improved = vr > best_val + 1e-4
            if improved:
                best_val, bad = vr, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"  epoch {ep:>3}/{args.epochs}  loss {tot/len(dl):.4f}  "
                  f"val_recall {vr*100:.2f}%  best {best_val*100:.2f}%  bad {bad}")
            if bad >= args.patience:
                print(f"  early stop at epoch {ep} (no val gain for {args.patience} checks)")
                break
        elif ep % 5 == 0:
            print(f"  epoch {ep:>3}/{args.epochs}  loss {tot/len(dl):.4f}")

    # restore the best-val checkpoint before embedding + saving
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val_recall {best_val*100:.2f}%")

    # embed the full canonical index once, save everything
    canon_emb = embed_all(model, canonicals, vocab, device)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "itos": vocab.itos,
        "config": {"emb": args.emb, "hidden": args.hidden, "out": args.out,
                   "layers": args.layers, "temp": args.temp},
        "canonicals": canonicals,
        "canonical_embeddings": canon_emb,
        "meta": {"max_epochs": args.epochs, "patience": args.patience,
                 "seed": args.seed, "train_pairs": len(train_pairs),
                 "best_val_recall": round(best_val, 4)},
    }, CKPT)
    print(f"saved: {CKPT}  (index: {tuple(canon_emb.shape)})")


if __name__ == "__main__":
    main()
