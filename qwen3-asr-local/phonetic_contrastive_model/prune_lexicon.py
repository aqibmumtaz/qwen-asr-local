"""
Build the PRUNED lexicon (v2.1) from the current v2, using the phonetic model.

Drops every single-word variant the model recovers CONFIDENTLY (top-1 == canonical
and cosine >= threshold). Those variants are redundant: at inference the model
(PHONETIC=1) recomputes them. Everything the model is NOT confident about — short
common words, low-confidence, mismatches, and the NORMALIZATIONS — is KEPT, plus all
phrases. Re-run this whenever v2 changes.

  python -m phonetic_contrastive_model.prune_lexicon            # threshold 0.90
  python -m phonetic_contrastive_model.prune_lexicon --threshold 0.90
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corrector import PhoneticContrastiveCorrector

DATA = Path(__file__).resolve().parent.parent / "data"
V2 = DATA / "lexicons_v2.json"
V21 = DATA / "lexicons_v21.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    corr = PhoneticContrastiveCorrector.load(threshold=args.threshold)
    raw = json.loads(V2.read_text(encoding="utf-8"))
    lex = raw["lexicons"]["lexicon"]
    phrases = raw["lexicons"]["phrases"]

    kept, dropped, total = {}, 0, 0
    for c, vs in lex.items():
        single = [v for v in vs if " " not in v]
        multi = [v for v in vs if " " in v]
        keep = list(multi)
        if single:
            emb = corr._encode(single)
            sims = emb @ corr.index.t()
            vals, idxs = sims.max(dim=1)
            for v, j, s in zip(single, idxs.tolist(), vals.tolist()):
                total += 1
                if corr.canonicals[j].lower() == c.lower() and s >= args.threshold:
                    dropped += 1                      # model recomputes it -> drop
                else:
                    keep.append(v)                    # keep (model not confident)
        kept[c] = keep

    out = {
        "_comment": f"v2 minus variants the phonetic model recovers >= {args.threshold}. "
                    f"Deploy with PHONETIC=1 so the model refills the dropped variants. "
                    f"Regenerate via phonetic_contrastive_model.prune_lexicon when v2 changes.",
        "lexicons": {"lexicon": kept, "phrases": phrases},
    }
    V21.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    kv = sum(len(v) for v in kept.values())
    print(f"  source v2 single-word variants : {total}")
    print(f"  dropped (model recovers >= {args.threshold}) : {dropped}  ({100*dropped/total:.0f}%)")
    print(f"  v2.1 keeps                     : {kv} variants + {len(phrases)} phrases")
    print(f"  written: {V21}")


if __name__ == "__main__":
    main()
