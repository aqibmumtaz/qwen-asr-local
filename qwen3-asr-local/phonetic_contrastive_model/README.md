# Phonetic Contrastive Model

A character-level **contrastive bi-encoder** that maps a Roman-Urdu spelling to its
canonical — the *learned* replacement for the hand-enumerated variant list and the
rule-based resolver.

Trained so a variant embeds next to its canonical in an L2-normalised space. Because the
ASR/transliterator noise is **systematic and word-independent** (`z↔j`, `aa↔a`, `ee↔i`,
doubled letters, `gh↔g`), the model learns the *transformation* and generalises to
spellings it never saw — so you stop adding variants by hand.

## What it does / does not do

- ✅ **Common-word spelling drift** (Bucket 2) — unseen garbles → canonical.
- ✅ **New canonicals** — encode once, add to the index, no retrain (`add_canonical`).
- ❌ **Names that are acoustically confusable** (`Siddique`/`Siddiqui`) — the distinguishing
  sound is in the audio, not the text; these need audio biasing (see the training plan).
- Safety: an **abstain threshold** leaves a word unchanged when no canonical is close
  enough — this is what stops the resolver-style corruption of real words.

## Files

| file | role |
|---|---|
| `model.py` | `CharEncoder` (Siamese BiGRU, masked mean+max pool, LayerNorm, L2-norm) + `info_nce` |
| `data.py` | v2 → pairs, char vocab, seeded train / held-out / held-out-name splits |
| `train.py` | InfoNCE (in-batch negatives over unique canonicals), saves weights + canonical index |
| `corrector.py` | `PhoneticContrastiveCorrector` — runtime inference, abstain, `add_canonical` |
| `eval.py` | held-out recall, abstain safety, exact-name, 80-call `diff_words` |
| `models/phonetic_contrastive_v1.pt` | checkpoint: weights + vocab + config + canonical index |

## Usage

```bash
# train (CPU is fine; ~minutes on 12k pairs)
python -m phonetic_contrastive_model.train --epochs 30 --device cpu

# evaluate the four decisive numbers
python -m phonetic_contrastive_model.eval --threshold 0.60
```

```python
from phonetic_contrastive_model.corrector import PhoneticContrastiveCorrector
c = PhoneticContrastiveCorrector.load()
c.resolve_word("chugataai")     # -> "Chughtai"
c.resolve_text("... chugataai lab se ...")
c.add_canonical("NewLabName")   # onboard a new canonical, no retraining
```

## Design notes

- **Char-level, not semantic** — the signal is phonetic spelling, not meaning; word/sentence
  embedders (BERT, sentence-transformers) are the wrong axis and don't cover Roman Urdu.
- **In-batch InfoNCE over *unique* canonicals** — avoids false negatives when two variants in
  a batch share a canonical.
- **Inference is one matmul** against the pre-computed canonical index cached in the
  checkpoint — no re-encoding of the gazetteer at runtime.
