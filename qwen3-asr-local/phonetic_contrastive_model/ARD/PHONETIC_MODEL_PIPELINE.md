# Phonetic Contrastive Model — Pipeline & Retraining Reference

**Last updated:** 2026-07-21
Runbook for the char-level contrastive model that replaces variant enumeration in the
Roman-Urdu pipeline. Read this before retraining, pruning, or onboarding new terms.

See also: `ARD/architecture.html` (visual, layer-by-layer) and `../README.md` (quick start).

---

## 1. What it is & why

A **character-level Siamese bi-encoder** trained with **InfoNCE** so a variant spelling
embeds next to its canonical. It learns the systematic ASR/transliterator noise
(`z↔j`, `aa↔a`, `ee↔i`, `gh↔g`, doubled letters) and **generalises to spellings never seen**
— so you stop hand-adding variants.

- **Value = maintainability, not accuracy.** On the 80-call benchmark it is accuracy-neutral
  (the fuzzy `diff_words` metric already forgives the spelling drift it fixes). Its job is to
  end the variant-enumeration treadmill and let a small canonical list stand in for a huge
  variant list. See `docs/ASR_ROMAN_URDU_MODEL_TRAINING_PLAN.md`.
- **Held-out recall 97.2%** (unseen variants → correct canonical); names 98.0%.
- **Does NOT fix mishearings** (audio) — that is the acoustic pipeline's job.

---

## 2. Data lineage — v2 → v2.1 → v2.2

| file | what | how made | canonicals / variants |
|---|---|---|---|
| `data/lexicons_v2.json` | the source lexicon `{canonical: [variants]}` | `clean_lexicon.py` from the Excel sources | 1,924 / 14,489 |
| `data/lexicons_v21.json` | **pruned** — drop variants the model recovers ≥0.90 | `prune_lexicon.py` (needs the trained model) | 1,924 / **4,055** |
| `data/lexicons_v22.json` | **v2 + new entity canonicals** (no variants) | `extend_canonicals.py` | **2,054** / 14,489 |

- The model **trains on v2's pairs**.
- **v2.1** deploys the pruned lexicon + the model (model refills the 73% dropped variants).
- **v2.2** adds entity canonicals; the model recovers their garbles from the extended index.

**Deploy choice:** keep **v2 full** (or v2.2) as the exact lexicon and set `PHONETIC=1`. The
model is a fallback for lexicon misses only. v2.1 is optional (smaller file, same accuracy).

---

## 3. Model architecture & the checkpoint

**Architecture** (`model.py`): Embedding(vocab→96) → BiGRU(96→128, 2 layers, bidirectional)
→ masked mean⊕max pool (→512) → Linear(512→128)+LayerNorm → L2-normalise. ≈0.5M params.
Loss: `info_nce` (in-batch + fresh sampled negatives). Same encoder embeds variant & canonical.

**Checkpoint** `models/phonetic_contrastive_v1.pt` — one self-contained file:
| key | contents |
|---|---|
| `state_dict` | trained weights |
| `itos` | char vocab (33) |
| `config` | `{emb:96, hidden:128, out:128, layers:2, temp:0.07}` |
| `canonicals` | list indexed by the model (currently **2,048** = 1,918 v2 single-word + 130 v2.2 entities) |
| `canonical_embeddings` | the (2048, 128) index — precomputed so inference is one matmul |
| `meta` | `{max_epochs:100, patience:5, seed:13, train_pairs:10512, best_val_recall:0.9692}` |

> The index (2048) is larger than v2's canonicals because `extend_canonicals --save-index`
> appended the 130 v2.2 entity names. A fresh `train.py` resets it to v2's canonicals only.

---

## 4. Training pipeline

**Data prep** (`data.py`): loads v2 → `(variant, canonical)` pairs; builds char vocab; splits
per-canonical: keep ≥1 variant in train, hold out ~20% as unseen-generalisation test. `train.py`
further carves ~10% of train as a **validation** set for early stopping.

**Loss** (`model.py:info_nce`): per batch, embed variants (anchors) + the batch's true
canonicals + a **fresh random sample of other canonicals** (`--neg-sample 256`, encoded with the
CURRENT model). InfoNCE pulls each variant to its canonical, pushes the negatives away.

> HISTORY / do not repeat: a **memory-bank** (frozen per-epoch negatives) DIVERGED — the model
> drifted fresh embeddings away from the stale bank and collapsed (val recall 70%→6%). The
> fix was **fresh sampled negatives** (all encoded with the current model). Keep it that way.

**Early stopping:** `--patience 5` epochs on val-recall, `--val-every 1` (so patience = real
epochs), `--epochs 100` max. Best-val checkpoint is restored before saving.

**Train command (CPU is fine, ~7 min; MPS is SLOWER for this small model — do not use):**
```bash
python -m phonetic_contrastive_model.train --device cpu
# defaults: epochs 100, patience 5, val-every 1, neg-sample 256, batch 256, lr 1e-3, temp 0.07
```
Watch live: `tail -f phonetic_contrastive_model/train.log`. It prints loss + val_recall + best +
patience counter each epoch, then saves the checkpoint (weights + freshly-embedded index).

---

## 5. Inference

`corrector.py :: PhoneticContrastiveCorrector` — loads the checkpoint, encodes a query word,
cosine vs the cached index, returns nearest canonical or **abstains** (threshold **0.90**).
Guards: already-a-canonical → untouched; non-alpha / <3 chars → untouched.

**Wired into the pipeline** (`hindi_to_roman_urdu.py`): set `PHONETIC=1` → the model runs ONLY
on words the exact lexicon missed. `PHONETIC_THRESHOLD` overrides 0.90.
```bash
LEXICON=v2 PHONETIC=1 python -c "import hindi_to_roman_urdu as H; print(H.transliterate('चुगताई लैब'))"
```
- **Threshold 0.90** is the measured safe point (0.60 corrupted real words; 0.80 over-fires;
  see `sweep.py`). Model-only (no lexicon) optimum is ~0.80, but deployed on-top-of-lexicon = 0.90.
- Onboard a **new canonical at runtime**: `corrector.add_canonical("NewName")` — encodes it,
  appends to the index, no retraining.

---

## 6. Maintenance workflows — "I want to… → do…"

**Source of truth = the Excel → `clean_lexicon.py` → v2.** Everything else derives from v2.

### A. New/updated Excel arrives (canonicals + a few variants changed)
```bash
python clean_lexicon.py --src data/lexicons_updated.json data/lexicons_clean.json \
                        --out data/lexicons_v2.json --write     # rebuild v2
python -m phonetic_contrastive_model.train --device cpu         # RETRAIN (index resets to v2)
python -m phonetic_contrastive_model.prune_lexicon               # regenerate v2.1
```
Retrain when the canonical set changes substantially, or the ASR/transliterator noise
distribution changes. Retraining is cheap (~7 min).

### B. Add NEW entity terms only (names/labs from Excel) — NO retraining
```bash
# put the new canonical names in a text file, one per line
python -m phonetic_contrastive_model.extend_canonicals --terms data/new_entities.txt --save-index
```
Validates each term on the 80-call set, **drops ambiguous ones** (that cause wrong captures),
writes `data/lexicons_v22.json` (canonicals only), and appends them to the model index
(`--save-index`). The model then recovers their garbled spellings immediately. Use this for
incremental name onboarding between retrains.

### C. Regenerate the pruned v2.1 (after v2 or the model changed)
```bash
python -m phonetic_contrastive_model.prune_lexicon --threshold 0.90
```
Deterministic (byte-for-byte reproducible). Deploy `LEXICON=v21 PHONETIC=1`.

### D. Retrain reproducibly
Same seed (13) + same v2 → same model. Note: retraining **overwrites** the index with v2's
canonicals only — re-run `extend_canonicals --save-index` afterward to re-add entity names,
OR bake those names into the Excel/v2 so they are trained in.

---

## 7. Evaluation

```bash
python -m phonetic_contrastive_model.eval --threshold 0.90   # held-out recall, abstain safety,
                                                             #   exact-name, 80-call diff_words
python -m phonetic_contrastive_model.sweep                   # threshold curve: recall/safety/dw
```
Current (best_val_recall 96.92%): held-out recall **97.2%** (names 98.0%), 80-call diff_words
neutral vs v2, safe operating threshold **0.90** (safety ~90%). Model recovers **98.4%** of all
v2 variants raw / **83.9%** at ≥0.90 → v2.1 drops 73% at the same accuracy.

**Metric:** `benchmark/test_accuracy.py :: diff_words` (fuzzy word recall ≥0.70), call-level on
the 80-call set. Same metric everywhere — do not change it.

---

## 8. Key decisions (don't re-litigate)

- **Char-level, not semantic embedders / fastText** — the signal is phonetic spelling; word
  embedders are the wrong axis and Roman Urdu is OOV for them.
- **Fresh sampled negatives, not a memory bank** — the bank collapses (see §4).
- **Threshold 0.90 on-top-of-lexicon** — measured; lower corrupts real words.
- **Names are NOT solved by this model** — dense name space (`Siddiqui`/`Siddique`) is
  acoustically distinguished, not textually; that is the acoustic pipeline's job.
- **Keep the lexicon** — the model alone is ~52% (abstains on short common words); the lexicon
  does the heavy lifting, the model is the long-tail generaliser.

---

## 9. File map

| file | role |
|---|---|
| `model.py` | `CharEncoder` + `info_nce` |
| `data.py` | v2 → pairs, vocab, seeded splits |
| `train.py` | training loop, fresh negatives, early stopping, saves checkpoint+index |
| `corrector.py` | inference: `resolve_word/resolve_text`, abstain, `add_canonical` |
| `eval.py` | held-out recall / abstain safety / exact-name / 80-call diff_words |
| `sweep.py` | abstain-threshold curve |
| `prune_lexicon.py` | build v2.1 (drop model-recoverable variants) |
| `extend_canonicals.py` | add entity canonicals → v2.2 + extend index (no retrain) |
| `models/phonetic_contrastive_v1.pt` | the checkpoint |
| `ARD/architecture.html` | visual layer-by-layer diagram |
| `ARD/PHONETIC_MODEL_PIPELINE.md` | this runbook |

---

## 10. Command cheat-sheet

```bash
# rebuild lexicon from Excel sources
python clean_lexicon.py --src data/lexicons_updated.json data/lexicons_clean.json --out data/lexicons_v2.json --write
# train (CPU) — 100 epochs max, early stop patience 5
python -m phonetic_contrastive_model.train --device cpu
# evaluate + threshold sweep
python -m phonetic_contrastive_model.eval --threshold 0.90
python -m phonetic_contrastive_model.sweep
# prune -> v2.1
python -m phonetic_contrastive_model.prune_lexicon --threshold 0.90
# add entity names -> v2.2 (no retrain)
python -m phonetic_contrastive_model.extend_canonicals --terms data/new_entities.txt --save-index
# deploy
LEXICON=v2 PHONETIC=1 PHONETIC_THRESHOLD=0.90 python your_app.py
```
