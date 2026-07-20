# ASR Roman Urdu Model — Training Plan

**Last updated:** 2026-07-20
**Scope:** improving name / entity accuracy in the Roman-Urdu ASR pipeline
(`Audio → Qwen3-ASR (Hindi Devanagari + per-word confidence) → transliterate() → Roman Urdu`),
built as a **maintainable** system: maintain canonicals (auto-synced), not variants; get
accuracy on **new** canonicals via audio biasing — see Target architecture.

---

## Benchmark harness & dataset (authoritative)

**Dataset:** `testing/lab_test_80_calls_urdu_roman_urdu.xlsx`
- **403 chunk rows across 72 calls.** Each row = a 25-second chunk.
- **Ground truth is per-CALL, not per-chunk:** `actual_urdu_transcript` (Urdu) and
  `benchmark_roman_urdu` (Roman) are filled on **one row per call — always
  `chunk_index == 0`** — and cover the *whole call* (verified: benchmark word count
  ≈ sum of all chunks' Hindi, not one chunk's).
- `model_output_hindi` and `model_output_roman_urdu` are **per-chunk** (all 403 filled),
  produced by the **previous** model.

**Audio:** `testing/lab_test_80_audios_chunks_25s/<call_id>/chunk_NNN.wav`,
**8kHz mono** (narrowband — the name-accuracy ceiling applies). All 403 present.
Chunk order + call grouping is in `_manifest.json`.

**Metric (official — use the dev's script, not edit-distance WER):**
`testing/test_accuracy.py :: diff_words(benchmark, hypothesis)`
- normalizes (lowercase, strip ASCII+Urdu punctuation, collapse whitespace),
- exact word alignment, then a **fuzzy pass** matching leftover words with
  char-similarity **≥ 0.70**,
- **`accuracy = matched / len(reference) × 100`** → fuzzy word **recall**,
- returns `mismatched_tokens` / `missing_tokens` for the downstream LLM classifier.

**Two properties to remember when reading numbers:**
1. **Recall-based, not WER** — over-generation (extra hyp words) is not directly
   penalized; only unmatched *reference* words cost accuracy. Different scale than
   the old `1 − edit/N`; the 47.51% below does **not** transfer to this metric.
2. **Fuzzy 0.70 already forgives spelling drift** (`daanish`≈`Danish`,
   `kar rahe`≈`kar raha`). The lexicon's value therefore looks *smaller* here — its
   remaining worth is on fixes that cross the 0.70 line or fix the wrong *word*, not
   minor romanization. A small delta ≠ the lexicon failing.

**Scoring granularity:** **call-level.** For each call, concatenate the per-chunk
column (Hindi or Roman) in `chunk_index` order → produce Roman → `diff_words` vs that
call's `benchmark_roman_urdu`.

**Baseline extraction (two numbers from one pass):**
- **Previous model** = concat `model_output_roman_urdu` vs benchmark → their baseline.
- **Our v2 pipeline** = our `transliterate()` on concat `model_output_hindi` vs
  benchmark → what our text layer adds over theirs.

**Where audio is used:** NOT for the text baseline (Hindi already exists). Audio (the
chunk wavs) is used only for a **fresh ASR pass** (running Qwen3-ASR ourselves to
benchmark a new end-to-end model) and for the **two-pass rungs** (re-decode).

**Output:** benchmark columns written per call (on the `chunk_index==0` row, mirroring
the GT layout) into a **copy** of the sheet — original manually-labeled file left
untouched unless explicitly told otherwise.

---

## Current state (Rung 0)

- Exact lexicon **v2** (`data/lexicons_v2.json`) is live and is the workhorse.
  **NOTE:** the headline **47.51% WER accuracy, +12.67 over baseline, 0 regressions**
  was measured on the OLD 178-turn set (`turnwise_results_eval_full.xlsx`) with
  edit-distance WER. It must be **re-baselined on the 80-call set with `diff_words`**
  (Rung 1b) before it is comparable to anything below. Self-test 62/62 still holds.
- Built by `clean_lexicon.py` from **both** sources
  (`lexicons_updated.json` + `lexicons_clean.json`, clean_new authoritative,
  gold-vouched restores, `ek→aik` injected).
- **Resolver** (`resolver.py`) exists but is **OFF** by default (`RESOLVER=1` to enable).

### Resolver investigation (2026-07-16, 80-call set) — keep OFF, do not patch

On the 80-call set the resolver fires on **53 of 72 calls** (~90 word changes) but nets
to zero: **helped 5 / hurt 4 / no-score-change 44** (`diff_words` 64.02 vs 64.03).

**Single root cause — no vocabulary guard.** `known_correct` (G2) contains only the
2,046 canonicals + phrases, NOT general vocabulary. So any real word that is not a
canonical is treated as a garble and snapped to the nearest canonical at edit-distance 1:

| input (a real word) | → wrong output | dist |
|---|---|---|
| `taareekh` (date) | `Tariq` (name) | 1 |
| `apareshan` (operation) | `pareshan` (worried) | 1 |
| `chaaineez` (Chinese) | `Chennai` (city) | 1 |
| `chaalaak` (clever) | `falak` (name) | 1 |
| `riyaayat` (discount) | `Riffat` (name) | 1 |

**Cannot be patched safe (tested):**
- Excluding names from the fuzzy index → helped 6 / hurt 4, corpus 64.03 vs 64.02: **no gain**
  (non-name corruptions remain).
- A vocabulary guard can't protect these — it would need to contain the *garbled* spelling
  (`apareshan`), not the dictionary word (`operation`).
- Structural: garbles sit 1 edit from wrong real words; edit-distance can't separate them,
  and the fuzzy metric already forgives the genuine fixes → **stuck at break-even.**

**Resolution (per ladder):** do NOT invest in patching the resolver.
- Its job (unseen-spelling repair) → **Rung 3b text-contrastive**, which learns a proper
  abstain from data ("`apareshan` is a real word, not a garble of `pareshan`") instead of a
  hand-maintained vocabulary.
- Names (its worst failures) → **Rung 2 audio two-pass**; acoustics disambiguate dense
  names, text fuzzy-matching cannot.
Keep the resolver as-is, OFF, as insurance only.

### The three error buckets (what each rung can and cannot fix)

| bucket | example | fixable by |
|---|---|---|
| **B2 — mis-spelling** (garbled non-word) | `chugataai` → Chughtai | text models (lexicon, resolver, text-contrastive) |
| **B1 — mis-hearing** (heard a different real word) | said "Aqeeb", heard "Aqib" | only audio (two-pass / BR-ASR) |
| **B3 — boundary collapse** (merges into a valid word) | — | upstream ASR / audio only |

Key rule: **text approaches fix B2. Audio approaches fix B1.** They are not substitutes.

---

## Target architecture — maintainable + accurate on NEW canonicals

**Goal:** stop hand-enumerating variants; maintain only a small canonical list that grows
with the business; and get accuracy on canonicals the system has *never seen* (new names,
labs, doctors) **without** adding their spellings by hand.

**The maintenance burden, precisely — it is NOT the 16k count.** The lexicon has two halves
that behave differently:
- **Closed vocabulary** (`hai`, `nahi`, `karna`, `report`) ≈ 1,168 words — **finite,
  saturates, maintained once.**
- **Entities** (names, places, labs, doctors) ≈ 500+ — **open-ended, grows forever** as new
  customers/labs appear. This is the real long-term burden.
- The **14,319 variants are a symptom** of ASR inconsistency, not a thing to maintain.
  Evidence: v2.1 (31% fewer variants) matched full-v2 accuracy — most variants earn nothing
  under the production metric.

### Three pillars

**1. Canonicals only, from the source Excel.**
- The lexicon is already built from an **Excel file** (via `clean_lexicon.py`). Going
  forward that Excel should hold **canonicals only** (correct names/labs/doctors/tests +
  closed vocab) — NOT hand-typed variant spellings.
- An **updated Excel** later = the maintenance action. Re-run the build + retrain (Rung 3b).
- This is the ONLY thing maintained — one canonical column, not thousands of variants.

**2. A learned generalizer replaces variant enumeration.**
- **Text-contrastive model** (Rung 3b), trained on canonical↔variant pairs, maps *unseen*
  spellings to their canonical. **Retrained automatically** when the canonical list changes.
- Has a **learned abstain** (leaves real words alone) — the safe version of the resolver,
  which corrupts because its abstain is rule-based (see Resolver investigation).

**3. Audio disambiguation for entities (where text cannot).**
- The gazetteer feeds **per-call contextual biasing / two-pass** (Rungs 2/5). The *audio*
  picks the right name; text spelling cannot (`Siddiqui` vs `Siddique`).

### How a NEW canonical gets accuracy — with zero variant enumeration

1. New entity enters the business DB → **auto-synced** into the canonical gazetteer (1 row).
2. On a call that mentions it:
   - per-call **biasing** nudges the ASR toward the now-present canonical → consistent output;
   - if still garbled, **two-pass** retrieves it from the gazetteer and re-decodes;
   - the **contrastive model** absorbs minor spelling drift → canonical.
3. The only action taken was the DB sync. **No human enumerated a variant.**

### Division of labour

| need | handled by | maintenance |
|---|---|---|
| closed-vocab spellings | fixed lexicon | done once |
| known-entity spelling drift | contrastive generalizer | auto-retrain on sync |
| **NEW entity recognition** | **DB sync + biasing/two-pass** | **automated sync** |
| mis-heard names | audio two-pass (Rung 2/5) | none |

**Where the accuracy on new canonicals actually comes from:** pillars 1–2 *hold* accuracy
while removing maintenance; the **accuracy gain on new canonicals comes from pillar 3**
(audio biasing/two-pass) plus the sync making the canonical *present*. Text alone stays
saturated (~64% on the 80-call set); **audio is the accuracy lever.** The ladder below is
the build order for these pillars.

---

## The ladder (cheapest first; each rung gated by the previous)

Each rung builds part of the **Target architecture** above:
- **Pillar 1 (canonicals from business data)** → Rung 0b below.
- **Pillar 2 (learned generalizer)** → Rung 3b.
- **Pillar 3 (audio for entities / new canonicals)** → Rungs 2 and 5.

### Rung 0b — canonicals from the source Excel *(foundation; do early)*

The lexicon is already built from an **Excel file**. Make that Excel the single source of
truth for **canonicals only** — the correct names/labs/doctors/tests — and stop adding
variant spellings to it. `clean_lexicon.py` already turns it into the lexicon + gazetteer;
keep that deterministic build.
- Deliverable: Excel (canonical column) → `clean_lexicon.py` → `lexicons_v2.json` +
  `entities.json`.
- Maintenance action: when an **updated Excel** arrives, re-run the build + retrain the
  Rung 3b generalizer. New canonicals become active; their variants are generalized, not
  typed.

### Rung 1 — measure before building *(½ day, nothing ships)*

Two free measurements that decide the whole ladder:

1. **Retrieval recall (Stage 0):** on existing first-pass text + gold, when a name
   is misheard, does `normalise()` + edit-distance surface the correct gazetteer
   term in top-K?
2. **Audio bandwidth:** capture one live WebRTC segment, run the FFT check — real
   Opus wideband, or PSTN-bridged 8kHz?

**Gate:**
- recall high (>85%) → retrieval is not the problem; audio is the limit → focus "Underneath".
- recall low → climb the retrieval rungs.
- wideband audio → audio rungs (2/5) worth it; narrowband → they are capped.

---

### Rung 1b — standalone text bench *(days, NO audio)*

Build the evaluation harness (dataset + `diff_words` from **Benchmark harness** above)
and run the **text-only** configs on the **80-call set**, call-level. This answers most
of the question with zero audio, and **re-baselines v2 under the official metric**.

See **Evaluation Matrix** below — configs **C0, C1, C3**, each in **full-eval + held-out** mode.

---

### Rung 2 — phonetic-hash two-pass *(no training; ~80% value / 5% effort — start here)*

Reuse `normalise()` + gazetteer. Pass 1 (no bias) → gate on **slot OR low-confidence**
→ retrieve candidates by edit-distance → re-decode the buffered segment with that
small glossary. Adds config **C2**.

- 2nd pass re-decodes the **whole utterance segment**, not the isolated word.
- Gate is **per-word**; decode is **per-segment**; glossary is targeted.
- Streaming: pass-1 = live partial, pass-2 = corrected final on endpoint.
- Latency ≈ `T × trigger_rate` average; `+T` tail on triggered finals only.

**Gate:** gain worth latency → keep, improve retrieval (Rung 3). No gain → audio is
the ceiling → "Underneath".

*Fixes: B1 for in-gazetteer names, within audio limits.*

---

### Rung 3 — better retrieval (two options, both cheap)

Only if Rung 1 showed edit-distance recall is weak. Both improve the candidate-pull step.

**3a — pretrained embedder (zero training).** Swap edit-distance for an off-the-shelf
multilingual phonetic/text embedder over the first-pass text. "Retrieval-augmented
biasing lite."

**3b — text-text contrastive (days, data you own).** Train a small char-level
bi-encoder on the ~14k v2 `(variant → canonical)` pairs + **abstain threshold**
(nothing close → leave unchanged). Produces config **C3** (standalone) and **C4** (two-pass).

Contrastive learning in one line: pull matching `(variant, canonical)` pairs close in
an embedding space, push mismatches apart, so any string that *looks like* a canonical
lands near it — including spellings never seen in training.

**Limits of text-contrastive (important):**
- Closed-set — a genuinely **new canonical** (not in v2) maps to the nearest existing
  one → wrong. Needs the abstain threshold or it always snaps.
- Fixes **B2 only** — a confident mis-hearing (B1) is already a valid word; text cannot fix it.

**Gate:** net positive with abstain on → keep the better of 3a/3b as the retriever.

---

### Rung 4 — decision point

With the best cheap text retriever in hand: **is retrieval still the bottleneck, or the audio?**
(Rungs 2–3 tests answer this.)
- bottleneck = 2nd-pass choice or audio → only Rung 5 / wideband helps.
- bottleneck = retrieval **and** wideband available → climb to Rung 5.
- neither → done; ship Rungs 2–3.

---

### Rung 5 — full BR-ASR audio-text contrastive *(weeks, research-grade)*

Real acoustic retriever without collecting real calls: **TTS the gazetteer in
Urdu/Hindi voices → apply 8kHz + GSM codec augmentation → contrastively train**
speech↔entity. Collapses two passes into one (retrieve from audio *before* decoding).

**Gate — climb only if BOTH:** Rung 4 said retrieval is the bottleneck **and** you have
wideband audio. Otherwise 8kHz eats the gain. Most call-centers never reach here.

*Fixes: B1 at scale, single-pass — only pays off with clean audio.*

---

## Underneath every rung — the two real levers

- **Audio quality.** If the live WebRTC path is real Opus wideband, that lifts the
  ceiling on Rungs 2/5 more than any model work. Consonant detail separating
  `Siddiqui`/`Siddique` lives above 4kHz — 8kHz deletes it.
- **Confirmation turn.** For form-critical fields (name, CNIC, address), have the caller
  spell/confirm. No ASR nails a novel spelled name from unknown telephony audio.

---

## Evaluation Matrix (independent observation of each model)

All configs scored with `diff_words` on the **80-call set, call-level**. Each text
model runs in **standalone** (text-only, corrects first-pass text directly) and
**two-pass** (feeds candidates to a second audio decode) mode, so each model's
contribution and the audio's added value are isolated.

| # | corrector / retriever | mode | uses audio? | isolates | measurable |
|---|---|---|---|---|---|
| C-prev | previous model (`model_output_roman_urdu`) | — | no | vendor baseline | **now** |
| C0 | none — our transliterate() + v2 exact | — | no | our text layer vs vendor | **now** |
| C1 | phonetic-hash | standalone | no | pure text-fix from edit-distance | **now** |
| C2 | phonetic-hash | two-pass | yes | what the 2nd decode adds | needs audio |
| C3 | text-contrastive | standalone | no | pure text-fix from learned model | after training |
| C4 | text-contrastive | two-pass | yes | learned retriever + 2nd decode | after training + audio |

**How to read it:**
- **C0 vs C-prev** → what our transliteration + lexicon adds over the vendor's Roman.
- **C1 vs C3** → which text model is the better corrector.
- **C2 − C1** and **C4 − C3** → how much the audio second pass adds on each.

**Two runs per standalone config** (because the 80-call GT words are what the metric
scores against, and the fuzzy 0.70 pass already forgives light drift):
- **(a) full set** — call-level `diff_words` with v2 present. Net production effect.
- **(b) held-out** — hide ~30% of v2 variants from lexicon + training, measure recovery
  on those. Honest test of generalization to *unseen* spellings.

---

## Benchmarking protocol — MANDATORY for every new model / rung

**Rule: nothing built on this plan is "done" until it is scored on the 80-call set with
`diff_words`, at call level, and its row is recorded below.** No new metric, no new
dataset, no per-chunk shortcuts — same harness every time, or the numbers are not
comparable across rungs.

For each new artifact (a lexicon version, a retriever, a contrastive model, a fresh ASR,
a two-pass config):
1. Produce its Roman-Urdu output per chunk.
2. Concatenate per call in `chunk_index` order.
3. `diff_words(benchmark_roman_urdu, our_output)` per call → accuracy, matched, total,
   mismatched/missing tokens.
4. Aggregate: **corpus accuracy = Σ matched / Σ total** (word-weighted across calls),
   and also report the simple mean of per-call accuracy.
5. Write the per-call columns into the benchmark **copy** of the sheet and append one
   row to the **Results log** below (date, config, corpus acc, mean acc, Δ vs C0, notes).
6. Regression gate: a change ships only if corpus accuracy is **non-decreasing** vs the
   current production config.

### Results log (fill as rungs are built)

Corpus acc = `diff_words` word-weighted (Σmatched/Σtotal); WER acc = edit-distance,
both call-level on the 80-call set via `testing/benchmark_baseline.py`.

| date | config | diff_words | Δ vs C0 | audio? | notes |
|---|---|---|---|---|---|
| 2026-07-16 | C0 — v2, resolver OFF | 64.03% | 0.00 | no | prior baseline |
| 2026-07-16 | C0+res — v2, resolver ON | 64.02% | −0.01 | no | resolver ≈ noise; DEPRECATED (corrupts) |
| 2026-07-20 | C3 — phonetic contrastive, standalone @0.90 | 63.27% | −0.76 | no | model on top of v2; held-out recall **97.2%** |
| 2026-07-20 | C3 — model-only (no lexicon) @0.80 | 55.57% | −8.5 | no | model alone ≈ raw+5; **lexicon is the workhorse** |
| 2026-07-20 | v2.1 pruned (−73%) + model @0.90 | 63.97% | −0.06 | no | model recomputes dropped variants → maintainability |
| **2026-07-21** | **C0′ — v2 + NORMALIZATIONS** | **64.85%** | **+0.82** | no | **new production baseline** (sar→sir, bahut→bohot) |
| _future_ | C2 — phonetic-hash / model two-pass | — | — | yes | audio needed |
| _future_ | C5 — BR-ASR (fresh ASR) | — | — | yes | audio needed |
| _skipped_ | C-prev — previous model | 64.37% | — | no | vendor's own Roman; tied with ours |

**Phonetic Contrastive Model (Rung 3b) — DONE.** `phonetic_contrastive_model/`, wired into
the pipeline as `PHONETIC=1` (preferred over the deprecated resolver). Trained on
positives + 256 fresh negatives, 100-epoch max / patience-5 early stop, best val recall
96.92%. Findings:
- **Held-out recall 97.2%** (names 98.0%) — generalises to unseen spellings; ends the
  enumeration treadmill. Recomputes **98.4%** of all variants (83.9% at ≥0.90).
- **Accuracy-neutral on the benchmark** — the fuzzy metric already forgives the drift it
  fixes; the sweep found NO threshold that beats baseline. Its value is **maintainability**,
  not a score gain. Deploy threshold **0.90** (safe); model-only optimum 0.80.
- **The lexicon is the workhorse** (+13 pts); the model alone adds only +1.7 over raw.
  Keep both: lexicon for the short common-word core, model for the long tail.

**Accuracy ceiling (measured):** text tops out ~**65%**. Error split on the 80-call set:
61% matched, **16% mishearings** (audio), **15% dropped words** (audio), ~7% near-miss
(mostly bidirectional particles, unsafe to map). Above ~65% requires improving the **ASR**
(biasing / two-pass / wideband), not text.

**Text-contrastive model gets three specific tests:**
1. **Held-out recall** — hidden variants still map to correct canonical.
2. **Abstain safety** — correct words not in v2 must NOT be snapped to a canonical.
3. **Standalone vs retriever** — test both as a post-ASR corrector alone and as the
   Rung-2 retriever.

---

## One-glance summary

| Rung | Approach | Pillar | Effort | Needs audio | Climb if |
|---|---|---|---|---|---|
| 0b | sync canonicals from business DB | **1** | days | no | foundation — do early |
| 1 | recall + audio check | — | ½ day | no | always — first |
| 1b | standalone text bench | — | days | no | — |
| 2 | phonetic-hash two-pass | **3** | days | yes | start here (audio) |
| 3a | pretrained embedder | 2 | days | no | edit-distance weak |
| 3b | text-text contrastive | **2** | days | C4 only | replaces variant enumeration |
| 4 | decision point | — | — | — | — |
| 5 | BR-ASR audio contrastive | **3** | weeks | yes | retrieval=bottleneck **and** wideband |
| — | audio quality + confirmation | 3 | ½ day | — | always, in parallel |

**Maintenance workflow after the pillars are built:** a new lab/doctor/name enters the
business DB → Rung 0b sync adds one canonical → Rung 3b model auto-retrains → Rungs 2/5
biasing pick it up acoustically. **No variant is ever hand-added.**

**Recommended next step:** Rung 1 (measurements) + Rung 1b (C0/C1 today), and scope Rung 0b
(the DB → gazetteer sync) since it is the foundation for both maintainability and
new-canonical accuracy.

---

## References

- BR-ASR: Bias Retrieval Framework for Contextual Biasing (Interspeech 2025) —
  https://arxiv.org/html/2505.19179v1
- Deliberation Model Based Two-Pass End-to-End ASR (Google) — https://arxiv.org/abs/2003.07962
- Transformer-based Deliberation for Two-Pass ASR — https://arxiv.org/pdf/2101.11577
- Qwen3-ASR Context Biasing — practical glossary guide —
  https://note.com/veltrea/n/n7dd0b7ffffe9?hl=en
