# Acoustic Contrastive Contextual Model — Finetuned

LoRA fine-tune of Qwen3-ASR-1.7B's **decoder** attention layers to natively
recognize this call center's names/places/orgs at the acoustic level, combined
with a runtime **two-pass selective-splice** inference design that guarantees
non-entity words are never sourced from the fine-tuned weights.

## STATUS (read this first — accurate as of the session that wrote it)

**All 7 scripts are implemented. 3 have been RUN with real data on this
machine (no GPU needed — pure data wrangling). 4 need a GPU and have NOT
been executed — training was never attempted, so nothing past that point is
verified end-to-end.**

| Script | Status |
|---|---|
| `build_entity_devanagari.py` | ✅ **Run, validated.** Went through 3 real bugs found and fixed this session (see §Data Prep Results below) — final methodology is sound, not just written. |
| `prepare_data.py` | ✅ **Run, validated.** Real output: 305 train / 53 eval / 106 adversarial examples. Documented caveats below — not perfect, needs the manual spot-check the plan already calls for. |
| `augment_audio.py` | ✅ **Run, validated.** 1300 total examples, 466MB augmented audio written to `data/augmented_audio/`. |
| `train_lora.py` | ⚠️ **Written, NOT run.** Grounded in the real `qwen_asr` package API (inspected directly this session, not guessed) — prompt format, processor call signature, and model structure are copied from `qwen_asr/inference/qwen3_asr.py`'s own inference path. Zero forward/backward passes executed. Loading the full model locally failed with an MPS buffer error earlier this session — expect device/dtype debugging on first real GPU run. |
| `eval_names.py` | ⚠️ **Written, NOT run.** Needs a trained adapter. |
| `splice_inference.py` | ⚠️ **Written, core logic unit-tested.** The word-alignment function (`SpliceASR._align_words`) was tested standalone against 4 realistic cases (identical, name-substitution, insertion, worst-case garbage) — all produced correct/sensible pairings. The confidence-scoring path in `_pass1` is a **placeholder** (uniform 1.0, never flags on confidence alone) — flagging currently works only via gazetteer phonetic-similarity, not low-confidence detection. Fix before relying on the confidence half of the flagging design. |
| `benchmark_acoustic_contrastive_contextual.py` (in `../benchmark/`) | ⚠️ **Written, NOT run.** Mirrors `benchmark_chunks_dynamic.py`'s structure exactly. Needs a trained adapter to produce anything. |

**What a GPU Claude session should actually do: start at §5 Step 3
(`train_lora.py`) below — data prep is done, don't redo it unless the
audio/gazetteer changed.** Read §"Gaps to close on the GPU machine" first.

---

## Gaps to close on the GPU machine (read before running anything)

1. **`train_lora.py`'s Dataset/collator has never executed.** The logic is
   grounded in the real processor API, but the first run will very likely
   surface shape/dtype/device mismatches. Budget real debugging time here.
2. **`splice_inference.py`'s confidence scoring is a placeholder.** Before
   trusting the splice design's confidence-based flagging (not just the
   gazetteer-similarity half), inspect whether `qwen_asr`'s `generate()`
   cleanly exposes per-token scores in a way that maps back to whole words
   (`output_scores=True, return_dict_in_generate=True` is wired in, but the
   score→word mapping is not implemented — see `_pass1`'s docstring).
3. **`prepare_data.py` needs a manual spot-check pass**, per the plan's own
   requirement — do not train on `train.jsonl` blindly:
   - Per-chunk name attribution is call-level, not verified per-chunk (a
     call's found names are attached to every chunk of that call).
   - `substitute_name_span()` uses a loose 0.55 similarity floor to find
     which Devanagari word to replace — verify a sample of positive
     examples' `target_hindi` actually has the substitution in a sane spot.
   - `build_entity_devanagari.py`'s methodology was already spot-checked
     and hardened this session (see §Data Prep Results — a punctuation bug
     and a per-call-vs-per-position matching bug were found and fixed in
     the script itself, not just patched in the output). Still worth
     reading the current 35 `"source": "corpus"` entries once before
     trusting them — it's a short list — but it's had real scrutiny, not
     none.
4. **`benchmark_acoustic_contrastive_contextual.py`'s Roman/v22ph columns
   are identical** (`model_output_roman_urdu` == `model_output_v22_phonetic`)
   — the v2.2+phonetic correction is already applied in one transliteration
   pass since `SpliceASR` outputs Hindi directly, there's no separate
   "raw vendor Roman" baseline the way the earlier pipelines had one. This
   is intentional, not a bug — just don't expect two different values there.
5. **`splice_inference.SpliceASR` loads two full model instances** (base +
   adapter) rather than toggling one via `enable_adapter_layers()` /
   `disable_adapter_layers()` — simpler to get right first, costs 2x VRAM.
   Revisit if that's tight on the training GPU.

---

## 1. Why this exists

Post-ASR text correction (`phonetic_contrastive_model/`) can only fix
**spelling** drift — it cannot recover a name the ASR never actually heard
correctly. Example, verified this session: gold says `Danish Ali`, the raw
Hindi ASR output (from `/chughtai` endpoint) is `दानिशली` (daanishli) — a
different, wrong word, not a misspelling of the right one. No lexicon can fix
that; the model needs to be taught to produce the right Devanagari at
decode time.

This project fine-tunes the **decoder**, not the audio encoder — the encoder
stays completely frozen. Precisely: `q_proj`, `v_proj`, `k_proj`, `o_proj`
inside the decoder's self-attention, via LoRA (rank 16).

Full rationale, risks, and the "why selective-splice not a raw adapter swap"
reasoning: `ard/qwen-asr-finetuning-plan.md` (repo root). Read that for
context; this file is the execution spec.

## 2. Directory layout

```
qwen3-asr-local/acoustic_contrastive_contextual_model_finetuned/
  README.md                     # this file
  requirements.txt
  build_entity_devanagari.py    # Step 0 -- DONE, run it, don't guess
  prepare_data.py                # Step 1 -- DONE
  augment_audio.py                # Step 2 -- DONE
  train_lora.py                    # Step 3 -- START HERE on GPU
  eval_names.py                     # Step 4
  splice_inference.py                # Step 5 (SpliceASR class)
  data/
    entities_devanagari.json        # ALREADY BUILT -- 35 corpus-sourced, 517 itrans_fallback
    train.jsonl                      # ALREADY BUILT -- 305 examples
    eval.jsonl                        # ALREADY BUILT -- 53 examples, 10 held-out calls
    adversarial.jsonl                  # ALREADY BUILT -- 106 examples
    train_augmented.jsonl               # ALREADY BUILT -- 1300 examples
    augmented_audio/                     # ALREADY BUILT -- 466MB
  adapters/
    <run_name>/
      phase1/ phase2/ phase3/            # NOT YET CREATED -- train_lora.py output
```

`benchmark/benchmark_acoustic_contrastive_contextual.py` lives outside this
directory, matching where every other benchmark script in this repo lives.

It imports, but does not duplicate, `acoustic_contextual_biasing/retriever.py`
(`NameRetriever`) — that stays the reusable library.

## 3. Prerequisites

```
pip install peft accelerate bitsandbytes   # torch/transformers/soundfile/
                                             # openpyxl already present
```
`audiomentations` from the original requirements.txt turned out unnecessary
— `augment_audio.py` was implemented with numpy+soundfile directly instead,
one fewer dependency.

- Base model: `Qwen/Qwen3-ASR-1.7B` (confirmed locally-loadable via the
  `qwen_asr` package this session, though full loading hit an MPS buffer
  error on this machine's GPU — a CUDA machine should not hit that specific
  issue, but budget time for it regardless).
- GPU with at least 16GB VRAM for LoRA (24GB comfortable, more if running
  `SpliceASR`'s two-full-model-instance design, see gap #5 above).

## 4. Data Prep Results (already run, real numbers)

### Step 0 — gazetteer coverage

```
python build_entity_devanagari.py
```

Confirmed this session: only entries where a Devanagari word's **raw,
uncorrected** transliteration is fuzzy-close (≥0.80) to the gazetteer term
**AND** the same call's independent gold text confirms the name was said
survive as `"source": "corpus"` — this two-part check exists because two
weaker versions were tried and failed:
- Naive fuzzy matching alone (threshold 0.82, no gold check): harvested the
  ASR's own errors as if correct (`ahsan` matched to a word romanizing to
  `aisi` — literally the mistake this project exists to fix).
- Exact-match-after-correction: the *learned phonetic model* and even the
  *curated v22 lexicon* both "fix up" wrong spellings on the way to Roman
  text, so testing post-correction Roman output isn't a valid check either
  — had to bypass all correction layers and use raw transliteration.

**Result: 35/552 gazetteer entries have real, gold-confirmed training
audio** (given names + places + orgs combined). Manual spot-check of the
first run (39 entries) found 3 bad matches, which led to fixing the SCRIPT
itself rather than one-off patching the output — now 35 entries, all
re-verified after the fix:
- `talha → कल्हा।` and `wazirabad → रज़िराबाद` were both real phonetic
  mismatches (wrong initial consonant) that only matched because
  gold-confirmation is per-CALL, not per-position -- the gazetteer term was
  said somewhere in that call, and a coincidentally similar-sounding
  Devanagari word existed elsewhere in the same call's Hindi text. Fixed by
  requiring the initial sound to agree (`term[0] == roman[0]`) before
  accepting a fuzzy match.
- `gujranwala → गुर्जरवाला।` had a trailing danda (।, U+0964) baked into
  the stored spelling -- Devanagari punctuation marks sit *inside* the same
  Unicode block as the letters, so a naive `[ऀ-ॿ]+` word regex swallows
  them. Fixed with a `clean_word()` strip step; `gujranwala` now correctly
  stays in the list at a clean `गुर्जरवाला` (its only problem was the
  stray punctuation, not the underlying match).

Rerunning `build_entity_devanagari.py` from scratch with both fixes
confirmed `talha`/`wazirabad` now correctly excluded and `gujranwala`
correctly retained, cleaned. The rest of the gazetteer (517 entries) are
ITRANS-generated fallbacks, usable only as adversarial decoys, never as
positive-example targets.

### Step 1 — training triplets

```
python prepare_data.py
```

62 calls → train, 10 calls → eval (holdout is by call, not chunk, to avoid
leakage). Output: **305 train (199 positive + 106 negative)**, **53 eval**,
**106 adversarial**. A Devanagari-script filter was added mid-session after
finding some `model_output_hindi` rows are already Roman/English text (ASR
hallucination in the source data, not something this script introduces) —
those rows are now excluded from target construction.

### Step 2 — augmentation

```
python augment_audio.py
```

**1300 total examples** (305 original + 995 augmented variants — speed
0.9x/1.1x, synthetic noise at 20dB SNR, ±3dB gain — applied to positive
examples only, per plan). Below the plan's ~2000-3000 estimate since
negatives were deliberately left unaugmented; revisit `--augment-negatives`
if more volume is wanted.

## 5. Step-by-step: what's left to run

### Step 3 — `train_lora.py`

```bash
python train_lora.py --phase 1 --data data/train_augmented.jsonl --run-name run1
python train_lora.py --phase 2 --data data/train_augmented.jsonl --resume-from adapters/run1/phase1 --run-name run1
python train_lora.py --phase 3 --data data/adversarial.jsonl     --resume-from adapters/run1/phase2 --run-name run1
```

LoRA config: `r=16, alpha=32, target=[q_proj,v_proj,k_proj,o_proj], dropout=0.05`.
Phase LRs: 1e-4 / 1e-4 / 5e-5 (phase 3 lower — corrective, not additive).
`--qlora` flag available for 4-bit loading on 12GB GPUs.

### Step 4 — `eval_names.py`

```bash
python eval_names.py --adapter adapters/run1/phase3 --eval data/eval.jsonl --adversarial data/adversarial.jsonl --base-only
python eval_names.py --adapter adapters/run1/phase3 --eval data/eval.jsonl --adversarial data/adversarial.jsonl
```

Run `--base-only` first to get the "before" row, then the adapter run.
**Do not proceed past this step if `general_accuracy_negative_examples`
dropped vs. the base-only run** — that's the regression gate.

### Step 5 — `splice_inference.py`

Not run standalone — used as a library by Step 6. Fix gap #2 (confidence
scoring) before trusting it fully; it will still function on
gazetteer-similarity flagging alone even unfixed.

### Step 6 — full benchmark

```bash
cd ../benchmark
python benchmark_acoustic_contrastive_contextual.py resume --adapter ../acoustic_contrastive_contextual_model_finetuned/adapters/run1/phase3 --calls 5   # smoke test first
python benchmark_acoustic_contrastive_contextual.py resume --adapter ../acoustic_contrastive_contextual_model_finetuned/adapters/run1/phase3             # full 72-call run
```

Writes `model_acoustic_contrastive_contextual` and
`benchmark_summary_acoustic_contrastive_contextual` into the same
`lab_test_80_calls_urdu_roman_urdu_benchmarked.xlsx` every other pipeline
this session wrote into — saves incrementally after every call, same as
`benchmark_chunks_dynamic.py`.

**Compare the resulting corpus accuracy against:**

| Pipeline already in that workbook | Corpus accuracy |
|---|---|
| Vendor Hindi + static chunks + local v2.2/phonetic | 64.92% |
| `/chughtai` HTTP + dynamic chunks, no fine-tune | **65.87%** |
| This fine-tune | *(run it to find out)* |

## 6. Explicit non-goals (unchanged from the plan)

- Only 35/552 gazetteer entries (6%) have real training audio — confirmed,
  not estimated, this session. Broader coverage needs more real calls or
  synthetic/TTS audio, out of scope here.
- Not addressing whole-sentence hallucination (the separate failure mode
  where the model fabricates an entire unrelated conversation on unclear
  audio) — name/entity recognition only.
- Not guaranteed to break the ~65-71% corpus accuracy ceiling seen across
  every pipeline tested this session (8kHz narrowband audio is a likely
  contributing limit) — this targets a specific, real gap, not a general
  accuracy ceiling.
