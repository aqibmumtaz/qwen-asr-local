# Qwen3-ASR Fine-Tuning Plan — Acoustic Contrastive Contextual Model

**Implementation lives separately, in `qwen3-asr-local/acoustic_contrastive_contextual_model_finetuned/README.md`
— a self-contained, step-by-step execution spec meant to be run by a Claude
Code session on the GPU machine, independent of this document.** This file
is the plan/rationale record: why this approach, what data, what risks, what
was decided and why. Read this for context; read the other README to
actually build and run it.

## Problem Statement

Qwen3-ASR-1.7B produces Hindi output from Urdu audio. It struggles with:
- **Person names**: "Shahid" → शहीद (martyr) instead of शाहिद (name)
- **Local places**: "Kharian" → खड़ियान instead of खारियां
- **Context biasing FP**: Retriever-based biasing introduces false positives because common words ("lekin", "mahine", "sahab") phonetically match entity names ("LinkedIn", "Mahineen", "Shahid")

Verified this session on real benchmark audio: gold says `Danish Ali`, raw
Hindi ASR output is `दानिशली` (daanishli) — a different wrong word, not a
misspelling of the right one. The text-side phonetic contrastive model
(`phonetic_contrastive_model/`) solves post-ASR spelling correction but
cannot fix this class of error, because by the time text arrives the model
never produced the right consonants/vowels in the first place.

## Goal

Fine-tune Qwen3-ASR to:
1. Natively output correct Hindi for Urdu names/locations it has actually
   heard in training audio (no post-processing needed for those).
2. Properly respect `context=` — use it only when the audio matches, never
   hallucinate from a hint.
3. Do both **without measurably degrading general (non-entity) transcription
   accuracy** — this was the driving constraint on the whole design, see
   "Runtime Design" below.

## Architecture

Qwen3-ASR is an encoder-decoder speech model:
```
Audio (mel spectrogram) → Audio Encoder → Cross-attention → Text Decoder → Hindi tokens
                                              ↑
                                    context= (text prefix/prompt)
```

**Fine-tuning targets the decoder's attention layers only** (`q_proj`,
`v_proj`, `k_proj`, `o_proj`), via LoRA. The audio encoder is never touched.
This is deliberately *not* called "acoustic model fine-tuning" in the
traditional ASR-research sense (that term means the encoder) — it's decoder/
LM fine-tuning aimed at fixing errors that manifest at the acoustic level.
"Acoustic Contrastive Contextual" describes the training *design*
(contrastive positive/adversarial pairs, context-conditioned), not a claim
about which architectural component is retrained.

## Training Data

### Source
- 80 lab-test calls, re-chunked with silence-aware boundaries
  (`benchmark/lab_test_80_audios_chunks_dynamic/`, ~400 chunks — preferred
  over the original hard-25.00s chunks, which measurably cut mid-speech)
- Ground-truth benchmark in Roman Urdu (`benchmark_roman_urdu` column)
- Entity gazetteer (`data/entities.json`): 357 given names, 148 places, 47
  organisations

### Gazetteer coverage — checked directly against the actual audio this session

| Gazetteer | Total entries | Actually appear in benchmark audio |
|---|---|---|
| Given names | 357 | **63 (17.6%)** |
| Places | 148 | **17 (11.5%)** |

This is the single most important constraint on this plan: **fine-tuning can
only teach the model acoustic patterns it has actually heard.** The ~82-88%
of the gazetteer absent from this audio gets zero benefit from this fine-tune
and stays dependent on the existing retrieval + phonetic-correction fallback.
Broader coverage requires either more real calls or synthetic/TTS audio for
the missing names — a separate, larger scope decision, not assumed here.

### Data Preparation Pipeline

```
Step 1: For each audio chunk, identify if it contains a name/place
        - Match benchmark_roman_urdu against gazetteer (only the 80
          audio-confirmed entities matter in practice)
        - Label: {audio_path, contains_names: ["Danish", "Ahsan"], correct_hindi: "..."}

Step 2: Create training triplets: (audio, context, target_hindi)

Step 3: Split into:
        - Name-bearing chunks (positive): context = names that appear → target has correct name
        - Name-free chunks (negative): context = "" → target has no names
        - Adversarial negatives: context = names that DON'T appear → target should NOT contain those names
```

### Training Set Structure

| Type | Audio | Context | Target | Purpose |
|------|-------|---------|--------|---------|
| Positive | chunk with "Ahsan" spoken | "Ahsan" | ...अहसन साहब ने कहा... | Learn to use context correctly |
| Negative (no ctx) | chunk without names | "" | normal transcription | Don't hallucinate names |
| Adversarial | chunk says "lekin" | "LinkedIn" | लेकिन (NOT LinkedIn) | Don't bias when audio doesn't match |

### Data Volume Estimate

From ~400 chunks: ~120 contain at least one (audio-confirmed) name (30%),
~280 have none (70%). Generate ~120 adversarial examples pairing non-name
audio with a random decoy name. Total ~520 raw examples — thin for a 1.7B
model, workable for LoRA specifically, augmented to ~2000-3000 (see below).
This is a genuinely borderline dataset size; treat early results as a
feasibility pilot, not a guarantee the target metrics below are reachable
on the first run.

### Ground-Truth Hindi Generation

Current benchmark only has Roman Urdu + vendor Hindi (which itself contains
the errors being fixed). To get correct Hindi targets:
1. Take vendor `model_output_hindi` as the base text (closer to correct than
   fresh ASR on non-name words).
2. Build a Roman→Devanagari lookup for just the 80 audio-confirmed gazetteer
   entries (not a general reverse-transliterator — only ~80 fixed strings
   need mapping, simpler and safer).
3. Replace only the mis-transcribed name span in the vendor Hindi with the
   correct Devanagari from that lookup — leave everything else as-is.
4. Manual spot-check of the ~120 name-bearing chunks before training.

## Fine-Tuning Method

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # decoder attention only
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 (5e-5 for Phase 3) | Standard for LoRA; lower for the corrective adversarial phase |
| Batch size | 4 (gradient accumulation 8) | Fit in 16-24GB VRAM |
| Epochs | 10-20 | Small dataset, need multiple passes, watch eval loss for early stop |
| Warmup | 10% of steps | Stable convergence |
| LoRA rank | 16 | Balance between capacity and overfitting |
| Max audio length | 25s (matches chunk size) | No truncation needed |
| Precision | bf16 | Speed + memory |

### Hardware Requirements

| Option | GPU | VRAM | Training Time |
|--------|-----|------|--------------|
| LoRA (recommended) | 1× RTX 4090 / A100 | 16-24 GB | 2-4 hours |
| QLoRA (4-bit) | 1× RTX 3090 | 12 GB | 3-5 hours |
| Full fine-tune | 1× A100 80GB | 40+ GB | 8-12 hours (not planned — LoRA is sufficient and lower-risk) |

Your GPU server (192.168.99.117) is the intended training host if it has
sufficient VRAM.

## Training Phases

### Phase 1: Domain Vocabulary
Teach correct Hindi tokens for the 63 confirmed names / 17 places:
audio contains "ahsan" → target अहसन (not अहसान or similar near-miss).

### Phase 2: Context-Conditioned
Teach the model to actually use `context=` rather than ignoring it:
- Positive: audio="ahsan", context="Ahsan" → अहसन ✓
- Negative: audio="lekin", context="LinkedIn" → लेकिन ✓ (ignore misleading context)
- No-context: audio="ahsan", context="" → whatever model produces (unconstrained)

### Phase 3: Adversarial Robustness
Hard negatives where context contains a phonetically-plausible but wrong
name relative to the actual audio — teaches the model to resist the exact
false-positive pattern already observed in the retriever-based biasing
benchmarks this session (e.g. "mahine" ≈ "Maheen").

## Runtime Design — why a raw adapter swap was rejected

**The core open question this plan had to resolve: if LoRA is attached for
a context-biased inference call, does it risk changing words that have
nothing to do with names?**

Answer, worked through explicitly: **yes, structurally, not just
empirically.** LoRA's effect is whole-sequence — every output token in an
adapted decode depends via attention on the hidden states of every preceding
token, all of which ran through the adapted weights. Adversarial training
(Phase 3) reduces this risk but cannot mathematically eliminate it. A raw
"trust pass-2's full output" design would leave general accuracy dependent
on training working perfectly, which is not an acceptable guarantee.

**Resolution: two-pass selective splice**, not a raw adapter swap:

```
Pass 1: audio → BASE model (adapter OFF, context="") → transcript + confidence
Flag: words that are low-confidence OR phonetically near a gazetteer entry
  no flags  → pass 1 IS final. LoRA never runs.
  flags found:
Pass 2: audio → LoRA-attached model (context=candidates) → biased transcript
Align pass1 <-> pass2 (word-level, LoRA can insert/drop/reorder — this is
  not a positional swap, needs sequence alignment)
Splice: ONLY the flagged positions take pass-2's word; everything else is
  pass-1/base, untouched.
```

This makes the guarantee structural rather than empirical: **non-entity
words are never sourced from the adapted model, period** — not "usually
similar to base," but literally never generated by it. The eval-time
General WER metric (below) is then a confirmation of training quality on
the entity-span words specifically, not a load-bearing safety mechanism for
everything else.

Corollary decision: **the adapter is never merged into the base model.** It
stays a small, separate, toggleable file (`peft` `enable_adapter_layers()` /
`disable_adapter_layers()` on one loaded base model instance — no reload,
cheap to switch per request). This is what makes "pass 1 uses literally
unmodified base weights" true rather than aspirational.

## Evaluation

### Metrics

| Metric | What it measures |
|--------|-----------------|
| Name WER | Word error rate on name tokens only |
| Name precision | % of output names that are correct |
| Name recall | % of spoken names that are captured |
| Context FP rate | % of context names that appear in output when NOT in audio |
| **General WER** | Overall word error rate on **non-entity words** — the regression gate |

### Test Set

Hold out 10 entire **calls** (not just chunks — chunks from the same call
share speaker/audio characteristics, chunk-level holdout would leak).
Evaluate: (1) without context, (2) with correct context, (3) with
adversarial context.

### Success Criteria

| Metric | Before (current) | Target |
|--------|-------------------|--------|
| Name recall (no context) | ~48% | 70%+ |
| Name recall (with correct context) | ~50% | 85%+ |
| Context FP rate | HIGH (causes corruption) | < 5% |
| General WER | baseline | no degradation |

These are the plan's targets, not measured outcomes — no training run has
happened yet.

### Full-pipeline benchmark (not just name-level eval)

Beyond the name-specific metrics above, the fine-tune's actual value is
judged the same way every other pipeline in this repo has been judged this
session: `benchmark_acoustic_contrastive_contextual.py` (see implementation
README) scores the full 72-call corpus with `diff_words`, written into the
same `lab_test_80_calls_urdu_roman_urdu_benchmarked.xlsx` workbook, directly
comparable against:

| Pipeline already in that workbook | Corpus accuracy |
|---|---|
| Vendor Hindi + static chunks + local v2.2/phonetic | 64.92% |
| `/chughtai` HTTP + dynamic chunks, no fine-tune | 65.87% |
| This fine-tune | target: meaningfully above 65.87%, not ~1pt |

## Deployment

1. Export LoRA adapter weights per phase — **not merged**, kept toggleable.
2. Serve locally via the `LocalASR` pattern already used elsewhere in this
   repo (`acoustic_contextual_biasing/asr.py`,
   `benchmark_acoustic_biasing.py`) — the remote GPU/HF servers used for the
   dynamic-chunking benchmarks this session run infrastructure this project
   doesn't control and can't attach a custom adapter to.
3. Inference always goes through `splice_inference.py` (two-pass selective
   splice), never a raw adapter-attached transcribe call.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Overfitting on ~500 raw examples | LoRA (low rank), early stopping, hold-out eval |
| Catastrophic forgetting (general speech degrades) | **Structurally prevented for non-entity words** by the selective-splice design — non-name output never comes from the adapted model. Low LR / phase-specific tuning is a secondary safeguard for the entity-span words themselves. |
| Model memorises specific audio clips | Data augmentation (speed perturbation, noise, volume) |
| Context hallucination persists | Adversarial training (Phase 3); selective splice also discards any hallucinated non-entity word from pass 2 regardless, since only flagged spans are ever taken from it |
| Gazetteer coverage gap | Only 63/357 names, 17/148 places in current audio — see Training Data above. Explicitly not solved by this plan; flagged as a scope boundary. |
| Whole-sentence hallucination (unrelated fabricated content, observed this session) | **Not addressed by this plan.** Separate failure mode from name-level errors; needs its own investigation (confidence-based segment rejection, audio preprocessing). |

## Data Augmentation

To avoid overfitting on ~500 raw examples:
- **Speed perturbation**: 0.9×, 1.0×, 1.1× → 3× data
- **Noise injection**: office noise, phone line noise at SNR 15-25dB
- **Volume variation**: ±3dB random gain
- **Name substitution**: swap a name label with other *audio-confirmed*
  gazetteer names only (not the full 357 — substituting in names the model
  will never hear real audio for doesn't help); re-synthesize via TTS only
  if quality is verified against real speech first.

Effective training set after augmentation: ~2000-3000 examples.

## Timeline

| Week | Task |
|------|------|
| 1 | Data preparation: build Devanagari lookup, extract triplets, manual QA on ~120 name-bearing chunks |
| 2 | Phase 1 training: domain vocabulary LoRA, evaluate name WER improvement |
| 3 | Phase 2 training: context-conditioned, evaluate biasing precision; Phase 3 adversarial |
| 4 | `splice_inference.py` + `benchmark_acoustic_contrastive_contextual.py`, full 72-call corpus run, compare against 64.92%/65.87% baselines |

## See Also

- `qwen3-asr-local/acoustic_contrastive_contextual_model_finetuned/README.md` — the
  actual step-by-step implementation spec, self-contained for a separate
  Claude Code session to execute on the GPU machine.
- `qwen3-asr-local/phonetic_contrastive_model/` — the existing text-level
  correction this project complements, not replaces.
- `qwen3-asr-local/acoustic_contextual_biasing/` — the reusable
  retriever/ASR-client library this project imports from, without duplicating.
