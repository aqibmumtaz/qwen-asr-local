# Qwen3-ASR Fine-Tuning Plan — Urdu Names & Locations

## Problem Statement

Qwen3-ASR-1.7B produces Hindi output from Urdu audio. It struggles with:
- **Person names**: "Shahid" → शहीद (martyr) instead of शाहिद (name)
- **Local places**: "Kharian" → खड़ियान instead of खारियां
- **Context biasing FP**: Retriever-based biasing introduces false positives because common words ("lekin", "mahine", "sahab") phonetically match entity names ("LinkedIn", "Mahineen", "Shahid")

The text-side phonetic contrastive model solves post-ASR correction but cannot fix acoustic-level confusion where the model doesn't even produce the right consonants/vowels.

## Goal

Fine-tune Qwen3-ASR to:
1. Natively output correct Hindi for Urdu names/locations (no post-processing needed)
2. Properly respect `context=` parameter — use context ONLY when audio matches, never hallucinate

## Architecture

Qwen3-ASR is an encoder-decoder speech model:
```
Audio (mel spectrogram) → Audio Encoder → Cross-attention → Text Decoder → Hindi tokens
                                              ↑
                                    context= (text prefix/prompt)
```

The `context=` parameter is injected as a text prefix that biases the decoder's attention. Fine-tuning targets the decoder (and optionally cross-attention layers).

## Training Data

### Source
- 80 lab-test calls (400+ chunks, 25s each)
- Ground-truth benchmark in Roman Urdu (`benchmark_roman_urdu` column)
- Entity gazetteer: 357 given names, 148 places, 47 organisations

### Data Preparation Pipeline

```
Step 1: For each audio chunk, identify if it contains a name/place
        - Match benchmark_roman_urdu against gazetteer
        - Label: {audio_path, contains_names: ["Shahid", "Lahore"], correct_hindi: "..."}

Step 2: Create training triplets:
        (audio, context, target_hindi)

Step 3: Split into:
        - Name-bearing chunks (positive): context = names that appear → target has correct name
        - Name-free chunks (negative): context = "" → target has no names
        - Adversarial negatives: context = names that DON'T appear → target should NOT contain those names
```

### Training Set Structure

| Type | Audio | Context | Target | Purpose |
|------|-------|---------|--------|---------|
| Positive | chunk with "Shahid" spoken | "Shahid" | शाहिद साहब ने कहा... | Learn to use context correctly |
| Negative (no ctx) | chunk without names | "" | normal transcription | Don't hallucinate names |
| Adversarial | chunk says "lekin" | "LinkedIn" | लेकिन (NOT LinkedIn) | Don't bias when audio doesn't match |

### Data Volume Estimate

From 80 calls × ~5-6 chunks/call = ~400 chunks:
- ~120 chunks contain at least one name (30%)
- ~280 chunks have no names (70%)
- Generate 120 adversarial examples (pair non-name audio with random context)

Total: ~520 training examples. Small but sufficient for LoRA fine-tuning.

### Ground-Truth Hindi Generation

Current benchmark only has Roman Urdu. To get correct Hindi targets:
1. Take vendor `model_output_hindi` as base
2. For name tokens only: reverse-transliterate from correct Roman Urdu → Hindi
   - "Shahid" → शाहिद
   - "Kharian" → خاریاں → खारियां
3. Replace incorrect name portions in vendor Hindi with correct Hindi names
4. Manual verification of ~120 name-bearing chunks

Script: `prepare_finetuning_data.py` (to be written)

## Fine-Tuning Method

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # decoder attention
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | Standard for LoRA |
| Batch size | 4 (gradient accumulation 8) | Fit in 24GB VRAM |
| Epochs | 10-20 | Small dataset, need multiple passes |
| Warmup | 10% of steps | Stable convergence |
| LoRA rank | 16 | Balance between capacity and overfitting |
| Max audio length | 25s (matches chunk size) | No truncation needed |
| Precision | bf16 | Speed + memory |

### Hardware Requirements

| Option | GPU | VRAM | Training Time |
|--------|-----|------|--------------|
| LoRA (recommended) | 1× RTX 4090 / A100 | 16-24 GB | 2-4 hours |
| Full fine-tune | 1× A100 80GB | 40+ GB | 8-12 hours |
| QLoRA (4-bit) | 1× RTX 3090 | 12 GB | 3-5 hours |

Your GPU server (192.168.99.117) can be used if it has sufficient VRAM.

## Training Phases

### Phase 1: Domain Vocabulary (names as text)

Fine-tune decoder to output correct Hindi tokens for Pakistani names:
- Input: audio containing "shahid"
- Target: शाहिद (not शहीद)

This teaches the model's vocabulary distribution for Urdu-specific names.

### Phase 2: Context-Conditioned (teach proper biasing)

Fine-tune with context= parameter:
- Positive: audio="shahid", context="Shahid" → शाहिद ✓
- Negative: audio="lekin", context="LinkedIn" → लेकिन ✓ (ignore misleading context)
- No-context: audio="shahid", context="" → whatever model produces (no constraint)

This teaches the model: "only use context when audio acoustically matches."

### Phase 3: Adversarial Robustness

Add hard negatives where:
- Context contains phonetically similar but wrong name (e.g., context="Maheen" but audio says "mahine")
- Target should produce महीने (months) NOT माहीन (name)

This specifically prevents the FP corruption we observed in the benchmark.

## Evaluation

### Metrics

| Metric | What it measures |
|--------|-----------------|
| Name WER | Word error rate on name tokens only |
| Name precision | % of output names that are correct |
| Name recall | % of spoken names that are captured |
| Context FP rate | % of context names that appear in output when NOT in audio |
| General WER | Overall word error rate (should not degrade) |

### Test Set

Hold out 10 calls (50-60 chunks) from training. Evaluate:
1. Without context (baseline improvement)
2. With correct context (biasing improvement)
3. With adversarial context (FP resistance)

### Success Criteria

| Metric | Before (current) | Target |
|--------|-------------------|--------|
| Name recall (no context) | ~48% | 70%+ |
| Name recall (with correct context) | ~50% | 85%+ |
| Context FP rate | HIGH (causes corruption) | < 5% |
| General WER | baseline | no degradation |

## Deployment

### After Training

1. Export LoRA adapter weights
2. Merge into base model: `model.merge_and_unload()`
3. Convert to GGUF for llama.cpp inference (if needed for local/edge)
4. Or serve via the GPU WebSocket server (replace base model with fine-tuned)

### Inference Pipeline (post fine-tune)

```
Audio → Fine-tuned Qwen3-ASR (context=per-call names if available, else "")
      → Hindi (with correct names natively)
      → hindi_to_roman_urdu (transliteration)
      → Phonetic corrector (catch remaining edge cases)
```

The fine-tuned model handles names at the acoustic level. The phonetic corrector handles any remaining text-level errors. No retriever needed.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Overfitting on 400 examples | LoRA (low rank), early stopping, hold-out eval |
| Catastrophic forgetting (general speech degrades) | Keep learning rate low, mix in general Hindi speech data |
| Model memorises specific audio clips | Data augmentation (speed perturbation, noise, volume) |
| Context hallucination persists | Adversarial training (Phase 3) with explicit negative examples |

## Data Augmentation

To avoid overfitting on 400 chunks:
- **Speed perturbation**: 0.9×, 1.0×, 1.1× → 3× data
- **Noise injection**: office noise, phone line noise at SNR 15-25dB
- **Volume variation**: ±3dB random gain
- **Name substitution**: swap "Shahid" label with other gazetteer names, re-synthesize if TTS available

Effective training set: ~2000-3000 examples after augmentation.

## Timeline

| Week | Task |
|------|------|
| 1 | Data preparation: extract name-bearing chunks, generate correct Hindi targets, manual QA |
| 2 | Phase 1 training: domain vocabulary LoRA, evaluate name WER improvement |
| 3 | Phase 2 training: context-conditioned, evaluate biasing precision |
| 4 | Phase 3: adversarial robustness, final evaluation, deployment |

## Files to Create

```
qwen3-asr-local/
  finetuning/
    prepare_data.py          # Extract training triplets from benchmark
    train_lora.py            # LoRA fine-tuning script
    eval_names.py            # Name-specific evaluation metrics
    augment_audio.py         # Speed/noise/volume augmentation
    data/
      train.jsonl            # {audio_path, context, target_hindi}
      eval.jsonl             # Hold-out test set
      adversarial.jsonl      # Hard negatives
```
