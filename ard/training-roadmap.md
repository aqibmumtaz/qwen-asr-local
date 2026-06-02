# Urdu ASR Training Roadmap

## Goal
Fine-tune Qwen3-ASR-1.7B to output **Roman Urdu** directly from 8kHz call-center audio, eliminating the Hindi→transliteration pipeline and its error-compounding problem.

---

## Key Findings

### The Core Problem
Hindi ASR models (Qwen3-ASR, Whisper) produce **approximated Hindi** when they hear Urdu speech. They hallucinate the closest-sounding Hindi word for Urdu-specific vocabulary (Arabic/Persian loanwords, religious terms, proper nouns). This wrong Hindi then transliterates to wrong Roman Urdu — the errors compound at every step.

### Model Landscape (researched May 2026)

| Model | Mean WER | Notes |
|-------|----------|-------|
| Qwen3-ASR-1.7B | 5.76% | Best open-source on HF leaderboard; supports Hindi, not Urdu |
| Whisper large-v3 | 7.44% | Strong multilingual; same Hindi-not-Urdu limitation |
| Whisper large-v3-turbo | 7.83% | Faster variant, similar quality |
| NVIDIA Canary-1B | 6.5% | Only en/de/es/fr |
| AI4Bharat IndicConformer | — | Hindi-specific, gated access |
| Meta MMS-1b-all | 22.54% | 1000+ languages but weak accuracy |
| SeamlessM4T v2 | — | Supports Urdu but not ASR-focused |

**Hinglish models investigated:** ujs/hinglish-asr and cricai-hinglish-asr-model — both weak/unvalidated, not production-ready.

**ShunyaLabs Zero Code Switch:** Commercial Hinglish API (model=zero-codeswitch, language_code=hi-en). Claims 3.10% WER on English benchmarks. Not open-source, not fine-tunable.

### Audio Quality Findings

| Finding | Detail |
|---------|--------|
| Call center recordings | 5/6 files are 8kHz mono 16-bit PCM (G.711 telephony standard) |
| One outlier | 44.1kHz — re-encoded (same phone number has another file at 8kHz) |
| 8kHz = genuine limitation | Phone networks (PSTN/G.711) physically limit audio to 0–4kHz |
| Upsampling helps inference | 16kHz WavePad upsample gives +3.1% avg confidence, +8.3% min confidence vs raw 8kHz |
| Can call center provide 16kHz? | Only if they use VoIP with G.722/Opus codec; G.711/PSTN = always 8kHz |

### Training Format Discovery
- Encoder-decoder models (Qwen3-ASR, Whisper) need only `(audio_clip, text)` pairs
- **No word-level timestamps required** — model learns alignment automatically via cross-attention
- Ideal clip length: 5–30 seconds (speaker-turn boundaries)
- Must resample all audio to 16kHz before training (model's native rate)

### Annotation Data Inventory (data/CLL audios/)
- **6 WAV files**, 15.3 minutes total audio
- **10 xlsx annotation files** — utterance-level (not word-level)
- Format: `[MM:SS] Speaker N: Nastaliq Urdu text` + corrected Nastaliq column + speaker role
- Some files have code-switched English preserved (e.g., "CT Scan", "MRI", "Night")
- Annotations are in Nastaliq Urdu script — need conversion to Roman Urdu for training target

### Three Solution Paths (ranked by ROI)

1. **Fine-tune to output Roman Urdu directly** (best long-term)
   - Skip Hindi entirely; audio → Roman Urdu tokens
   - Requires 100–300h corrected Roman Urdu transcripts
   - Permanent solution — no transliteration errors

2. **LLM post-correction layer** (fastest to try now)
   - Keep: Qwen3-ASR → Hindi
   - Add: LLM (Qwen2.5-7B) that fixes noisy Hindi → proper Roman Urdu
   - Zero training needed; use lexicon patterns as few-shot examples
   - Immediate improvement, production-ready in days

3. **Expand lexicon + rules** (current approach, has ceiling)
   - Growing lexicons.json with word-level corrections
   - Doesn't scale — every new error needs a manual entry
   - Will always chase the long tail

### 8kHz Training Strategy
- **Standard industry practice:** Google, AWS Transcribe, Azure Speech all have "telephony" models trained on 8kHz-upsampled data
- **Correct approach:** Upsample 8kHz→16kHz for both training AND inference
- **The model learns telephony audio characteristics** — it adapts to the empty 4–8kHz band
- **Don't mix sample rates:** All training data should be consistently 8kHz-upsampled-to-16kHz

### Pseudo-Labeling Quality Concern
- Current Hindi output on 8kHz Urdu audio is unreliable
- **Mitigation:** Confidence-based filtering — only auto-accept segments with avg confidence > 0.85
- Human correction for the uncertain middle band (0.65–0.85)
- Discard segments below 0.65 (garbage in = garbage out)

---

## Current State

| Asset | Details |
|-------|---------|
| Unlabeled audio | ~20,000 call-center recordings (8kHz mono 16-bit PCM) |
| Annotated audio | 6 WAV files, 15.3 min total, with xlsx annotations (Nastaliq Urdu per speaker turn) |
| Annotation format | Utterance-level: `[MM:SS] Speaker N: text` — no word-level timestamps |
| Audio quality | 5/6 files at 8kHz (G.711 telephony), 1 file at 44.1kHz (re-encoded) |
| Current pipeline | Qwen3-ASR → Hindi → Nastaliq → Roman Urdu (lexicon-corrected) |
| Core problem | Hindi ASR hallucinates wrong Hindi for Urdu words → bad transliteration |

---

## Training Data Format

Each sample = one `(audio_clip, text)` pair:

```json
{"audio": "chunk_0001.wav", "text": "assalam o alaikum chughtai lab se baat kar raha hoon"}
```

**Requirements:**
- Audio: WAV, mono, **16kHz** (upsample all 8kHz with `ffmpeg -ar 16000`)
- Duration: 5–30 seconds per clip (speaker-turn boundaries)
- Text: Roman Urdu, normalized, no timestamps
- No word-level alignment needed — model learns alignment via cross-attention

---

## Phase 1: Data Preparation (Week 1–2)

### 1.1 Normalize existing annotations
- Parse xlsx files → extract `(start_time, end_time, text, speaker)` tuples
- Convert Nastaliq Urdu text → Roman Urdu (use existing pipeline + manual correction)
- Split full audio at speaker-turn boundaries into individual clips
- Upsample all clips to 16kHz: `ffmpeg -i input.wav -ar 16000 -ac 1 output.wav`

### 1.2 Build annotation tool
- Simple web UI or CLI that plays audio chunk + shows auto-generated Roman Urdu
- Annotator corrects text → saves to metadata.jsonl
- Priority: speed (annotator hears + edits, not transcribes from scratch)

### 1.3 Output
```
dataset/
├── train/
│   ├── metadata.jsonl
│   └── audio/
│       ├── chunk_0001.wav
│       ├── chunk_0002.wav
│       └── ...
├── val/
│   ├── metadata.jsonl
│   └── audio/
└── test/
    ├── metadata.jsonl
    └── audio/
```

---

## Phase 2: Pseudo-Labeling at Scale (Week 2–4)

### 2.1 Bulk transcription
- Run Qwen3-ASR (Hindi mode) + full pipeline on all 20K files
- Extract per-word confidence scores via `hf_asr_with_confidence()`
- Store: `{audio_path, hindi_text, roman_urdu_text, avg_confidence, min_confidence}`

### 2.2 Confidence-based filtering
- **Tier A (auto-accept):** avg_confidence > 0.85 → use as-is for training
- **Tier B (human review):** 0.65 < avg_confidence < 0.85 → send to annotation queue
- **Tier C (discard):** avg_confidence < 0.65 → skip (too noisy/uncertain)

### 2.3 Segment splitting
- Split long recordings at silence boundaries (VAD: silero-vad or webrtcvad)
- Target: 5–15s segments per training sample
- Discard segments < 1s (noise/clicks) or > 30s (too long for GPU memory)

### 2.4 Expected yield
- 20K files × ~3 min avg = ~1000 hours raw audio
- After VAD splitting + confidence filtering: estimate 300–500 hours usable
- Tier A alone: likely 100–200 hours (sufficient for meaningful fine-tuning)

---

## Phase 3: Human Annotation Loop (Week 3–6, ongoing)

### 3.1 Correct Tier B segments
- Annotators see: audio + auto-generated Roman Urdu
- Annotators fix: wrong words, missing words, code-switched English
- Output: corrected `(audio, roman_urdu_text)` pairs

### 3.2 Validate Tier A samples (spot-check)
- Random 5–10% of auto-accepted segments go to human review
- Measures pseudo-label quality → adjusts confidence threshold if needed

### 3.3 Active learning priority
- Sort Tier B by confidence ascending (most uncertain first)
- These are the highest-value samples for the model to learn from
- Each corrected sample teaches the model its biggest blind spots

---

## Phase 4: Fine-Tuning (Week 5–8)

### 4.1 Base model
- **Qwen3-ASR-1.7B** (encoder-decoder, attention-based alignment)
- Start from pretrained weights (Hindi/multilingual knowledge transfers)

### 4.2 Training config
| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-5 to 5e-5 | LoRA: 1e-4 |
| Batch size | 8–16 | Per GPU, gradient accumulation as needed |
| Epochs | 3–10 | Early stopping on val WER |
| Method | Full fine-tune or LoRA (rank 16–64) | LoRA for quick iteration, full for final |
| GPU | 1× A100 80GB or 2× A10 24GB | bf16 training |
| Audio preprocessing | Resample to 16kHz, normalize amplitude | Match inference pipeline |
| Target text | Roman Urdu (direct output) | No Hindi intermediary |
| Validation | Hold-out 6 gold-annotated files (15.3 min) | WER on Roman Urdu |

### 4.3 Training stages
1. **Stage 1 — Telephony adaptation:** Fine-tune on Tier A (auto-labeled, high-confidence) only. Goal: adapt acoustic model to 8kHz-upsampled telephony audio.
2. **Stage 2 — Urdu vocabulary:** Add human-corrected Tier B data. Goal: learn Urdu-specific words the base model gets wrong.
3. **Stage 3 — Domain specialization:** Add domain-specific corrections (medical terms for Chughtai Lab context). Goal: production accuracy on call-center vocabulary.

---

## Phase 5: Evaluation & Iteration (Week 8+)

### 5.1 Metrics
| Metric | Target | How |
|--------|--------|-----|
| WER (Roman Urdu) | < 15% | Compare model output vs human-corrected ground truth |
| Domain accuracy | > 90% | Medical terms, proper nouns (Chughtai, CT scan, MRI) |
| Latency | < 2× real-time | 10s audio transcribed in < 20s (GPU inference) |
| Confidence calibration | Pearson > 0.7 | Confidence score correlates with actual correctness |

### 5.2 Iteration loop
```
Train → Evaluate on test set → Identify failure patterns →
→ Add targeted training data for failures → Retrain
```

### 5.3 A/B comparison
- Run same test audio through:
  - Old pipeline: Qwen3-ASR → Hindi → transliterate → Roman Urdu
  - New model: Qwen3-ASR-finetuned → Roman Urdu directly
- Measure WER improvement, latency improvement, and error type reduction

---

## Infrastructure Requirements

| Resource | Purpose | Estimated Cost |
|----------|---------|---------------|
| GPU server (A100 80GB) | Training | ~$2/hr spot, ~$50–100 per training run |
| Storage | 20K audio files (~50–100 GB) | Minimal |
| Annotation tool | Human correction UI | Build in-house (1–2 days) |
| Annotators | 2–3 Urdu speakers | Correct Tier B segments |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Pseudo-labels too noisy | Strict confidence threshold (0.85); human spot-checks |
| 8kHz audio quality ceiling | Train on 8kHz-upsampled (matches production); acceptable for telephony |
| Insufficient training data | Start with Tier A only (100–200h); iterate as more corrections come in |
| Model forgets English | Include 10–20% English/code-switched samples in training mix |
| Overfitting to Chughtai domain | Add generic Urdu conversation data if available; regularization via LoRA |

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1–2 | Data prep | Annotation parser, audio splitter, 16kHz pipeline |
| 2–4 | Pseudo-labeling | Bulk transcription of 20K files, confidence-filtered dataset |
| 3–6 | Human annotation | Corrected Tier B segments, validated Tier A |
| 5–8 | Training | Fine-tuned model (3 stages) |
| 8+ | Evaluation | A/B tests, WER benchmarks, production deployment |

**First measurable improvement:** After Phase 4 Stage 1 (~Week 6), the model should already outperform the Hindi→transliteration pipeline on telephony Urdu.
