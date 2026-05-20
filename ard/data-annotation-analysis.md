# Audio Data & Annotation Analysis
## File: `in-4234500300-+923000754715-20260501-221336-1777655616.988465`

---

## 1. Audio Quality Assessment

### Physical Characteristics
| Property | Value | Assessment |
|----------|-------|------------|
| **Sample Rate** | 8000 Hz | ⚠️ **Telephony quality** — narrow band |
| **Channels** | Mono (1) | ✓ Standard for call recordings |
| **Bit Depth** | 16-bit | ✓ Adequate for speech |
| **Duration** | 96.07 seconds (1.60 min) | ⚠️ **Very short sample** |
| **Bitrate** | 128 kbps | ✓ Reasonable for 8kHz mono |
| **File Size** | 1.47 MB | ✓ Compact |

### Quality Grade: **B- (Acceptable but Limited)**

**Pros:**
- ✓ Clean recording with minimal background noise
- ✓ Natural conversational speech (not scripted)
- ✓ Real call center interaction (authentic domain data)
- ✓ Dual speaker (agent + customer) with clear speaker separation

**Cons:**
- ⚠️ **8kHz sample rate:** Loses frequencies >4kHz (speech intelligibility affected)
- ⚠️ **Very short:** 96 seconds = 22 dialogue turns (insufficient for training)
- ⚠️ **Telephony compression:** Phone line bandwidth limitations reduce audio clarity
- ⚠️ **Limited acoustic diversity:** Single call, likely from same location/device

---

## 2. Annotation Structure & Quality

### Schema
| Column | Field | Example | Type |
|--------|-------|---------|------|
| Col1 | (empty) | — | — |
| Col2 | Raw ASR Transcription | "اسلام علیکم چغتائی لیب سے حسن بات کر رہا ہوں please بتائیے گا" | Mixed Urdu+English |
| Col3 | Clean Transcription | "اسلام علیکم چغتائی لیب سے حسن بات کر رہا ہوں پلیز بتائیے گا" | Urdu only |
| Col4 | Speaker Role | "ایجنٹ" (Agent) or "کسٹمر" (Customer) | Categorical |
| Col5 | Turn Number | 1, 2, 3, ... 22 | Sequential |

### Data Statistics
| Metric | Value |
|--------|-------|
| **Total rows** | 32 (including headers) |
| **Annotated turns** | 22 dialogue turns |
| **Pure Urdu** | 5 (22.7%) |
| **Code-mixed (Urdu+English)** | 17 (77.3%) |
| **Pure English** | 0 |
| **Agent turns** | 11 |
| **Customer turns** | 11 |

### Annotation Quality Grade: **A- (Good)**

**Strengths:**
- ✓ **Dual annotation:** Raw (with English) + clean (Urdu only)
- ✓ **Speaker roles tracked:** ایجنٹ (Agent) vs کسٹمر (Customer) — useful for speaker diarization
- ✓ **Turn-by-turn:** Sequential turn numbers enable temporal alignment
- ✓ **Realistic code-mixing:** 77% of turns mix Urdu+English (realistic for Pakistani call centers)
- ✓ **Consistent format:** All turns follow same annotation pattern

**Limitations:**
- ⚠️ No timestamps for alignment with audio
- ⚠️ No confidence scores or notes on uncertain words
- ⚠️ Single annotator (no inter-annotator agreement data)
- ⚠️ Spelling/transliteration consistency (e.g., "please" → "پلیز", "test" → "ٹیسٹ", "Sure" → "شور")

---

## 3. Domain Analysis

### Domain: **Customer Service / Medical Call Center**
- **Organization:** Chughtai Lab (medical/diagnostic laboratory)
- **Language:** Urdu (Pakistan) with high code-mixing
- **Call Type:** Inbound customer inquiry (report status, test results)
- **Topics Covered:**
  - Greeting & formalities (السلام علیکم)
  - Patient identification & details
  - Lab report status inquiries
  - Test results discussions (HBA1C, blood sugar)
  - Wait times & follow-up procedures
  - Closing & feedback transfer

### Real-World Value: **HIGH**
This is **genuine call center data**, not scripted. Key characteristics:
- Natural speech patterns (hesitations, repetitions, filler words)
- Code-switching (Urdu↔English loanwords)
- Regional dialect (Pakistani Urdu)
- Real background context (lab operations, report systems)

---

## 4. Feasibility for Training Qwen3-ASR

### Sample Size Assessment

| Use Case | Required Data | This Dataset | Verdict |
|----------|---------------|--------------|---------|
| **Single sample eval** | 10-60 sec | 96 sec ✓ | ✓ **Sufficient** |
| **Fine-tuning baseline** | 10-50 hours | 0.027 hours ✗ | ✗ **Severely insufficient** |
| **Domain adaptation** | 100+ hours | 0.027 hours ✗ | ✗ **Insufficient** |
| **Distillation pseudo-label** | 100-500 hours unlabeled | 0.027 hours ✗ | ⚠️ **Needs scaling** |
| **Confidence validation** | Any size | 96 sec ✓ | ✓ **Useful for validation** |

### Training Feasibility: **LOW (for standalone model training)**

**❌ NOT suitable for:**
1. **Fine-tuning Qwen3** — 96 seconds is ~0.027 hours; need 10-50+ hours minimum
2. **Building new model** — Far too small (need 100+ hours)
3. **Reducing model size** — Would overfit catastrophically

**✅ SUITABLE for:**
1. **Validation & benchmarking** — Test Qwen3 on this specific domain
2. **Confidence calibration** — Verify confidence extraction quality on call center audio
3. **Code-mixing evaluation** — Measure performance on Urdu-English mixed speech
4. **Baseline establishment** — Measure current WER before any improvements

---

## 5. Audio Quality Issues & Mitigations

### Issue 1: **8kHz Sample Rate (Telephony Compression)**
- **Impact:** Loses high-frequency consonants (e.g., /s/, /ʃ/, /ð/)
- **Current Qwen3 assumption:** 16kHz stereo waveforms
- **Risk:** Potential accuracy drop on this domain
- **Mitigation:**
  ```bash
  # Upsample to 16kHz before feeding to Qwen3
  ffmpeg -i audio.wav -ar 16000 audio_16k.wav
  ```
- **Note:** Upsampling won't recover lost information, but helps Qwen3 process expected sample rate

### Issue 2: **Very Short Duration**
- **Impact:** No statistical significance for model training
- **Current use:** Only for testing/validation
- **Mitigation:**
  - Collect 50-100+ more similar call recordings
  - Combine with existing Hindi/Urdu datasets (Common Voice, ULCA, etc.)
  - Use **unlabeled audio** for distillation training (Phase 2 of knowledge distillation)

### Issue 3: **Code-Mixing (77%)**
- **Impact:** Qwen3 trained mainly on Hindi; Urdu+English mix is domain-specific
- **Advantage:** Highly realistic for Pakistani call centers
- **Training strategy:**
  - Use confidence filtering (Section 4 below) to identify weak areas in code-mixed words
  - Fine-tune on mixed Urdu-English data (need more samples)
  - Or: Use Whisper-large-v3-turbo-urdu (multilingual model better at code-mixing)

---

## 6. Immediate Use: Confidence Validation

### Test Qwen3-ASR on this data

```bash
cd qwen3-asr-local

# Process the call center audio
conda run -n base python asr_transcribe_and_transliterate.py \
  --conf-table \
  data/audio/in-4234500300-+923000754715-20260501-221336-1777655616.988465.wav

# Compare ASR output with annotation (Col3)
# Measure WER: word error rate against gold standard
```

**What to measure:**
1. **Overall WER** on this call center domain
2. **Code-mixed word accuracy** (does Qwen3 handle "please" → "پلیز" correctly?)
3. **Confidence calibration** (do low-confidence scores match actual errors?)
4. **Speaker robustness** (does performance differ between agent/customer?)

---

## 7. Scaling Strategy: From This Sample to Production

### Phase 1: Validate Current Model (This Sample)
- **Goal:** Baseline WER on call center data
- **Timeline:** 1 day
- **Output:** Confidence report, error analysis
- **Data:** 96 seconds (this file)

### Phase 2: Collect & Annotate (2-4 weeks)
- **Goal:** Gather 50-100+ similar call recordings
- **Sources:**
  - Chughtai Lab (existing recordings)
  - Other Pakistani call centers (consents permitting)
  - Synthetic data generation (TTS + recordings)
- **Annotation:** Simple format (turn-by-turn like this sample)
- **Data:** 5-10 hours
- **Cost:** Low (reuse existing annotation template)

### Phase 3: Pseudo-Labeling & Distillation (2-4 weeks)
- **Goal:** Improve Qwen3 without manual annotation
- **Method:** Knowledge distillation from Whisper
  1. Run Whisper-large-v3-turbo-urdu on unlabeled data
  2. Filter outputs by confidence (your existing pipeline!)
  3. Train Qwen3 student on filtered pseudo-labels
  4. Validate on this call center sample
- **Data:** 20-50 hours unlabeled audio
- **Improvement:** +5-15% WER reduction (distillation effect)

### Phase 4: Fine-tuning (2-4 weeks)
- **Goal:** Domain-specific tuning for call center speech
- **Method:** Fine-tune on Phase 2 collected + annotated data
- **Data:** 5-10 hours labeled + 20-50 hours pseudo-labeled
- **Improvement:** +10-20% WER reduction
- **Result:** Qwen3 specialized for Urdu call center ASR

---

## 8. Recommendation Summary

### Current Data Assessment
| Aspect | Grade | Notes |
|--------|-------|-------|
| **Audio Quality** | B- | 8kHz telephony, clean, but limited bandwidth |
| **Annotation Quality** | A- | Well-structured, dual-level, speaker roles tracked |
| **Sample Size** | D | 96 seconds—only for validation, not training |
| **Domain Value** | A+ | Real call center data, highly relevant |
| **Code-mixing Realism** | A+ | 77% mixed—authentic Pakistani Urdu |

### What to Do Now

**✅ DO (Immediately):**
1. **Run Qwen3-ASR on this sample** → measure baseline WER
2. **Compare ASR output to annotation (Col3)** → identify domain errors
3. **Analyze confidence scores** on code-mixed words → validate your confidence extraction
4. **Document findings** in a call-center-specific error taxonomy

**❌ DO NOT (Don't Waste Effort):**
1. ❌ Try to fine-tune Qwen3 on 96 seconds (will overfit)
2. ❌ Treat this as a complete dataset (need 100x more data)
3. ❌ Expect high accuracy without domain-specific training (8kHz + code-mixing)

**🚀 DO NEXT (2-4 weeks):**
1. **Collect 50-100 more call recordings** (same lab, same domain)
2. **Annotate in same format** (Col2 raw, Col3 clean, Col4 role, Col5 turn#)
3. **Implement distillation pipeline** (Whisper teacher → Qwen3 student, filtered by confidence)
4. **Validate on combined dataset** (this sample + new samples)
5. **Measure improvement** (baseline WER → post-distillation WER)

---

## 9. Code: Extract & Validate Annotations

Use this script to extract gold-standard transcriptions and measure ASR accuracy:

```python
import openpyxl
from jiwer import wer, cer

xlsx_path = "in-4234500300-+923000754715-20260501-221336-1777655616.988465.xlsx"
wb = openpyxl.load_workbook(xlsx_path)
ws = wb['Sheet1']

gold_standard = []
for row in range(3, ws.max_row + 1):  # Skip headers
    col2 = ws.cell(row, 2).value  # Raw (with English)
    col3 = ws.cell(row, 3).value  # Clean (Urdu only)
    col4 = ws.cell(row, 4).value  # Role
    
    if col3:
        gold_standard.append({
            'turn': row - 2,
            'raw': col2 or "",
            'clean': col3,
            'role': col4,
        })

# Later: Compare Qwen3 output against col3 (clean)
# asr_output = "... your ASR output here ..."
# error_rate = wer(col3, asr_output)
# print(f"WER: {error_rate:.1%}")
```

---

## 10. Related Documentation

- [word-level-confidence.md](word-level-confidence.md) — Your confidence extraction (applies to this data!)
- [knowledge-distillation-asr-roadmap.md](knowledge-distillation-asr-roadmap.md) — How to scale this to 100+ hours
- [hindi-to-roman-urdu-design.md](hindi-to-roman-urdu-design.md) — Transliteration (relevant for cleaning annotations)

---

## Summary Table: This Data's Role in Your Pipeline

| Stage | Input | This Data | Output | Status |
|-------|-------|-----------|--------|--------|
| **Baseline** | Raw Qwen3 | ✓ Validation set | WER% | 🟢 Ready |
| **Confidence Validation** | Qwen3 + confidence scores | ✓ Test domain | Calibration report | 🟢 Ready |
| **Pseudo-labeling** | Whisper teacher | ⚠️ Too small alone | Filtered labels | 🟡 Useful after scaling |
| **Fine-tuning** | Annotated data | ✗ Insufficient | Improved Qwen3 | 🔴 Need 100x more |
| **Evaluation** | Gold standard | ✓ Annotation ready | Final WER | 🟢 Ready |
