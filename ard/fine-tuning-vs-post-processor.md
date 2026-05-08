# Fine-tuning vs Post-processor — Decision Guide

## The Problem
Qwen3-ASR outputs Hindi Devanagari for Urdu audio.
Goal: Get Urdu Nastaliq text out of the pipeline efficiently.

---

## Option A — Post-processor (Simple Transliteration)

### What it is
A deterministic script conversion layer after ASR output.
Hindi Devanagari → Urdu Nastaliq via character mapping rules.

### How it works
```
Audio → Qwen3-ASR → "आज का मौसम اچھا ہے" → [GokulNC rules] → "آج کا موسم اچھا ہے"
```

### Pros
- Works right now — no training, no data collection
- Fast: <1ms (rule-based) or ~5ms (lightweight ML transliteration)
- No GPU needed for the conversion step
- Interpretable — you can see and fix the rules
- Deterministic — same input always gives same output
- Easy to ship (pure Python or Rust)

### Cons
- Schwa syncope not perfectly handled by rules (needs ML for edge cases)
- Homographs: same Devanagari spelling → multiple valid Urdu spellings (context-dependent)
- Rare/archaic characters may not be covered
- Still two inference steps (ASR + mapping, though mapping is near-zero cost)

### Best library for this today (Python)
**GokulNC `indo-arabic-transliteration`** — positional rules, schwa deletion, nuqta consonants.
For production accuracy, use its ML mode (IndTrans, IIT Bombay).

---

## Option B — Fine-tuning

### What it is
Fine-tune Qwen3-ASR (or Whisper) to output Urdu Nastaliq directly from Urdu audio.
One model, one inference step, native output.

### Two sub-options:

**B1 — Fine-tune Qwen3-ASR on Urdu data**
- Add Urdu audio + Urdu text pairs to training
- Model learns to output Nastaliq directly
- Requires significant Urdu ASR dataset (Mozilla Common Voice Urdu has ~100h)
- Needs GPU for training (at minimum A100 for 1.7B model fine-tune)

**B2 — Use already fine-tuned Whisper-Urdu (no training needed)**
- `kingabzpro/whisper-large-v3-turbo-urdu` — 26.2% WER
- Already trained, ready to download
- Single inference step, outputs Urdu natively
- This is actually "fine-tuning done by someone else"

### Pros
- Single inference step — cleanest architecture
- No post-processing at all
- Better accuracy on real Urdu speech patterns (not transliterated Hindi)
- B2 is available right now (no training required from your side)

### Cons
- B1 requires training data, GPU compute, experimentation time
- B2 (Whisper-urdu) requires switching ASR engine (away from Qwen3-ASR)
- Fine-tuned model may degrade on non-Urdu languages

---

## Recommendation

### Phase 1 — Right now (Python): Post-processor
Use GokulNC transliteration (ML mode) as a drop-in replacement for the 8B LLM step.

```
Audio → Qwen3-ASR → Hindi text → GokulNC transliterate() → Urdu text
Time: ~18–20s total  (was ~200s)
```

**This is a 10x speedup with zero model changes and zero training.**

### Phase 2 — Better (Python or Rust): Whisper-Urdu fine-tune
Replace Qwen3-ASR entirely with `kingabzpro/whisper-large-v3-turdu-urdu`.

```
Audio → whisper-large-v3-turbo-urdu → Urdu text directly
Time: ~1–3s (MLX/Apple Silicon) or ~15s (CPU)
```

No transliteration needed at all.

### Phase 3 — Production (Rust): Static lookup table
In the final Rust app, replace GokulNC with a native Rust `match` table.
100 lines of code, zero dependencies, <1ms, fully offline.

---

## Decision Matrix

| Factor | Post-processor | Fine-tune Qwen3-ASR | Whisper-Urdu (pre-trained) |
|---|---|---|---|
| Ready now? | ✓ Yes | ✗ Weeks of work | ✓ Yes |
| Training needed? | No | Yes | No |
| Speed | ~18s | ~18s | ~1–3s (MLX) |
| Accuracy | Good (ML mode) | Best (native Urdu) | Very good (26.2% WER) |
| Rust-compatible | ✓ (lookup table) | ✓ (llama-cpp-2) | ✓ (whisper-rs) |
| Architecture complexity | Low | High | Low |

**Short answer: Start with post-processor (immediate win), plan for Whisper-Urdu (long-term).**
