# Urdu ASR Improvement Plan

## Problem

Current pipeline is slow and error-prone:

```
Audio → Qwen3-ASR (~18s) → 5–8B LLM translation (~180s) → Urdu text
Total: ~200s per sample
```

**Root cause**: Qwen3-ASR was not trained on Urdu. Urdu and Hindi share identical phonemes (spoken sounds) — only the script differs. Qwen3-ASR maps Urdu audio to Hindi/Devanagari script. The LLM "translation" step is really just a script conversion, which is massive overkill.

---

## What Phonemes Means

Phonemes = the individual sound units of a language.

Urdu and Hindi are the same spoken language — different alphabets only:
- Hindi writes: **आज का موسم اچھا ہے** (Devanagari)
- Urdu writes: **آج کا موسم اچھا ہے** (Nastaliq/Perso-Arabic)

The ASR hears the audio correctly but writes it in the wrong script.

---

## Three-Tier Improvement Plan

### Tier 1 — Quick Win: Replace LLM with Script Transliteration (~10x speedup)

Replace the 5–8B LLM translation step with a lightweight rule-based transliteration library.
Since it's the same language, no "translation" of meaning is needed — just convert the alphabet.

**Python:**
```bash
pip install urduhack
# or
pip install indic-transliteration
```

```python
import urduhack
urdu = urduhack.transliterate(hindi_devanagari_text)
# آج کا موسم اچھا ہے
```

**Speed**: <1ms. Zero model inference.
**Expected pipeline time**: ~200s → ~18s (10x faster, zero quality loss)

---

### Tier 2 — Better: Replace Qwen3-ASR with Urdu Fine-tuned Whisper (~1 inference step)

A community of researchers fine-tuned OpenAI Whisper on Urdu (Common Voice 17 dataset).
These models output **Urdu Nastaliq directly** — no translation step at all.

**Best model**: [`kingabzpro/whisper-large-v3-turbo-urdu`](https://huggingface.co/kingabzpro/whisper-large-v3-turbo-urdu)
- 26.2% WER, ChrF 81.6
- Based on `large-v3-turbo` — 8x faster decoder than `large-v3`
- ~800M params, ~1.6 GB

**Smaller options:**
| Model | WER | Size |
|---|---|---|
| `kingabzpro/whisper-large-v3-urdu` | ~20–22% est. | ~1.5 GB |
| `kingabzpro/whisper-large-v3-turbo-urdu` | 26.2% | ~1.6 GB |
| `kingabzpro/whisper-base-urdu-full` | 39.1% | ~290 MB |
| `kingabzpro/whisper-tiny-urdu` | higher | ~150 MB |

**Inference backends:**

| Backend | Speed (CPU) | Speed (Apple Silicon/MLX) |
|---|---|---|
| faster-whisper int8 | ~15s/sample | — |
| **MLX 4-bit (Mac)** | — | **~1–3s/sample** |

```bash
# Convert once (MLX)
python -m mlx_audio.convert \
  --hf-path kingabzpro/whisper-large-v3-turbo-urdu \
  --mlx-path ./whisper-turdu-urdu-mlx \
  --quantize --q-bits 4

# Run
mlx_whisper audio.wav --model ./whisper-turdu-urdu-mlx --language ur
```

---

### Tier 3 — Innovative: Lightweight Transliteration Head on Qwen3-ASR

Keep Qwen3-ASR frozen (no retraining the large model).
Attach a tiny character-level seq2seq head (~10M params) on top of the decoder output.
Maps Devanagari token sequences → Nastaliq token sequences.

```
Audio → Encoder → Qwen3-ASR Decoder (Hindi tokens) → Transliteration Head (~5ms) → Urdu tokens
```

Benefits:
- Keeps Qwen3-ASR's multilingual accuracy
- Adds only ~5ms transliteration overhead
- Only the tiny head needs training (frozen backbone)

---

## What is MLX

MLX = Apple's ML framework for Apple Silicon (M1/M2/M3/M4).

Apple Silicon has **unified memory** — CPU, GPU, and Neural Engine share the same RAM pool.
No data copying between CPU RAM and GPU VRAM (a major bottleneck on normal PCs).

```
Normal PC:                   Mac Apple Silicon:
CPU RAM                      Unified Memory Pool
    ↕ (slow copy)                ↙    ↓    ↘
GPU VRAM                    CPU   GPU   Neural Engine
```

`mlx-whisper` runs Whisper entirely on Apple Silicon using all three compute units simultaneously.
Benchmarked: **~1 second per audio clip** for large-v3-turbo on M-series Mac vs 60s on CPU PyTorch.

---

## Speed Comparison Summary

| Approach | WER | Speed (CPU) | Speed (Mac Metal/MLX) | Script | Translation needed |
|---|---|---|---|---|---|
| Current: Qwen3 + LLM | High (compounding) | ~200s | ~200s | Urdu (via LLM) | Yes |
| Tier 1: Qwen3 + urduhack | Same as ASR WER | ~18s | ~18s | Urdu (transliteration) | No |
| **Tier 2: whisper-turbo-urdu + MLX** | **26.2%** | ~15s | **~1–3s** | **Urdu native** | **No** |
| Tier 2: whisper-large-v3-urdu + MLX | ~20% | ~30s | ~2–5s | Urdu native | No |

---

## Other Models Evaluated

- **Meta MMS** (`facebook/mms-1b-all`, `urd-script_arabic`) — supports Urdu natively but benchmarks worse than fine-tuned Whisper
- **SeamlessM4T-large** — 17% WER on read speech, beats Whisper, but larger/slower and harder to run locally
- **wav2vec2 Urdu models** — higher WER, older approach, not recommended
