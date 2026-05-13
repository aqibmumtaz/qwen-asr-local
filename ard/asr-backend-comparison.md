# ASR Backend Comparison — Qwen3-ASR-1.7B Across Three Inference Paths

**Last updated:** 2026-05-13
**Test audio:** `qwen3-asr-local/samples/sample_ur1.wav` (Urdu speech, ~12s)
**Reference:** [`asr_transcribe_and_transliterate.py`](../qwen3-asr-local/asr_transcribe_and_transliterate.py), [`transcribe.sh`](../qwen3-asr-local/transcribe.sh), [`app.py`](https://huggingface.co/spaces/AqibMumtaz/...) (production server)

---

## Summary — same model, three different outputs

All three paths load the **same Qwen3-ASR-1.7B model weights**. They produce
**three different transcriptions** of the identical audio file. This document
catalogues why, and what (doesn't) help.

```
sample_ur1.wav (Urdu: "My name is Aqib. This is a test of code-ki-zaban for voice recognition.")

GPU vLLM (app.py):     मेरा नाम अपीब है। ये उर्दू ज़बान में आवाज़ की शनाख्त का टेस्ट है।
CPU transformers fp32: मेरा नाम अपीब है। ये उर्दू जान में आवाज की शनाख्त का टेस्ट है।
CPU transformers bf16: मेरा नाम अपीब है। ये उर्दू जान में आवाज की शनाख्त का टेस्ट है।   ← identical to fp32
llama.cpp Q8_0 GGUF:   मेरा नाम अकीब है। ये कोर्डुज़बान में आवाज की शनात का दैस है।
llama.cpp BF16 GGUF:   मेरा नाम अकीब है। ये कोर्डुज़बान में आवाज की शनात का दैस है।   ← identical to Q8_0
```

Notable differences:

| Word position | GPU vLLM | CPU transformers | llama.cpp |
|---|---|---|---|
| `Aqib` (proper name) | `अपीब` | `अपीब` | `अकीब` ← only llama.cpp lands here |
| `code-ki-zaban` (3 words) | `उर्दू ज़बान` (2 words, wrong but split) | `उर्दू जान` (wrong word, no nukta) | `कोर्डुज़बान` (mashed into 1 word) |
| `awaz` (voice) | `आवाज़` (with nukta) | `आवाज` (no nukta) | `आवाज` (no nukta) |
| `shanakht` (recognition) | `शनाख्त` ✓ | `शनाख्त` ✓ | `शनात` (kh dropped) |
| `test` | `टेस्ट` ✓ | `टेस्ट` ✓ | `दैस` (entirely wrong word) |

The CPU transformers paths are very close to the GPU reference (only nukta marks differ).
llama.cpp diverges most significantly.

---

## Architecture comparison

| | GPU vLLM (app.py) | CPU transformers | llama.cpp |
|---|---|---|---|
| **Weights** | Qwen/Qwen3-ASR-1.7B safetensors (~4.4 GB) | Same safetensors | Qwen3-ASR-1.7B-Q8_0-new.gguf (2 GB) or BF16-new.gguf (3.8 GB) + mmproj BF16 (612 MB) |
| **Inference library** | `vllm` + `qwen-asr` Python | `transformers` + `qwen-asr` Python | `llama-mtmd-cli` (C++) |
| **Attention** | PagedAttention (CUDA fused) | PyTorch SDPA | ggml attention (C++) |
| **Audio preprocessing** | qwen-asr Python pipeline (torchaudio mel) | qwen-asr Python pipeline (torchaudio mel) | mtmd library mel (C, custom) |
| **Sampling/decoding** | vLLM `SamplingParams` (CUDA) | HF `generate()` | llama.cpp sampler (C++) |
| **Precision** | bf16 (CUDA SM ≥ 80) | fp32 or bf16 (emulated on CPU) | bf16 or Q8_0 weights, fp32 accumulators |
| **Hardware** | NVIDIA GPU | Apple Silicon CPU | CPU (Metal optional) |

The lower three rows (attention, preprocessing, sampling, precision, hardware) all differ.
The only thing held constant is the trained weights.

---

## Root cause — why same weights → different output

The model is **genuinely uncertain** at certain token positions:

1. The Devanagari nukta `़` (U+093C) is a separate token in the tokenizer.
2. At decode step for `ज़बान`, the model must pick:
   - token A: `़` (emit nukta) → continues with "zaban"
   - token B: end-of-word → moves to "jaan" path
3. Logits at these positions are **nearly tied** (e.g. A = 0.501, B = 0.499).
4. Greedy decoding picks the larger. Any numerical drift flips the choice.

Sources of numerical drift between backends:

- bf16 vs fp32 vs Q8_0 quantisation
- Fused CUDA kernels vs PyTorch ops vs ggml ops
- PagedAttention vs SDPA vs ggml attention
- vLLM sampler vs HF generate vs llama.cpp sampler
- HF mel-spectrogram vs mtmd C mel-spectrogram

**Each path lands on a different side of the same coin-flip token decisions.**

---

## What we tested — and what did NOT help

### Test 1: Match dtype between CPU and GPU (`DTYPE=bfloat16`)

```bash
DTYPE=bfloat16 python3 asr_transcribe_and_transliterate.py samples/sample_ur1.wav
```

**Result:** Output **identical to fp32 run**. The nuktas were still dropped.

| | Hindi output | Time |
|---|---|---|
| CPU fp32 | …**जान**…**आवाज**… | 30.7s |
| CPU bf16 | …**जान**…**आवाज**… | **141.5s** (4.6× slower) |

CPU PyTorch's bf16 implementation emulates via fp32 internally — no speedup,
and the numerics still differ from a CUDA bf16 path with fused kernels.
**Same dtype label ≠ same compute.**

### Test 2: Use BF16 GGUF in llama.cpp instead of Q8_0

```bash
bash transcribe.sh --asr-model Qwen3-ASR-1.7B-BF16-new samples/sample_ur1.wav
```

**Result:** Output **byte-identical to Q8_0 GGUF**. Only the runtime was 1.7× slower.

**Conclusion:** llama.cpp's errors aren't from quantisation — they're from the
ggml C++ inference path itself (different from PyTorch / vLLM).

---

## What WOULD help

| Strategy | Cost | Effect |
|---|---|---|
| **Run on a GPU** with vLLM | Hardware required | Reproduces app.py output exactly |
| **Switch to Urdu-tuned Whisper** (`kingabzpro/whisper-large-v3-turbo-urdu`) | New model, new pipeline | Native Urdu phonology — likely avoids the nukta + word-confusion issues at source |
| **Use Qwen3-ASR-0.6B** on MPS | Smaller model fits MPS buffer cap | Different model, different output; faster on Mac but accuracy TBD |
| **Beam search instead of greedy** | Same backend, different decoding | Sometimes catches near-tie tokens; needs API support in qwen-asr |
| **Better audio** | Recording quality | Reduces model uncertainty → fewer borderline tokens |

---

## What does NOT belong in the lexicon

The temptation when seeing these diffs is to "patch" them via `data/lexicons.json`:

```json
{
  "jaan": "zaban",           // ✗ different words! जان = life, ज़बान = language
  "shanaat": "shanakht",     // ✗ shanaat isn't a word at all
  "korduzabaan": "code ki zaban",  // ✗ ASR garble, not a transliteration
  "dais": "test"             // ✗ dais is a real English word (raised platform)
}
```

**This is wrong.** The lexicon's purpose is mapping **spelling variants of the
same word**:

- `meraa` → `mera` ✓ (long ā shortened — same word)
- `kaa` → `ka` ✓
- `achchaa` → `acha` ✓
- `akeeb` → `Aqib` ✓ (same name, ASR phonetic spelling vs proper transliteration)

Adding ASR-error patches would:
1. Mistranslate legitimate audio where the patched word actually means what it says (e.g. someone saying `jaan` = "life" gets converted to "language")
2. Pollute the linguistic data with non-linguistic mishears
3. Mask the underlying ASR quality problem instead of fixing it

**Fix ASR errors at the ASR layer, not in the lexicon.**

---

## Recommendation

For this project's Chughtai Lab use case:

1. **Production (cloud / GPU):** Use app.py with vLLM. Best accuracy.
2. **Local dev / Mac:** Use HF transformers backend (`BACKEND=auto` in our script).
   Accept that nuktas may drop on borderline tokens — it doesn't change
   downstream Roman Urdu accuracy much because the lexicon catches
   legitimate spelling variants.
3. **Offline embedded:** Use llama.cpp GGUF path. Faster but lossier; the
   model-level errors (`दैस`, `शनात`) need to be tolerated, not patched.
4. **For target deployment:** evaluate `whisper-large-v3-turbo-urdu` as a
   replacement — Urdu-native ASR avoids most of these problems at source.

**The current pipeline (any backend) gives usable output; the lexicon stays
clean for legitimate transliteration corrections only.**

---

## Status

Investigation closed 2026-05-13. No code changes recommended for backend
parity. Future work: evaluate Urdu-tuned Whisper as a Qwen3-ASR alternative.
