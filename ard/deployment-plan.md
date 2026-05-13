# Deployment Plan — Rust Integration via GGUF (CPU + GPU matched accuracy)

**Target hardware:** Intel i7-14700KF (28 threads, Raptor Lake) + NVIDIA RTX 3090 (24 GB, Ampere SM 8.6) + 32 GB RAM + Ubuntu 22.04
**Production runtime:** GGUF model files via `llama-cpp-2` Rust bindings
**Accuracy goal:** CPU output matches GPU output ≥ 95% token-exact match on a 50-clip corpus

---

## Why your target system is the easy case

| Aspect | Mac (current dev) | Target Linux + RTX 3090 |
|---|---|---|
| GPU available | No (Metal-only, 2 GB MPS cap) | ✓ RTX 3090, 24 GB, SM 8.6 (native bf16) |
| CPU bf16 support | ✗ Apple Silicon no BFMMLA | ✓ AVX-VNNI INT8 + fast AVX2 GEMM |
| vLLM available | ✗ no wheels for arm64 | ✓ `pip install vllm` works |
| llama.cpp CUDA | n/a | ✓ cuBLAS / cuDNN backends |
| llama.cpp CPU | ggml ARM kernels | ✓ ggml AVX2/AVX-VNNI kernels (battle-tested) |
| MLX option | ✓ (Mac-only) | n/a |

On your Linux target, you can run **any** of the inference paths and they all work well. The CPU↔GPU parity question collapses to a much simpler one.

---

## Phase 0 — Establish the reference (1 day)

**Goal:** Define what "correct output" means for accuracy comparison.

```bash
# On RTX 3090, run the gold-standard pipeline used in app.py
pip install qwen-asr[vllm]
python3 -c "
from qwen_asr import Qwen3ASRModel
m = Qwen3ASRModel.LLM(
    model='Qwen/Qwen3-ASR-1.7B',
    gpu_memory_utilization=0.7,
    dtype='bfloat16',
    max_new_tokens=1024,
)
results = m.transcribe(audio=['sample_ur1.wav', ...], language=['Hindi'])
for r in results: print(r.text)
"
```

**Deliverables:**
- `reference/` directory: 50 audio clips + their GPU-vLLM-bf16 Hindi transcriptions
- This becomes the **ground truth** for all later comparisons

---

## Phase 1 — Test GGUF on GPU (1 day)

**Goal:** Confirm llama.cpp CUDA path matches the vLLM reference.

```bash
# Build llama.cpp with CUDA
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# Run with same audio
./build/bin/llama-mtmd-cli \
    -m Qwen3-ASR-1.7B-BF16-new.gguf \
    --mmproj mmproj-Qwen3-ASR-1.7B-bf16-new.gguf \
    --image sample_ur1.wav \
    -ngl 999 \
    -p '<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage Hindi<asr_text>' \
    -n 256
```

**Measure:** token-exact match % vs Phase 0 reference on all 50 clips.

**Expected:** llama.cpp CUDA + BF16 GGUF should give ≥ 95% match (same precision, similar attention impl as vLLM at the kernel level). If it doesn't, file an issue or pick a different quantisation.

---

## Phase 2 — Test GGUF on CPU (i7-14700KF) (1 day)

**Goal:** Confirm CPU path is acceptable for the Rust deployment.

```bash
# Same llama.cpp build with CPU only (no -DGGML_CUDA)
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release -j

# Test BF16 (most accurate, slower)
./build-cpu/bin/llama-mtmd-cli -m ...-BF16-new.gguf ... -t 16

# Test Q8_0 (8-bit weights, ~2× faster)
./build-cpu/bin/llama-mtmd-cli -m ...-Q8_0-new.gguf ... -t 16

# Test Q5_K_M (5-bit weights, ~3× faster, slight accuracy loss)
./build-cpu/bin/llama-mtmd-cli -m ...-Q5_K_M.gguf ... -t 16
```

**Measure for each quantisation:**
1. Token-exact match % vs Phase 0 reference
2. Wall-clock seconds per audio second (RTF — real-time factor)
3. RAM usage

**Expected matrix:**

| Quant | Size | Speed (RTF) | Accuracy vs GPU |
|---|---|---|---|
| BF16 | 3.8 GB | ~0.3× (slower than realtime) | best |
| Q8_0 | 2.0 GB | ~1× | very close |
| Q5_K_M | 1.4 GB | ~2× | slight degradation |

Pick the smallest quant that hits ≥ 95% match.

---

## Phase 3 — Decide deployment config (½ day)

**Decision tree:**

```
RTF < 1.0× on CPU?
├── No  → ship GPU-only build, Rust calls cuBLAS
└── Yes
    └── Accuracy ≥ 95% on chosen CPU quant?
        ├── Yes → ship CPU-only build, Rust calls AVX2 path
        └── No  → ship hybrid: GPU primary, CPU fallback, OR find a better quant
```

For your hardware (i7-14700KF, fast x86 CPU), **Q8_0 on CPU is very likely to land at >95% accuracy + RTF < 1.0×**. The Mac CPU problems we hit don't apply.

---

## Phase 4 — Rust integration handover (1 week)

### Deliverables to the Rust team

1. **Model files** (place in `models/` of Rust repo)
   ```
   models/
     Qwen3-ASR-1.7B-Q8_0-new.gguf           ← decoder (chosen quant)
     mmproj-Qwen3-ASR-1.7B-bf16-new.gguf    ← encoder (always bf16)
   ```

2. **Lexicon as data file**
   ```
   data/lexicons.json   ← 1166 corrections + 265 proper nouns (sorted JSON)
   ```
   Loaded once at startup; embed via `include_str!()` for single-binary deploy.

3. **Rust crate skeleton** (`qwen-asr-rust/`)
   ```toml
   [dependencies]
   llama-cpp-2     = { version = "0.1.146", features = ["cuda"] }  # also has "metal"
   serde_json      = "1"
   anyhow          = "1"
   ```

   ```rust
   use llama_cpp_2::{
       context::params::LlamaContextParams,
       llama_backend::LlamaBackend,
       model::{params::LlamaModelParams, LlamaModel},
       mtmd::{MtmdContext, MtmdBitmap},
   };

   pub fn transcribe(audio_path: &str, language: &str) -> Result<String> {
       // 1. Init llama backend + load decoder + mmproj
       // 2. Load audio via mtmd (single multimodal context)
       // 3. Build prompt: format!("<|im_start|>system\n<|im_end|>\n…\nlanguage {language}<asr_text>")
       // 4. Decode greedy, n_predict=256
       // 5. Extract everything after "<asr_text>" tag
       Ok(hindi_text)
   }
   ```

4. **Transliteration crate** (`transliterate/`)
   - Port `hindi_to_roman_urdu.py` algorithm to Rust (~2-3 days)
   - Dependencies: `unicode-normalization`, `fancy-regex` (for lookbehind), `phf` (compile-time map for phoneme tables)
   - Load `lexicons.json` via `include_str!()` + `serde_json`
   - See [`rust-architecture-plan.md`](rust-architecture-plan.md) for the full porting outline

5. **Test corpus** (`tests/corpus/`)
   - The 50-clip audio set from Phase 0
   - `expected.json` with reference transcriptions
   - Integration test that runs the full pipeline and asserts ≥ 95% match

6. **Documentation** (`README.md`)
   - Inference flags (matters for accuracy):
     ```
     n_threads     = 16   # = number of physical P-cores on i7-14700KF
     n_predict     = 256  # max output tokens
     temperature   = 0    # greedy
     top_p         = 1.0
     top_k         = -1
     repeat_penalty = 1.0
     ```
   - System prompt + assistant suffix exactly as `transcribe.sh` uses them
   - Language list (Qwen3-ASR supports 30 languages; use "Hindi" for Urdu audio)

---

## Phase 5 — Production hardening (Rust team, ~2 weeks)

Hand-off; the Rust team owns these. Listed here so the plan is complete:

- Graceful error handling (audio decode failures, mtmd init errors)
- Streaming inference (if real-time call-center transcription is needed) — `llama-cpp-2` has streaming support; UtteranceSession pattern from `app.py` translates directly
- VAD integration (Silero VAD via `vad-rs` or port the `RMSVad` fallback from `app.py`)
- Concurrency: thread-pool of `LlamaContext` instances (one per call), shared `LlamaModel` (the weights are read-only)
- Metrics: latency p50/p95/p99, RTF, errors, lexicon hit rate
- Telemetry: log Hindi + Roman Urdu output for ongoing lexicon growth

---

## Does CPU choice matter for accuracy?

Yes, slightly. Across modern x86 CPUs running the same llama.cpp build + same GGUF:

| CPU | Notable instructions | Accuracy vs GPU | Speed (RTF) |
|---|---|---|---|
| **i7-14700KF (your target)** | AVX2, AVX-VNNI (INT8 fused) | very close (same kernel paths as GPU GEMM in fp32 fallback) | ~1× on Q8_0 with 16 threads |
| Xeon Sapphire Rapids+ | AVX-512 + AMX + BFMMLA | identical numerics to NVIDIA bf16 (true bf16 hardware) | ~1.5× on Q8_0 |
| AMD Zen 4 / EPYC Genoa | AVX-512 + native BF16 | identical to GPU bf16 | ~1.3× on Q8_0 |
| Apple M3/M4 | NEON, AMX2 (fp32/fp16 only) | drops nuktas on borderline tokens | ~0.3× (much slower) |
| ARM Graviton 3/4 | SVE + BF16 | matches GPU | similar to Zen 4 |

**Your i7-14700KF is fine.** It doesn't have BFMMLA so true bf16 GEMM isn't accelerated, but llama.cpp dequantises Q8_0 → fp32 → AVX2 GEMM, which is fast and numerically stable. The accuracy you'll measure in Phase 2 will hold for any modern x86 deployment.

If you later deploy to a wider CPU SKU range (cloud servers, edge devices), the same GGUF should work on any system that runs llama.cpp — accuracy varies only by a few percent.

---

## Timeline summary

| Phase | Duration | Owner |
|---|---|---|
| 0 — Reference corpus | 1 day | You |
| 1 — GGUF on GPU validation | 1 day | You |
| 2 — GGUF on CPU validation | 1 day | You |
| 3 — Decision | ½ day | You |
| 4 — Rust handover bundle | 1 week | You |
| 5 — Production hardening | 2 weeks | Rust team |

**Total: ~3 weeks from today** to a production Rust binary.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| llama.cpp BF16 GGUF accuracy gap vs vLLM reference | Medium | Use Q8_0 instead; or fall back to GPU runtime |
| Q8_0 hits < 95% match | Low | Try Q5_K_M, Q4_K_M, or BF16; or larger model (Qwen3-ASR-3B if released) |
| llama-cpp-2 mtmd API changes | Low | Pin version; vendor the crate if needed |
| Rust port of transliterator has subtle bugs | Medium | Port test suite first (62 self-tests + 50-clip corpus); fail CI on regression |
| Lexicon coverage gaps in production | Low | Growth process documented; log unknown outputs for review |

---

## TL;DR

1. Your **i7-14700KF + RTX 3090** target is well-suited for both CPU and GPU paths. Mac was the hard case.
2. **Build llama.cpp on the target, run Phase 0–2** (3 days) to measure actual accuracy + speed on your hardware.
3. Pick **Q8_0 on CPU** unless measurements say otherwise.
4. Hand the Rust team the GGUF + lexicons.json + algorithm-port spec.
5. Done in ~3 weeks total.
