# Rust Architecture Plan — ASR + Urdu Pipeline

## Goal

Build a production Rust application that:
1. Runs ASR via llama.cpp multimodal (Qwen3-ASR) or Whisper
2. Handles Hindi Devanagari → Urdu Nastaliq transliteration natively
3. Calls llama.cpp LLM models for any downstream translation/processing

---

## Rust Crate Landscape

### llama.cpp Bindings

| Crate | Version | Status | Multimodal |
|---|---|---|---|
| **`llama-cpp-2`** | 0.1.146 | Active (weekly updates, 227K downloads/mo) | ✓ `mtmd` module |
| `llama_cpp` 0.3.x | 0.3.2 | Stale — last updated Apr 2024 | ✗ |
| `llm` (rustformers) | — | Archived/dead | ✗ |

Use **`llama-cpp-2`**. It exposes `MtmdContext`, `MtmdBitmap`, `MtmdInputChunks` for audio/image input — supports Qwen3-ASR GGUF via the same `libmtmd` that `llama-mtmd-cli` uses.

### Whisper in Rust

| Crate | Version | Status |
|---|---|---|
| **`whisper-rs`** | 0.16.0 | Active (Mar 2026) — Metal/CUDA/CoreML |

Wraps whisper.cpp. Load any GGUF/GGML Whisper checkpoint including Urdu fine-tunes.

### Transliteration (Devanagari → Urdu Nastaliq)

| Crate | Status | Notes |
|---|---|---|
| **`rust_icu_utrans`** v5.6.0 | Active (Google, Mar 2026) | Wraps C++ ICU4C. Has built-in `Devanagari-Arabic` transform. Requires `libicu-dev`. |
| `icu::experimental::transliterate` v2.2.0 | Experimental | Pure Rust but no built-in Devanagari→Urdu rules — must write custom CLDR rules |
| `indicscriptswap` | Abandoned (May 2023) | Do not use |

**Note**: Nastaliq is a *font/rendering* style of Perso-Arabic script — handled at display layer (HarfBuzz + Nastaliq font). The text data output is standard Unicode Perso-Arabic, same as any Arabic text.

---

## Two Pipeline Paths

### Path A — Whisper + Urdu Fine-tune (Recommended for Urdu audio)

```
[Audio]
   │
   ▼
whisper-rs (0.16.0)
+ whisper-large-v3-turbo-urdu GGUF
   │
   ▼
[Urdu Perso-Arabic Unicode text]
```

- No transliteration step needed
- Simple dependency tree (no ICU system lib)
- Best when input audio is Urdu speech

### Path B — Qwen3-ASR + ICU Transliteration (for Hindi audio → Urdu output)

```
[Audio]
   │
   ▼
llama-cpp-2 (mtmd module)
+ Qwen3-ASR-1.7B GGUF
   │
   ▼
[Hindi Devanagari text]
   │
   ▼
rust_icu_utrans
UTransliterator::new("Devanagari-Arabic", None, UTRANS_FORWARD)
   │
   ▼
[Urdu Perso-Arabic Unicode text]
   │
   ▼
[HarfBuzz + Nastaliq font — display layer only]
```

- Use when input is Hindi speech and Urdu script output is needed
- Requires `libicu-dev` system package

---

## Cargo.toml

```toml
[dependencies]
# ASR via llama.cpp multimodal (Qwen3-ASR)
llama-cpp-2 = "0.1.146"

# OR: ASR via Whisper (Urdu fine-tuned)
whisper-rs = "0.16.0"

# Devanagari→Arabic transliteration (Path B only)
rust_icu_utrans = "5.6.0"
rust_icu_common = "5.6.0"
rust_icu_sys    = "5.6.0"
```

For Metal (Mac):
```toml
llama-cpp-2 = { version = "0.1.146", features = ["metal"] }
whisper-rs  = { version = "0.16.0",  features = ["metal"] }
```

---

## Key Design Notes

1. **`llm` crate is dead** — archived by rustformers. Do not use.
2. **`llama_cpp` 0.3.x is stale** — missing all 2025/2026 llama.cpp features including libmtmd multimodal support.
3. **Nastaliq rendering**: purely a font concern. Output Unicode codepoints are standard Perso-Arabic (U+0600–U+06FF range). Use a Nastaliq OpenType font (Noto Nastaliq Urdu, Jameel Noori Nastaleeq) with HarfBuzz shaping for display.
4. **ICU4X transliteration** is pure Rust but has no built-in Devanagari→Urdu transform yet — marked experimental in ICU 2.2.0 (Apr 2026). Monitor for updates.
5. **Qwen3-ASR GGUF source**: `ggml-org/Qwen3-ASR-1.7B-GGUF` on Hugging Face.
6. **Whisper Urdu fine-tune source**: `kingabzpro/whisper-large-v3-turbo-urdu` on Hugging Face.
