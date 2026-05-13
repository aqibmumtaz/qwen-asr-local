# Rust Architecture Plan — ASR + Urdu Pipeline

**Status:** DEFERRED — current Python/shell implementation works (97.5% accuracy
on Chughtai Lab vocabulary). Revisit when production hardening, single-binary
deployment, or per-call latency reduction below 100ms becomes critical.

## Goal

Build a production Rust application that:
1. Runs ASR via llama.cpp multimodal (Qwen3-ASR) or Whisper
2. Handles Hindi Devanagari → Urdu Nastaliq + Roman Urdu transliteration natively
3. Calls llama.cpp LLM models for any downstream translation/processing
4. Ships as a single static binary — no Python, no shell glue

---

## What we have today (to port from)

| Component | Current implementation | Lines | Maturity |
|---|---|---|---|
| ASR | `transcribe.sh` → llama.cpp `llama-mtmd-cli` subprocess | ~100 | Production |
| Hindi → Nastaliq | GokulNC `indo_arabic_transliteration` (Python) | external lib | Production |
| Hindi → Roman Urdu | `hindi_to_roman_urdu.py` algorithm + `data/lexicons.json` | 434 + 1431 entries | Production (62/62 tests) |
| Shell pipeline | `transcribe_and_transliterate.sh` (.sh → .sh) | ~110 | Production |
| Python pipeline | `transcribe_and_transliterate.py` (subprocess to .sh) | ~115 | Production |

All shell entry points are clean (`transcribe.sh`, `hindi_to_roman_urdu.sh`)
and ready to be dropped in favour of a single Rust binary.

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

---

## Roman Urdu transliterator port — concrete plan

The single biggest decision: how to port `hindi_to_roman_urdu.py` to Rust.
That file has the schwa-syncope algorithm + 4 regex rules + 1431 lexicon entries.

### Option A — Pure Rust port (recommended)

```toml
unicode-normalization = "0.1"   # NFC normalisation
regex                 = "1"     # supports lookaround via fancy-regex below
fancy-regex           = "0.13"  # lookbehind/lookahead support
```

Port outline:
1. **Phoneme tables** — embed as `phf::Map` (compile-time perfect hash maps).
   The 5 dicts (CONSONANTS, NUKTA_MAP, EXTENDED, INDEPENDENT_VOWELS, MATRAS)
   become `static CONSONANTS: phf::Map<&str, &str> = phf_map!{ ... };`
2. **Char iteration** — `text.chars()` with `unicode_normalization::UnicodeNormalization::nfc()`
3. **Schwa syncope state machine** — straight Rust loop, no extra deps. The
   Python `_emit_vowel` function is ~40 lines, ports cleanly.
4. **Regex normalize_endings** — use `fancy-regex` for the lookbehind rules
   `(?<=[bcdfghjklmnpqrstvwxyz])aa\b → a`.
5. **Lexicon** — embed `data/lexicons.json` via `include_str!()` and parse at
   startup with `serde_json`. Or pre-bake into `phf::Map` at build time via
   `build.rs` for zero-cost lookups.

Estimated effort: 2–3 days for a working port + tests passing.

### Option B — ICU transliterator (faster but Urdu-quality-dependent)

Use `rust_icu_utrans` with the built-in `Devanagari-Latin` transform. But:
- ICU's Devanagari→Latin produces academic ITRANS-style output (e.g. `mErA nAma akiba hai`), not natural Roman Urdu
- Would still need our lexicon corrections on top
- Adds `libicu-dev` system dependency

Not recommended unless ICU quality is tested and found acceptable.

---

## Nastaliq (GokulNC) port

The GokulNC `indo_arabic_transliteration` library is pure Python with ~95 CSV
mapping entries + virama state machine + positional rules. Equivalent to our
Devanagari→Roman algorithm but targeting Perso-Arabic output.

Two options:
1. **`rust_icu_utrans` "Devanagari-Arabic"** — built-in ICU transform, works
   out of the box but quality TBD (similar concerns as ICU Devanagari-Latin)
2. **Port GokulNC algorithm** — port the CSV tables + virama state machine
   directly. ~150 lines of Rust.

Defer until tested. If ICU output is acceptable, save the effort.

---

## Cargo workspace layout

```
qwen-asr-rust/
├── Cargo.toml                  # workspace
├── crates/
│   ├── transcribe/             # bin: ASR via llama-cpp-2
│   ├── transliterate/          # lib: Hindi → Roman Urdu (ported from hindi_to_roman_urdu.py)
│   │   ├── src/lib.rs
│   │   └── data/lexicons.json  # symlink or copy from Python project
│   ├── nastaliq/               # lib: Hindi → Urdu Nastaliq
│   └── pipeline/               # bin: full audio → roman_urdu + nastaliq
└── tests/
    └── corpus.rs               # the 62-test suite ported from hindi_to_roman_urdu.py
```

---

## Triggers — when to start the port

Begin Rust port when any of these become true:

1. **Latency** — per-call overhead of subprocess (~100ms) becomes a bottleneck
   (e.g. real-time streaming where every ms matters)
2. **Deployment** — need a single static binary for distribution
   (no Python runtime, no shell, no `data/lexicons.json` file at install time)
3. **Production hardening** — embedded systems, mobile, or air-gapped servers
4. **Performance** — need to process >100 calls/sec on a single host
5. **Cross-platform** — need Windows native support without WSL/MSYS

Until then: the Python + shell pipeline is honest about what it is, easy to
edit (just modify `data/lexicons.json`), and runs at ~100ms per call which is
negligible vs the 20s ASR step.

---

## Migration path

When the trigger fires:

1. **Phase 1 (1 week)** — Port `hindi_to_roman_urdu.py` → Rust crate.
   Embed `data/lexicons.json` via `include_str!`. Pass all 62 self-tests.
2. **Phase 2 (1 week)** — Port ASR step using `llama-cpp-2` mtmd module.
   Match output of current `transcribe.sh` byte-for-byte on the sample set.
3. **Phase 3 (1 week)** — Decide Nastaliq strategy (ICU vs port GokulNC).
   Add to pipeline.
4. **Phase 4 (1 week)** — Integration tests, cross-platform builds, CI.

Total: ~4 weeks for a feature-complete Rust port at current accuracy.

---

## Status

**Deferred** as of 2026-05-13. Current Python/shell pipeline serves the
Chughtai Lab call-centre use case with 100% accuracy on the realistic test
corpus. Revisit when triggers above fire.
6. **Whisper Urdu fine-tune source**: `kingabzpro/whisper-large-v3-turbo-urdu` on Hugging Face.
