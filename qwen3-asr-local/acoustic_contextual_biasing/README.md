# Acoustic Contextual Biasing

Reduce **name mishearings** at the ASR stage — the errors text post-processing cannot fix
(the ASR heard a *different* word, e.g. `shaam` for `Ehtesham`). Feeds relevant names to
Qwen3-ASR's `context=` biasing hook so recognition is nudged toward them.

This is the **audio** counterpart to `phonetic_contrastive_model/` (which handles *spelling*).

## How it works

```
audio ─► Pass 1 (no context) ─► rough hypothesis
                                      │
                       retrieve relevant gazetteer names (BR-ASR-lite,
                       text-keyed via the phonetic encoder — small list,
                       avoids the dilution of biasing the whole gazetteer)
                                      │
        Pass 2 (context = retrieved names) ─► biased transcript
```

Training-free approximation of BR-ASR: retrieval is keyed on the first-pass text (not a
learned audio embedding), so it costs two decodes but no training.

## Files

| file | role |
|---|---|
| `asr.py` | `BiasedASR` — loads base Qwen3-ASR (non-quantized), `transcribe(audio, context)` |
| `retriever.py` | `NameRetriever` — embeds the gazetteer, retrieves names relevant to a hypothesis |
| `two_pass.py` | `TwoPass` — pass1 → retrieve → pass2 |
| `benchmark.py` | baseline vs two-pass vs oracle, `diff_words` + **name-recovery**, on the 80-call set |

## Usage

```bash
# thorough benchmark (scope with --calls / --limit-chunks; CPU is slow, mps faster)
python -m acoustic_contextual_biasing.benchmark --calls 6 --device cpu
```

```python
from acoustic_contextual_biasing.two_pass import TwoPass
tp = TwoPass()
r = tp.transcribe("chunk_000.wav")
r["pass1"], r["names"], r["pass2"]
```

## The honest gate — 8kHz

Biasing can only recover a name if the acoustic detail that distinguishes it **survived the
codec**. The 80-call audio is **8kHz narrowband**; a scoped test showed that even with the
exact name in context (`Shahid`), the model still heard `Shaher` — the detail is gone. So on
this audio, biasing gives only a **marginal nudge** (structure, not names).

**Expected result on 8kHz:** small `diff_words` gain, low name recovery — proof the ceiling is
the audio. The `oracle` column (biasing with the gold names) is the *upper bound*: if even the
oracle can't recover names, wideband audio is required. Biasing/two-pass pay off once the
source audio is **16kHz / Opus wideband**.
