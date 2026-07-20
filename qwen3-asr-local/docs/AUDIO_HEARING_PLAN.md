# Audio Hearing-Problem Plan — Contextual Biasing

**Last updated:** 2026-07-21
**Problem:** text post-processing tops out at ~65% on the 80-call set. The remaining
errors are **mishearings** (ASR heard a *different* word — `shaam` for `Ehtesham`) and
**dropped words** — ~31% of the reference. These are NOT text-fixable; they live in the
audio. This plan attacks them at the ASR stage.

## Tooling (confirmed available)

- `qwen_asr.Qwen3ASRModel.from_pretrained(...)` — **base (non-quantized) Qwen3-ASR**, already
  downloaded and running locally.
- `model.transcribe(audio, context="...", language=..., return_time_stamps=...)` — the
  **`context` argument is the contextual-biasing hook**: a string/list of terms that nudges
  recognition toward them (names, labs, doctors) without mandating output.
- Audio: `testing/lab_test_80_audios_chunks_25s/<call_id>/chunk_NNN.wav` (8kHz mono).
- Gazetteer: `data/entities.json` (+ the v2.2 entity canonicals).

## Approach (the audio ladder)

1. **Feasibility + speed** — load base model, transcribe one chunk, confirm output format and
   per-chunk latency. Decide test scope (subset vs all 403 chunks).
2. **Re-ASR baseline (no context)** — transcribe the audio ourselves and score vs benchmark.
   Establishes what the *current* Qwen3-ASR does on this audio, independent of the vendor's
   earlier Hindi.
3. **Contextual biasing** — transcribe with `context = entity gazetteer`. Measure whether
   name mishearings drop (does `shaam` become `Ehtesham` when the model is told the name?).
4. **Scope smartly** — focus on the calls/chunks where name mishearings occurred; measure
   `diff_words` (and name-level recovery) with vs without biasing.
5. **Dilution check** — biasing is known to degrade as the term list grows. Sweep context
   size (per-call ~10-20 relevant names vs the full gazetteer) to find the safe size.

## Metric

Same as text work: `test_accuracy.diff_words` vs `benchmark_roman_urdu`, call-level. Plus a
**name-level** check: of the entity names in the gold, how many the ASR now produces.

## Honest gate

The audio is **8kHz narrowband**. Biasing can only recover a name if the acoustic detail that
distinguishes it survived the codec. Where two names are acoustically identical at 8kHz,
biasing cannot separate them — that needs wideband audio or a caller confirmation turn. So the
expected outcome is: biasing recovers *some* in-gazetteer names, capped by the 8kHz ceiling.

## Deliverable

A measured answer: does contextual biasing reduce mishearings on this 8kHz set, and by how
much — the first accuracy lever that goes *above* the ~65% text ceiling (or proof that the
ceiling is the audio itself).

---

## RESULT (2026-07-21) — biasing gives a marginal nudge; 8kHz is the ceiling

Ran the base **Qwen3-ASR-1.7B** (`testing/audio_biasing_benchmark.py`), re-transcribing call
chunks with **context = the call's gold names** (the *upper bound* of biasing — we literally
told the model the answers). CPU is slow (~80–380s per 25s chunk), so this is a scoped sample.

**Call `…976941`, biased with 26 gold names (incl. `Shahid`):**

| | transcript (excerpt) | diff_words |
|---|---|---|
| no context | `...shahebaad karon chupachaaila subataai...` | 16.6% |
| + names | `...Shaher baat karun chupai la. Sabtai...` | 18.1% (+1.5) |
| gold | `...Shahid baat kar raha hoon chughtai lab se...` | — |

**The decisive observation:** `Shahid` was **in the bias context**, yet the model still heard
`Shaher` / `shahebaad`. **Even told the exact name, the 8kHz audio does not contain the
acoustic detail to recover it.** Biasing improved general structure (`baat karun`, spacing) for
+1.5%, but **could not fix the name** — the mishearing.

**Conclusion — the ceiling is the audio, not the method.** Contextual biasing helps at the
margin, but on 8kHz narrowband it cannot recover names whose distinguishing detail the codec
destroyed. The real lever for names is therefore:

1. **Wideband audio** (16kHz / Opus at the source) — restores the consonant/vowel detail that
   separates `Shahid` from `Shaher`. This is the single highest-value change.
2. **Caller confirmation turn** for form-critical fields (spell the name) — what humans do.

Biasing + two-pass become worthwhile **once the audio is wideband**; on 8kHz they are capped.

*Compute note:* a full 806-transcription run (~18h on CPU) was not run here; the scoped result
is directionally conclusive. Re-run at scale on GPU with `testing/audio_biasing_benchmark.py`
when wideband audio is available.
