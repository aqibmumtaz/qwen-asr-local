# Session Context — Urdu ASR Low-Confidence Correction

> **Purpose:** Full handoff of an analysis/design session so this work can resume in another
> project. This directory is meant to be **moved as a self-contained unit**. Drop it into the
> new project and point a fresh Claude session at this file.
>
> **Date of session:** 2026-06-02
> **Source project:** AI-Clinical-Triage-System (Urdu-first, real-time voice clinical triage)

---

## 0. TL;DR — where we landed

- We analyzed an ASR eval spreadsheet (`turnwise_results_eval_full.xlsx`) for an Urdu
  call-center transcription pipeline and reverse-engineered what every accuracy/confidence
  column actually means.
- We classified the transcription errors into **3 buckets** and proved which signals can and
  cannot catch each.
- We designed a **text-only "fix stage"** that sits after the existing lexicon step to correct
  low-confidence words, and weighed it against **fine-tuning Qwen2-Audio**.
- **Key decision:** start with a **text2text / prompt-based corrector** (≈0 training data to
  start; pairs already exist in the spreadsheet). Audio fine-tuning is the only way to fix the
  hardest error type but needs 10–30 hrs of annotated audio.

---

## 1. The pipeline (current state)

```
Audio (8 kHz mono telephony)
   → Qwen STT  ──────────────►  Hindi (Devanagari) text + per-word confidence
   → Hindi→Roman lexicon script ►  Roman Urdu  (column: roman_urdu_model) + conf carried through
   → [ FIX STAGE — to be built ] ►  corrected Roman Urdu   ← THIS is the open work
```

- **Qwen STT** outputs **best hypothesis + confidence only** (NO n-best / alternatives — confirmed by user).
- The **lexicon script already exists and works** — it romanizes Hindi→Roman Urdu and applies
  known normalizations (see `lexicon_updates` column, e.g. `pahle -> pehle`, `aisan -> ahsan`).
- The **fix stage does NOT yet exist.** Its job: correct the remaining low-confidence /
  wrong words in `roman_urdu_model`.
- Because Qwen exposes no n-best and the fix stage runs on text, **the fix stage is text-only
  and cannot consult the audio.** That single fact bounds what it can achieve (see §4, §6).

> Note: the *live* triage product uses Deepgram STT (see project CLAUDE.md). The Qwen track here
> is the R&D effort to build a better Urdu STT. Don't conflate the two.

---

## 2. The dataset — `turnwise_results_eval_full.xlsx`

One sheet `asr_results`, **183 data rows × 27 columns**, **8 calls**, one row per conversation turn.
This is both the **eval set** and (for text2text) a ready-made **training set**.

> ⚠️ This dir (`CLL data/`) currently contains audio for **only ONE call**
> (`in-4234500300-+393928520852-...`, 20 turns). The xlsx covers all 8 calls. If you need the
> other 7 calls' audio, copy them from the source project's `training/data/CLL data/` too.

### Column dictionary (verified this session)

| Column | Meaning |
|---|---|
| `audio_name` | e.g. `turn_001_agent.wav` |
| `actual_urdu_transcript` | Urdu-script reference (human) |
| `model_output_hindi` | Qwen STT raw output, Devanagari |
| `model_output_roman_urdu` / `roman_urdu_model` | Lexicon-romanized model output (**the input to the fix stage**) |
| `word_scores` | JSON per word: `{hindi, roman, min_conf, geo_conf, low}` |
| `word_conf_readable` | `word:geo_conf` pairs, space-separated |
| `min_conf_row`, `geo_conf_row` | **ADDED THIS SESSION** — row-wise min and geometric-mean of word confidences (derived from `word_conf_readable`). Placed right after `word_conf_readable`. |
| `roman_urdu_reference` | **GOLD** Roman Urdu (human reference) — the text2text target |
| `lexicon_updates` | `model_word -> corrected` pairs the lexicon applied |
| `incorrect_words` | diff list `model | reference` — the error log (gold-labeled) |
| `word_level_accuracy_row` | see §3 — confidence-correlated, **NOT** true accuracy |
| `roman_urdu_accuracy_row` | see §3 — **true WER accuracy** |
| `confidence_score_pct` / `mean_word_conf` | mean of per-word `geo_conf` |
| `n_words`, `latency_s`, `duration_s`, `speaker`, `turn`, `call_id`, `start_s`, `end_s`, `status` | metadata |

### Dataset stats
- Word-level accuracy: mean 75.8%, median 84.2%
- Roman-Urdu (WER) accuracy: mean 66.1%, median 71.4%
- Confidence: mean 89.5% · Latency: mean 0.59 s · Duration: mean 5.3 s/turn
- ~20% of all words (652 / 3231) are wrong per `incorrect_words`.

---

## 3. What the accuracy columns actually compute (reverse-engineered)

- **`confidence_score_pct` = `mean_word_conf` = mean of per-word `geo_conf`.** (Exact match.)
- **`word_level_accuracy_row` = `(n_words − incorrect_count) / n_words`** — a model-word-count
  metric. It correlates with confidence (r=0.80) more than with the true reference diff (r=0.58).
  **Treat it as a confidence proxy, NOT real accuracy.** (Row 2 in the sheet is an off-by-one
  spreadsheet-formula artifact; ignore.)
- **`roman_urdu_accuracy_row` = WER accuracy = `1 − edit_distance/ref_words`** computed on
  `roman_urdu_model` vs `roman_urdu_reference`, **after** applying `lexicon_updates` to the model
  output. This is the **true** accuracy (r=0.86 with the reference diff). Use THIS as the metric
  to optimize. (Minor implementation quirk in how one-model-word→many-ref-words mismatches are
  penalized, but the formula is confirmed.)

---

## 4. Error taxonomy — the 3 buckets (the core mental model)

Worked from **row 1** (`turn_001_agent.wav`, this call). Reference (13 words):
`ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon`
Model (9 words): `ji es romalikoom chukaai lab se daanishli baat karun`
→ WER accuracy 30.77% (only `ji, lab, se, baat` survived).

| Bucket | Example | min_conf | geo_conf | Caught by confidence? | Caught by LM? |
|---|---|---|---|---|---|
| **1. OOV hallucination** | `daanishli` ← "danish ali" | **0.82** (looks fine!) | 0.86 | ❌ (high conf) | ✅ surprisal 11.9 |
| **1. OOV hallucination** | `romalikoom` ← "alaikum" | 0.43 | 0.85 | ✅ | ✅ 9.0 |
| **2. Garbled non-word** | `chukaai` ← "chughtai" | 0.54 | 0.69 | ✅ | ✅ 10.6 |
| **2. Garbled non-word** | `es` ← "assalam o" | 0.58 | 0.65 | ✅ | ✅ 8.8 |
| **3. Boundary collapse → valid word** | `karun` ← "kar raha hoon" | 0.69 | 0.75 | borderline | ❌ surprisal 2.69 (fluent!) |

**The critical insight:** `karun` is a real, common word and `baat karun` is grammatically
fluent, so **no text signal (confidence, dictionary, or LM) can flag it.** It needs **acoustic
evidence** (word-count/duration vs audio, or forced alignment, or Qwen n-best). This is the
ceiling on every text-only approach.

### WER-faithful word-level table (row 1)

The 9 model words actually align to **13 reference words** — three model words each swallowed a
multi-word span (the `WER ops` column shows the substitutions `S` and deletions `D`). `LM surp` is
the masked-LM surprisal (`xlm-roberta-base`); `Caught by` is which detection signal flags the error.

| # | Model | Ground Truth | ref# | WER ops | min_conf | geo_conf | LM surp |  | Caught by |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ji | ji | 1 | · | 0.967 | 0.973 | 4.68 | ✓ | — |
| 2 | es | assalam o | 2 | 1S+1D | 0.576 | 0.647 | 8.78 | ✗ | conf + LM |
| 3 | romalikoom | alaikum | 1 | 1S | 0.428 | 0.851 | 9.02 | ✗ | conf + LM |
| 4 | chukaai | chughtai | 1 | 1S | 0.541 | 0.690 | 10.61 | ✗ | conf + LM |
| 5 | lab | lab | 1 | · | 0.995 | 0.996 | 2.03 | ✓ | — |
| 6 | se | se | 1 | · | 0.999 | 0.999 | 3.84 | ✓ | — |
| 7 | daanishli | danish ali | 2 | 1S+1D | 0.821 | 0.858 | 11.91 | ✗ | conf + LM |
| 8 | baat | baat | 1 | · | 0.999 | 1.000 | 5.43 | ✓ | — |
| 9 | karun | kar raha hoon | 3 | 1S+2D | 0.694 | 0.751 | 2.69 | ✗ | **conf only** |

Totals: N=13 ref words, S=5, D=4, I=0 → correct = 13−5−4 = **4** → WER acc = 4/13 = **30.77%**
(= `roman_urdu_accuracy_row`). The model-word view (4/9 = 44.44%) is `word_level_accuracy_row`, the
misleading one. Note rows 2, 7, 9 are the boundary-collapse words — they are **both** substitutions
**and** the source of every deletion, and `karun` (row 9) is the lone error no LM can catch.

Root causes of high error rate overall: **8 kHz telephony audio** (model trained on 16 kHz),
**OOV proper nouns** (lab/doctor names), and **fast speech collapsing word boundaries**.

---

## 5. Detection findings (which words to send to the fixer)

- **Confidence alone caps at F1 ≈ 0.50 / recall ≈ 0.60** across the full 3231 words — a hard
  ceiling, because bucket-1 OOV words can be confidently wrong (`daanishli`).
- Best single confidence signal: **`geo_conf < 0.90`** (F1 0.51).
- The "gap" idea (`geo − min` large ⇒ wrong) looked great on row 1 but **did NOT generalize**
  (logistic-regression gap coefficient was negative). Don't rely on it.
- **LM surprisal** (masked-LM, `xlm-roberta-base`; MuRIL blocked by torch<2.6 + no safetensors)
  catches buckets 1 & 2 strongly, **misses bucket 3**.
- **Recommended detection gate (dual, not low-conf-only):**

  ```
  flag word  if  geo_conf < 0.90  OR  min_conf < 0.60  OR  LM_surprisal > τ
  ```

  The LM term is essential — without it the confidently-wrong OOV words (`daanishli`) are never
  even seen by the fixer.

LM surprisal = `−log2 P(word | rest of sentence)`, computed by masking each word's subword
tokens and reading the model's log-prob. High = unexpected = likely wrong.

---

## 6. The Fix-Stage track (text-only post-fix) — recommended first build

Detection and correction use **different tools** (each weak at the other's job):

1. **Detect** — dual gate above (confidence finds acoustic doubt; LM surprisal finds linguistic
   impossibility).
2. **Reconstruct** — a **generative LLM** infills the flagged span from
   *context + the garbled token (keeps phonetic shape) + a domain glossary* (lab names, doctor
   names, drugs, greetings). Must be **generative** (not masked-LM) because fixes are
   variable-length: `es → assalam o` (1→2), `karun → kar raha hoon` (1→3).
   **Surgical: only rewrite flagged spans**, leave everything else byte-for-byte (clinical safety).
3. **Validate** — re-score with LM; **accept only if surprisal drops**. If unsure, keep original
   and flag for human review (never silently guess on a clinical word).
4. **Learn** — log accepted fixes → auto-grow the lexicon (recurring LLM fixes become
   deterministic rules → less cost/latency over time). This is what `lexicon_updates` already is.
5. **Measure** — score every change on this xlsx via `roman_urdu_accuracy_row`.

**Ceiling:** fixes buckets 1 & 2 (~80% of errors). Does **not** fix `karun` (bucket 3, needs audio).

---

## 6.1 Fixing the Roman incorrect words — deeper discussion

This is the heart of the open work. The goal: turn `roman_urdu_model` into something closer to
`roman_urdu_reference` by repairing the flagged words, **without** access to the audio.

### What makes a Roman-Urdu word "wrong" — and how to repair each kind

| Sub-type | Row-1 example | What it really is | Repair strategy |
|---|---|---|---|
| **Known recurring error** | `pahle→pehle`, `bradar→brother` | spelling/normalization the lexicon already handles | **Deterministic lexicon map** (free, instant) |
| **OOV proper noun** | `chukaai`→`chughtai`, `daanishli`→`danish ali` | a real entity the STT never learned | **Glossary match** (phonetic) + LLM context |
| **Garbled non-word** | `romalikoom`→`alaikum`, `es`→`assalam o` | acoustic mush romanized literally | **LLM contextual infill** (the garble still carries phonetic shape) |
| **Boundary collapse → valid word** | `karun`→`kar raha hoon` | a fluent word in a fluent slot | **Unfixable from text** — needs audio (bucket 3) |

### The correction order (cheap → expensive, stop when resolved)

1. **Deterministic lexicon** — exact-match replace for known errors. Zero risk, zero latency.
   Most production traffic is repeats of the same names/greetings, so this carries a lot.
2. **Phonetic glossary match** — Roman Urdu has **no standard spelling**, so compare the flagged
   token to a domain entity list using a phonetic key (Soundex / Double-Metaphone / edit distance
   on a normalized phonetic form). `chukaai` ≈ `chughtai`, `daanishli` ≈ `danish ali` resolve here
   *deterministically* if the entity is in the glossary. This is the single highest-leverage piece
   for the OOV bucket, because those are **names** and names live in a finite list.
3. **LLM contextual infill** — only for what 1–2 can't resolve. The model gets:
   - the **full sentence** (context tells it slot 2–4 is a greeting),
   - the **garbled token itself** (keep it — `romalikoom` rhymes with `alaikum`; throwing it away
     loses the only phonetic clue),
   - the **Hindi original** (`रॉमलिकूम` — sometimes the Devanagari preserves detail the romanizer dropped),
   - the **glossary** (bias toward real entities).

### Why a *generative* model, not masked-LM, for the rewrite
Corrections are **variable-length**: `es → assalam o` (1→2 words), `karun → kar raha hoon` (1→3).
A masked-LM has a fixed number of `[MASK]` slots and cannot expand one token into three. Use the
masked-LM only for **detection/validation** (surprisal); use a generative LLM for the **rewrite**.

### Worked expectation on row 1
- `es`, `romalikoom` → greeting context + phonetic shape → LLM reconstructs `assalam o alaikum`. ✅
- `chukaai`, `daanishli` → phonetic glossary match to `chughtai`, `danish ali` (if in glossary). ✅
- `karun` → context `baat karun` is fluent; **left unchanged** (correctly flagged but unfixable). ❌
- Net: 4 of 5 errors repairable from text; `roman_urdu_accuracy_row` would jump from 30.77% toward
  ~85–92% on this turn.

### Guardrails (non-negotiable for a clinical system)
- **Surgical edits only** — rewrite the flagged span, leave every other word byte-for-byte.
- **LM-validation** — accept a fix only if sentence surprisal **drops**; otherwise revert.
- **No silent guessing** — if the corrector's own confidence is low, keep the original and route to
  human review. A confidently-wrong "fix" of a symptom word is worse than a flagged uncertainty.
- **Never touch** high-conf + low-surprisal words.

### The flywheel (cost goes down over time)
Every accepted LLM fix is logged as a `model→correct` pair (exactly the `incorrect_words` /
`lexicon_updates` shape). Recurring ones get **promoted to deterministic lexicon rules**, so the
expensive LLM step is called less and less. The same log is also the **text2text training set** and
the **error model for synthesizing** more training data (§8).

### Sketch of the LLM corrector contract
```
INPUT  (per flagged span):
  context_left, garbled_token, context_right, hindi_original, glossary[]
OUTPUT (structured):
  { corrected_span: str, confidence: float, used_glossary_entity: str|null }
RULES:
  - only the span may change; if unsure, return the original with low confidence
  - prefer a glossary entity when phonetically plausible
```

### Bottom line
The text-stage fixer is a **layered repair**: deterministic lexicon → phonetic glossary match →
constrained generative LLM, gated by the dual detector and validated by LM surprisal. It recovers
the OOV/garbled buckets (~80% of errors). `karun`-type collapses stay for the audio track.

---

## 7. Architecture decisions (settled this session)

User repeatedly asked about "appending LLM layers to the Qwen ASR." Conclusions:

- **"Appending LLM layers" is never the right move.** Appended layers are randomly initialized →
  need the *same* training data as fine-tuning, plus they degrade output until trained. The
  architecture wiring does **not** reduce the data requirement.
- **Qwen2-Audio is already `audio encoder → projector → full Qwen-7B LLM decoder`.** There is no
  empty socket; the LLM is already the output head. More layers on a 32-layer LLM ≠ better.
- **Placement "after lexicon" forces text-only** — the audio is already gone by then, so such a
  head has the same `karun` ceiling as a plain correction LLM, with more cost. No benefit.
- **The real fork:**

  | | Text-only correction | Audio-aware correction |
  |---|---|---|
  | Sees | Roman text | Qwen acoustic features |
  | How | separate LLM (prompt or small fine-tune) — **don't fuse** | **LoRA-tune existing Qwen2-Audio** decoder |
  | Data | text pairs (cheap, already have) | audio-paired (10–30 hrs) |
  | Fixes `karun`? | ❌ | ✅ |

- **The data requirement is set by the signal you need (text vs audio), not by the wiring.**

Existing fine-tune script in source project: `training/scripts/train_qwen_audio.py`
(`Qwen/Qwen2-Audio-7B-Instruct`, LoRA rank 64, QLoRA via `--load_in_4bit`, needs A100-80GB or
4090). Lighter alternative: `train_whisper.py` (smaller, weaker built-in LLM).

---

## 8. Data requirements

### Text2text corrector (recommended path)
- **Prompt-based (few-shot): ≈0 training** — 10–20 examples + glossary in the prompt. Ship now.
- **Fine-tune small text2text (mT5/ByT5-base): floor ~1–2k pairs, usable ~5–10k, strong ~20k+.**
- **You already have the pairs:** `(roman_urdu_model → roman_urdu_reference)` = 183 real pairs in
  the xlsx, produced as a byproduct. No new annotation modality.
- **Synthesis unlock:** build an error model from the `incorrect_words` column, inject those
  realistic errors into clean `roman_urdu_reference` text → manufacture tens of thousands of
  `(corrupted, clean)` pairs. (Errors are repetitive — same names/greetings/telephony garbles —
  so the task is data-efficient.)
- **Ceiling:** still text-only → still can't fix `karun`.

### Audio LoRA (Qwen2-Audio) — only way to fix bucket 3
- Tier 1 (proof): ~500–1k pairs (1–3 hrs). Tier 2 (beats pipeline): ~3–10k pairs (10–30 hrs).
  Tier 3 (production): 50–100+ hrs.
- You have ~183 gold turns — **below Tier 1.** Enough to *evaluate*, not yet to *train*.
- Annotate by **post-editing** (correct the draft) not transcribing from scratch — 5–10× faster.
- Augment: simulate telephony (downsample→8kHz→μ-law→back), speed/pitch perturb.

---

## 9. Recommended next steps (in order)

1. **Build the error-injection synthesizer** — turn `incorrect_words` into a large
   `(corrupted, clean)` text2text training set. (User was about to greenlight this.)
2. **Ship prompt-based corrector** on the live pipeline: dual detection gate + glossary +
   surgical span rewrite + LM-validation guardrail. Measure on `roman_urdu_accuracy_row`.
3. **Collect** human-accepted corrections → grow real pair set.
4. **Fine-tune small ByT5/mT5** once volume justifies replacing the API (cheaper/lower-latency).
5. **(Stretch) Audio:** accumulate post-edited turns toward Tier-1, then LoRA-tune Qwen2-Audio to
   attack bucket 3 (`karun`-type collapses). Evaluate against this xlsx as held-out.

### Open questions to resolve in the new project
- How many gold `(audio, roman_urdu)` pairs exist across the full source `training/data/`? (Only
  1 of 8 calls' audio is in this dir.) Needed to know distance to Tier-1.
- Glossary contents — collect the domain entity list (lab names, doctor names, drugs, greetings).
- Latency budget for the live pipeline — decides prompt-API vs local small model.
- Can Qwen STT be coaxed to emit n-best later? That would unlock bucket-3 detection cheaply.

---

## 10. Files in this directory

- `turnwise_results_eval_full.xlsx` — the eval/training spreadsheet (27 cols, incl. the
  `min_conf_row`/`geo_conf_row` columns added this session). **Synced to latest version.**
- `in-4234500300-+393928520852-20260503-134723-1777798043.1006725/` — audio for 1 call
  (20 turns, `turn_XXX_<speaker>.wav`, 8 kHz mono PCM). This is the call analyzed in §4.
- `SESSION_CONTEXT.md` — this file.

## 11. Environment notes
- Source project conda env: `ai-clinical-triage` (python 3.10).
- Reading the xlsx: `openpyxl` (installed this session). `pandas` was NOT installed.
- LM surprisal demo used `transformers` + `torch 2.2.2`. ⚠️ torch 2.2.2 **refuses** to load
  `.bin`-only models (CVE-2025-32434) — use models that ship **safetensors**
  (`xlm-roberta-base` works; `google/muril-base-cased` was blocked). Upgrade torch ≥2.6 to use MuRIL.
- An OpenAI API key was present in the source project `.env` (value never read). No Anthropic key.
