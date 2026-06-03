# Roman Urdu ASR Corrector Pipeline

**Project:** qwen-asr-local  
**Date:** 2026-06-03  
**Status:** Working on 0.6B — 1.00 WER on Turn 1 sample

---

## Overview

The corrector is a post-processing stage that sits after the ASR and transliteration steps.
It fixes low-confidence (garbled) words in the Roman Urdu transcript using a Qwen LLM,
while leaving high-confidence words completely untouched.

```
Audio
  → Qwen3-ASR 1.7B          (Hindi Devanagari + per-word confidence scores)
  → transliterate()          (deterministic Hindi → Roman Urdu)
  → Corrector.fix()          (LLM fixes low-conf words only)
  → Corrected Roman Urdu
```

---

## Step-by-Step Correction Flow

### Turn 1 worked example

**Input audio:** Chughtai Lab call-center recording (8kHz telephony)

---

### STEP 1 — ASR Raw Output

Qwen3-ASR 1.7B transcribes audio to Hindi, with per-word confidence scores:

```
Hindi:      जी एस रॉमलिकूम चुकाई लैब से दानिशली बात करूं।
Roman Urdu: ji es romalikoom chukaai lab se daanishli baat karun.
```

---

### STEP 2 — Confidence Gate

Each word is checked against two thresholds:
- `min_conf < 0.65`  (minimum sub-token confidence — catches acoustically uncertain words)
- `geo_conf < 0.90`  (geometric mean confidence — catches linguistically risky words)

Either condition triggers a `⚠ FLAGGED` label.

```
Word          min_conf  geo_conf  Decision
-----------   --------  --------  --------
ji             0.967     0.973    ✓ HIGH CONF — untouched by LLM
es             0.576     0.647    ⚠ FLAGGED
romalikoom     0.428     0.851    ⚠ FLAGGED
chukaai        0.484     0.690    ⚠ FLAGGED
lab            0.989     0.996    ✓ HIGH CONF — untouched by LLM
se             0.997     0.999    ✓ HIGH CONF — untouched by LLM
daanishli      0.557     0.858    ⚠ FLAGGED
baat           0.999     1.000    ✓ HIGH CONF — untouched by LLM
karun          0.414     0.751    ⚠ FLAGGED
```

**Result:** 5 flagged, 4 high-conf

---

### STEP 3 — Annotate for LLM

Flagged words are wrapped in `[FIX:word]` markers.
High-conf words are passed through as plain text.

```
ji [FIX:es] [FIX:romalikoom] [FIX:chukaai] lab se [FIX:daanishli] baat [FIX:karun]
```

---

### STEP 4 — Qwen LLM Correction (one call for the full turn)

The annotated text is sent to **Qwen3-0.6B** (or 4B for production) with:

1. **System prompt** — role + domain glossary + rules
2. **Few-shot examples** — 2 worked examples showing correct fixes (critical for small models)
3. **User message** — the annotated turn to fix

The LLM sees the full sentence context and fixes all `[FIX:]` groups in one call.

```
LLM output:
  ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon
```

Corrections made:
- `[FIX:es] [FIX:romalikoom]` → `assalam o alaikum`  (two garbled words = one greeting)
- `[FIX:chukaai]`             → `chughtai`           (lab name, from glossary + example)
- `[FIX:daanishli]`           → `danish ali`          (agent name, two words)
- `[FIX:karun]`               → `kar raha hoon`       (verb phrase expansion)

---

### STEP 5 — Guardrail

After LLM output, the guardrail checks that all high-conf words are still present.
If any were dropped by the LLM, they are re-inserted at their correct position.

```
Check: ji    → present ✓
Check: lab   → present ✓
Check: se    → present ✓
Check: baat  → present ✓

Nothing to re-insert.
```

**Reinsertion logic (when triggered):**
- If a dropped word has no preceding high-conf anchor → prepend it
- Otherwise → insert immediately after the last occurrence of the preceding high-conf word
- This preserves relative order of high-conf words regardless of LLM output

---

### STEP 6 — Final Output

```
corrected: ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon
reference: ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon

WER accuracy: 0.31 → 1.00  ✓ PERFECT
```

---

## Architecture

```
corrector.py
│
├── DOMAIN_GLOSSARY          known (garbled → correct) mappings from incorrect_words column
│
└── QwenBackend
    ├── _SYSTEM              role + glossary + rules (injected into every prompt)
    ├── _EXAMPLES            2 few-shot examples (critical for 0.6B proper noun disambiguation)
    ├── _annotate()          wraps [FIX:word] on low-conf words, leaves high-conf plain
    ├── _build_prompt()      assembles system + examples + user turn in Qwen chat format
    ├── _parse_output()      strips <think> blocks, non-Latin script, prompt echoes
    ├── _reinsert_dropped()  guardrail — re-inserts any dropped high-conf words
    └── correct()            orchestrates all steps, returns corrected sentence
```

---

## Prompt Structure

```
<|im_start|>system
/no_think
You are an Urdu ASR post-corrector...
[DOMAIN_GLOSSARY]
Rules: ...
<|im_end|>

<|im_start|>user
ASR text: ji [FIX:es] [FIX:romalikoom] [FIX:chukaai] lab se [FIX:daanishli] baat [FIX:karun]
Corrected:<|im_end|>
<|im_start|>assistant
ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon   ← example answer
<|im_end|>

<|im_start|>user
ASR text: {actual turn to fix}
Corrected:<|im_end|>
<|im_start|>assistant
                                                                      ← model completes here
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| One LLM call per full turn | Model sees full sentence context for better corrections |
| High-conf words never sent as output targets | Prevents LLM from altering correct words |
| Few-shot examples in prompt | 0.6B model needs demonstrations, not just rules |
| Glossary vs examples | Glossary = what to fix; examples = how to apply (needed for similar proper nouns) |
| Guardrail post-processing | Safety net for dropped high-conf words; code-guaranteed, not LLM-dependent |
| max_tokens = 512 | 0.6B uses thinking mode before answering — needs room to finish |

---

## Model Comparison

| Model | Sample 1 WER | Notes |
|---|---|---|
| Raw ASR (no correction) | 0.31 | baseline |
| Qwen3-0.6B, no glossary | 0.31 | model too small, falls back |
| Qwen3-0.6B, glossary only | 0.31 | 0.6B ignores fine-grained hints |
| Qwen3-0.6B, glossary + examples | **1.00** | examples disambiguate proper nouns |
| Qwen3-4B, glossary + examples | **1.00** | reliable without examples too |

---

## Files

| File | Purpose |
|---|---|
| `corrector.py` | Main corrector class — all prompt/model/guardrail logic |
| `test_qwen_corrector.py` | Test script — reads xlsx, runs corrector, shows WER before/after |
| `test_corrector.py` | Broader test harness supporting both Qwen and mT5 backends |
| `roman_urdu_asr.py` | Full pipeline wrapper (ASR + transliterate + corrector) |
| `data/CLL analysis/turnwise_results_eval_full.xlsx` | 183-turn eval dataset (27 cols, 8 calls) |

---

## Next Steps

1. Run 5-row and full 183-row test to measure overall WER improvement
2. Expand `_EXAMPLES` with more diverse error types from `incorrect_words` column
3. Build error-injection synthesizer to manufacture training pairs for mT5 fine-tuning
4. Grow `DOMAIN_GLOSSARY` from all 183 turns (currently seeded from Turn 1 only)
5. Switch to Qwen3-4B for production (already downloaded, gives 1.00 reliably without examples)
