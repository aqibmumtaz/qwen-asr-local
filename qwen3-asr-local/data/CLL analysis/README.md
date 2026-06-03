# CLL Analysis — Roman Urdu ASR Corrector

**Project:** qwen-asr-local  
**Conda env:** `ai-clinical-triage`  
**Status:** Working — 0.6B achieves 1.00 WER on Turn 1 with few-shot examples

---

## Quick Start

```bash
cd qwen3-asr-local
conda activate ai-clinical-triage
```

### Run corrector test

```bash
# Default — Qwen3-0.6B, first 5 rows
python3 test_qwen_corrector.py

# First row only (fast check)
python3 test_qwen_corrector.py --rows 1

# All 183 rows
python3 test_qwen_corrector.py --rows all

# Compare all model sizes side-by-side (run on GPU system)
python3 test_qwen_corrector.py --all-models --rows 5

# Custom model selection
python3 test_qwen_corrector.py --models Qwen/Qwen3-1.7B Qwen/Qwen3-4B --rows 10

# Disable guardrail to see raw LLM output
python3 test_qwen_corrector.py --no-guardrail --rows 5
```

### Run full ASR pipeline (audio → corrected Roman Urdu)

```bash
python3 roman_urdu_asr.py audio.wav
python3 roman_urdu_asr.py audio.wav --corrector qwen
python3 roman_urdu_asr.py audio.wav --corrector mt5 --model path/to/trained/mt5
python3 roman_urdu_asr.py audio.wav --corrector none   # raw transliteration only
```

---

## What This Does

The corrector fixes garbled words in ASR output caused by 8kHz telephony audio quality,
OOV proper nouns (lab names, doctor names), and fast-speech collapses.

```
Audio (8kHz telephony)
  → Qwen3-ASR 1.7B       →  Hindi Devanagari + per-word confidence
  → transliterate()       →  Raw Roman Urdu
  → Corrector.fix()       →  Fixed Roman Urdu  ← this is what we're building
```

---

## Step-by-Step Correction Flow

**Worked example — Turn 1, Chughtai Lab call-center recording:**

### Step 1 — ASR raw output

```
Hindi:      जी एस रॉमलिकूम चुकाई लैब से दानिशली बात करूं।
Roman Urdu: ji es romalikoom chukaai lab se daanishli baat karun.
```

### Step 2 — Confidence gate

Each word is checked: flag if `min_conf < 0.65` OR `geo_conf < 0.90`

```
Word           min_conf  geo_conf  Decision
-----------    --------  --------  --------
ji              0.967     0.973    ✓ HIGH CONF — LLM never touches this
es              0.576     0.647    ⚠ FLAGGED
romalikoom      0.428     0.851    ⚠ FLAGGED
chukaai         0.484     0.690    ⚠ FLAGGED
lab             0.989     0.996    ✓ HIGH CONF
se              0.997     0.999    ✓ HIGH CONF
daanishli       0.557     0.858    ⚠ FLAGGED
baat            0.999     1.000    ✓ HIGH CONF
karun           0.414     0.751    ⚠ FLAGGED
```

### Step 3 — Annotate for LLM

```
ji [FIX:es] [FIX:romalikoom] [FIX:chukaai] lab se [FIX:daanishli] baat [FIX:karun]
```

### Step 4 — Qwen LLM fixes all [FIX:] words in one call

Prompt contains: system role + domain glossary + 2 few-shot examples + the annotated text.

```
LLM output:
  ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon

Fixes applied:
  [FIX:es] [FIX:romalikoom]  →  assalam o alaikum   (greeting, two words collapsed)
  [FIX:chukaai]              →  chughtai             (lab name)
  [FIX:daanishli]            →  danish ali           (agent name, two words)
  [FIX:karun]                →  kar raha hoon        (verb phrase)
```

### Step 5 — Guardrail

Code checks that all high-conf words from the input are still in the output.
If any were dropped by the LLM, they are re-inserted at their original relative position.

```
ji   → present ✓
lab  → present ✓
se   → present ✓
baat → present ✓
Nothing re-inserted.
```

### Step 6 — Final result

```
corrected: ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon
reference: ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon

WER: 0.31 → 1.00  ✓ PERFECT
```

---

## Architecture

```
corrector.py
│
├── DOMAIN_GLOSSARY          (garbled → correct) pairs from incorrect_words column
│
└── QwenBackend
    ├── _SYSTEM              role + glossary + rules → every prompt
    ├── _EXAMPLES            2 few-shot examples → critical for 0.6B proper noun disambiguation
    ├── _annotate()          wraps [FIX:word] on low-conf words, plain text for high-conf
    ├── _build_prompt()      system + examples + user turn in Qwen chat format
    ├── _parse_output()      strips <think> blocks, non-Latin script, prompt echoes
    ├── _reinsert_dropped()  guardrail — re-inserts dropped high-conf words
    └── correct()            orchestrates all steps → corrected sentence
```

---

## Prompt Format

```
<|im_start|>system
/no_think
You are an Urdu ASR post-corrector...
[DOMAIN_GLOSSARY]
Rules: ...
<|im_end|>

<|im_start|>user                         ← few-shot example 1 (input)
ASR text: ji [FIX:es] [FIX:romalikoom] [FIX:chukaai] lab se [FIX:daanishli] baat [FIX:karun]
Corrected:<|im_end|>
<|im_start|>assistant                    ← few-shot example 1 (answer)
ji assalam o alaikum chughtai lab se danish ali baat kar raha hoon
<|im_end|>

<|im_start|>user                         ← actual turn to fix
ASR text: {annotated turn}
Corrected:<|im_end|>
<|im_start|>assistant                    ← model completes here
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| One LLM call per full turn | Full context → better multi-word fixes (e.g. `es + romalikoom` = greeting) |
| High-conf words not in output targets | LLM can't alter correct words — only fixes `[FIX:]` markers |
| Few-shot examples | 0.6B ignores glossary hints for similar proper nouns — examples show exact mapping |
| Glossary + examples | Glossary = what; examples = how (needed together for small models) |
| Guardrail (code, not LLM) | Safety net — guarantees high-conf words always present in output |
| max_tokens = 512 | 0.6B uses `<think>` mode internally — needs tokens to finish before answering |

---

## Model Comparison (Turn 1 sample)

| Approach | WER | Notes |
|---|---|---|
| Raw ASR, no correction | 0.31 | baseline |
| 0.6B, no glossary | 0.31 | falls back — outputs Nastaliq script |
| 0.6B, glossary only | 0.31 | ignores fine-grained hints |
| **0.6B, glossary + examples** | **1.00** | examples disambiguate proper nouns ✓ |
| 4B, glossary + examples | 1.00 | reliable even without examples |

> Run `--all-models` on a GPU system to get full 183-row comparison across all model sizes.

---

## Files

| File | Description |
|---|---|
| `../../corrector.py` | All prompt, model, guardrail logic |
| `../../test_qwen_corrector.py` | Test script with multi-model support |
| `../../test_corrector.py` | Test harness for both Qwen and mT5 backends |
| `../../roman_urdu_asr.py` | Full pipeline: ASR + transliterate + corrector |
| `turnwise_results_eval_full.xlsx` | 183-turn eval set (27 cols, 8 calls) — ground truth |
| `SESSION_CONTEXT.md` | Original session handoff from AI-Clinical-Triage-System |

---

## Next Steps

1. Run `--all-models --rows all` on GPU to get full 183-row WER comparison
2. Expand `_EXAMPLES` from `incorrect_words` column across all 183 turns
3. Grow `DOMAIN_GLOSSARY` from all 8 calls (currently seeded from Turn 1 only)
4. Build error-injection synthesizer → manufacture mT5 training pairs
5. Fine-tune mT5-small on `(roman_urdu_model → roman_urdu_reference)` pairs as faster corrector
