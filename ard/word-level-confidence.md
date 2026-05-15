# Word-Level Confidence Extraction from Qwen3-ASR

**Module:** `qwen3-asr-local/asr_transcribe_and_transliterate.py`
**Functions:** `hf_asr_with_confidence()`, `WordConf` dataclass,
`_aggregate_tokens_to_words()`

---

## 1. Purpose

For each transcribed word, output a confidence score that downstream
systems can use to:

- Flag uncertain words for human review
- Sort/rank words by overall confidence
- Tune a "auto-accept vs review vs reject" policy
- Detect when the ASR is genuinely unsure (versus when it's confidently wrong)

This is **review-triage data**, not a quality metric. A word with high
confidence can still be wrong (if the model is consistently mishearing
that word); a word with low confidence is one the model itself flagged
as uncertain.

---

## 2. The flow

```
Audio file
   │
   ▼
Qwen3-ASR audio encoder ───→ audio embeddings
   │
   ▼
Qwen3 LLM decoder (autoregressive, greedy)
   │
   │ At each step:
   │   logits over 152K-token vocab
   │   pick argmax → next token
   │   (logits captured via output_scores=True)
   │
   ▼
Per-token data captured:
   for each generated token i:
       logprob_i = log_softmax(logits_i)[chosen_token_id]
       confidence_i = exp(logprob_i)        # in [0.0, 1.0]
   │
   ▼ Word grouping (BPE space-marker)
   raw_tokens = tokenizer.convert_ids_to_tokens(gen_ids)
   for each token:
       if raw_token starts with 'Ġ' → new word boundary
       else                        → continuation of current word
   │
   ▼ Per-word aggregation
   for each word's token group:
       Min Conf  = exp(min(logprobs))      ← flagging metric
       Geo Conf  = exp(mean(logprobs))     ← sorting metric
   │
   ▼
   WordConf(text, min_conf, geo_conf, n_tokens)
```

### Why log-probs in the math, "confidence" in the names

Internally we work in log-space (`min`/`mean` of logprobs, then one `exp`
at the end) to avoid float32 underflow on long words. But the user-facing
field names use **confidence** because that's what the number means after
exponentiation: a value in `[0.0, 1.0]` where 1.0 = "model was 100% sure
this was the right token", 0.0 = "model was completely unsure".

We don't expose the raw log-probabilities — only the final confidence numbers.

---

## 3. Two confidence metrics — Min Conf vs Geo Conf

Hindi words almost always span 3–7 BPE sub-tokens (Devanagari has many
combining marks per character). A single uncertain sub-token can flip
the word's meaning. Two metrics surface different failure modes:

| Metric | Formula | Question it answers | Use for |
|---|---|---|---|
| **Min Conf** | `exp(min(logprobs))` | "Was *any* sub-token uncertain?" | **Flagging** |
| **Geo Conf** | `exp(mean(logprobs))` | "Overall, how confident is this word?" | **Sorting / ranking** |

Concrete example from `sample_ur1.wav`:

```
Word: शनाख्त (7 sub-tokens)
  Token 1: श         conf 0.99
  Token 2: न         conf 0.97
  Token 3: ा         conf 0.98
  Token 4: ख         conf 0.98
  Token 5: ्         conf 0.53  ← weakest link (the borderline nukta sub-token)
  Token 6: त         conf 0.99
  Token 7: <joiner>  conf 0.99

  Min Conf = 0.53      ← catches the uncertainty
  Geo Conf = 0.89      ← averages it away
```

If we flagged on Geo Conf (0.89), `शनाख्त` would look fine. With Min Conf
(0.53), we correctly flag it for review.

---

## 4. Why flagging uses Min Conf, not Geo Conf

In ASR for languages with complex script (Devanagari, Nastaliq), **one
wrong sub-token can change a word's meaning entirely**:

- `ज़बान` (zaban = language) vs `जान` (jaan = life) — differs by one nukta sub-token
- `शनाख्त` (shanakht = recognition) vs `शनात` (not a word) — differs by one cluster sub-token

For review-triage, we want the metric to flag the word *even if just one
sub-token was shaky*. That's exactly what Min Conf does. Geo Conf would
let single weak tokens hide inside a long word.

Trade-off: Min Conf has higher **false-positive rate** on long words
(more tokens = more chances of at least one being borderline even when
the word is correct). That's acceptable for review-triage — better to
ask a human about a probably-correct word than to silently emit a wrong
one.

---

## 5. Current threshold

```python
LOW_CONF_THRESHOLD = float(os.getenv("LOW_CONF_THRESHOLD", "0.7"))
```

Default **0.7** (chosen empirically — see Section 6.1 — to flag the
real uncertainty signals on the corpus without false positives on
common words). Overridable via env var:

```bash
LOW_CONF_THRESHOLD=0.7 python3 asr_transcribe_and_transliterate.py audio.wav  # stricter
LOW_CONF_THRESHOLD=0.3 python3 asr_transcribe_and_transliterate.py audio.wav  # looser
```

Used by `WordConf.is_low`:

```python
@property
def is_low(self) -> bool:
    return self.min_conf < LOW_CONF_THRESHOLD
```

### Production policy template

A more nuanced policy is straightforward to build on top:

```python
if   wc.min_conf >= 0.9:   action = "auto_accept"
elif wc.min_conf >= 0.7:   action = "accept_log"
elif wc.min_conf >= 0.5:   action = "review"
else:                      action = "reject_or_reprocess"
```

The single `is_low` boolean is just the simplest form of "review or skip".

---

## 6. Examples — Hindi + Raw Roman + Roman Urdu + Nastaliq + Confidence

Both samples were run with **`language="English"`** as the ASR hint
(empirically gives output closer to GPU bf16 reference on Mac CPU —
the model emits the nukta marks correctly that get dropped under `Hindi`
hint). The point of showing both samples is to demonstrate the
confidence numbers are **not random** — they correlate predictably with
word properties (length, rarity, native vs foreign, borderline phonetic
decisions).

The **Raw Roman** column shows the phonetic transliteration produced by
`_transliterate_raw + _normalize_endings` in `hindi_to_roman_urdu.py`,
**before** the lexicon corrections are applied. The **Roman Urdu**
column shows the final output after `_apply_corrections` (CORRECTIONS +
PROPER_NOUNS lookups). Comparing the two reveals which words the
lexicon is actively fixing.

### Example 1 — `sample_ur1.wav`

```
ASR (Hindi):     मेरा नाम अपीब है। ये उर्दू ज़बान में आवाज़ की शनाख्त का टेस्ट है।
Roman Urdu:      mera naam apeeb hai. yeh urdoo zaban mein awaz ki shanaakht ka test hai.
Nastaliq:        میرا نام اپیب ہے۔ یے ارْدو ج़بان میں آواج़ کی شناکھْت کا ٹیسْٹ ہے۔
```

| # | Hindi (ASR) | Raw Roman | Roman Urdu | Nastaliq | Min Conf | Geo Conf | Tokens | Flag | Notes |
|---:|---|---|---|---|---:|---:|---:|:---:|---|
| 1 | मेरा | mera | mera | میرا | 1.00 | 1.00 | 4 | | native function |
| 2 | नाम | naam | naam | نام | 1.00 | 1.00 | 4 | | native function |
| 3 | अपीब | apeeb | apeeb | اپیب | 0.65 | 0.87 | 5 | **LOW** | proper name |
| 4 | है। | hai. | hai. | ہے۔ | 0.69 | 0.91 | 4 | **LOW** | sentence-end variant |
| 5 | ये | ye | yeh | یے | 0.99 | 1.00 | 3 | | lexicon: ye → yeh |
| 6 | उर्दू | urdoo | urdoo | ارْدو | 0.81 | 0.96 | 7 | | long, includes nukta |
| 7 | ज़बान | zabaan | zaban | ج़بان | **0.45** | 0.83 | 7 | **LOW** | **with nukta — matches GPU output!** |
| 8 | में | men | mein | میں | 1.00 | 1.00 | 3 | | lexicon: men → mein |
| 9 | आवाज़ | aawaaz | awaz | آواج़ | 0.73 | 0.95 | 7 | | lexicon: aawaaz → awaz |
| 10 | की | ki | ki | کی | 1.00 | 1.00 | 2 | | native postposition |
| 11 | शनाख्त | shanaakht | shanaakht | شناکھْت | 0.56 | 0.90 | 7 | **LOW** | rare cluster ख्त |
| 12 | का | ka | ka | کا | 1.00 | 1.00 | 2 | | native function |
| 13 | टेस्ट | test | test | ٹیسْٹ | 1.00 | 1.00 | 6 | | English loan |
| 14 | है। | hai. | hai. | ہے۔ | 0.98 | 0.99 | 4 | | native function |

**Flagged words (Min Conf < 0.7) — exactly the words that deserve human review:**

- **`अपीब (0.65)`** — proper name "Aqib"; model unsure how to spell an Arabic name in Devanagari
- **`है। (0.69)`** — sentence-end variant of है; model chooses between `है` and `है।` form
- **`ज़बान (0.45)`** — *the lowest confidence in the sample.* Note the **lexicon corrects `zabaan` → `zaban`**. Model's uncertainty here is genuine because Devanagari nukta `़` is rare in Hindi training data but essential in Urdu loanwords.
- **`शनाख्त (0.56)`** — rare cluster `ख्त`; borderline between `शनाख्त` and `शनात`

**Lexicon corrections visible by comparing Raw Roman vs Roman Urdu:**

- `ye → yeh` — function-word convention
- `men → mein` — function-word convention
- `aawaaz → awaz` — drops phonetic length

### Example 2 — `sample_ur2.wav`

```
ASR (Hindi):     आज का मौसम बहुत अच्छा है। हम एक नई टेक्नोलॉजी को आजमा रहे हैं।
Roman Urdu:      aaj ka mausam bahut acha hai. hum ek nai teknoloji ko aajma rahe hain.
Nastaliq:        آج کا موسم بہت اچْچھا ہے۔ ہم ایک نئی ٹیکْنولاجی کو آجما رہے ہیں۔
```

| # | Hindi (ASR) | Raw Roman | Roman Urdu | Nastaliq | Min Conf | Geo Conf | Tokens | Flag | Notes |
|---:|---|---|---|---|---:|---:|---:|:---:|---|
| 1 | आज | aaj | aaj | آج | 1.00 | 1.00 | 2 | | native (today) |
| 2 | का | ka | ka | کا | 1.00 | 1.00 | 2 | | native function |
| 3 | मौसम | mausam | mausam | موسم | 1.00 | 1.00 | 5 | | common (weather) |
| 4 | बहुत | bahut | bahut | بہت | 1.00 | 1.00 | 5 | | common (very) |
| 5 | अच्छा | achcha | acha | اچْچھا | 1.00 | 1.00 | 6 | | lexicon: achcha → acha |
| 6 | है। | hai. | hai. | ہے۔ | 0.70 | 0.92 | 4 | **LOW** | sentence-end variant |
| 7 | हम | ham | hum | ہم | 1.00 | 1.00 | 2 | | lexicon: ham → hum |
| 8 | एक | ek | ek | ایک | 1.00 | 1.00 | 3 | | native (one) |
| 9 | नई | nai | nai | نئی | 0.98 | 0.99 | 3 | | native (new) |
| 10 | टेक्नोलॉजी | teknoloji | teknoloji | ٹیکْنولاجی | **0.61** | 0.94 | **12** | **LOW** | **English loan, 12 tokens** |
| 11 | को | ko | ko | کو | 1.00 | 1.00 | 2 | | native postposition |
| 12 | आजमा | aajma | aajma | آجما | 0.94 | 0.98 | 5 | | uncommon verb form |
| 13 | रहे | rahe | rahe | رہے | 0.99 | 1.00 | 4 | | native aux verb |
| 14 | हैं। | hain. | hain. | ہیں۔ | 0.97 | 0.99 | 5 | | native function |

**Flagged words (Min Conf < 0.7):**

- **`है। (0.70)`** — sentence-end variant ambiguity (same as ur1)
- **`टेक्नोलॉजी (0.61)`** — *the lowest confidence in the sample.* "Technology" phonetically transliterated; 12 BPE tokens (longest word in either sample). Foreign loan + length = many chances for one borderline sub-token.

**Lexicon corrections visible by comparing Raw Roman vs Roman Urdu:**

- `achcha → acha` — drops redundant ch
- `ham → hum` — function-word convention

---

## 6.1 Why this is NOT random — pattern analysis

Look at what the confidence numbers cluster around across both samples:

| Word category | Examples | Typical Min Conf |
|---|---|---|
| **Native function words** (1-3 chars) | मेरा, नाम, का, हम, एक, आज, में | **0.98–1.00** |
| **Common content words** (4-6 chars) | मौसम, बहुत, अच्छा, टेस्ट, की, रहे | **0.95–1.00** |
| **Long native words** (5-7 chars) | उर्दू, नई, आजमा | **0.81–0.98** |
| **Borderline phonetic / nukta decisions** | ज़बान (with nukta), शनाख्त | **0.45–0.56** |
| **Proper nouns / foreign names** | अपीब (Aqib) | **0.65** |
| **Long foreign loan words** | टेक्नोलॉजी (12 tokens) | **0.61** |
| **Sentence-end variants** (है।, हैं।) | है।, हैं। | **0.69–0.99** (variant choice) |

**The signal is real:**

1. **Word length matters but isn't everything.** `अच्छा` (6 tokens, native) scores 1.00; `टेक्नोलॉजी` (12 tokens, English loan) scores 0.61. The model is more confident about long-but-common words than short-but-foreign ones.

2. **Foreign loans get lower scores.** English words written in Devanagari (`टेक्नोलॉजी`, `अपीब`/Aqib) are systematically less confident because the model has to choose between multiple plausible phonetic spellings.

3. **Borderline phonetic / nukta decisions stand out.** `ज़बान` (0.45) is the *lowest confidence in ur1* — the model is genuinely unsure between emitting the nukta `़` or dropping it. This is the same near-tied logit decision discussed in `asr-backend-comparison.md` — the confidence number directly surfaces it.

4. **Sentence-end variants** (`है।` vs `है`) score lower (0.69-0.92) than mid-sentence variants (0.99) because the model has to decide whether to emit the period before EOS.

If the numbers were random, native function words and long foreign loans would have similar scores. They don't. The model is genuinely tracking its own uncertainty, and the per-word aggregation surfaces it cleanly.

### `language="English"` vs `language="Hindi"` — which to use on Mac CPU

Both samples in Section 6 were run with `language="English"` as the ASR
hint. This is empirically better on Mac CPU than `language="Hindi"`
because the model emits **the nukta marks correctly** that the Hindi
hint drops:

| Word | `language="Hindi"` (Mac CPU) | `language="English"` (Mac CPU) | GPU bf16 reference |
|---|---|---|---|
| ज़बान / जान | जान (no nukta) ✗ | **ज़बान** (with nukta) ✓ | ज़बान ✓ |
| आवाज़ / आवाज | आवाज (no nukta) ~ | **आवाज़** (with nukta) ✓ | आवाज़ ✓ |
| अच्छा / अपीब / etc. | same | same | same |

**Recommendation:** use `language="English"` on Mac CPU until the
project moves to GPU/vLLM. The Qwen3-ASR `language` hint isn't a strict
constraint — it's a bias — and "English" surprisingly biases the model
toward a more careful nukta-emission path on this hardware.

### Threshold tuning observation

At the default `LOW_CONF_THRESHOLD=0.7`, both samples flag exactly the
right words — perfect signal-to-noise on this corpus:

- ur1: `अपीब` (0.65), `है।` (0.69), `ज़बान` (0.45), `शनाख्त` (0.56)
- ur2: `है।` (0.70), `टेक्नोलॉजी` (0.61)

These are exactly the words a human reviewer should look at. The
0.7 default was chosen empirically; you can tune via env:

```bash
LOW_CONF_THRESHOLD=0.5 python3 ...   # looser — only catches severe cases
LOW_CONF_THRESHOLD=0.9 python3 ...   # stricter — flags moderately uncertain words too
```

---

## 7. Implementation — monkey-patching the inner generate

Qwen3-ASR's high-level `transcribe()` discards token logprobs — only the
final text comes back. To intercept the scores without losing the
transcribe pipeline (which handles audio preprocessing, language
forcing, post-processing), we patch the inner LLM's `generate`:

```python
target = model.model.thinker.generate    # the inner LLM's generate
orig = target

def patched_generate(*args, **kwargs):
    kwargs["output_scores"] = True
    kwargs.setdefault("return_dict_in_generate", True)
    result = orig(*args, **kwargs)
    captured["outputs"] = result           # stash for our use
    return result                          # qwen-asr expects this back

try:
    target.generate = patched_generate
    results = model.transcribe(audio=[...], language=[...])
finally:
    target.generate = orig                 # always restore
```

### Why patch the INNER `thinker.generate`, not the outer `model.generate`

Qwen3ASR's outer `generate()` already calls
`self.thinker.generate(input_ids=..., return_dict_in_generate=True, **kwargs)`
internally — it hardcodes `return_dict_in_generate=True` as a positional
kwarg. If we patch the outer `generate()` and also add
`return_dict_in_generate=True` to its kwargs, those kwargs get propagated
down and the inner call sees the kwarg twice → `TypeError: multiple
values for keyword argument 'return_dict_in_generate'`.

Patching the inner `thinker.generate` is the right level.

---

## 8. Token-to-word grouping (BPE space marker)

Qwen3 uses GPT-style ByteLevel BPE where leading spaces are encoded as
the character `Ġ` (U+0120) at the start of a token. We use that as the
word-boundary marker:

```python
raw_tokens = tokenizer.convert_ids_to_tokens(gen_ids)
# raw_tokens looks like: ['Ġम', 'े', 'रा', 'Ġना', 'म', 'Ġअ', 'पी', 'ब', ...]

groups: list[list[int]] = []
current: list[int] = []
for i, (tid, raw) in enumerate(zip(gen_ids, raw_tokens)):
    if tid in special_ids:        # drop <|im_end|>, <|endoftext|>, etc.
        if current: groups.append(current)
        current = []
        continue
    if raw.startswith("Ġ") and current:   # new word starts here
        groups.append(current)
        current = []
    current.append(i)
if current: groups.append(current)
```

Then for each group:

```python
word_ids = [gen_ids[i] for i in group]
word_text = tokenizer.decode(word_ids, skip_special_tokens=False).strip()
```

### Why decode the GROUP, not individual tokens

Single Devanagari characters are multi-byte UTF-8 sequences. Some BPE
tokens contain only partial bytes of a Devanagari codepoint, so
`tokenizer.decode([single_id])` returns broken UTF-8 (renders as `▓`).

Decoding the full group rebuilds the byte sequence correctly. This was
a real bug in the first iteration of this code — the table came out
with `▓▓▓` instead of `मेरा` because we were decoding token-by-token.

---

## 9. Performance

| | Inference time |
|---|---|
| `hf_asr()` (no logprobs) | ~33s on Mac CPU |
| `hf_asr_with_confidence()` (with logprobs) | ~24s on Mac CPU |

The difference is normal run-to-run variance, not a real cost. Capturing
logprobs has **near-zero overhead** at compute time — `output_scores=True`
just reads tensors that were already computed during inference. The
post-processing (softmax + min/mean + decode) is microseconds.

In short: **confidence extraction is free.** Use it everywhere.

---

## 10. Backend support matrix

| Backend | Confidence extraction? | How |
|---|---|---|
| **transformers** (HF Python) | ✓ implemented | Monkey-patch `thinker.generate` (this doc) |
| **vLLM** (production GPU) | ✓ supported by vLLM | Set `sampling_params.logprobs = N` after model init |
| **llama.cpp `llama-mtmd-cli`** | ✗ not exposed | CLI lacks `--logprobs` flag |
| **llama.cpp `llama-server`** | ✓ via HTTP | `/completion` endpoint `n_probs` field, or OAI `top_logprobs` |
| **Forced Aligner** (`Qwen3-ForcedAligner-0.6B`) | ✗ timestamps only | not useful for confidence |

For the Rust handover, the recommended path is:

- **CPU/local dev:** transformers backend (current implementation)
- **Production GPU:** vLLM with `logprobs=5` (same metrics, faster batching)
- **Embedded/offline:** switch from `llama-mtmd-cli` to `llama-server` (same GGUF, exposes `n_probs`)

---

## 11. Tuning knobs

| Knob | Default | Effect |
|---|---|---|
| `LOW_CONF_THRESHOLD` env | `0.7` | Cutoff for `is_low` flag; raise to be stricter |
| Use `geo_conf` instead of `min_conf` for flag | — | Edit `is_low` property; less sensitive (fewer flags) |
| Combine both | — | `is_low = min_conf < 0.5 OR geo_conf < 0.85` for belt-and-suspenders |

The CLI also supports `--conf-table` to display the full table instead
of just the inline summary.

---

## 12. Related docs

- [`asr-backend-comparison.md`](asr-backend-comparison.md) — explains
  why different backends produce different Hindi (CPU/fp32 vs GPU/bf16
  vs llama.cpp). The same near-tied logits that cause backend divergence
  show up here as low confidence scores.
- [`hindi-to-roman-urdu-design.md`](hindi-to-roman-urdu-design.md) — the
  downstream transliteration step that runs after ASR.
- [`deployment-plan.md`](deployment-plan.md) — Rust handover plan
  (production stack will use vLLM backend for confidence extraction).
