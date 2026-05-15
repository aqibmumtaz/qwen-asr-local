# Hindi → Roman Urdu Transliteration — Design & Implementation

**Module:** `qwen3-asr-local/hindi_to_roman_urdu.py`
**Lexicon:** `qwen3-asr-local/data/lexicons.json`
**Convention spec:** `ard/roman-urdu-convention.md`
**Status:** production, 62/62 self-tests passing, 100% on Chughtai Lab corpus

---

## 1. Purpose

Convert Hindi text in Devanagari script (the output of Qwen3-ASR for Urdu
audio) into Roman Urdu — the WhatsApp / SMS-style Latin spelling that
Pakistani and Indian users actually type.

```
मेरा नाम अकीब है। ये टेस्ट का सैंपल है।
                       ↓
mera naam Aqib hai. yeh test ka sample hai.
```

Hindi and Urdu are phonetically identical — same vocabulary, same grammar,
different script. So this is **script conversion**, not translation. The
algorithm is fully rule-based and deterministic: same input → same output,
every time, no ML, no randomness.

---

## 2. Architecture — 3-layer pipeline

```
Hindi Devanagari input
     │
     ▼
┌─────────────────────────────────────────────┐
│ Layer 1 — Phoneme mapping                   │
│   character-by-character Unicode walk       │
│   + schwa syncope state machine             │
│   + nuqta + virama + matra handling         │
│   + special conjuncts (ज्ञ → gy)           │
└────────────┬────────────────────────────────┘
             │
             ▼ raw Roman output
┌─────────────────────────────────────────────┐
│ Layer 2 — Vowel-ending normalisation        │
│   4 regex rules on Roman output:            │
│     ee\b           → i                       │
│     (C)aa\b        → a                       │
│     (C)aa+CV\b     → a+CV                    │
│     (C)aao\b       → ao                      │
└────────────┬────────────────────────────────┘
             │
             ▼ normalised Roman
┌─────────────────────────────────────────────┐
│ Layer 3 — Lexicon corrections               │
│   PROPER_NOUNS lookup (names, acronyms)     │
│   CORRECTIONS  lookup (word + phrase)       │
└────────────┬────────────────────────────────┘
             │
             ▼
       Final Roman Urdu
```

Each layer handles a different kind of work:

- **Layer 1** does the heavy lifting — Unicode-aware character iteration.
- **Layer 2** is universal post-processing (long vowels in Roman Urdu shorten
  predictably at word ends).
- **Layer 3** is the data-driven part — every wrong output found in production
  becomes one JSON entry and a permanent fix.

---

## 3. Layer 1 — Phoneme mapping in detail

### 3.1 Consonants (35 entries)

Devanagari consonants map to Roman Urdu in a single table:

```
क→k    ख→kh   ग→g    घ→gh   ङ→n
च→ch   छ→ch   ज→j    झ→jh   ञ→n
ट→t    ठ→th   ड→d    ढ→dh   ण→n
त→t    थ→th   द→d    ध→dh   न→n
प→p    फ→ph   ब→b    भ→bh   म→m
य→y    र→r    ल→l    व→w
श→sh   ष→sh   स→s    ह→h    ळ→l
```

Notable choices:
- **छ → `ch`** (not `chh`) — Roman Urdu convention drops aspirate distinction
- **व → `w`** (not `v`) — Urdu speakers default to `w` in Roman Urdu

### 3.2 Nuqta consonants (Arabic-origin sounds, 8 entries)

The nuqta `़` (U+093C) modifies a base consonant to produce sounds from
Arabic / Persian loan words:

```
क़→q    ख़→kh   ग़→gh   ज़→z
ड़→r    ढ़→rh   फ़→f    य़→y
```

These are crucial for Urdu vocabulary — words like ज़बान (`zaban` =
language), आवाज़ (`awaz` = voice), क़िस्मत (`qismat` = fate) only work
correctly with the nuqta.

### 3.3 Vowels — independent (15 entries) and matras (12 entries)

Independent vowels appear word-initial or standalone:

```
अ→a    आ→aa   इ→i    ई→ee
उ→u    ऊ→oo   ए→e    ऐ→ai
ओ→o    औ→au   ऋ→ri
```

Matras attach to a consonant:

```
ा→aa   ि→i    ी→ee   ु→u    ू→oo
े→e    ै→ai   ो→o    ौ→au
```

### 3.4 Special characters

```
् (virama) — suppresses inherent vowel after consonant
ं (anusvara) — nasalisation, emits 'n'
ँ (chandrabindu) — same as anusvara for our purposes
ः (visarga) — dropped in casual Urdu
। ॥ (danda) — sentence/paragraph end → '.'
```

### 3.5 Schwa syncope — the tricky rule

In Hindi/Urdu, every consonant carries an inherent `a` vowel unless
something suppresses it. The classic case:

```
करना  (verb infinitive: to do)
  k+a + r+a + n+a → karana   ← naive output
  but the schwa after र should drop
  → karna                    ← what we want
```

The rule fires when the next pattern is:
**consonant + consonant + matra + word-end**
AND a vowel has been emitted earlier in the word.

```
karna ✓    ← fires on र (next: न+ा+end), vowel 'a' emitted before
aadmi ✓    ← fires on द (next: म+ी+end), vowel 'aa' emitted before
bolna ✓    ← fires on ल (next: न+ा+end)
```

The guard prevents over-deletion in 2-syllable words:
```
nadi    ← would WRONGLY shorten to 'ndi' without guard; न is first consonant
bara    ← would WRONGLY shorten to 'bra'; ब is first consonant
jagah   ← stays unchanged; ह has no matra, rule doesn't fire
manana  ← stays unchanged; first matra has another consonant after
```

### 3.6 Special conjuncts

```
ज्ञ → gy    (not jn — Hindi convention)
क्ष → ksh
त्र → tr
```

---

## 4. Layer 2 — Vowel-ending normalisation

Four regex rules run on the raw Roman output. They handle the most common
"Roman Urdu prefers shorter forms" patterns universally, without per-word
entries.

| Rule | Pattern | Examples |
|---|---|---|
| **1.** ee\b → i | word-final long ī | `paanee → paani`, `nadee → nadi` |
| **2.** (C)aa\b → a | word-final long ā preceded by consonant | `meraa → mera`, `kaa → ka` |
| **3.** (C)aa+CV\b → a+CV | aa + consonant + vowel at word-end | `khaana → khana`, `paani → pani` |
| **4.** (C)aao\b → ao | aa + o at word-end (imperatives) | `lagaao → lagao`, `khaao → khao` |

Important: rules 2-4 only fire when **preceded by a consonant**. This keeps
standalone vowels intact:

- `कब आ सकते हैं` (`kab aa sakte hain`) — standalone `आ` keeps as `aa`
- `मेरा` (raw `meraa`) — `r` before `aa` → shortens to `mera`

### 4.1 ī + nasalisation special case

When ī matra precedes anusvara (`ी` + `ं`/`ँ`), we emit `in` not `een`:

```
नहीं → nahi   (after correction; raw: nahin)
कहीं → kahin
यहीं → yahin
```

Other long vowels keep their length:
- ā + ं → `aan` (हाँ → haan)
- ū + ँ → `oon` (हूँ → hoon)
- ai + ं → `ain` (हैं → hain)

---

## 5. Layer 3 — Lexicon (`data/lexicons.json`)

Pure data file, no code. Two maintenance dictionaries:

```json
{
  "lexicons": {
    "corrections": {
      "apointament": "appointment",
      "chaahie":     "chahiye",
      "meraa":       "mera",
      "achchaa":     "acha",
      "hepatitis si":"hepatitis C",
      ...1166 entries...
    },
    "proper_nouns": {
      "akeeb":       "Aqib",
      "muhammad":    "Muhammad",
      "karaachee":   "Karachi",
      "seebeesi":    "CBC",
      ...265 entries...
    }
  }
}
```

### 5.1 Two dict types, same purpose

| Dict | Holds | Capitalisation |
|---|---|---|
| `corrections` | Common words + multi-word phrases | Preserves caller's case |
| `proper_nouns` | Names, places, acronyms, brands | Forces caps from dict value |

The algorithm checks `proper_nouns` first (for capitalisation), then
`corrections`. Each dict has both single-word keys and multi-word keys —
keys with spaces get a phrase-level regex pass after the word pass.

### 5.2 Why keys are *raw Roman output*, not Hindi input

This was a deliberate design choice that simplifies the lexicon dramatically.

```
ASR variant 1: अकीब  → akeeb  ───┐
ASR variant 2: अक़ीब → aqeeb  ───┼─→ all converge on similar raw Roman
ASR variant 3: आक़िब → aqib   ───┘    after Layer 1+2

We add ONE entry per spelling variant in the OUTPUT:
  "akeeb": "Aqib"
  "aqeeb": "Aqib"
  "aqib":  "Aqib"
```

If we keyed by Hindi input, we'd need to anticipate every ASR mishearing
in Devanagari — much harder. Keying by raw Roman output is stable and
lets one entry cover many ASR variants.

### 5.3 What does NOT belong in the lexicon

The lexicon's purpose is mapping **spelling variants of the same word**.
ASR mishears that produce different words must NOT be patched here:

| ✗ Bad lexicon entry | Reason |
|---|---|
| `"jaan": "zaban"` | जान (life) and ज़बान (language) are different words |
| `"shanaat": "shanakht"` | `shanaat` isn't a word — it's an ASR hallucination |
| `"dais": "test"` | `dais` is a real English word; force-mapping breaks legitimate uses |

ASR errors get fixed **at the ASR layer** (better model, better audio,
better decoding) — never via lexicon hacks.

---

## 6. Worked example — `sample_ur1.wav`

Real audio sample. Recorded Urdu sentence; Qwen3-ASR transcribed it into
Hindi Devanagari.

```
Audio: qwen3-asr-local/samples/sample_ur1.wav (~12s)
Spoken Urdu:    میرا نام عاقب ہے۔ یہ کوڈ کی زبان میں آواز کی شناخت کا ٹیسٹ ہے۔
                (My name is Aqib. This is a test of voice recognition in code language.)
ASR (Hindi):    मेरा नाम अकीब है। ये कोर्डुज़बान में आवाज की शनात का दैस है।
```

The transliterator's job: convert this Hindi to natural Roman Urdu.

### Step 1 — `_transliterate_raw()` walks each character

Each Devanagari character is converted using the phoneme tables + schwa
syncope rule.

| Hindi word | Character breakdown | Raw Roman |
|---|---|---|
| मेरा | म+े + र+ा | `meraa` |
| नाम | न+ा + म (word-end, schwa dropped) | `naam` |
| अकीब | अ + क+ी + ब (word-end) | `akeeb` |
| है | ह+ै | `hai` |
| ।  | Devanagari danda | `.` |
| ये | य+े | `ye` |
| कोर्डुज़बान | क+ो + र+्(virama) + ड+ु + ज+़(nukta=z) + ब+ा + न | `korduzabaan` |
| में | म+े + ं(anusvara=n) | `men` |
| आवाज | आ + व+ा + ज (word-end, no nukta) | `aawaaj` |
| की | क+ी (word-end) | `kee` |
| शनात | श + न+ा + त (word-end) | `shanaat` |
| का | क+ा | `kaa` |
| दैस | द+ै + स (word-end) | `dais` |
| है | ह+ै | `hai` |
| ।  | | `.` |

```
Raw output:
'meraa naam akeeb hai. ye korduzabaan men aawaaj kee shanaat kaa dais hai.'
```

### Step 2 — `_normalize_endings()` applies 4 regex rules

```
'meraa'    → 'mera'   (rule 2: C+aa\b → a; 'r' before aa, word-final)
'kaa'      → 'ka'     (rule 2: 'k' before aa, word-final)
'kee'      → 'ki'     (rule 1: ee\b → i)
'aawaaj'   → no change (ends in 'j', not 'aa' or 'ee')
'shanaat'  → no change (ends in 't')
'akeeb'    → no change (internal ee, not word-final)

Normed:
'mera naam akeeb hai. ye korduzabaan men aawaaj ki shanaat ka dais hai.'
```

### Step 3 — `_apply_corrections()` lookup

**Word-level pass** — each `[A-Za-z0-9]+` token is checked against
`PROPER_NOUNS` then `CORRECTIONS`:

| Word | Source | Mapped to | Notes |
|---|---|---|---|
| `mera` | (already correct) | `mera` | not in dict, kept as-is |
| `naam` | (correct) | `naam` | not in dict, kept |
| `akeeb` | `PROPER_NOUNS['akeeb']` | **`Aqib`** | proper noun, capitalised |
| `hai` | (correct) | `hai` | kept |
| `ye` | `CORRECTIONS['ye']` | **`yeh`** | conventional Roman Urdu |
| `korduzabaan` | not in dict | `korduzabaan` | ASR garbled input; we leave it |
| `men` | `CORRECTIONS['men']` | **`mein`** | function-word correction |
| `aawaaj` | `CORRECTIONS['aawaaj']` | **`awaz`** | ASR-produced variant of آواز |
| `ki` | (correct) | `ki` | kept |
| `shanaat` | not in dict | `shanaat` | ASR mishear; not patched (different word) |
| `ka` | (correct) | `ka` | kept |
| `dais` | not in dict | `dais` | ASR mishear; not patched (real English word) |

**Phrase-level pass** — no multi-word matches in this output.

```
Final Roman Urdu:
'mera naam Aqib hai. yeh korduzabaan mein awaz ki shanaat ka dais hai.'
```

### What worked, what didn't

✓ **Transliteration was correct for every word the ASR got right.**
- `meraa → mera`, `kee → ki`, `kaa → ka` (Layer 2 rules)
- `akeeb → Aqib` (proper noun lookup)
- `ye → yeh`, `men → mein`, `aawaaj → awaz` (Layer 3 lexicon)

✗ **ASR-level errors propagate unchanged through transliterator.**
- `korduzabaan` — Qwen3-ASR mashed three words ("code ki zaban")
- `shanaat` — Qwen3-ASR dropped the `kh` syllable from `shanakht`
- `dais` — Qwen3-ASR misheard `test`

These are NOT transliteration bugs — the transliterator faithfully converts
whatever Hindi it receives. Fixing them requires improving the ASR layer
(better model, GPU bf16 numerics, or Urdu-tuned Whisper), not patching
the lexicon with cross-word mappings.

### Saved output

The full pipeline writes to `transcriptions/sample_ur1_post_processor.txt`:
```
Hindi (ASR)          : मेरा नाम अकीब है। ये कोर्डुज़बान में आवाज की शनात का दैस है।
Urdu Nastaliq        : میرا نام اکیب ہے۔ یے کورْڈج़بان میں آواج کی شنات کا دیس ہے۔
Roman Urdu           : mera naam Aqib hai. yeh korduzabaan mein awaz ki shanaat ka dais hai.
```

---

## 7. Convention summary

For ambiguous cases (Roman Urdu has no single standard), we picked these:

| Hindi | Our pick | Alternatives rejected |
|---|---|---|
| क्या | `kya` | kia, kyaa |
| है | `hai` | he, hae |
| नहीं | `nahi` | nahin, nai |
| में | `mein` | men, main, mai |
| हम | `hum` | ham |
| यह | `yeh` | ye, yah |
| वह | `woh` | wo, wah, vo |
| का | `ka` | kaa |
| की | `ki` | kee, kii |

Full convention spec in `ard/roman-urdu-convention.md`.

---

## 8. Accuracy

Tested on a 129-item realistic Chughtai Lab call-centre corpus:

| Category | Accuracy |
|---|---|
| Function words | 100% |
| Nouns | 100% |
| Lab/medical terms | 100% |
| Realistic sentences | 100% |
| Verbs | 100% |
| Numbers | 100% |
| Proper nouns | 100% |
| **OVERALL** | **100%** |

Plus 62/62 embedded self-tests covering edge cases (schwa syncope guards,
nukta handling, conjuncts, vowel hiatus, etc.).

---

## 9. Maintenance — how to add a new correction

When you find a wrong word in real-world audio output:

### Step 1 — Identify the wrong output
```
Audio:        "मेरी रिपोर्ट कब आएगा"
Pipeline out: "meri report kab aaega"
Wrong word:   "aaega"  (should be "aayega")
```

### Step 2 — Find the raw Roman key
```python
from hindi_to_roman_urdu import _transliterate_raw, _normalize_endings
raw = _normalize_endings(_transliterate_raw("आएगा"))
print(raw)   # → 'aaega'
```

### Step 3 — Edit `data/lexicons.json`
```json
{
  "lexicons": {
    "corrections": {
      ...
      "aaega": "aayega",      ← add this line
      ...
    }
  }
}
```

### Step 4 — Verify
```bash
python3 hindi_to_roman_urdu.py   # should still pass all 62 self-tests
```

Done — that word is permanently fixed for every future audio.

---

## 10. Determinism guarantee

Same input → same output, every time:

- Zero randomness in any layer
- No ML / no probabilistic decoding
- Pure function over Unicode codepoints + dict lookups
- Reproducible across runs, machines, hardware

Safe to embed in a deterministic Rust pipeline later — the entire
`lexicons.json` can be embedded as a `phf::Map` literal via `include_str!()`.

---

## 11. Files

| File | Lines | Role |
|---|---|---|
| `hindi_to_roman_urdu.py` | 434 | Algorithm — Layer 1+2 + lexicon loader |
| `hindi_to_roman_urdu.sh` | 33 | Shell CLI wrapper for the engine |
| `data/lexicons.json` | ~35 KB | Layer 3 data: 1166 corrections + 265 proper nouns |
| `ard/roman-urdu-convention.md` | — | Full convention spec |
| `ard/transliterator-findings.md` | — | Findings & accuracy report |
| `ard/asr-backend-comparison.md` | — | Why same model gives different Hindi on different backends |
| `ard/deployment-plan.md` | — | Rust handover plan with deliverables |

---

## 12. Status

Production-ready as of 2026-05-13. Maintained by adding entries to
`data/lexicons.json` as new vocabulary appears in real-world audio.
No algorithm changes anticipated.
