# Roman Urdu Convention — Hindi→Roman Urdu Transliterator

**Single chosen convention.** Pick one, stick with it. This is what
`hindi_to_roman_urdu.py` outputs. If you want a different convention,
adjust the rules + corrections dict.

---

## 1. Long vowels (the main convention choices)

| Devanagari | Position | Output | Example |
|---|---|---|---|
| आ / ा | Word-final | `a` | मेरा → mera, का → ka |
| आ / ा | Internal, before final CV syllable | `a` | खाना → khana, पानी → pani |
| आ / ा | Internal, before final consonant | `aa` | नाम → naam, काम → kaam |
| ई / ी | Word-final | `i` | नदी → nadi, ज़िंदगी → zindagi |
| ई / ी | Internal | `ee` | तीन → teen, पीछे → peechhe |
| ऊ / ू | Word-final | `oo` (or `un` with anusvara) | हूँ → hoon |
| ऊ / ू | Internal | `oo` | शुक्रिया → shukriya |

**Why:** Long ā shortens at word-end because Roman Urdu speakers
universally write `mera` not `meraa`. The CV-shortening (khana not
khaana) matches conversational style. But internal `aa` before a
final consonant (naam, kaam) is kept because shortening would create
ambiguity with English words.

---

## 2. Short vowels

| Devanagari | Output |
|---|---|
| अ (inherent) | `a` |
| इ / ि | `i` |
| उ / ु | `u` |
| ए / े | `e` |
| ओ / ो | `o` |

---

## 3. Diphthongs

| Devanagari | Output |
|---|---|
| ऐ / ै | `ai` |
| औ / ौ | `au` |

---

## 4. Consonants

### Plain
| Devanagari | Output |
|---|---|
| क | `k` |
| ग | `g` |
| च | `ch` |
| ज | `j` |
| ट / त | `t` |
| ड / द | `d` |
| न / ण / ञ / ङ | `n` |
| प | `p` |
| ब | `b` |
| म | `m` |
| य | `y` |
| र | `r` |
| ल / ळ | `l` |
| व | `w` |
| श / ष | `sh` |
| स | `s` |
| ह | `h` |

### Aspirated (h-suffix)
| Devanagari | Output |
|---|---|
| ख | `kh` |
| घ | `gh` |
| **छ** | **`ch`** (NOT `chh` — Roman Urdu convention) |
| झ | `jh` |
| ठ / थ | `th` |
| ढ / ध | `dh` |
| फ | `ph` |
| भ | `bh` |

### Nuqta (Urdu-origin sounds)
| Devanagari | Output |
|---|---|
| क़ | `q` |
| ख़ | `kh` |
| ग़ | `gh` |
| ज़ | `z` |
| ड़ | `r` (retroflex flap) |
| ढ़ | `rh` |
| फ़ | `f` |

### Conjuncts (special-cased)
| Devanagari | Output |
|---|---|
| ज्ञ | `gy` (not `jn` — Hindi convention) |
| क्ष | `ksh` |
| त्र | `tr` |

---

## 5. Nasalisation

| Devanagari | Output | Note |
|---|---|---|
| ं (anusvara) | `n` | hain, mein, kahan |
| ँ (chandrabindu) | `n` | hoon, kahan |
| ः (visarga) | (dropped) | Rare in casual Urdu |

---

## 6. Punctuation

| Devanagari | Output |
|---|---|
| । (danda) | `.` |
| ॥ (double danda) | `.` |
| Devanagari digits ०–९ | ASCII `0–9` |

---

## 7. Schwa syncope (the hard rule)

Inherent `a` after a consonant is **deleted** when the next pattern is
`consonant + matra + word-end` AND a vowel has already been emitted
earlier in the word.

Examples:
- करना → **karna** (delete schwa after र — vowel `a` before from क)
- आदमी → **aadmi** (delete schwa after द — vowel `aa` before from आ)
- बोलना → **bolna**

Does NOT fire (the guard prevents it):
- नदी → **nadi** (न is first consonant, no prior vowel — keep schwa)
- बड़ा → **bara** (ब is first consonant)
- जगह → **jagah** (no matra after final consonant — keep schwa)
- मनाना → **manana** (first matra has another consonant after, not word-end)

---

## 8. Function words (corrections override phonetics)

For very common words, conventional Roman Urdu spelling beats strict
phonetics:

| Devanagari | Phonetic | Convention |
|---|---|---|
| हम | `ham` | **hum** |
| यह | `yah` | **yeh** |
| वह | `wah` | **woh** |
| नहीं | `nahin` | **nahi** |
| में | `men` | **mein** |
| सिर्फ | `sirph` | **sirf** |
| आदाब | `aadaab` | **adaab** |

Full list in `CORRECTIONS` dict.

---

## 9. Proper nouns (names)

ASR transcribes names phonetically in Devanagari (Hindi has no `q`),
so names need explicit mapping:

| ASR Hindi | Phonetic | Proper Roman |
|---|---|---|
| अकीब | akeeb | **Aqib** (or Aqib) |
| आक़िब | aaqib | **Aqib** |
| अली | ali | **Ali** |
| मोहम्मद | mohammad | **Mohammad** |

Defined in `PROPER_NOUNS` dict. Capitalisation preserved.

---

## 10. Style summary

- **Phonetic by default** — preserve long vowels (`aa`, `ee`, `oo`)
- **Shorten at word-end** — final ā → a, final ī → i
- **Override common words** — function words use conventional spellings
- **Proper nouns capitalised** — names look like names

If a word doesn't appear in `CORRECTIONS` or `PROPER_NOUNS`, the
phonetic output is what you get. Find a wrong word in real usage →
add it to the dict → it stays fixed forever.
