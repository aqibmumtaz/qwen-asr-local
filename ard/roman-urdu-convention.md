# Roman Urdu Convention — Hindi→Roman Urdu Transliterator

**Single chosen convention.** Pick one, stick with it. This is what
`hindi_to_roman_urdu.py` outputs. To override any choice, edit the
`CORRECTIONS` or `PROPER_NOUNS` dict — no code changes needed.

---

## Quick review table — every ambiguous case

Mark any pick you want flipped and the dict will be updated.

### Function words (used in every sentence)

| Hindi | My pick | Alternatives rejected | Why |
|---|---|---|---|
| क्या | `kya` | kia, kyaa | Most universal; matches WhatsApp norms |
| है | `hai` | he, hae | `he` clashes with English pronoun |
| हैं | `hain` | hai, hen | Distinguishes from singular `hai` |
| नहीं | `nahi` | nahin, nai | `nai` clashes with नई (= new) |
| हाँ | `haan` | haa, han | Preserves long ā |
| में | `mein` | main, mai, men | Distinguishes from pronoun `main` |
| मैं | `main` | mae, men | Standard Roman Urdu |
| हम | `hum` | ham | Urdu convention (ہم has dama vowel) |
| यह | `yeh` | ye, yah | Most common in writing |
| वह | `woh` | wo, wah, vo | Distinguishes from English `wo` |

### Postpositions (ka/ki/ke pattern)

| Hindi | My pick | Alternatives rejected |
|---|---|---|
| का | `ka` | kaa |
| की | `ki` | kee, kii |
| के | `ke` | kay, key |
| को | `ko` | kau, koh |
| से | `se` | sey |
| पर | `par` | par (same) |
| तक | `tak` | takk |

### Question words

| Hindi | My pick | Alternatives rejected |
|---|---|---|
| कब | `kab` | kub |
| कहाँ | `kahan` | kahaan, kahaa |
| कैसे | `kaise` | kaisay |
| क्यों | `kyon` | kyun, kyoon |
| क्योंकि | `kyonki` | kyunki, kyoonki |
| कौन | `kaun` | kon |
| कितना | `kitna` | kitnaa, kitana |

### Pronouns / possessives

| Hindi | My pick | Alternatives rejected |
|---|---|---|
| तू | `tu` | too |
| तुम | `tum` | toom |
| आप | `aap` | ap |
| मेरा | `mera` | meraa |
| मेरी | `meri` | meree |
| तेरा | `tera` | teraa |
| हमारा | `hamara` | hamaaraa |
| तुम्हारा | `tumhara` | tumhaaraa |
| अपना | `apna` | apnaa, apanaa |

### Numbers (debatable — flip if you prefer phonetic)

| Hindi | My pick | Alternatives rejected | Note |
|---|---|---|---|
| एक | `ek` | aek | |
| दो | `do` | dou | |
| तीन | `teen` | tin | Long ī kept internal |
| चार | `char` | chaar | **Shortened** — flip if you prefer `chaar` |
| पाँच | `panch` | paanch | **Shortened** — flip if you prefer `paanch` |
| छह | `chhe` | chha, chhah | |
| सात | `saat` | sat | |
| आठ | `aath` | ath | |
| नौ | `nau` | naw | |
| दस | `das` | dass | |
| सौ | `sau` | so | |
| हज़ार | `hazar` | hazaar | **Shortened** |
| लाख | `lakh` | laakh | |

### Conjunctions

| Hindi | My pick | Alternatives rejected |
|---|---|---|
| और | `aur` | or, our |
| लेकिन | `lekin` | laikin |
| मगर | `magar` | mager |
| अगर | `agar` | ager |
| तो | `to` | toh |
| इसलिए | `isliye` | isalie, islie |
| क्योंकि | `kyonki` | kyunki |

---

## 1. Long vowels — the core rules

| Pattern | Output | Examples |
|---|---|---|
| आ / ा word-final | `a` | मेरा → mera, का → ka |
| आ / ा internal, before final CV | `a` | खाना → khana, पानी → pani |
| आ / ा internal, before final C | `aa` | नाम → naam, काम → kaam |
| ई / ी word-final | `i` | नदी → nadi, ज़िंदगी → zindagi |
| ई / ी word-internal | `ee` | तीन → teen, पीछे → peechhe |
| ी + ं/ँ (nasalised) | `in` | नहीं → nahi, कहीं → kahin |
| ऊ / ू | `oo` | हूँ → hoon, शुक्रिया → shukriya |

**Reasoning:** Long ā shortens at word-end because every Roman Urdu
speaker writes `mera`, not `meraa`. CV-shortening matches conversational
style. But internal `aa` before a final consonant (naam, kaam) is kept
because shortening loses the long-vowel signal entirely.

---

## 2. Short vowels

| Devanagari | Output |
|---|---|
| अ (inherent) | `a` |
| इ / ि | `i` |
| उ / ु | `u` |
| ए / े | `e` |
| ओ / ो | `o` |

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
| **छ** | **`ch`** (NOT `chh` — Roman Urdu convention; chota not chhota) |
| झ | `jh` |
| ठ / थ | `th` |
| ढ / ध | `dh` |
| फ | `ph` (but Urdu loan words → `f` via corrections: sirf, farq) |
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

| Pattern | Output | Example |
|---|---|---|
| Most vowels + ं/ँ | vowel + `n` | हाँ → haan, हैं → hain |
| ī + ं/ँ | `in` (not `een`) | नहीं → nahi, कहीं → kahin |
| ū + ँ | `oon` | हूँ → hoon |
| Standalone ं/ँ | `n` | |
| ः (visarga) | dropped | |

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

Examples that fire:
- करना → **karna** (delete schwa after र — vowel `a` before from क)
- आदमी → **aadmi** (delete schwa after द — vowel `aa` before from आ)
- बोलना → **bolna**
- देखना → **dekhna**

Does NOT fire (the guard prevents over-deletion):
- नदी → **nadi** (न is first consonant — no prior vowel — keep schwa)
- बड़ा → **bara** (ब is first consonant)
- जगह → **jagah** (no matra after final consonant — keep schwa)
- मनाना → **manana** (first matra has another consonant after, not word-end)

---

## 8. Function-word overrides (corrections beat phonetics)

For common words where conventional Roman Urdu differs from strict
phonetics, the `CORRECTIONS` dict wins:

| Hindi | Phonetic output | Convention |
|---|---|---|
| हम | `ham` | **hum** |
| यह | `yah` | **yeh** |
| वह | `wah` | **woh** |
| नहीं | `naheen` | **nahi** |
| में | `men` | **mein** |
| सिर्फ | `sirph` | **sirf** |
| आदाब | `aadaab` | **adaab** |
| आवाज़ | `aawaaz` | **awaz** |
| ज़बान | `zabaan` | **zaban** |
| शिनाख़्त | `shinaakht` | **shanakht** |
| बच्चा | `bachcha` | **bacha** |

Full list (~330 entries) in [`CORRECTIONS`](../qwen3-asr-local/hindi_to_roman_urdu.py).

---

## 9. Proper nouns (names)

ASR transcribes names phonetically in Devanagari (Hindi has no `q`),
so names need explicit mapping. **`PROPER_NOUNS` dict wins over
`CORRECTIONS`** and preserves capitalisation.

### Male names
| Hindi variants (from ASR) | Roman Urdu |
|---|---|
| अकीब, अक़ीब, आक़िब, अकेब, अक़ीब | **Aqib** |
| अली, अलि | Ali |
| उमर, ओमर | Umar |
| उस्मान | Usman |
| मुहम्मद, मोहम्मद | **Muhammad** (canonical) |
| अहमद | Ahmad |
| हसन | Hassan |
| हुसैन | Hussain |
| इब्राहिम | Ibrahim |
| इस्माइल | Ismail |
| यूसुफ़ | Yusuf |
| यूसफ़ | Yousaf |
| तारिक़ | Tariq |
| इमरान | Imran |
| कामरान | Kamran |
| सलमान | Salman |
| आरिफ़ | Arif |
| आसिफ़ | Asif |
| काशिफ़ | Kashif |
| शाहिद | Shahid |
| राशिद | Rashid |
| ख़ालिद | Khalid |
| मजीद | Majeed |
| बिलाल | Bilal |
| अब्दुल्लाह | Abdullah |
| नदीम | Nadeem |
| वसीम | Waseem |
| फ़ैसल | Faisal |
| नईम | Naeem |

### Female names
| Hindi variants | Roman Urdu |
|---|---|
| आयेशा, आइशा | Ayesha |
| फ़ातिमा | Fatima |
| मरयम | Maryam |
| ख़दीजा | Khadija |
| ज़ैनब | Zainab |
| सादिया | Sadia |
| आम्ना | Amna |
| सफ़िया | Safia |
| राबिया | Rabia |

**Removed:** `Sara` — conflicts with adjective सारा ("all/whole"). If
the female name is needed, capitalise manually after transliteration.

### Place names
| Hindi | Roman |
|---|---|
| कराची, करांची | Karachi |
| लाहौर | Lahore |
| इस्लामाबाद | Islamabad |
| पाकिस्तान | Pakistan |
| हिंदुस्तान | Hindustan |
| दिल्ली | Delhi |
| मुंबई | Mumbai |

---

## 10. How to override any choice

To change a single word, edit the dict:

```python
# To make चार output 'chaar' instead of 'char':
# In hindi_to_roman_urdu.py CORRECTIONS dict:
'chaar': 'chaar',   # was 'chaar': 'char'
```

To remove a proper noun mapping:

```python
# Delete the relevant key from PROPER_NOUNS dict
```

Run `python3 hindi_to_roman_urdu.py` to verify the embedded self-test
still passes (62 cases).

---

## 11. Summary of style choices

- **Phonetic by default** — preserve long vowels (`aa`, `ee`, `oo`)
- **Shorten at word-end** — final ā → a, final ī → i
- **Override common words** — function words use conventional spellings
- **Proper nouns capitalised** — names look like names
- **No fake alternatives** — `Aqeeb` removed (not a standard name)

If a word doesn't appear in `CORRECTIONS` or `PROPER_NOUNS`, the
phonetic output is what you get. Find a wrong word in real usage →
add it to the dict → it stays fixed forever.

**Last reviewed:** 2026-05-12. Tests: 62/62 passing.
