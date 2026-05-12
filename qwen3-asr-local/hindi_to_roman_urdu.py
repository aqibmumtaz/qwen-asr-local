#!/usr/bin/env python3
"""
Devanagari (Hindi) → Roman Urdu transliterator
Direct single-step conversion — no intermediate Nastaliq.

Three-layer pipeline:
  Layer 1 — phoneme map (consonants, vowels, nuqta)
  Layer 2 — schwa deletion (word-final, virama-explicit, before non-Devanagari)
  Layer 3 — word endings normalisation + corrections dict

Usage:
    from hindi_to_roman_urdu import transliterate
    print(transliterate("मेरा नाम اکیب ہے"))   # "mera naam Aqib hai"

    python3 hindi_to_roman_urdu.py "मेरा नाम اکیب ہے"
"""

import re
import unicodedata

# ── Unicode constants ────────────────────────────────────────────────────────
NUKTA        = '़'  # ़  combining nukta
VIRAMA       = '्'  # ्  suppresses inherent vowel
ANUSVARA     = 'ं'  # ं  nasalisation
CHANDRABINDU = 'ँ'  # ँ  nasalisation
VISARGA      = 'ः'  # ः  (dropped in casual Urdu)

# Devanagari Unicode block boundaries — used for non-Devanagari boundary check
_DEVA_START = 0x0900
_DEVA_END   = 0x097F

# ── Layer 1: consonant tables ────────────────────────────────────────────────
CONSONANTS = {
    'क': 'k',   'ख': 'kh',  'ग': 'g',   'घ': 'gh',  'ङ': 'n',
    'च': 'ch',  'छ': 'ch',  'ज': 'j',   'झ': 'jh',  'ञ': 'n',
    'ट': 't',   'ठ': 'th',  'ड': 'd',   'ढ': 'dh',  'ण': 'n',
    'त': 't',   'थ': 'th',  'द': 'd',   'ध': 'dh',  'न': 'n',
    'प': 'p',   'फ': 'ph',  'ब': 'b',   'भ': 'bh',  'म': 'm',
    'य': 'y',   'र': 'r',   'ल': 'l',   'व': 'w',
    'श': 'sh',  'ष': 'sh',  'स': 's',   'ह': 'h',
    'ळ': 'l',
}

# base + nukta (U+093C) combinations — keyed as two-char strings
_N = NUKTA
NUKTA_MAP = {
    'क' + _N: 'q',   # क़  ka + nukta = qa
    'ख' + _N: 'kh',  # ख़  kha + nukta = kha (ghain-adjacent)
    'ग' + _N: 'gh',  # ग़  ga + nukta = ghain
    'ज' + _N: 'z',   # ज़  ja + nukta = za
    'ड' + _N: 'r',   # ड़  da + nukta = retroflex flap
    'ढ' + _N: 'rh',  # ढ़  dha + nukta
    'फ' + _N: 'f',   # फ़  pha + nukta = fa
    'य' + _N: 'y',   # य़  ya + nukta
}

# pre-composed extended range U+0958–U+095F (single codepoints)
EXTENDED = {
    'क़': 'q',   # क़
    'ख़': 'kh',  # ख़
    'ग़': 'gh',  # ग़
    'ज़': 'z',   # ज़
    'ड़': 'r',   # ड़
    'ढ़': 'rh',  # ढ़
    'फ़': 'f',   # फ़
    'य़': 'y',   # य़
}

# ── Layer 1: vowel tables ────────────────────────────────────────────────────
INDEPENDENT_VOWELS = {
    'अ': 'a',   'आ': 'aa',  'इ': 'i',   'ई': 'ee',
    'उ': 'u',   'ऊ': 'oo',  'ए': 'e',   'ऐ': 'ai',
    'ओ': 'o',   'औ': 'au',  'ऋ': 'ri',  'ॠ': 'ri',
    'ऍ': 'e',   'ऑ': 'o',   'ऌ': 'li',
}

MATRAS = {
    'ा': 'aa',  # ा
    'ि': 'i',   # ि
    'ी': 'ee',  # ी
    'ु': 'u',   # ु
    'ू': 'oo',  # ू
    'ृ': 'ri',  # ृ
    'े': 'e',   # े
    'ै': 'ai',  # ै
    'ो': 'o',   # ो
    'ौ': 'au',  # ौ
    'ॅ': 'e',   # ॅ
    'ॉ': 'o',   # ॉ
}

# word-boundary punctuation — schwa deleted before these
_PUNCT_BOUNDARY = set(' \t\n.,!?;:।॥"\'()[]{}')


# ── Lexicon (CORRECTIONS + PROPER_NOUNS) — loaded from data/lexicon.json ─────
# Plain JSON data file, no Python. Edit directly to add/remove entries.
import json as _json
from pathlib import Path as _Path

_LEXICON_PATH = _Path(__file__).resolve().parent / 'data' / 'lexicon.json'
_lex = _json.loads(_LEXICON_PATH.read_text(encoding='utf-8'))
CORRECTIONS  = _lex['corrections']
PROPER_NOUNS = _lex['proper_nouns']



# ── Core algorithm ────────────────────────────────────────────────────────────

def _is_deva(ch: str) -> bool:
    return _DEVA_START <= ord(ch) <= _DEVA_END


def _emit_vowel(chars: list, i: int, n: int, roman_c: str, result: list) -> int:
    """
    After consuming a consonant, determine and emit roman_c + its vowel.
    Schwa (inherent 'a') is deleted when:
      - end of string
      - followed by virama (explicit suppression)
      - followed by punctuation / space
      - followed by a non-Devanagari character (digit, ASCII, Arabic script, etc.)
    Returns updated index.
    """
    if i >= n:
        result.append(roman_c)
        return i

    ch = chars[i]

    # explicit virama — no vowel
    if ch == VIRAMA:
        result.append(roman_c)
        return i + 1

    # dependent vowel matra
    if ch in MATRAS:
        vowel = MATRAS[ch]
        i += 1
        if i < n and chars[i] in (ANUSVARA, CHANDRABINDU):
            # ī + nasalisation → 'in' (नहीं→nahin, कहीं→kahin, यहीं→yahin)
            # other long vowels keep their length (हाँ→haan, हूँ→hoon, हैं→hain)
            if vowel == 'ee':
                result.append(roman_c + 'in')
            else:
                result.append(roman_c + vowel + 'n')
            return i + 1
        result.append(roman_c + vowel)
        return i

    # anusvara / chandrabindu directly after consonant → inherent 'a' + nasal
    if ch in (ANUSVARA, CHANDRABINDU):
        result.append(roman_c + 'an')
        return i + 1

    # punctuation or space — word boundary, delete schwa
    if ch in _PUNCT_BOUNDARY:
        result.append(roman_c)
        return i

    # non-Devanagari character (digit, ASCII, Urdu Nastaliq, etc.) — word boundary
    if not _is_deva(ch):
        result.append(roman_c)
        return i

    # ── Schwa syncope before a final consonant+matra cluster ─────────────
    # Pattern: C + C + matra + (boundary)  →  delete C's inherent schwa
    # Fires on verb infinitives and similar: करना→karna, आदमी→aadmi, बोलना→bolna
    # Guard: only fires when a vowel has already been emitted in this word.
    # Otherwise it would over-delete first-syllable schwas (नदी→ndi, बड़ा→bra).
    # Does NOT fire when:
    #   - we're at the first consonant of the word (no vowel before)
    #   - next consonant has no matra (जगह→jagah stays with schwas)
    #   - matra is not word-final (मनाना: first matra has another consonant after)
    if (ch in CONSONANTS or ch in EXTENDED) \
            and result and result[-1] and result[-1][-1] in 'aeiouy':
        j = i + 1
        if j < n and chars[j] == NUKTA:
            j += 1
        if j < n and chars[j] in MATRAS:
            k = j + 1
            if k >= n or chars[k] in _PUNCT_BOUNDARY or not _is_deva(chars[k]):
                result.append(roman_c)
                return i

    # medial position within Devanagari word — keep inherent 'a'
    result.append(roman_c + 'a')
    return i


def _transliterate_raw(text: str) -> str:
    """Layer 1 + 2: character-level phoneme mapping with schwa deletion."""
    text = unicodedata.normalize('NFC', text)
    chars = list(text)
    result = []
    i = 0
    n = len(chars)

    while i < n:
        ch = chars[i]

        # ── Devanagari digits → ASCII ────────────────────────────────────
        if '०' <= ch <= '९':
            result.append(str(ord(ch) - 0x0966))
            i += 1
            continue

        # ── Pre-composed extended nuqta consonants (U+0958–U+095F) ───────
        if ch in EXTENDED:
            roman_c = EXTENDED[ch]
            i += 1
            i = _emit_vowel(chars, i, n, roman_c, result)
            continue

        # ── Special conjunct: ज् + ञ → 'gy' ─────────────────────────────
        if (ch == 'ज' and i + 2 < n
                and chars[i + 1] == VIRAMA and chars[i + 2] == 'ञ'):
            i += 3
            i = _emit_vowel(chars, i, n, 'gy', result)
            continue

        # ── Regular consonants ────────────────────────────────────────────
        if ch in CONSONANTS:
            roman_c = CONSONANTS[ch]
            # combining nukta follows → override mapping
            if i + 1 < n and chars[i + 1] == NUKTA:
                key = ch + NUKTA
                roman_c = NUKTA_MAP.get(key, roman_c)
                i += 1  # consume nukta
            i += 1
            i = _emit_vowel(chars, i, n, roman_c, result)
            continue

        # ── Independent vowels ────────────────────────────────────────────
        if ch in INDEPENDENT_VOWELS:
            result.append(INDEPENDENT_VOWELS[ch])
            i += 1
            if i < n and chars[i] in (ANUSVARA, CHANDRABINDU):
                result.append('n')
                i += 1
            continue

        # ── Standalone anusvara / chandrabindu ───────────────────────────
        if ch in (ANUSVARA, CHANDRABINDU):
            result.append('n')
            i += 1
            continue

        # ── Visarga — drop ────────────────────────────────────────────────
        if ch == VISARGA:
            i += 1
            continue

        # ── Devanagari danda ─────────────────────────────────────────────
        if ch in ('।', '॥'):  # । ॥
            result.append('.')
            i += 1
            continue

        # ── Pass through (ASCII, spaces, Urdu Nastaliq, etc.) ────────────
        result.append(ch)
        i += 1

    return ''.join(result)


def _normalize_endings(text: str) -> str:
    """
    Normalise long vowels to natural Roman Urdu conventions.

    Rules applied in order:
      1. word-final 'ee' → 'i'   (paanee→paani, nadee→nadi)
      2. word-final 'aa' → 'a'   (khaanaa→khaana, meraa→mera)
      3. aa + consonant + vowel at word-end → a + consonant + vowel
         Runs AFTER rules 1+2 so it sees 'paani'/'khaana' not 'paanee'/'khaanaa'
         Handles: paani→pani, khaana→khana, bataana→batana
         Does NOT affect: naam ('m' not followed by vowel), pyaar (corrected separately)
    """
    _C = '[bcdfghjklmnpqrstvwxyz]'
    _V = '[aeiouy]'
    text = re.sub(r'ee\b', 'i', text)
    # word-final 'aa' → 'a' ONLY when preceded by a consonant
    # (otherwise standalone आ would wrongly shorten: "कब आ" → "kab a")
    text = re.sub(rf'(?<={_C})aa\b', 'a', text)
    # aa+CV at word-end → a+CV ONLY when preceded by a consonant
    # Prevents wrongly shortening word-initial 'aa' (आता→ata, आगे→age)
    text = re.sub(rf'(?<={_C})aa({_C}{_V})\b', r'a\1', text)
    # word-final 'aao' → 'ao' after consonant (imperatives: लगाओ→lagao, खाओ→khao)
    text = re.sub(rf'(?<={_C})aao\b', 'ao', text)
    return text


def _apply_corrections(text: str) -> str:
    """
    Layer 3: replace known wrong phonetic words with natural Roman Urdu.

    Two lookup tables (both held in CORRECTIONS unless capitalisation matters):
      - PROPER_NOUNS: names / places / acronyms — explicit caps from dict value
      - CORRECTIONS:  common words AND multi-word phrases
                      - keys without spaces → word-level lookup
                      - keys with spaces    → phrase-level regex replacement
                        (runs after word-level so phrases see corrected words)
    """
    # Single-word pass first (so multi-word patterns can match the corrected forms)
    def fix_word(m):
        w = m.group(0)
        lower = w.lower()
        if lower in PROPER_NOUNS:
            return PROPER_NOUNS[lower]
        corrected = CORRECTIONS.get(lower)
        if not corrected or ' ' in lower:  # skip phrase keys
            return w
        if w[0].isupper():
            return corrected[0].upper() + corrected[1:]
        return corrected

    text = re.sub(r'[A-Za-z0-9]+', fix_word, text)

    # Phrase pass: any CORRECTIONS key containing a space.
    # Longest keys first so 'hepatitis B' beats 'hepatitis' if both existed.
    phrase_keys = sorted(
        (k for k in CORRECTIONS if ' ' in k),
        key=lambda k: -len(k),
    )
    for phrase in phrase_keys:
        text = re.sub(
            rf'\b{re.escape(phrase)}\b',
            CORRECTIONS[phrase],
            text,
            flags=re.IGNORECASE,
        )
    return text


def transliterate(text: str) -> str:
    """
    Convert Hindi Devanagari text to Roman Urdu.
    Non-Devanagari characters (ASCII, Urdu Nastaliq, digits) pass through unchanged.
    """
    raw    = _transliterate_raw(text)
    normed = _normalize_endings(raw)
    return _apply_corrections(normed)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    tests = [
        # (input, expected)
        ("मेरा नाम اکیب ہے۔",            "mera naam اکیب ہے۔"),
        ("आज का मौसम بہت اچھا ہے۔",      "aaj ka mausam بہت اچھا ہے۔"),
        ("बहुत अच्छा है।",                "bahut acha hai."),
        ("यह कोड की ज़बान में आवाज़ की शिनाख़्त का टेस्ट है।",
                                           "yeh kod ki zaban mein awaz ki shanakht ka test hai."),
        # vowel ending normalisation
        ("नदी",      "nadi"),
        ("ज़िंदगी",  "zindagi"),
        ("बड़ा",     "bara"),
        ("छोटा",     "chota"),
        ("पानी",     "pani"),
        ("खाना",     "khana"),
        ("बताना",    "batana"),
        ("लड़की",    "larki"),
        ("प्यार",    "pyar"),
        ("आसमान",    "aasman"),
        # conjuncts / clusters
        ("ज्ञान",    "gyan"),
        ("धर्म",     "dharm"),
        ("शुक्रिया", "shukriya"),
        # function words
        ("हम",       "hum"),
        ("आवाज",     "awaz"),
        ("आवाज़",    "awaz"),
        ("ज़बान",    "zaban"),
        # stable words that must NOT change
        ("नाम",      "naam"),
        ("हूँ",      "hoon"),
        ("हैं",      "hain"),
        ("रात",      "raat"),
        ("काम",      "kaam"),
        # digits + mixed
        ("नाम123",   "naam123"),
        # schwa syncope (verb infinitives + CCV-matra word-end)
        ("करना",     "karna"),
        ("देखना",    "dekhna"),
        ("सुनना",    "sunna"),
        ("बोलना",    "bolna"),
        ("लिखना",    "likhna"),
        ("समझना",    "samajhna"),
        ("आदमी",     "aadmi"),
        # first-syllable schwa MUST NOT delete (regression guard)
        ("नदी",      "nadi"),
        ("बड़ा",     "bara"),
        ("गली",      "gali"),
        # consonant cluster cleanup
        ("बच्चा",    "bacha"),
        # corrections
        ("इसलिए",    "isliye"),
        ("सिर्फ",    "sirf"),
        ("आदाब",     "adaab"),
        ("क्या तुम जाओगे?", "kya tum jaoge?"),
        # CC word-final without matra (must keep schwa)
        ("जगह",      "jagah"),
        # ── Proper nouns (PROPER_NOUNS dict) ─────────────────────────────
        ("अकीब",     "Aqib"),
        ("अली",      "Ali"),
        ("मोहम्मद",  "Muhammad"),
        ("अहमद",     "Ahmad"),
        ("करांची",   "Karachi"),
        ("मेरा नाम अकीब है।", "mera naam Aqib hai."),
        # ── Comprehensive corrections (CORRECTIONS dict) ─────────────────
        ("चार",      "char"),
        ("पाँच",     "panch"),
        ("कितना",    "kitna"),
        ("कितनी",    "kitni"),
        ("बेटा",     "beta"),
        ("बेटी",     "beti"),
        ("भाई",      "bhai"),
        ("बहन",      "behan"),
        ("नया",      "naya"),
        ("ठंडा",     "thanda"),
        ("गरम",      "garam"),
        # ── Vowel hiatus (aa+e, ee+o) ────────────────────────────────────
        ("क्या तुम पानी पीओगे?", "kya tum pani peoge?"),
        ("हम कल जाएंगे।",        "hum kal jayenge."),
    ]

    if len(sys.argv) > 1:
        print(transliterate(' '.join(sys.argv[1:])))
    else:
        print("── Self-test ──────────────────────────────────")
        passed = 0
        for inp, expected in tests:
            result = transliterate(inp)
            ok = result == expected
            if ok:
                passed += 1
            mark = '✓' if ok else '✗'
            print(f"{mark}  {inp!r:35} → {result!r:30}  (expected {expected!r})")
        print(f"\n{passed}/{len(tests)} passed")
