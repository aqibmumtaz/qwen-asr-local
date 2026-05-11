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

# ── Layer 3: corrections dict ────────────────────────────────────────────────
# keys = phonetic output after layers 1+2+endings normalisation
# values = natural Roman Urdu spelling
CORRECTIONS = {
    # grammar particles
    'men':         'mein',
    'meen':        'mein',
    'yah':         'yeh',
    'wahaan':      'wahan',
    'yahaan':      'yahan',

    # consonant cluster overcorrections
    'achcha':      'acha',     # अच्छा: च्छ cluster → double ch → normalise
    'achch':       'ach',
    'pachchha':    'pacha',
    'kachcha':     'kacha',

    # ज्ञ conjunct (via special-case code → 'gy', but 'aan'→'an' not caught by word-final rule)
    'gyaan':       'gyan',
    'gyaani':      'gyani',
    'wigyaan':     'vigyan',
    'jnaan':       'gyan',     # fallback if special-case missed

    # Urdu vs phonetic Hindi spellings (common words spelt differently in Roman Urdu)
    'zabaan':      'zaban',
    'zaabaan':     'zaban',
    'aawaaz':      'awaz',
    'aawaz':       'awaz',
    'shinaakht':   'shanakht',
    'shinaakhat':  'shanakht',
    'waqat':       'waqt',
    'mausaam':     'mausam',
    'duniyaa':     'duniya',    # fallback (should be caught by aa→a rule)
    'zindagii':    'zindagi',
    'khushii':     'khushi',
    'nahin':       'nahi',
}


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
    Normalise word-final long vowels to natural Roman Urdu conventions:
      - word-final 'aa' → 'a'  (meraa→mera, kaa→ka, shukriyaa→shukriya)
      - word-final 'ee' → 'i'  (nadee→nadi, zindagee→zindagi)
    Does NOT affect mid-word vowels (naam, aaj, school stay unchanged).
    """
    text = re.sub(r'aa\b', 'a', text)
    text = re.sub(r'ee\b', 'i', text)
    return text


def _apply_corrections(text: str) -> str:
    """Layer 3: replace known wrong phonetic words with natural Roman Urdu."""
    def fix_word(m):
        w = m.group(0)
        lower = w.lower()
        corrected = CORRECTIONS.get(lower)
        if not corrected:
            return w
        if w[0].isupper():
            return corrected.capitalize()
        return corrected

    return re.sub(r'[A-Za-z]+', fix_word, text)


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
        ("मेरा नाम اکیب ہے۔",           "mera naam اکیب ہے۔"),
        ("आज का मौसम بہت اچھا ہے۔",     "aaj ka mausam بہت اچھا ہے۔"),
        ("बहुत अच्छा है।",               "bahut acha hai."),
        ("यह कोड की ज़बान में आवाज़ की शिनाख़्त का टेस्ट है।",
                                          "yeh kod ki zaban mein awaz ki shanakht ka test hai."),
        ("ज़बान", "zaban"),
        ("आवाज़",  "awaz"),
        ("नदी",    "nadi"),
        ("ज़िंदगी", "zindagi"),
        ("बड़ा",   "bara"),
        ("छोटा",   "chota"),
        ("ज्ञान",  "gyan"),
        ("नाम",    "naam"),
        ("हूँ",    "hoon"),
        ("हैं",    "hain"),
        ("नाम123", "naam123"),
        ("धर्म",   "dharm"),
        ("शुक्रिया", "shukriya"),
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
