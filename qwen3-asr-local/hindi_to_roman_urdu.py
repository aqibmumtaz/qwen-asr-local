#!/usr/bin/env python3
"""
Devanagari (Hindi) → Roman Urdu transliterator
Direct single-step conversion — no intermediate Nastaliq.

Three-layer pipeline:
  Layer 1 — phoneme map (consonants, vowels, nuqta)
  Layer 2 — schwa deletion (word-final, virama-explicit)
  Layer 3 — word-level corrections (known wrong → right)

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

# ── Layer 1: consonant tables ────────────────────────────────────────────────
CONSONANTS = {
    'क': 'k',   'ख': 'kh',  'ग': 'g',   'घ': 'gh',  'ङ': 'n',
    'च': 'ch',  'छ': 'chh', 'ज': 'j',   'झ': 'jh',  'ञ': 'n',
    'ट': 't',   'ठ': 'th',  'ड': 'd',   'ढ': 'dh',  'ण': 'n',
    'त': 't',   'थ': 'th',  'द': 'd',   'ध': 'dh',  'न': 'n',
    'प': 'p',   'फ': 'ph',  'ब': 'b',   'भ': 'bh',  'म': 'm',
    'य': 'y',   'र': 'r',   'ल': 'l',   'व': 'w',
    'श': 'sh',  'ष': 'sh',  'स': 's',   'ह': 'h',
    'ळ': 'l',
}

# base + nukta (U+093C) combinations
NUKTA_MAP = {
    'क़': 'q',   'ख़': 'kh',  'ग़': 'gh',  'ज़': 'z',
    'ड़': 'r',   'ढ़': 'rh',  'फ़': 'f',   'य़': 'y',
}

# pre-composed extended range U+0958–U+095F
EXTENDED = {
    'क़': 'q',   'ख़': 'kh',  'ग़': 'gh',  'ज़': 'z',
    'ड़': 'r',   'ढ़': 'rh',  'फ़': 'f',   'य़': 'y',
}

# ── Layer 1: vowel tables ────────────────────────────────────────────────────
INDEPENDENT_VOWELS = {
    'अ': 'a',   'आ': 'aa',  'इ': 'i',   'ई': 'ee',
    'उ': 'u',   'ऊ': 'oo',  'ए': 'e',   'ऐ': 'ai',
    'ओ': 'o',   'औ': 'au',  'ऋ': 'ri',  'ॠ': 'ri',
    'ऍ': 'e',   'ऑ': 'o',
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

# word-boundary characters — schwa deleted before these
_BOUNDARY = set(' \t\n.,!?;:।॥"\'()[]{}')

# ── Layer 3: word-level corrections ─────────────────────────────────────────
# keys are phonetic output from layers 1+2, values are natural Roman Urdu
CORRECTIONS = {
    # consonant cluster overcorrections
    'achhaa':      'acha',
    'achhchhaa':   'acha',
    'achha':       'acha',
    'achhcha':     'acha',
    'achchhaa':    'acha',
    'achchha':     'acha',

    # common grammar particles (long vowel → natural short form)
    'kaa':         'ka',
    'kee':         'ki',
    'koo':         'ko',
    'sae':         'se',
    'nae':         'ne',
    'par':         'par',
    'men':         'mein',
    'meen':        'mein',
    'yah':         'yeh',
    'wahaa':       'wahan',
    'yahaan':      'yahan',

    # pronouns
    'meraa':       'mera',
    'teraa':       'tera',
    'hamaaraa':    'hamara',
    'tumhaaraa':   'tumhara',
    'unakaa':      'unka',
    'isekaa':      'iska',
    'usekaa':      'uska',

    # common verbs / endings
    'kyaa':        'kya',
    'nahin':       'nahi',
    'nahiin':      'nahi',
    'kahnaa':      'kehna',
    'jaanaa':      'jana',
    'aanaa':       'aana',
    'jaataa':      'jaata',
    'hotaa':       'hota',
    'kartaa':      'karta',
    'rahaa':       'raha',
    'gayaa':       'gaya',
    'aayaa':       'aaya',
    'haen':        'hain',

    # common nouns / adjectives
    'bahuut':      'bahut',
    'aajaa':       'aaja',
    'kahaan':      'kahan',
    'waqat':       'waqt',
    'mausaam':     'mausam',
    'aawaz':       'awaz',
    'aawaaz':      'awaz',
    'zaabaan':     'zaban',
    'zabaan':      'zaban',
    'shinaakht':   'shanakht',
    'paanee':      'pani',
    'khaanaa':     'khana',
    'paanaa':      'pana',
    'raastaа':     'rasta',
    'duniiyaa':    'duniya',
    'duniyaa':     'duniya',
    'zindagee':    'zindagi',
    'zindagii':    'zindagi',
}


# ── Core algorithm ────────────────────────────────────────────────────────────

def _emit_vowel(chars: list, i: int, n: int, roman_c: str, result: list) -> int:
    """
    After consuming a consonant at position i-1, look ahead and emit
    roman_c + the appropriate vowel (or nothing for word-final schwa deletion).
    Returns updated index.
    """
    if i >= n:
        # word-final: delete inherent schwa
        result.append(roman_c)
        return i

    ch = chars[i]

    if ch == VIRAMA:
        result.append(roman_c)
        return i + 1

    if ch in MATRAS:
        vowel = MATRAS[ch]
        i += 1
        # anusvara/chandrabindu after matra → nasalise
        if i < n and chars[i] in (ANUSVARA, CHANDRABINDU):
            result.append(roman_c + vowel + 'n')
            return i + 1
        result.append(roman_c + vowel)
        return i

    if ch in (ANUSVARA, CHANDRABINDU):
        result.append(roman_c + 'an')
        return i + 1

    if ch in _BOUNDARY:
        # before a word boundary: delete inherent schwa
        result.append(roman_c)
        return i

    # medial position before another consonant or vowel: keep inherent 'a'
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

        # pre-composed extended nuqta consonants
        if ch in EXTENDED:
            roman_c = EXTENDED[ch]
            i += 1
            i = _emit_vowel(chars, i, n, roman_c, result)
            continue

        # regular consonants
        if ch in CONSONANTS:
            roman_c = CONSONANTS[ch]
            # check for following combining nukta
            if i + 1 < n and chars[i + 1] == NUKTA:
                key = ch + NUKTA
                roman_c = NUKTA_MAP.get(key, roman_c)
                i += 1  # consume nukta
            i += 1
            i = _emit_vowel(chars, i, n, roman_c, result)
            continue

        # independent vowels
        if ch in INDEPENDENT_VOWELS:
            result.append(INDEPENDENT_VOWELS[ch])
            i += 1
            if i < n and chars[i] in (ANUSVARA, CHANDRABINDU):
                result.append('n')
                i += 1
            continue

        # standalone anusvara / chandrabindu
        if ch in (ANUSVARA, CHANDRABINDU):
            result.append('n')
            i += 1
            continue

        # visarga — drop
        if ch == VISARGA:
            i += 1
            continue

        # Devanagari danda
        if ch in ('।', '॥'):
            result.append('.')
            i += 1
            continue

        # pass through (ASCII, spaces, digits, Urdu Nastaliq if mixed, etc.)
        result.append(ch)
        i += 1

    return ''.join(result)


def _apply_corrections(text: str) -> str:
    """Layer 3: replace known wrong phonetic words with natural Roman Urdu."""
    def fix_word(m):
        w = m.group(0)
        lower = w.lower()
        corrected = CORRECTIONS.get(lower)
        if not corrected:
            return w
        # preserve original capitalisation
        if w[0].isupper():
            return corrected.capitalize()
        return corrected

    return re.sub(r"[A-Za-z]+", fix_word, text)


def transliterate(text: str) -> str:
    """
    Convert Hindi Devanagari text to Roman Urdu.
    Non-Devanagari characters (ASCII, Urdu Nastaliq, digits) pass through unchanged.
    """
    raw = _transliterate_raw(text)
    return _apply_corrections(raw)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    test_sentences = [
        "मेरा नाम اکیب ہے۔",
        "आज का मौसम بہت اچھا ہے۔",
        "यह कोड की ज़बान में आवाज़ की शिनाख़्त का टेस्ट है।",
        "बहुत अच्छा है।",
        "नमस्ते، میں ٹھیک ہوں۔",
    ]

    if len(sys.argv) > 1:
        print(transliterate(' '.join(sys.argv[1:])))
    else:
        print("── Self-test ──────────────────────────────────")
        for s in test_sentences:
            print(f"IN : {s}")
            print(f"OUT: {transliterate(s)}")
            print()
