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


# ── Lexicon ──────────────────────────────────────────────────────────────────
# Default: data/lexicons_v2.json — {canonical: [variants]}, one entry per real
# word. Inverted here to a flat {variant: canonical} map, so runtime lookup is
# O(1) exactly as before.
#
# v2 replaces the old lexicons.json + lexicons_updated.json. See clean_lexicon.py.
# Set  LEXICON=old  to fall back to data/lexicons.json (for A/B comparison).
import json as _json
import os as _os
from pathlib import Path as _Path

_DATA = _Path(__file__).resolve().parent / 'data'
LEXICON_VERSION = _os.getenv('LEXICON', 'v2').lower()


def _load_lexicon(version: str = 'v2'):
    """Returns (word_map, phrase_map) as flat {variant_lower: canonical}."""
    if version == 'old':
        raw = _json.loads((_DATA / 'lexicons.json').read_text(encoding='utf-8'))['lexicons']
        words = {k.lower(): v for k, v in raw['proper_nouns'].items()}
        # PROPER_NOUNS won on collisions in the old code, so load it first
        for k, v in raw['corrections'].items():
            words.setdefault(k.lower(), v)
        phrases = {k.lower(): v for k, v in raw['corrections'].items() if ' ' in k}
        return words, phrases

    # any other version loads data/lexicons_<version>.json in the v2 layout
    raw = _json.loads(
        (_DATA / f'lexicons_{version}.json').read_text(encoding='utf-8')
    )['lexicons']
    words, phrases = {}, {}
    for canon, variants in raw['lexicon'].items():
        for v in variants:
            words[v.lower()] = canon
    for canon, variants in raw['phrases'].items():
        for v in variants:
            phrases[v.lower()] = canon
    return words, phrases


WORD_MAP, PHRASE_MAP = _load_lexicon(LEXICON_VERSION)

# Back-compat aliases — some callers still import these names.
CORRECTIONS  = WORD_MAP
PROPER_NOUNS = {}

# ── Optional Layer 4a: the RESOLVER (see resolver.py) — DEPRECATED ────────────
# Rule-based edit-distance fallback. Superseded by the phonetic model (below):
# on the 80-call set the resolver nets ~0 (helped 5 / hurt 4) because it corrupts
# real words (taareekh->Tariq). Kept only for A/B. Enable with RESOLVER=1.
_RESOLVER = None
if _os.getenv('RESOLVER', '').lower() in ('1', 'true', 'yes', 'on'):
    try:
        from resolver import Resolver as _R
        _RESOLVER = _R()
    except Exception as _e:          # never break the pipeline over an optional layer
        import sys as _sys
        print(f"[warn] resolver unavailable: {_e}", file=_sys.stderr)

# ── Optional Layer 4b: the PHONETIC CONTRASTIVE MODEL (learned generalizer) ───
# The exact lexicon only fixes spellings someone listed. The phonetic model
# COMPUTES the canonical for unseen spellings — 97% held-out recall vs the
# resolver's 30%. It is the maintainable replacement for enumerating variants.
# Runs only on words the exact lexicon MISSED, with a 0.90 abstain threshold so
# it never corrupts real words. Enable with  PHONETIC=1.
# NOTE: on this eval it is accuracy-neutral (the fuzzy metric already forgives the
# spelling drift it fixes); its value is maintainability, not a score gain.
_PHONETIC = None
if _os.getenv('PHONETIC', '').lower() in ('1', 'true', 'yes', 'on'):
    try:
        from phonetic_contrastive_model.corrector import PhoneticContrastiveCorrector as _P
        _PHONETIC = _P.load(threshold=float(_os.getenv('PHONETIC_THRESHOLD', '0.90')))
    except Exception as _e:          # never break the pipeline over an optional layer
        import sys as _sys
        print(f"[warn] phonetic model unavailable: {_e}", file=_sys.stderr)



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


# phrase keys, longest first, so "blood culture" beats "blood" if both existed
_PHRASE_KEYS = sorted(PHRASE_MAP, key=len, reverse=True)


_TOKEN = re.compile(r'[A-Za-z0-9]+')


def _expand(canon: str, following: str) -> str:
    """
    Expand a variant to its canonical WITHOUT duplicating words the ASR already
    transcribed.

    A single-word variant may expand to several words:
        "chughtai lab": ["chukaai", "chuppaai", ...]
    Most of those garble only the NAME. When the ASR did hear "lab" as its own
    token, a blind expansion emits it twice:
        chukaai laib se  ->  chughtai lab | lab se       <- "lab" duplicated

    So: if the tail of the expansion is immediately repeated in the text that
    FOLLOWS the match, emit only the head and let the text's own copy stand.
        chukaai laib se  ->  chughtai | lab se           <- correct

    The following tokens are compared through WORD_MAP, because the phrase pass
    runs FIRST — at this point the text still reads "laib", and it only becomes
    "lab" in the word pass below. Comparing the raw token would miss the clash.

    Only an UNBROKEN run of words is considered (nothing but whitespace between
    the match and the repeat), so genuine reduplication across a clause boundary
    is never collapsed. At least one word is always emitted.
    """
    words = canon.split()
    if len(words) < 2:
        return canon
    if following[:1] and not following[:1].isspace():
        return canon                     # glued to punctuation — not a word repeat

    nxt = [(WORD_MAP.get(t.lower()) or t).lower()
           for t in _TOKEN.findall(following)[: len(words) - 1]]
    for k in range(len(words) - 1, 0, -1):          # longest tail first
        if nxt[:k] == [w.lower() for w in words[-k:]]:
            return ' '.join(words[:-k])
    return canon


def _apply_corrections(text: str) -> str:
    """
    Layer 3: map ASR misspellings onto the correct Roman Urdu word.

    Both maps are flat {variant_lower: canonical}, inverted from v2's
    {canonical: [variants]} at import time.

    PHRASE PASS RUNS FIRST. A multi-word garble like "beeta echaseeji" must be
    matched while it is still intact — if the word pass ran first it could
    rewrite "beeta" on its own and the phrase would never match.

    A canonical may itself be several words (assalaamualaikum ->
    "assalam o alaikum"); _expand() keeps that from duplicating a word the ASR
    already emitted.
    """
    # 1. phrase pass — on the raw text, before any word is touched
    for phrase in _PHRASE_KEYS:
        canon = PHRASE_MAP[phrase]
        text = re.sub(
            rf'\b{re.escape(phrase)}\b',
            lambda m, c=canon: _expand(c, m.string[m.end():]),
            text,
            flags=re.IGNORECASE,
        )

    # 2. word pass
    def fix_word(m):
        w = m.group(0)
        canon = WORD_MAP.get(w.lower())
        if not canon:
            # 3. learned fallback — ONLY for words the exact lexicon MISSED.
            #    Prefer the phonetic model (generalizes, abstains safely); the
            #    resolver is the deprecated rule-based alternative. Neither can
            #    override an exact lexicon hit.
            if _PHONETIC is not None:
                return _PHONETIC.resolve_word(w)
            if _RESOLVER is not None:
                return _RESOLVER.resolve_word(w)
            return w
        canon = _expand(canon, m.string[m.end():])
        # the canonical carries its own correct case (CNIC, Chughtai, area);
        # only mirror an incoming capital when the canonical is all-lowercase
        if w[0].isupper() and canon == canon.lower():
            return canon[0].upper() + canon[1:]
        return canon

    return re.sub(r"[A-Za-z0-9]+", fix_word, text)


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
        ("बहुत अच्छा है।",                "bohot acha hai."),   # NORMALIZATIONS: bahut->bohot (gold spelling)
        # gold annotators write "main" for में, not "mein" — the gold is the WER target
        ("यह कोड की ज़बान में आवाज़ की शिनाख़्त का टेस्ट है।",
                                           "yeh kod ki zaban main awaz ki shanakht ka test hai."),
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
        # gold annotators write "paanch", not "panch" — the gold is the WER target
        ("पाँच",     "paanch"),
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
