# Transliteration Libraries — Internal Analysis

## urduhack — NOT useful for this
No Devanagari→Urdu transliteration. Only normalizes within Urdu script (malformed Unicode, digit systems).
Ignore it for Hindi→Urdu conversion.

## indic-transliteration — Character table + virama state machine
- Each script (Devanagari, Urdu, Latin etc.) defined as a TOML file of ~95 char mappings
- Stateful greedy tokenizer: walks string left-to-right, longest-match first
- Handles multi-codepoint sequences: क्ष → کْشَ (consonant + sukun + consonant)
- Virama (्) maps to sukun (ْ) — suppresses the inherent vowel
- **No ML. Fully deterministic.**
- **Problem**: Always outputs full diacritics (zabar/zer/pesh everywhere) → کَ instead of ک → unnatural in real Urdu prose

## GokulNC/Indic-PersoArabic-Script-Converter — Best rule-based option
- Regex passes with **positional rules** (word-initial / medial / word-final)
- Handles schwa deletion/abjadification (removes short vowels → matches Urdu abjad nature)
- CSV mapping tables: consonants, vowels, initial_vowels, final_vowels, hamza
- Nuqta consonants handled: ज़→ز, ड़→ڑ, फ़→ف
- Example positional logic:
  - अ word-initially → ا
  - ा medially       → ا
  - ी word-finally   → ی
- Optional ML mode (IndTrans, IIT Bombay) for better schwa handling
- Still deterministic in rule-based mode

## Key Limitation of ALL Rule-based Approaches
**Schwa syncope**: In Hindi, the inherent 'a' vowel is silently deleted in certain syllable positions
("vidyalay" not "vidyalaya"). No lookup table can solve this without morphological context.
- indic-transliteration: doesn't delete it — outputs full diacritics (wrong for prose)
- GokulNC: deletes ALL short vowels uniformly — overcorrects
- Only ML (IndTrans / trained model) gets schwa deletion right

## Rust Architecture for Transliteration
See rust-architecture-plan.md. Recommendation: embed a static Rust `match` table (100 lines, <1ms, zero deps).
The LLM should never touch transliteration — it's a deterministic char lookup, not a language task.
