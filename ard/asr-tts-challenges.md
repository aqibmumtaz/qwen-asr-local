# CLL Agent — STT & TTS Challenges
### by AI Team · BitLogix

---

## Speech-to-Text (STT) Challenges

### 1. Audio Quality: 8 kHz Telephony vs 16 kHz ASR

| Factor | Detail |
|---|---|
| Recorded sample rate | 8 kHz (telephony standard) |
| Model requirement | 16 kHz (Whisper, Qwen3-ASR, all major ASR models) |
| Problem | Upsampling from 8 → 16 kHz adds no information — high-frequency band (4–8 kHz) containing Urdu fricatives (ش, س, ز, ف, خ, غ) is permanently lost at the codec stage |
| Impact | Higher WER on fricative-heavy Urdu medical vocabulary; muffled/distorted signal degrades confidence across the board |

**Mitigation options:** Request 16 kHz recording from PBX before codec encoding; use telephony-trained acoustic model variant; apply spectral enhancement pre-processing.

---

### 2. Dual-Speaker Audio: Agent + Caller on One Channel

| Speaker | Characteristics | Challenge |
|---|---|---|
| Caller (patient) | Normal volume, variable pace, may have accent | Natural variation — handled reasonably by model |
| Agent (helpdesk staff) | **Low volume** (distant mic), **very fast speech** (experienced professional cadence), domain jargon | Under-represented in training data for fast low-volume speech |
| Combined | Single mixed audio channel | No speaker separation; ASR sees both speakers as one signal |

**Impact:** Agent utterances get transcribed with higher error rate; fast speech causes word merging; low-volume agent audio may be partially drowned by caller.

**Mitigation options:** Separate recording channels (two-channel PBX recording); voice activity detection + volume normalisation; diarisation before ASR.

---

### 3. Language-Hint Dilemma: Hindi vs English Prompt

The ASR model (Qwen3-ASR) requires a language hint in its prompt. Neither available option is correct for Urdu:

| Language Hint | Behaviour | Problem |
|---|---|---|
| `language Hindi` | Outputs Hindi Devanagari script; preserves Urdu word content | **Drops nuktas** (़) — क़→क, ज़→ज, फ़→फ — so قابل→kaabil not qaabil, زندگی→jindagi not zindagi |
| `language English` | Preserves nuktas; phonetically more accurate Devanagari | **Transcribes code-switched or ambiguous segments in Latin/English** — entire Urdu phrases can come out in English |

> **Root cause:** No `language Urdu` option — Urdu is not in the model's supported language list. Urdu uses Perso-Arabic script; the model transcribes it as Hindi/Devanagari. Nuktas distinguish Perso-Arabic phonemes but Hindi training data omits them.

**Mitigation options:** Fine-tune model on Urdu audio with nukta-annotated Devanagari targets; post-process with rule-based nukta restoration (ph→f closed set + lexicon); submit Urdu support request upstream.

---

### 4. Code-Switching and Medical Vocabulary

- Callers mix Urdu, English medical terms (blood test, CBC, ultrasound, creatinine) and sometimes Punjabi
- Medical terminology (lab test names, drug names, anatomical terms) is out-of-distribution for general ASR models
- Names: doctor names, test names, branch names (Johar Town, Gulberg, Liberty) are frequently mis-transcribed

---

### 5. STT Model Training Challenges

**Lack of Urdu-specific training data:**
- Most open-source ASR models (Whisper, Qwen3-ASR) are trained predominantly on English and high-resource languages; Urdu is severely under-represented
- No large-scale, clean, transcribed Urdu telephony corpus exists publicly — CommonVoice Urdu is small (~100 hours) and read-speech only, not conversational
- Collecting and annotating domain-specific Urdu helpdesk audio (medical terms, lab tests, proper nouns) requires significant manual effort and native-speaker expertise

**Fine-tuning complexity:**
- Fine-tuning Whisper or Qwen3-ASR on Urdu requires GPU infrastructure, careful learning rate scheduling, and risk of catastrophic forgetting on other languages
- Urdu ground-truth transcripts must be in Nastaliq script with consistent orthographic conventions — inconsistency in training labels directly causes WER regression
- Evaluation requires native Urdu annotators; standard WER metrics do not account for script normalisation (e.g., alef variants: ا، آ، أ)

**Domain mismatch:**
- General ASR models are trained on news, podcasts, and read speech — not live telephony conversations with interruptions, hesitations, and background noise
- Medical vocabulary (haematology, biochemistry, radiology terms) is out-of-vocabulary in all general models; fine-tuning on in-domain data is essential but expensive
- Accents across Pakistani cities (Lahore, Karachi, Peshawar, Faisalabad) are not uniformly represented — model generalises poorly to under-represented regional accents

---

### 6. Progress: Hindi → Roman Urdu

Since no native Urdu ASR model exists on-premises, the current approach bridges the gap through a custom post-processing layer built at BitLogix:

**How it works:**

1. **ASR transcribes in Devanagari** — Qwen3-ASR outputs Hindi/Devanagari text for Urdu speech (e.g., "زندگی" is transcribed as "ज़िंदगी")
2. **Phoneme mapping (Layer 1)** — Each Devanagari character is mapped to its Roman Urdu equivalent using a hand-crafted consonant and vowel map; nukta characters (़) are resolved to their Perso-Arabic phonemes (फ़→f, ज़→z, क़→q)
3. **Schwa deletion (Layer 2)** — Hindi inherits Sanskrit schwa (implicit /ə/ vowel) which is silent in many Urdu words; rule-based schwa deletion removes it to produce correct Roman spelling (e.g., "करम" → "karam" not "karama")
4. **Lexicon correction (Layer 3)** — A domain lexicon of ~985 word corrections and ~286 proper nouns fixes residual errors that rules cannot handle (medical terms, Urdu-specific phonology, names)

**What it solves:**

| Problem | Solution |
|---|---|
| No native Urdu ASR | Devanagari output re-mapped to Roman Urdu — downstream NLP works on Roman script |
| Nukta omission (Hindi hint) | English language hint used on CPU — better nukta retention; nukta-aware phoneme map handles फ़/ज़/क़ correctly |
| Urdu-specific phonology (ph→f, j→z) | Closed rule sets handle systematic substitutions; open sets handled by lexicon |
| Medical and proper nouns | Proper nouns dictionary covers lab branch names, doctor names, test names |
| Script mismatch for NLP | Output is clean Roman Urdu — compatible with intent classifiers and LLM context |

**Current limitations:** Lexicon must be maintained manually as new vocabulary is encountered; regional accent variation can produce Devanagari forms that miss the lexicon; nukta emission remains inconsistent across audio quality levels.

---

## Text-to-Speech (TTS) Challenges

### 1. Telephony Downsampling: 16 kHz → 8 kHz Output

- TTS models generate audio at 16–24 kHz (full bandwidth, natural-sounding)
- PTCL SIP / G.711 delivers to caller at 8 kHz
- Downsampling collapses the high-frequency band: synthesised voice loses naturalness, sounds muffled over phone
- Urdu fricatives (ش، س، ز، خ) are especially degraded — clarity of dental and emphatic consonants reduced

---

### 2. Scarcity of High-Quality On-Premises Urdu Models

| Tier | Examples | Gap |
|---|---|---|
| Cloud (excluded) | Google WaveNet Urdu, Azure Neural Urdu, Amazon Polly | Privacy / on-prem policy |
| Research models | IMS-Toucan (multilingual), MMS-TTS | Urdu quality inconsistent; not production-ready |
| Open on-prem | Coqui XTTS (zero-shot), Piper TTS | Limited Urdu training data; accent/naturalness issues |
| Target | VITS or XTTS fine-tuned on Urdu | Requires custom dataset + GPU training |

Fine-tuning a neural TTS requires 5–20 hours of clean, professionally recorded Urdu speech — not readily available.

---

### 3. Missing Diacritics (Aerab) Problem

- Urdu Nastaliq script is routinely written **without aerab** (short vowels: زبر، زیر، پیش) in all practical text sources
- Without aerab, the same written word has multiple valid pronunciations — TTS must infer the correct one from context
- Example: "کرم" = karam (grace) or kirm (worm) depending on context; TTS with no aerab will guess
- LLM-generated response text will always lack aerab — no practical way to enforce diacritic annotation at scale

**Mitigation:** Diacritic prediction model (grapheme-to-phoneme with context); constrained vocabulary for known domain terms.

---

### 4. Mixed-Language Pronunciation and Script Conflicts

- Response text from LLM contains Urdu (Nastaliq), English terms, numbers, and sometimes Devanagari
- TTS must correctly pronounce English words embedded in Urdu (e.g., "CBC test کے لیے Johar Town آئیں")
- Naive single-model TTS applies Urdu phonology to English words → heavy foreign accent on medical terms
- Solution requires per-token language ID + multilingual TTS or a separate English synthesis pass with prosodic blending

---

### 5. Number, Price, and Date Normalisation

- LLM outputs may include: `Rs. 1,450`, `12 May 2026`, `10:30 AM`, `CBC (Complete Blood Count)`
- TTS requires normalised text-to-speech form: "ایک ہزار چار سو پچاس روپے", "بارہ مئی دو ہزار چھبیس"
- No robust Urdu text normalisation library exists; must be built or adapted from multilingual normalisation tools
- Edge cases: ranges ("500–1500"), abbreviations ("Dr.", "Dept."), mixed units

---

### 6. Naturalness, Prosody, and Real-Time Latency

- Conversational helpdesk requires TTS response within ~1 second of intent completion
- Neural TTS batch-processes full sentences — streaming requires chunked synthesis with sentence boundary detection
- Helpdesk callers are patients — empathetic, clear, unhurried delivery matters for trust
- Current on-prem Urdu models produce flat monotone speech; sentence-final intonation for questions vs statements is often incorrect

| Model Type | RTF (approx.) | GPU Requirement | Notes |
|---|---|---|---|
| HMM/Concatenative | < 0.1× | CPU only | Robotic; acceptable latency |
| VITS (fine-tuned) | ~0.3–0.5× | Mid-range GPU | Best quality/speed tradeoff |
| XTTS v2 (zero-shot) | ~1–2× | 8 GB VRAM | High quality; slow without batching |
| Large diffusion TTS | > 2× | High-end GPU | Not viable for real-time |

---

### 7. No Custom Brand Voice

- Chughtai Lab may require a consistent, recognisable helpdesk voice identity
- Custom voice cloning (XTTS speaker conditioning, VALL-E style) requires 5–30 minutes of target speaker recordings
- Voice consistency across sessions, call types, and languages must be maintained
- Requires a dedicated voice recording session with a professional Urdu speaker

---

*Prepared for Chughtai Lab AI Helpdesk · Chughtai Lab Voice Pipeline Technical Review*
