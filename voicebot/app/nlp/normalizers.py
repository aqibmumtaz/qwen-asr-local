"""Code-side normalizers for spoken Roman Urdu: numbers, phone, age, dates, names, yes/no.

The LLM never decides these values; it may suggest raw text, code validates and converts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from rapidfuzz import fuzz, process

# ── number words ────────────────────────────────────────────────────────────
_UNITS: dict[str, int] = {}


def _add(value: int, *words: str) -> None:
    for w in words:
        _UNITS[w] = value


_add(0, "zero", "sifar", "siffar", "sifr")
_add(1, "ek", "aik", "one")
_add(2, "do", "two", "dou")
_add(3, "teen", "tin", "three")
_add(4, "char", "chaar", "four")
_add(5, "paanch", "panch", "paanchh", "five")
_add(6, "chay", "chhe", "che", "chah", "chhay", "six")
_add(7, "saat", "sat", "seven")
_add(8, "aath", "ath", "eight")
_add(9, "nau", "no", "nine")
_add(10, "das", "dus", "ten")
_add(11, "gyarah", "gyara", "giyarah", "gyarha", "eleven")
_add(12, "barah", "bara", "twelve")
_add(13, "terah", "tera", "thirteen")
_add(14, "chaudah", "chowdah", "chaudha", "fourteen")
_add(15, "pandrah", "pandra", "fifteen")
_add(16, "solah", "sola", "sixteen")
_add(17, "satrah", "satra", "seventeen")
_add(18, "atharah", "athara", "eighteen")
_add(19, "unnees", "unnis", "unees", "nineteen")
_add(20, "bees", "bis", "twenty")
_add(21, "ikkees", "ikkis", "ikis")
_add(22, "baees", "baais", "bais")
_add(23, "teis", "taees", "teyees")
_add(24, "chaubees", "chobees", "chaubis")
_add(25, "pachees", "pachis", "pacchis", "pachchees")
_add(26, "chhabbees", "chabbees", "chhabbis")
_add(27, "sattaees", "satais", "sattais")
_add(28, "atthaees", "athais", "atthais")
_add(29, "untees", "unattees", "unatis")
_add(30, "tees", "tis", "thirty")
_add(31, "ikattees", "iktees")
_add(32, "battees", "batees")
_add(33, "taintees", "tentees")
_add(34, "chauntees", "chauntis")
_add(35, "paintees", "painteesh", "painties", "pentees", "paintis")
_add(36, "chhattees", "chhatis")
_add(37, "saintees", "saintis")
_add(38, "artees", "artis")
_add(39, "untaalees", "unchaalees", "untalis")
_add(40, "chaalees", "chalis", "chaleez", "forty")
_add(41, "iktaalees", "iktalis")
_add(42, "bayaalees", "bayalis")
_add(43, "taintaalees", "tentalis")
_add(44, "chauwaalees", "chauvalis")
_add(45, "paintaalees", "paintalis", "pentalis")
_add(46, "chhiyaalees", "chiyalis")
_add(47, "saintaalees", "saintalis")
_add(48, "artaalees", "artalis")
_add(49, "unchaas", "unanchas")
_add(50, "pachaas", "pachas", "pachaass", "fifty")
_add(51, "ikyawan", "ikyavan")
_add(52, "baawan", "bawan")
_add(53, "tirpan", "tirpun")
_add(54, "chauwan", "chauvan")
_add(55, "pachpan", "pachpun")
_add(56, "chhappan", "chappan")
_add(57, "sattawan", "satavan")
_add(58, "atthawan", "athavan")
_add(59, "unsath", "unsat")
_add(60, "saath", "sath", "sixty")
_add(61, "iksath", "iksat")
_add(62, "baasath", "basath")
_add(63, "tirsath", "tirsat")
_add(64, "chaunsath", "chounsath")
_add(65, "painsath", "painsat")
_add(66, "chhiyasath", "chiyasath")
_add(67, "sarsath", "sarsat")
_add(68, "arsath", "arsat")
_add(69, "unhattar", "unhatar")
_add(70, "sattar", "satar", "seventy")
_add(71, "ikhattar")
_add(72, "bahattar", "bahatar")
_add(73, "tihattar")
_add(74, "chauhattar")
_add(75, "pachhattar", "pachattar")
_add(76, "chhihattar")
_add(77, "satattar")
_add(78, "athhattar")
_add(79, "unasi", "unnasi")
_add(80, "assi", "asi", "eighty")
_add(81, "ikyasi")
_add(82, "bayasi")
_add(83, "tirasi")
_add(84, "chaurasi")
_add(85, "pachasi")
_add(86, "chhiyasi")
_add(87, "sattasi")
_add(88, "athasi")
_add(89, "nawasi", "navasi")
_add(90, "nabbay", "nabbe", "ninety")
_add(91, "ikyanbay")
_add(92, "bayanbay")
_add(93, "tiranbay")
_add(94, "chauranbay")
_add(95, "pachanbay")
_add(96, "chhiyanbay")
_add(97, "sattanbay")
_add(98, "athanbay")
_add(99, "ninyanbay", "ninanbay")

_MULT = {"sau": 100, "so": 100, "hundred": 100, "hazaar": 1000, "hazar": 1000, "thousand": 1000,
         "lakh": 100000, "lac": 100000}
_FRACT = {"dedh": 1.5, "derh": 1.5, "dhai": 2.5, "dhaai": 2.5, "sawa": 1.25, "paune": 0.75}
_DIGIT_WORDS = {"zero": 0, "sifar": 0, "siffar": 0, "o": 0, "oh": 0, "ek": 1, "aik": 1, "one": 1, "do": 2, "two": 2,
                "teen": 3, "tin": 3, "three": 3, "char": 4, "chaar": 4, "four": 4, "paanch": 5, "panch": 5,
                "five": 5, "chay": 6, "chhe": 6, "che": 6, "chah": 6, "six": 6, "saat": 7, "sat": 7, "seven": 7,
                "aath": 8, "ath": 8, "eight": 8, "nau": 9, "nine": 9}
_REPEAT = {"double": 2, "dubal": 2, "triple": 3, "tripal": 3}

_TOKEN_RE = re.compile(r"[a-z]+|\d+", re.I)


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _unit_value(tok: str) -> int | None:
    if tok in _UNITS:
        return _UNITS[tok]
    if len(tok) >= 4:
        m = process.extractOne(tok, _UNITS.keys(), scorer=fuzz.ratio, score_cutoff=86)
        if m:
            return _UNITS[m[0]]
    return None


def words_to_int(text: str) -> int | None:
    """'paanch sau' -> 500, 'dedh hazaar' -> 1500, 'painteesh' -> 35, '1450' -> 1450."""
    toks = tokens(text)
    if not toks:
        return None
    total = 0.0
    current = 0.0
    seen = False
    pending_fraction: float | None = None
    for t in toks:
        if t.isdigit():
            current += int(t)
            seen = True
        elif t in _FRACT:
            pending_fraction = _FRACT[t]
            seen = True
        elif t in _MULT:
            mult = _MULT[t]
            base = pending_fraction if pending_fraction is not None else (current or 1)
            if pending_fraction is not None and current:
                base = pending_fraction  # "dhai sau": fraction applies to multiplier
            if mult >= 1000:
                total += base * mult
                current = 0
            else:
                current = base * mult
            pending_fraction = None
            seen = True
        else:
            v = _unit_value(t)
            if v is None:
                continue
            current += v
            seen = True
    if not seen:
        return None
    return int(round(total + current))


def find_numbers(text: str) -> list[int]:
    """All numbers in text, digits or words, in order (greedy grouping of adjacent number words)."""
    out: list[int] = []
    group: list[str] = []

    def flush() -> None:
        if group:
            v = words_to_int(" ".join(group))
            if v is not None:
                out.append(v)
            group.clear()

    for t in tokens(text):
        is_num = t.isdigit() or t in _MULT or t in _FRACT or _unit_value(t) is not None
        # short words are only numbers when exact matches (avoid 'do' in sentences being fuzzy)
        if is_num and not t.isdigit() and t not in _UNITS and t not in _MULT and t not in _FRACT:
            is_num = len(t) >= 5
        if is_num:
            group.append(t)
        else:
            flush()
    flush()
    return out


# ── phone ───────────────────────────────────────────────────────────────────
@dataclass
class PhoneResult:
    digits: str
    valid: bool
    reason: str = ""


def extract_digits(text: str) -> str:
    """Digits spoken or typed: '0300 12 34 567', 'zero teen sau', 'double paanch', 'triple saat'."""
    toks = tokens(text)
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.isdigit():
            out.append(t)
        elif t in _REPEAT and i + 1 < len(toks):
            nxt = toks[i + 1]
            d = int(nxt) if nxt.isdigit() and len(nxt) == 1 else _DIGIT_WORDS.get(nxt)
            if d is not None:
                out.append(str(d) * _REPEAT[t])
                i += 1
        elif t in _DIGIT_WORDS:
            out.append(str(_DIGIT_WORDS[t]))
        else:
            v = _unit_value(t) if len(t) >= 5 else None
            if v is not None and v >= 10:
                out.append(str(v))
        i += 1
    return "".join(out)


def normalize_phone(text: str) -> PhoneResult:
    d = extract_digits(text)
    if d.startswith("0092"):
        d = "0" + d[4:]
    elif d.startswith("92") and len(d) >= 12:
        d = "0" + d[2:]
    elif d.startswith("3") and len(d) == 10:
        d = "0" + d
    if not d:
        return PhoneResult("", False, "empty")
    if len(d) < 11:
        return PhoneResult(d, False, "short")
    if len(d) > 11:
        return PhoneResult(d, False, "long")
    if not d.startswith("03") and not d.startswith("02") and not d.startswith("04") and not d.startswith("05") \
            and not d.startswith("06") and not d.startswith("07") and not d.startswith("08") and not d.startswith("09"):
        return PhoneResult(d, False, "prefix")
    return PhoneResult(d, True)


def speak_digits(digits: str) -> str:
    """'03001234567' -> 'zero teen zero zero ek do teen char paanch chay saat' for read-back."""
    names = ["zero", "ek", "do", "teen", "char", "paanch", "chay", "saat", "aath", "nau"]
    return " ".join(names[int(c)] for c in digits if c.isdigit())


def speak_number(n: int) -> str:
    """Amounts for TTS in simple Roman Urdu: 1450 -> 'ek hazaar chaar sau pachaas'."""
    if n == 0:
        return "zero"
    inv = {v: k for k, v in sorted(_UNITS.items(), reverse=True) if len(k) > 2 or v < 10}
    pref = {1: "ek", 2: "do", 3: "teen", 4: "char", 5: "paanch", 6: "chay", 7: "saat", 8: "aath", 9: "nau",
            10: "das", 11: "gyarah", 12: "barah", 13: "terah", 14: "chaudah", 15: "pandrah", 16: "solah",
            17: "satrah", 18: "atharah", 19: "unnees", 20: "bees", 30: "tees", 40: "chaalees", 50: "pachaas",
            60: "saath", 70: "sattar", 80: "assi", 90: "nabbay"}
    pref.update({v: inv[v] for v in range(21, 100) if v in inv and v not in pref})
    parts: list[str] = []
    for unit, name in ((100000, "lakh"), (1000, "hazaar"), (100, "sau")):
        if n >= unit:
            q, n = divmod(n, unit)
            parts.append(f"{speak_number(q)} {name}")
    if n:
        parts.append(pref.get(n, str(n)))
    return " ".join(parts)


# ── age / gender / yes-no ───────────────────────────────────────────────────
def extract_age(text: str) -> int | None:
    nums = [n for n in find_numbers(text) if 0 < n <= 120]
    return nums[0] if nums else None


_MALE = {"mard", "male", "aadmi", "admi", "larka", "ladka", "beta", "boy", "man", "mr", "sahab", "bhai"}
_FEMALE = {"aurat", "female", "khatoon", "larki", "ladki", "beti", "girl", "woman", "mrs", "bibi", "behan", "khawateen"}


def extract_gender(text: str) -> str | None:
    toks = set(tokens(text))
    m, f = bool(toks & _MALE), bool(toks & _FEMALE)
    if m and not f:
        return "male"
    if f and not m:
        return "female"
    return None


_YES = {"haan", "han", "ha", "haa", "ji", "jee", "yes", "yeah", "theek", "thik", "sahi", "bilkul", "zaroor", "ok",
        "okay", "kar", "karo", "kardo", "karen", "kijiye", "kijye", "done", "confirm", "pakka", "accha", "acha"}
_NO = {"nahi", "nahin", "nai", "na", "no", "nhi", "galat", "ghalat", "ruko", "rukiye", "mat", "cancel", "wrong"}
_YES_PHRASES = ("theek hai", "thik hai", "ji haan", "ji han", "kar do", "kar den", "sahi hai", "bilkul theek",
                "haan ji", "ji kar do", "confirm kar")


def yes_no(text: str) -> str | None:
    t = " ".join(tokens(text))
    toks = tokens(text)
    if not toks:
        return None
    nos = sum(1 for x in toks if x in _NO)
    yes = sum(1 for x in toks if x in _YES) + sum(2 for p in _YES_PHRASES if p in t)
    if nos and not yes:
        return "no"
    if yes and not nos:
        return "yes"
    if nos and yes:
        # decided by the first decisive word ("haan nahi ..." ambiguous -> first token wins)
        for x in toks:
            if x in _NO:
                return "no"
            if x in _YES:
                return "yes"
    return None


# ── names ───────────────────────────────────────────────────────────────────
_NAME_FILLERS = {"mera", "meri", "naam", "nam", "name", "ka", "ki", "hai", "hain", "hoon", "hun", "main", "mein", "me",
                 "mareez", "mariz", "patient", "umar", "age", "saal", "sal", "year", "years", "old", "ye", "yeh", "ji",
                 "jee", "ka", "ke", "the", "is", "my", "aur", "and", "sample", "lena", "hai", "mere", "walid",
                 "walida", "ammi", "abbu", "ke", "liye", "wala", "wali", "ap", "aap", "to", "toh", "bas", "bhai"}


def normalize_name(text: str) -> str | None:
    """Keep the name as heard: strip fillers and numbers, title-case, at most 4 words."""
    out = []
    for t in tokens(text):
        if t in _NAME_FILLERS or t.isdigit() or t in _UNITS or t in _MULT:
            continue
        out.append(t)
    if not out:
        return None
    return " ".join(w.capitalize() for w in out[:4])


# ── dates and times ─────────────────────────────────────────────────────────
_WEEKDAYS = {
    "somwar": 0, "peer": 0, "pir": 0, "monday": 0, "mon": 0,
    "mangal": 1, "mangar": 1, "tuesday": 1, "tue": 1,
    "budh": 2, "budhwar": 2, "wednesday": 2, "wed": 2,
    "jumerat": 3, "jumeraat": 3, "thursday": 3, "thu": 3,
    "juma": 4, "jumma": 4, "friday": 4, "fri": 4,
    "hafta": 5, "haftah": 5, "saturday": 5, "sat": 5,
    "itwar": 6, "ittwar": 6, "itwaar": 6, "sunday": 6, "sun": 6,
}
_MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
           "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
           "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}
_PARTS = {"subah": "morning", "subha": "morning", "sawere": "morning", "morning": "morning",
          "dopahar": "afternoon", "dophar": "afternoon", "dupahar": "afternoon", "afternoon": "afternoon",
          "shaam": "evening", "sham": "evening", "evening": "evening",
          "raat": "night", "rat": "night", "night": "night"}


@dataclass
class DateSel:
    day: date | None = None
    part: str | None = None      # morning | afternoon | evening | night
    hour: int | None = None      # 0-23 when a clock time was given
    label: str = ""

    @property
    def has_day(self) -> bool:
        return self.day is not None

    @property
    def complete(self) -> bool:
        return self.day is not None and (self.part is not None or self.hour is not None)


_DATE_VOCAB = {"aaj", "aj", "kal", "parson", "parso", "narson", "tarson", "subah", "subha", "dopahar", "shaam", "sham",
               "raat", "baje", "agle", "agli", "somwar", "mangal", "budh", "jumerat", "juma", "jumma", "itwar", "hafta"}


def _split_merged(toks: list[str]) -> list[str]:
    """'parsonshaam' -> ['parson', 'shaam'] (speech recognisers often glue neighbouring words)."""
    out: list[str] = []
    for t in toks:
        if t in _DATE_VOCAB or t.isdigit() or len(t) < 6:
            out.append(t)
            continue
        for i in range(3, len(t) - 2):
            if t[:i] in _DATE_VOCAB and t[i:] in _DATE_VOCAB:
                out += [t[:i], t[i:]]
                break
        else:
            out.append(t)
    return out


_NEG = {"nahi", "nahin", "nai", "nhi", "na"}
_FUZZY_DATE_TARGETS = ["subah", "dopahar", "shaam", "raat", "parson", "narson", "somwar", "mangal", "budhwar", "jumerat",
                       "jumma", "itwar", "tareekh", "baje", "agle", "agli", "january", "february", "march", "april", "june",
                       "july", "august", "september", "october", "november", "december"]
_KNOWN_NOT_DATE = {"subhan", "shamim", "ratan", "mangla", "rahat", "aqsa", "shaan"}


def _fuzzy_date_words(toks: list[str]) -> list[str]:
    """Repair misspelled day/part words ('subeh' -> 'subah', 'shaaam' -> 'shaam'); short words are left alone."""
    out = []
    for t in toks:
        if len(t) >= 4 and not t.isdigit() and t not in _DATE_VOCAB and t not in _WEEKDAYS and t not in _MONTHS \
                and t not in _PARTS and t not in _KNOWN_NOT_DATE:
            m = process.extractOne(t, [w for w in _FUZZY_DATE_TARGETS if w[0] == t[0]], scorer=fuzz.ratio, score_cutoff=78)
            if m:
                out.append(m[0])
                continue
        out.append(t)
    return out


def resolve_date(text: str, today: date | None = None) -> DateSel:
    today = today or datetime.now().date()
    toks = _fuzzy_date_words(_split_merged(tokens(text)))
    # "kal nahi parson shaam": a day word followed by a negation is rejected, not chosen
    skip = {i for i, t in enumerate(toks[:-1]) if t in _DATE_VOCAB and toks[i + 1] in _NEG}
    toks = [t for i, t in enumerate(toks) if i not in skip]
    sel = DateSel()
    nxt = "agle" in toks or "agli" in toks or "next" in toks
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("aaj", "aj", "today"):
            sel.day, sel.label = today, "aaj"
        elif t in ("kal", "kl", "tomorrow"):
            sel.day, sel.label = today + timedelta(days=1), "kal"
        elif t in ("parson", "parso", "parsoon"):
            sel.day, sel.label = today + timedelta(days=2), "parson"
        elif t in ("narson", "tarson"):
            sel.day, sel.label = today + timedelta(days=3), "narson"
        elif t in _WEEKDAYS and t not in ("sat", "sun", "mon", "tue", "wed", "thu", "fri") or \
                (t in ("sat", "sun", "mon", "tue", "wed", "thu", "fri") and len(toks) <= 3):
            target = _WEEKDAYS[t]
            delta = (target - today.weekday()) % 7
            if delta == 0 and nxt:
                delta = 7
            sel.day = today + timedelta(days=delta)
            sel.label = t
        elif t in _MONTHS:
            # "15 november" or "november 15"
            day_n = None
            if i > 0 and toks[i - 1].isdigit():
                day_n = int(toks[i - 1])
            elif i + 1 < len(toks) and toks[i + 1].isdigit():
                day_n = int(toks[i + 1])
            if day_n and 1 <= day_n <= 31:
                y = today.year
                try:
                    d = date(y, _MONTHS[t], day_n)
                    if d < today:
                        d = date(y + 1, _MONTHS[t], day_n)
                    sel.day, sel.label = d, f"{day_n} {t}"
                except ValueError:
                    pass
        elif t == "tareekh" or t == "tarikh":
            if i > 0:
                n = words_to_int(toks[i - 1])
                if n and 1 <= n <= 31:
                    d = _next_day_of_month(today, n)
                    if d:
                        sel.day, sel.label = d, f"{n} tareekh"
        if t in _PARTS:
            sel.part = _PARTS[t]
        i += 1
    # clock time: "10 baje", "dus baje", "saat se gyarah"
    for idx, t in enumerate(toks):
        if t == "baje" or t == "bje":
            h = words_to_int(toks[idx - 1]) if idx > 0 else None
            if h is not None and 0 < h <= 24:
                sel.hour = _to_24h(h, sel.part)
    if sel.hour is not None and sel.part is None:
        sel.part = _part_of_hour(sel.hour)
    if sel.part and sel.hour is None and sel.day is None and "aaj" in toks:
        sel.day = today
    return sel


def _next_day_of_month(today: date, n: int) -> date | None:
    for add_month in (0, 1):
        m = today.month + add_month
        y = today.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        try:
            d = date(y, m, n)
        except ValueError:
            continue
        if d >= today:
            return d
    return None


def _to_24h(h: int, part: str | None) -> int:
    if h > 12:
        return h % 24
    if part in ("afternoon", "evening", "night") and h < 12:
        return h + 12 if not (part == "night" and h >= 10) else h + 12 if h < 12 else h
    if part == "morning" and h == 12:
        return 0
    return h


def _part_of_hour(h: int) -> str:
    if h < 12:
        return "morning"
    if h < 16:
        return "afternoon"
    if h < 19:
        return "evening"
    return "night"


def part_label(part: str | None) -> str:
    return {"morning": "subah", "afternoon": "dopahar", "evening": "shaam", "night": "raat"}.get(part or "", "")


# ── misc text cleanup (light ASR clean-up before routing/extraction) ────────
_VARIANTS = {"nhi": "nahi", "nai": "nahi", "nahin": "nahi", "kl": "kal", "krna": "karna", "krni": "karni",
             "kro": "karo", "kr": "kar", "hn": "haan", "han": "haan", "jee": "ji", "thk": "theek", "thik": "theek",
             "tkk": "theek", "plz": "please", "pls": "please", "bje": "baje", "mje": "mujhe", "mjhe": "mujhe",
             "chahiye": "chahiye", "chaiye": "chahiye", "chahye": "chahiye", "chahiay": "chahiye"}


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, unify the most common spelling variants."""
    t = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return " ".join(_VARIANTS.get(w, w) for w in t.split())
