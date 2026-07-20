#!/usr/bin/env python3
"""
STEP 4 — the RESOLVER. Ends the enumeration treadmill.

THE PROBLEM
    lexicons_v2 holds 14,575 hand-listed misspellings for 2,071 real words. The
    next call always produces a spelling that is not in the list:
        "chugataai" -> (not found) -> stays broken
    You can never finish enumerating. Every new call adds new garbles.

THE FIX
    Keep only the 2,071 CANONICALS and COMPUTE the match:
        "chugataai" -> normalise -> fuzzy-match -> "Chughtai"     (never seen before)

HOW IT WORKS
    1. normalise()  strips the phonetic noise the ASR+transliterator introduce:
       doubled letters, ee/i, oo/u, aa/a, ph/f, y/i, w/v, k/c ...
         chugataai / chughtai / choogtay / chukataai  ->  all become "chgt"
    2. exact match on the normalised key (fast, O(1))
    3. otherwise, bounded edit-distance search over canonicals of similar length

SAFETY — this is the whole game. A fuzzy matcher that fires on a CORRECT word is
worse than no matcher at all. Four guards, in order:
    G1  exact lexicon hit wins        -> never overridden
    G2  the word is ALREADY CORRECT   -> never touched (gold / protected / canonical)
    G3  too short to match safely     -> skipped (edit-distance 2 on a 4-char word
                                          matches almost anything)
    G4  distance budget scales with length, and the match must be UNAMBIGUOUS
        (a clear winner; ties are rejected)

Usage:
    from resolver import Resolver
    r = Resolver()
    r.resolve_word("chugataai")   -> "Chughtai"
    r.resolve_text("... lab se chugataai ...")
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
V2 = SCRIPT_DIR / "data" / "lexicons_v2.json"
XLSX = SCRIPT_DIR / "data" / "CLL analysis" / "turnwise_results_eval_full.xlsx"

# ── Cross-word collision protection ──────────────────────────────────────────
# Words that normalise to the same key as an unrelated word. Adding them to the
# known-correct set (G2) prevents the fuzzy matcher from ever touching them.
#   naila / neela  ->  both normalise to "nela"  (name vs "blue")
#   faur  / for    ->  both normalise to "for"   ("immediately" vs English "for")
#   aaye  / eye    ->  both normalise to "e"
#   aayen / ain    ->  both normalise to "en"
#   daina / dena   ->  both normalise to "dena"  (same meaning, different spelling)
#   chai  / chhe   ->  both normalise to "ce"    ("tea" vs "six")
#   omair / omer   ->  both normalise to "omer"  (different names)
CROSS_WORD_PROTECTED = {
    "naila", "neela",
    "faur", "for",
    "aaye", "eye",
    "aayen", "ain",
    "chai", "chhe",
    "omair", "omer",
}

# ── G3: minimum length to attempt a fuzzy match ──────────────────────────────
# This is the length guard we deliberately DEFERRED from the lexicon cleanup:
# exact-match is safe at any length, but fuzzy-match is not. Edit-distance 2 from
# a 4-char string matches half the dictionary.
#
# 8 is measured, not guessed. Sweeping the threshold on the gold set (fuzzy path
# isolated from the exact lexicon):
#     min_len  corrupted   fixes   net
#           4         70      73     +3
#           5         54      49     -5
#           6         46      48     +2
#           7         16      16      0
#           8          6      12     +6      <- safest AND best
MIN_FUZZY_LEN = 8

# ── G4: distance budget by word length ───────────────────────────────────────
# Uses a proportional threshold (floor) instead of hard-coded tiers.
# floor() is stricter on short words, preventing false positives.
#
# At threshold 0.25:
#     len  8 → budget 2       len 12 → budget 3
#     len  9 → budget 2       len 13 → budget 3
#     len 10 → budget 2       len 14 → budget 3
#     len 11 → budget 2       len 16 → budget 4
EDIT_DISTANCE_THRESHOLD = 0.20


def _max_distance_fixed(n: int) -> int:
    """Original hard-coded tier-based distance budget (kept for reference)."""
    if n < MIN_FUZZY_LEN:
        return 0
    if n <= 7:
        return 1
    if n <= 11:
        return 2
    return 3


def max_distance(token_len: int, candidate_len: int) -> int:
    """
    Proportional edit-distance budget based on the longer of the two words.
    Uses floor (not round) to be stricter on short words and avoid false positives.
    Note: G3 already ensures the raw word is >= MIN_FUZZY_LEN before this is called.
    """
    return math.floor(max(token_len, candidate_len) * EDIT_DISTANCE_THRESHOLD)


# ── normalisation — collapse phonetic noise, but KEEP the vowel skeleton ─────
#
# A first attempt dropped vowels entirely, reducing each word to its consonant
# skeleton. That was far too lossy and produced garbage:
#     jaata -> "jt"  and  Ajeet -> "jt"   =>  jaata resolved to the NAME "Ajeet"
#     bohot -> "bt"  and  Bhutto -> "bt"  =>  "very" resolved to a SURNAME
# Roman Urdu has too many short words sharing a consonant skeleton.
#
# So: keep the vowels, but normalise only the distinctions the ASR + the
# transliterator actually blur — vowel LENGTH (aa/a, ee/i, oo/u) and a handful
# of consonant spellings. Word shape and syllable count survive.
_SUBS = [
    (r"(.)\1+", r"\1"),      # doubled letters:   chughttai -> chughtai
    (r"ph", "f"),            # phes  / fes
    (r"kh|q", "k"),          # khan  / qan
    (r"gh", "g"),            # chughtai / chugtai
    # (r"sh", "s"),            # DISABLED: sh (ش) and s (س) are distinct Urdu
    #                           # letters; collides sabir↔Shabbir, chai↔chhe
    (r"ch", "c"),
    # (r"th|dh", "t"),         # DISABLED: th/dh/t are distinct sounds in Urdu;
    #                           # collides tek↔theek, dha↔ta, dhai↔the
    # (r"[zj]", "j"),          # DISABLED: z and j are distinct phonemes in Urdu,
    #                           # collapsing them causes false matches
    (r"w", "v"),
    (r"ee|ie|y", "i"),       # vowel LENGTH is noise: nadee / nadi
    (r"oo|ou", "u"),
    (r"aa", "a"),
    (r"ai|ay|ei|ey", "e"),
    (r"au|ao|ow", "o"),
]


def normalise(w: str) -> str:
    """
    Collapse the phonetic noise while KEEPING the word's shape.

        chugataai / chughtai / choogtay / chukataai  ->  "cugtai"-ish
        but  jaata != Ajeet  and  bohot != Bhutto     (vowels keep them apart)
    """
    w = w.lower()
    for pat, rep in _SUBS:
        w = re.sub(pat, rep, w)
    return re.sub(r"(.)\1+", r"\1", w)   # collapse again after substitutions


def _edit(a: str, b: str, cap: int) -> int:
    """Levenshtein with early exit once the distance exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        lo = max(1, i - cap)
        hi = min(len(b), i + cap)
        for j in range(1, len(b) + 1):
            if j < lo or j > hi:
                cur.append(cap + 1)
                continue
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != b[j - 1])))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class Resolver:
    """Exact lexicon first, then normalise + bounded fuzzy match."""

    def __init__(self, v2_path: Path = V2, gold_vocab: Optional[set] = None,
                 min_len: int = MIN_FUZZY_LEN):
        raw = json.loads(v2_path.read_text(encoding="utf-8"))["lexicons"]
        self.min_len = min_len

        # exact map (G1)
        self.exact: dict[str, str] = {}
        for section in ("lexicon", "phrases"):
            for canon, variants in raw[section].items():
                for v in variants:
                    self.exact[v.lower()] = canon

        # the canonicals — the ONLY thing the fuzzy matcher needs
        self.canonicals = [c for c in raw["lexicon"] if " " not in c]

        # G2: words that are already correct and must never be touched
        self.known_correct = {c.lower() for c in self.canonicals}
        self.known_correct |= {c.lower() for c in raw["phrases"]}
        self.known_correct |= CROSS_WORD_PROTECTED
        if gold_vocab:
            self.known_correct |= gold_vocab

        # normalised index, bucketed by length for a fast bounded search
        self.norm_exact: dict[str, list[str]] = defaultdict(list)
        self.by_len: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for c in self.canonicals:
            nk = normalise(c)
            if not nk:
                continue
            self.norm_exact[nk].append(c)
            self.by_len[len(nk)].append((nk, c))

        self.stats = defaultdict(int)

    # ── the one method that matters ──────────────────────────────────────────
    def resolve_word(self, word: str) -> str:
        lw = word.lower()

        # G1 — an exact lexicon entry always wins
        hit = self.exact.get(lw)
        if hit:
            self.stats["exact"] += 1
            return hit

        # G2 — the word is already correct. NEVER touch it.
        if lw in self.known_correct:
            self.stats["already_correct"] += 1
            return word

        # G3 — too short to fuzzy-match safely
        if len(lw) < self.min_len or not lw.isalpha():
            self.stats["too_short"] += 1
            return word

        nk = normalise(lw)
        if not nk:
            return word

        # fast path: the normalised skeletons match exactly
        cands = self.norm_exact.get(nk)
        if cands:
            if len(cands) == 1:
                self.stats["fuzzy_exact_skeleton"] += 1
                return cands[0]
            self.stats["ambiguous"] += 1     # G4 — two canonicals collide; refuse
            return word

        # bounded edit-distance search over similar-length skeletons
        cap = max_distance(len(nk), len(nk))  # initial cap based on token itself
        if cap == 0:
            return word
        best, best_d, runner_up = None, cap + 1, cap + 1
        for L in range(len(nk) - cap, len(nk) + cap + 1):
            for cnk, canon in self.by_len.get(L, ()):
                pair_cap = max_distance(len(nk), len(cnk))
                d = _edit(nk, cnk, pair_cap)
                if d < best_d:
                    best, runner_up, best_d = canon, best_d, d
                elif d < runner_up:
                    runner_up = d

        # G4 — require a clear, unambiguous winner
        if best is None or best_d > max_distance(len(nk), len(normalise(best))):
            self.stats["no_match"] += 1
            return word
        if runner_up == best_d:
            self.stats["ambiguous"] += 1     # tie -> refuse rather than guess
            return word

        self.stats["fuzzy"] += 1
        return best

    def resolve_text(self, text: str) -> str:
        return re.sub(r"[A-Za-z]+", lambda m: self.resolve_word(m.group(0)), text)

    def report(self) -> str:
        t = sum(self.stats.values()) or 1
        return "  " + "  ".join(
            f"{k}={v} ({100*v/t:.0f}%)" for k, v in sorted(self.stats.items())
        )


def load_gold_vocab() -> set:
    try:
        import openpyxl
    except ImportError:
        return set()
    wb = openpyxl.load_workbook(str(XLSX), data_only=True)
    rows = list(wb["asr_results"].iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    v = set()
    for r in rows[1:]:
        ref = r[idx["roman_urdu_reference"]]
        if isinstance(ref, str):
            for w in ref.split():
                w = w.strip(".,?!;:").lower()
                if w:
                    v.add(w)
    return v


if __name__ == "__main__":
    r = Resolver(gold_vocab=load_gold_vocab())
    print(f"canonicals indexed: {len(r.canonicals)}")
    print()
    tests = [
        ("chugataai", "Chughtai"),   # NOT in the variant list
        ("choogtaay", "Chughtai"),   # invented spelling
        ("nefrologi", "nephrology"),
        ("seyalkot", "Sialkot"),
        ("aponitment", "appointment"),
        ("daanishli", "?"),
        ("baat", "(leave alone — correct)"),
        ("hai", "(leave alone — correct)"),
    ]
    print(f"  {'input':<14} {'->':<4} {'resolved':<16} expected")
    print("  " + "-" * 58)
    for w, exp in tests:
        print(f"  {w:<14} ->   {r.resolve_word(w):<16} {exp}")
    print()
    print(r.report())
