"""ASR-style noise for Roman Urdu text (used to test robustness against speech-recognition errors).

Patterns follow what the domain-tuned Qwen3-ASR produces on real call audio: vowel confusions, dropped or doubled
letters, 'h' toggles, merged or split words, English words spelled phonetically.
"""
from __future__ import annotations

import random
import re

_VOWEL_SWAP = {"a": "e", "e": "a", "i": "e", "o": "u", "u": "o"}
_PROTECT = re.compile(r"^\d+$")


def noisy_word(w: str, rng: random.Random) -> str:
    if len(w) < 4 or _PROTECT.match(w):
        return w
    kind = rng.choice(["vowel", "drop", "dup", "h", "swap"])
    i = rng.randrange(1, len(w) - 1)
    if kind == "vowel":
        idx = [j for j, c in enumerate(w) if c in _VOWEL_SWAP and 0 < j]
        if idx:
            j = rng.choice(idx)
            return w[:j] + _VOWEL_SWAP[w[j]] + w[j + 1:]
    if kind == "drop":
        return w[:i] + w[i + 1:]
    if kind == "dup":
        return w[:i] + w[i] + w[i:]
    if kind == "h":
        return w[:-1] if w.endswith("h") else w + "h" if w[-1] in "aeiou" else w
    if kind == "swap" and len(w) > 4:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w


def add_noise(text: str, level: float, seed: int) -> str:
    """level = probability that a word gets one error; words are also occasionally merged or split."""
    rng = random.Random(seed)
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        r = rng.random()
        if r < level:
            w = noisy_word(w, rng)
        if rng.random() < level * 0.15 and i + 1 < len(words) and len(w) + len(words[i + 1]) < 12:
            w = w + words[i + 1]
            i += 1
        out.append(w)
        i += 1
    return " ".join(out)
