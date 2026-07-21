"""
Word-level diff between two Roman Urdu strings.

Copied (lightly adapted) from batch_transcription/evaluation/accuracy.py so the
lab-test pipeline doesn't depend on importing the heavier package. Same
semantics: exact alignment first, then a fuzzy pass (char-similarity ≥ 0.70)
inside replace/insert/delete spans to absorb romanizer spelling drift.

Returns matched_count, total_benchmark_count, leftover unmatched-from-hypothesis
words, and leftover missing-from-benchmark words — that latter pair is what the
LLM phonetic classifier in llm_client.py operates on.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher

_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "।॥،؛؟–—…“”‘’«»")
_FUZZY_THRESHOLD = 0.70


def normalize_tokens(text: str) -> list[str]:
    if not text:
        return []
    cleaned = text.lower().translate(_PUNCT_TABLE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.split()


@dataclass
class WordDiff:
    accuracy: float
    matched: int
    total: int
    mismatched_tokens: list[str]
    missing_tokens: list[str]


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def diff_words(benchmark: str, hypothesis: str, *,
               fuzzy: bool = True, fuzzy_threshold: float = _FUZZY_THRESHOLD) -> WordDiff:
    ref = normalize_tokens(benchmark)
    hyp = normalize_tokens(hypothesis)
    if not ref and not hyp:
        return WordDiff(100.0, 0, 0, [], [])
    if not ref:
        return WordDiff(0.0, 0, 0, hyp, [])
    if not hyp:
        return WordDiff(0.0, 0, len(ref), [], ref)

    matcher = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = 0
    matched_hyp: set[int] = set()
    matched_ref: set[int] = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
            matched_hyp.update(range(j1, j2))
            matched_ref.update(range(i1, i2))

    if fuzzy:
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            r = [k for k in range(i1, i2) if k not in matched_ref]
            h = [k for k in range(j1, j2) if k not in matched_hyp]
            if not r or not h:
                continue
            cands = sorted(
                ((_sim(ref[ri], hyp[hi]), ri, hi) for ri in r for hi in h),
                reverse=True,
            )
            for sim, ri, hi in cands:
                if sim < fuzzy_threshold:
                    break
                if ri in matched_ref or hi in matched_hyp:
                    continue
                matched += 1
                matched_ref.add(ri)
                matched_hyp.add(hi)

    return WordDiff(
        accuracy=round(matched / len(ref) * 100, 2),
        matched=matched,
        total=len(ref),
        mismatched_tokens=[hyp[k] for k in range(len(hyp)) if k not in matched_hyp],
        missing_tokens=[ref[k] for k in range(len(ref)) if k not in matched_ref],
    )
