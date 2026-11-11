"""FAQ layer: match a caller question to an approved FAQ and speak its stored answer word for word.

Two thresholds: high score + clear gap -> answer; middle -> ask to confirm; low -> no match.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from app.config import Settings, get_settings
from app.nlp.normalizers import clean_text

_STOP = {"ka", "ki", "ke", "hai", "hain", "kya", "kitna", "kitne", "kitni", "mujhe", "aap", "ap", "main", "mein",
         "bata", "batao", "dein", "do", "please", "se", "ko", "to", "toh"}


def _canon(s: str) -> str:
    return " ".join(w for w in clean_text(s).split() if w not in _STOP) or clean_text(s)


@dataclass
class FaqMatch:
    status: str                  # answer | confirm | none
    entry: dict | None = None
    score: float = 0.0
    second: float = 0.0
    text: str = ""               # rendered answer


class FaqBook:
    def __init__(self, settings: Settings | None = None, path: Path | None = None):
        self.s = settings or get_settings()
        self.path = path or self.s.faq_file
        self.answer_min = 0.72
        self.confirm_min = 0.55
        self.gap = 0.10
        self.entries: list[dict] = []
        self.pricing: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self.pricing = {}
        pf = self.s.entity_data_dir / "pricing.csv"
        if pf.exists():
            with pf.open(newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    self.pricing[r["key"]] = r["value"]
        self.entries = []
        if self.path.exists():
            for e in yaml.safe_load(self.path.read_text(encoding="utf-8")) or []:
                if e.get("status") == "approved":
                    e["_canon"] = [_canon(q) for q in e.get("questions", [])]
                    self.entries.append(e)

    def render(self, entry: dict) -> str:
        vals = {}
        for k, ref in (entry.get("data") or {}).items():
            src, _, key = str(ref).partition(".")
            vals[k] = self.pricing.get(key, "") if src == "pricing" else ""
        try:
            return entry["answer"].format(**vals)
        except (KeyError, IndexError):
            return entry["answer"]

    def _idf(self) -> dict[str, float]:
        import math
        if getattr(self, "_idf_cache", None) is not None and self._idf_n == len(self.entries):
            return self._idf_cache
        df: dict[str, int] = {}
        for e in self.entries:
            seen = set()
            for c in e["_canon"]:
                seen |= set(c.split())
            for t in seen:
                df[t] = df.get(t, 0) + 1
        n = max(len(self.entries), 1)
        self._idf_cache = {t: math.log(1 + n / d) for t, d in df.items()}
        self._idf_n = len(self.entries)
        return self._idf_cache

    @staticmethod
    def _tok_match(t: str, pool: list[str]) -> bool:
        return any(t == p or (len(t) > 3 and fuzz.ratio(t, p) >= 84) for p in pool)

    def _score(self, q: str, cq: str) -> float:
        """Weighted token F1 between the utterance and one FAQ phrasing (rare tokens count more)."""
        idf = self._idf()
        qt, ct = q.split(), cq.split()
        if not qt or not ct:
            return 0.0
        w = lambda t: idf.get(t, 1.0)
        hit_q = sum(w(t) for t in qt if self._tok_match(t, ct))
        hit_c = sum(w(t) for t in ct if self._tok_match(t, qt))
        tot_q = sum(w(t) for t in qt)
        tot_c = sum(w(t) for t in ct)
        p, r = hit_q / tot_q, hit_c / tot_c
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    def match(self, text: str) -> FaqMatch:
        q = _canon(text)
        if len(q) < 3:
            return FaqMatch("none")
        scored = []
        for e in self.entries:
            best = max((self._score(q, c) for c in e["_canon"]), default=0.0)
            scored.append((best, e))
        scored.sort(key=lambda x: -x[0])
        if not scored or scored[0][0] < self.confirm_min:
            return FaqMatch("none")
        top, entry = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if top >= self.answer_min and top - second >= self.gap:
            return FaqMatch("answer", entry, top, second, self.render(entry))
        return FaqMatch("confirm", entry, top, second, self.render(entry))

    def check(self) -> list[str]:
        """Load-time checks: long answers, near-duplicate questions across FAQs, unknown placeholders."""
        issues = []
        import re as _re
        for e in self.entries:
            for ph in _re.findall(r"{(\w+)}", e["answer"]):
                src = (e.get("data") or {}).get(ph)
                if not src:
                    issues.append(f"{e['id']}: placeholder {{{ph}}} has no data source")
                elif str(src).startswith("pricing.") and str(src).split(".", 1)[1] not in self.pricing:
                    issues.append(f"{e['id']}: {src} not found in data/pricing.csv")
        for e in self.entries:
            if len([x for x in e["answer"].split(".") if x.strip()]) > 3:
                issues.append(f"{e['id']}: answer longer than two spoken sentences")
        for i, a in enumerate(self.entries):
            for b in self.entries[i + 1:]:
                for qa in a["_canon"]:
                    for qb in b["_canon"]:
                        if fuzz.ratio(qa, qb) >= 92:
                            issues.append(f"{a['id']} / {b['id']}: near-duplicate questions '{qa}' ~ '{qb}'")
        return issues


    def sample_count(self) -> int:
        return sum(1 for e in self.entries if e.get("sample"))

    def entry_by_id(self, id_: str) -> dict | None:
        return next((e for e in self.entries if e["id"] == id_), None)
