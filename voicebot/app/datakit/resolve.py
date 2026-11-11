"""Entity resolver: spoken text -> ranked candidates from a data list, with ambiguity handling.

Public API (kept stable; the inside can change later):
    DataKit.resolve(entity, text, context=None) -> Resolution
    DataKit.get(entity, id) -> dict | None
    DataKit.facts(entity, id, fields) -> dict
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from app.config import Settings, get_settings
from app.datakit.loader import EntityConfig, Record, load_config, load_records
from app.nlp.normalizers import clean_text

_GLOBAL_STOP = {"wala", "wali", "wale", "waala", "ji", "to", "toh", "wo", "woh", "ye", "yeh", "mera", "meri", "mere", "main", "mein", "ko", "se",
                "the", "of", "for", "is", "my"}   # note: single letters such as "a" are kept ("hepatitis a", "vitamin a")
_VOWELS = re.compile(r"[aeiouy]")
_SPLIT = re.compile(r"\b(?:aur|and|plus|tatha)\b|,|;|\+|/|\bya\b|\bor\b")


def _skeleton(s: str) -> str:
    s = re.sub(r"(.)\1+", r"\1", s)
    s = s.replace("ph", "f").replace("sh", "s").replace("kh", "k").replace("ch", "c").replace("q", "k")
    s = s.replace("w", "v").replace("z", "s")
    return s[:1] + _VOWELS.sub("", s[1:])


@dataclass
class Candidate:
    id: str
    name: str
    score: float
    record: dict[str, Any]
    matched: str = ""


@dataclass
class Resolution:
    status: str                              # resolved | confirm | ambiguous | none
    candidates: list[Candidate] = field(default_factory=list)
    query: str = ""
    entity: str = ""
    more: bool = False                       # more near-matches than max_choices

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


class _Entity:
    def __init__(self, cfg: EntityConfig, records: list[Record], settings: Settings):
        self.cfg = cfg
        self.records = records
        self.by_id = {r.id: r for r in records}
        th = cfg.thresholds
        self.accept = float(th.get("accept", settings.ent_accept))
        self.gap = float(th.get("gap", settings.ent_gap))
        self.confirm = float(th.get("confirm", settings.ent_confirm))
        self.max_choices = int(th.get("max_choices", settings.ent_max_choices))
        self.max_rounds = int(th.get("max_rounds", settings.ent_max_rounds))
        self.stop = _GLOBAL_STOP | cfg.stopwords
        self._canon_names: list[tuple[Record, str, list[str], str]] = []
        for r in records:
            for n in r.names:
                variants = {self.canon(n, strip_stop=False), self.canon(n, strip_stop=True)}
                for c in variants:
                    if c:
                        self._canon_names.append((r, n, c.split(), _skeleton(c.replace(" ", ""))))

    def canon(self, text: str, strip_stop: bool = True, fallback: bool = True) -> str:
        t = clean_text(text.replace("-", " ").replace("/", " "))
        toks = t.split()
        if strip_stop:
            kept = [w for w in toks if w not in self.stop]
            toks = kept or (toks if fallback else [])
        return " ".join(toks)

    def score(self, q: str, qt: list[str], qs: str, ct: list[str], cs: str) -> float:
        c = " ".join(ct)
        if q == c:
            return 1.0
        qset, cset = set(qt), set(ct)
        if qset and qset <= cset:
            return max(0.0, 0.97 - 0.03 * min(len(ct) - len(qt), 4))
        if cset and cset <= qset:
            # alias appears inside a longer utterance (e.g. an address)
            return 0.96 - 0.01 * min(len(qt) - len(ct), 4)
        n = len(ct)
        best = 0.0
        if len(qt) > n:
            for i in range(0, len(qt) - n + 1):
                w = " ".join(qt[i:i + n])
                best = max(best, fuzz.ratio(w, c) / 100.0)
            best *= 0.95
        s = max(
            fuzz.token_sort_ratio(q, c) / 100.0,
            fuzz.ratio(q.replace(" ", ""), c.replace(" ", "")) / 100.0,
            (fuzz.ratio(qs, cs) / 100.0) * 0.92 if len(qs) >= 4 and len(cs) >= 4 else 0.0,
        ) * 0.95
        return max(best, s)

    def rank(self, text: str) -> list[Candidate]:
        q = self.canon(text, fallback=False)
        if len(q.replace(" ", "")) < 2:
            return []
        qt = q.split()
        qs = _skeleton(q.replace(" ", ""))
        best: dict[str, Candidate] = {}
        # variant without bare numbers / single letters (house numbers, blocks) so addresses still match
        qt2 = [w for w in qt if not w.isdigit()]
        q2 = " ".join(qt2)
        qs2 = _skeleton(q2.replace(" ", "")) if q2 else ""
        for r, n, ct, cs in self._canon_names:
            s = self.score(q, qt, qs, ct, cs)
            if q2 and q2 != q and len(q2.replace(" ", "")) >= 2:
                s = max(s, self.score(q2, qt2, qs2, ct, cs))
            cur = best.get(r.id)
            if cur is None or s > cur.score:
                best[r.id] = Candidate(id=r.id, name=r.name, score=round(s, 4), record=r.data, matched=n)
        return sorted(best.values(), key=lambda c: (-c.score, c.name))


class DataKit:
    def __init__(self, config_dir: Path | None = None, data_dir: Path | None = None,
                 settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.config_dir = config_dir or self.settings.entity_config_dir
        self.data_dir = data_dir or self.settings.entity_data_dir
        self._entities: dict[str, _Entity] = {}
        self.reload()

    def reload(self) -> None:
        self._entities = {}
        for p in sorted(self.config_dir.glob("*.yaml")):
            cfg = load_config(p, self.data_dir)
            self._entities[cfg.name] = _Entity(cfg, load_records(cfg), self.settings)

    # ── introspection ──
    def entities(self) -> list[str]:
        return list(self._entities)

    def config(self, entity: str) -> EntityConfig:
        return self._entities[entity].cfg

    def max_rounds(self, entity: str) -> int:
        return self._entities[entity].max_rounds

    # ── interface ──
    def resolve(self, entity: str, text: str, context: dict | None = None) -> Resolution:
        e = self._entities[entity]
        ranked = e.rank(text)
        q = e.canon(text, fallback=False)
        if not ranked or ranked[0].score < e.confirm:
            return Resolution("none", [], q, entity)
        top = ranked[0]
        exact = [c for c in ranked if c.score >= 0.999]
        if len(exact) == 1:
            return Resolution("resolved", [top], q, entity)
        near = [c for c in ranked if c.score >= e.confirm and top.score - c.score < e.gap]
        if len(near) == 1:
            status = "resolved" if top.score >= e.accept else "confirm"
            return Resolution(status, [top], q, entity)
        return Resolution("ambiguous", near[:e.max_choices], q, entity, more=len(near) > e.max_choices)

    def clean_mention(self, entity: str, text: str) -> str:
        return self._entities[entity].canon(text, fallback=False)

    def resolve_spelled(self, entity: str, text: str) -> Resolution:
        """Letters spoken one by one ('c b c', 'e s r') are joined before searching."""
        toks = clean_text(text).split()
        joined = "".join(t for t in toks if len(t) == 1) if toks and all(len(t) == 1 for t in toks) else text
        return self.resolve(entity, joined)

    def split_mentions(self, entity: str, text: str) -> list[str]:
        parts = [p.strip() for p in _SPLIT.split(clean_text(text) if False else text.lower()) if p and p.strip()]
        return [p for p in parts if p not in ("aur", "and", "plus")]

    def resolve_many(self, entity: str, text: str) -> list[Resolution]:
        out = []
        for part in self.split_mentions(entity, text):
            e = self._entities[entity]
            if len(e.canon(part, fallback=False).replace(" ", "")) >= 2:
                out.append(self.resolve(entity, part))
        return out

    def get(self, entity: str, id_: str) -> dict | None:
        r = self._entities[entity].by_id.get(id_)
        return dict(r.data) if r else None

    def facts(self, entity: str, id_: str, fields: list[str] | None = None) -> dict:
        r = self._entities[entity].by_id.get(id_)
        if not r:
            return {}
        fields = fields or self._entities[entity].cfg.facts
        return {f: r.data.get(f, "") for f in fields}

    def describe(self, entity: str, cand: Candidate, others: list[Candidate]) -> str:
        """Spoken label for a candidate that tells it apart from the others."""
        cfg = self._entities[entity].cfg
        label = cand.name
        for attr in cfg.disambiguate_by:
            vals = {o.record.get(attr, "") for o in others + [cand]}
            if len(vals) > 1 and cand.record.get(attr):
                label += f" ({cand.record[attr]})"
                break
        return label
