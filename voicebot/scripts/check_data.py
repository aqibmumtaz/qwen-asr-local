#!/usr/bin/env python3
"""Validate data lists: duplicate ids, empty names, aliases pointing to several ids, near-duplicate names.

Usage: python scripts/check_data.py [entity ...]    (exit code 1 if errors)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz import fuzz  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.datakit.loader import load_config, load_records  # noqa: E402
from app.nlp.normalizers import clean_text  # noqa: E402


def check(path: Path, data_dir: Path) -> tuple[list[str], list[str]]:
    cfg = load_config(path, data_dir)
    errors: list[str] = []
    warns: list[str] = []
    if not cfg.source_path.exists():
        return [f"{cfg.name}: data file missing: {cfg.source_path}"], []
    recs = load_records(cfg)
    if not recs:
        errors.append(f"{cfg.name}: no records")
    seen_ids: dict[str, int] = defaultdict(int)
    alias_owner: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        seen_ids[r.id] += 1
        for n in r.names:
            alias_owner[clean_text(n)].add(r.id)
        if len(r.names) == 1:
            warns.append(f"{cfg.name}: {r.id} '{r.name}' has no aliases (matching will be weaker)")
    errors += [f"{cfg.name}: duplicate id {i}" for i, c in seen_ids.items() if c > 1]
    for a, ids in alias_owner.items():
        if len(ids) > 1:
            errors.append(f"{cfg.name}: alias '{a}' points to several ids {sorted(ids)}")
    names = [(r.id, clean_text(r.name)) for r in recs]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if fuzz.ratio(names[i][1], names[j][1]) >= 96:
                warns.append(f"{cfg.name}: near-duplicate names {names[i][0]} / {names[j][0]}: '{names[i][1]}' ~ '{names[j][1]}'")
    for fld in cfg.facts:
        if recs and all(fld not in r.data for r in recs):
            errors.append(f"{cfg.name}: fact column '{fld}' not in data")
    return errors, warns


def check_faq() -> int:
    from app.faq import FaqBook
    fb = FaqBook()
    issues = fb.check()
    print(f"[faq] {len(fb.entries)} approved entries, {len(issues)} issues, {fb.sample_count()} marked sample")
    for i in issues:
        print("  ERROR", i)
    if fb.sample_count():
        print("  warn  sample FAQ content in use: replace with approved answers before real use")
    return 1 if issues else 0


def main(argv: list[str]) -> int:
    s = get_settings()
    wanted = set(argv)
    rc = 0
    for p in sorted(s.entity_config_dir.glob("*.yaml")):
        if wanted and p.stem not in wanted:
            continue
        errs, warns = check(p, s.entity_data_dir)
        print(f"[{p.stem}] {len(errs)} errors, {len(warns)} warnings")
        for e in errs:
            print("  ERROR", e)
        for w in warns:
            print("  warn ", w)
        rc = rc or (1 if errs else 0)
    if not wanted or "faq" in wanted:
        rc = rc or check_faq()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
