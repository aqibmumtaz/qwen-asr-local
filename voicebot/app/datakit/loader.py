"""Read entity configs (config/entities/*.yaml) and their data files into records."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:  # Excel is optional
    import openpyxl
except Exception:  # pragma: no cover
    openpyxl = None


@dataclass
class EntityConfig:
    name: str
    source_path: Path
    id_field: str
    display_field: str
    alias_fields: list[str] = field(default_factory=list)
    list_separator: str = "|"
    facts: list[str] = field(default_factory=list)
    disambiguate_by: list[str] = field(default_factory=list)
    stopwords: set[str] = field(default_factory=set)
    thresholds: dict[str, float] = field(default_factory=dict)
    confirm_before_use: bool = False
    spell_fallback: bool = True


@dataclass
class Record:
    id: str
    name: str
    names: list[str]            # display name + aliases
    data: dict[str, Any]


def load_config(path: Path, data_dir: Path) -> EntityConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    src = raw.get("source", {})
    return EntityConfig(
        name=raw["name"],
        source_path=(data_dir / src["path"]) if not Path(src["path"]).is_absolute() else Path(src["path"]),
        id_field=raw["id_field"],
        display_field=raw["display_field"],
        alias_fields=list(raw.get("alias_fields", [])),
        list_separator=raw.get("list_separator", "|"),
        facts=list(raw.get("facts", [])),
        disambiguate_by=list(raw.get("disambiguate_by", [])),
        stopwords=set(raw.get("stopwords", [])),
        thresholds=dict(raw.get("thresholds", {})),
        confirm_before_use=bool(raw.get("confirm_before_use", False)),
        spell_fallback=bool(raw.get("spell_fallback", True)),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        if openpyxl is None:
            raise RuntimeError("openpyxl is required for Excel lists")
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
        rows = list(ws.iter_rows(values_only=True))
        header = [str(h).strip() for h in rows[0]]
        return [{h: ("" if v is None else str(v).strip()) for h, v in zip(header, r)} for r in rows[1:] if any(r)]
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


def load_records(cfg: EntityConfig) -> list[Record]:
    out: list[Record] = []
    for row in _read_rows(cfg.source_path):
        rid = row.get(cfg.id_field, "").strip()
        name = row.get(cfg.display_field, "").strip()
        if not rid or not name:
            continue
        names = [name]
        for fld in cfg.alias_fields:
            for a in row.get(fld, "").split(cfg.list_separator):
                a = a.strip()
                if a:
                    names.append(a)
        out.append(Record(id=rid, name=name, names=names, data=row))
    return out
