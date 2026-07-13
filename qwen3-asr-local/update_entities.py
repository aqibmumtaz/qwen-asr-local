#!/usr/bin/env python3
"""
Extend data/entities.json from a classification of the lexicon's canonicals.

Names, places and organisations get capitalised in the lexicon; ordinary Urdu and
English words stay lowercase. Capitalising a common word (aasman, chahiye) would
corrupt real transcripts, so anything not confidently a proper noun stays a word.

Input : a JSON file  {"PERSON": [...], "PLACE": [...], "ORG": [...], "WORD": [...]}
        (or a directory of such files, which are merged)

Usage:
    python3 update_entities.py /tmp/classified.json           # dry-run
    python3 update_entities.py /tmp/classified.json --write   # update entities.json
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ENTITIES = SCRIPT_DIR / "data" / "entities.json"
CLEAN = SCRIPT_DIR / "data" / "lexicons_clean.json"

# Terms that must NEVER be capitalised even if a classifier says PERSON/PLACE/ORG.
# These are ordinary words that happen to look name-like. Capitalising them would
# rewrite correct transcript words.
NEVER_CAPITALISE = {
    "acha", "aasan", "aasman", "apna", "apni", "bhai", "biwi", "chacha", "chahiye",
    "darwaza", "dekhiye", "dijiye", "farmaiye", "farq", "farz", "gaye", "gehra",
    "hamara", "hamesha", "kahiye", "keliye", "kitna", "kitni", "kuch", "larka",
    "larki", "lijiye", "maujood", "mazeed", "musalsal", "nahi", "okay", "online",
    "pareshan", "rabta", "rasta", "shukriya", "suniye", "waqt", "andaza", "awaz",
    "email", "faisla", "intezar", "jaiye", "konsa", "meharbani", "aayenge",
    "ultrasound", "echocardiography", "gynaecologist", "gynaeclogist", "clinics",
    "clinic", "timings", "charges", "levels", "tests", "located", "centre",
    "baseline", "gastro", "integrated", "changed", "names", "numainda", "padhai",
    "shanakht", "mithai", "biryani", "pulao", "machhli", "mitha", "narangi",
    "phool", "gulab", "dil", "kher", "ehsan", "sehat", "illaj", "med", "govt",
    "stat", "upto", "opp", "ip", "ul", "uz", "ch", "gh", "lt", "roa",
    "pre-employment", "pre-marriage", "medicare", "serivce", "serivces", "civel",
    "brewary", "queens", "girls", "family", "university", "vaccination", "vaccine",
    "diagnostic", "examination", "cardiology", "international", "civic", "cavalry",
    "bodevolution", "essing", "iving", "iver", "arit", "akchah", "hadded", "flp",
    "halar", "khadda", "mahl", "mahma", "chutta", "dhora", "nala", "pacha",
    "sanda", "tek", "roa", "ugs", "pak", "isb", "khi", "afb", "kda", "uet", "ajk",
}


def load_classification(path: Path) -> dict:
    """Accept a single JSON file or a directory of them; merge."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    merged = {"PERSON": [], "PLACE": [], "ORG": [], "WORD": []}
    for f in files:
        d = json.loads(f.read_text())
        for k in merged:
            merged[k].extend(d.get(k, []))
    return {k: sorted(set(v)) for k, v in merged.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("classification", help="JSON file or dir of classification files")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cls = load_classification(Path(args.classification))
    ent = json.loads(ENTITIES.read_text(encoding="utf-8"))

    print("=" * 70)
    print("  EXTEND GAZETTEER — capitalise names / places / orgs")
    print("=" * 70)
    for k in ("PERSON", "PLACE", "ORG", "WORD"):
        print(f"  {k:<8} {len(cls[k]):>4}")
    print()

    # apply the safety blocklist
    blocked = []
    for cat in ("PERSON", "PLACE", "ORG"):
        keep = []
        for t in cls[cat]:
            if t.lower() in NEVER_CAPITALISE:
                blocked.append(t)
            else:
                keep.append(t)
        cls[cat] = keep
    if blocked:
        print(f"  BLOCKED from capitalisation ({len(blocked)}) — ordinary words:")
        print(f"    {', '.join(sorted(blocked))}")
        print()

    before = {
        "given_names": len(ent["given_names"]),
        "places": len(ent["places"]),
        "organisations": len(ent["organisations"]),
    }
    ent["given_names"] = sorted(set(ent["given_names"]) | set(cls["PERSON"]))
    ent["places"] = sorted(set(ent["places"]) | set(cls["PLACE"]))
    ent["organisations"] = sorted(set(ent["organisations"]) | set(cls["ORG"]))

    print("  GAZETTEER GROWTH")
    for k in ("given_names", "places", "organisations"):
        print(f"    {k:<15} {before[k]:>4} -> {len(ent[k]):>4}  (+{len(ent[k]) - before[k]})")
    total = sum(len(ent[k]) for k in ("given_names", "places", "organisations", "religious"))
    print(f"    {'TOTAL':<15} {total:>4} entities")
    print()

    if args.write:
        ENTITIES.write_text(json.dumps(ent, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ written: {ENTITIES}")
        print("  Now run: python3 clean_lexicon.py --write && python3 verify_lexicon.py")
    else:
        print("  (dry run — pass --write to update entities.json)")


if __name__ == "__main__":
    main()
