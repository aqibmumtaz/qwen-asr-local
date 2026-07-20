"""
Extend the canonical set with NEW entity terms — the maintainability tool.

Add names / places / orgs (in production: from your Excel gazetteer) as canonicals
WITHOUT retraining. The model encodes each into its index (weights unchanged) so it
recovers their garbled spellings immediately.

SAFETY: each candidate is validated on the 80-call set. A term that causes a WRONG
capture (a garble maps to it but the gold says otherwise) is AMBIGUOUS and dropped.

Outputs:
  - data/lexicons_v22.json    v2 + the safe new canonicals (garbles found become variants)
  - models/phonetic_contrastive_v1.pt  updated in place with the extended index (--save-index)
  - a report: which terms added / dropped / captured, and accuracy before vs after

  python -m phonetic_contrastive_model.extend_canonicals --terms new_entities.txt
  python -m phonetic_contrastive_model.extend_canonicals --terms new_entities.txt --save-index
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "testing"))

import openpyxl  # noqa: E402
from test_accuracy import diff_words, normalize_tokens  # noqa: E402
from phonetic_contrastive_model.corrector import PhoneticContrastiveCorrector, CKPT  # noqa: E402

DATA = ROOT / "data"
XLSX = ROOT / "testing" / "lab_test_80_calls_urdu_roman_urdu.xlsx"


def load_calls():
    wb = openpyxl.load_workbook(str(XLSX)); ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True)); idx = {h: i for i, h in enumerate(rows[0])}
    calls = defaultdict(list)
    for r in rows[1:]:
        calls[r[idx["call_id"]]].append(r)
    for cid in calls:
        calls[cid].sort(key=lambda x: x[idx["chunk_index"]] or 0)
    out = {}
    for cid, cr in calls.items():
        b = next((x[idx["benchmark_roman_urdu"]] for x in cr
                  if isinstance(x[idx["benchmark_roman_urdu"]], str)
                  and x[idx["benchmark_roman_urdu"]].strip()), None)
        if not b:
            continue
        hindi = " ".join(str(x[idx["model_output_hindi"]]).strip() for x in cr
                         if isinstance(x[idx["model_output_hindi"]], str)
                         and x[idx["model_output_hindi"]].strip())
        out[cid] = (b, hindi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", type=Path, required=True,
                    help="text file of new canonical terms, one per line")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--out-lexicon", type=Path, default=DATA / "lexicons_v22.json")
    ap.add_argument("--save-index", action="store_true",
                    help="persist the extended index into the model checkpoint")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the ambiguity check (add all terms)")
    args = ap.parse_args()

    import os
    os.environ["LEXICON"] = "v2"; os.environ["RESOLVER"] = "0"; os.environ["PHONETIC"] = ""
    import hindi_to_roman_urdu as H
    import importlib; importlib.reload(H)

    corr = PhoneticContrastiveCorrector.load(threshold=args.threshold)
    terms = [t.strip() for t in args.terms.read_text(encoding="utf-8").splitlines() if t.strip()]
    terms = [t for t in terms if t.lower() not in corr._known]     # skip already-known
    data = load_calls()

    def transliterate_all():
        return {cid: corr.resolve_text(H.transliterate(h)) if corr else H.transliterate(h)
                for cid, (b, h) in data.items()}

    def score(outs):
        M = T = 0
        for cid, (b, h) in data.items():
            d = diff_words(b, outs[cid]); M += d.matched; T += d.total
        return 100 * M / T

    # baseline output = v2 + model on misses.  We emulate "on misses" via the pipeline:
    os.environ["PHONETIC"] = "1"; os.environ["PHONETIC_THRESHOLD"] = str(args.threshold)
    importlib.reload(H)
    # point the pipeline's corrector at OUR instance so add_canonical is visible
    H._PHONETIC = corr
    before = {cid: H.transliterate(h) for cid, (b, h) in data.items()}
    base_acc = score(before)

    # add candidates, then find which cause a WRONG capture -> ambiguous -> drop
    for t in terms:
        corr.add_canonical(t)
    after = {cid: H.transliterate(h) for cid, (b, h) in data.items()}
    cand_low = {t.lower(): t for t in terms}
    hits, bad, hit_ex = set(), set(), []
    for cid, (b, h) in data.items():
        bench = set(normalize_tokens(b))
        for x, y in zip(before[cid].split(), after[cid].split()):
            if x != y and y.lower() in cand_low:
                if y.lower() in bench:
                    hits.add(y); hit_ex.append(f"{x}->{y}")
                else:
                    bad.add(y)

    safe = [t for t in terms if t not in bad] if not args.no_validate else terms

    # rebuild corrector index with only the SAFE terms (drop ambiguous)
    corr2 = PhoneticContrastiveCorrector.load(threshold=args.threshold)
    for t in safe:
        corr2.add_canonical(t)
    H._PHONETIC = corr2
    after_safe = {cid: H.transliterate(h) for cid, (b, h) in data.items()}
    safe_acc = score(after_safe)

    print("=" * 68)
    print("  EXTEND CANONICALS — maintainability (add terms, no retraining)")
    print("=" * 68)
    print(f"  candidate terms                 : {len(terms)}")
    print(f"  dropped as ambiguous (wrong cap): {len(bad)}  {sorted(bad)}")
    print(f"  ADDED (safe)                    : {len(safe)}")
    print(f"  captured correctly (HITs)       : {sorted(hits - bad)}")
    print(f"  HIT examples (garble->canonical): {hit_ex[:12]}")
    print()
    print(f"  accuracy  base {base_acc:.2f}%  ->  +safe terms {safe_acc:.2f}%  ({safe_acc-base_acc:+.2f})")
    print()

    # write v2.2 lexicon = v2 + safe terms as CANONICALS ONLY (no variants).
    # Deliberately NOT folding the captured garbles in as exact variants — that would
    # be re-enumerating (the thing the model exists to avoid) and would overfit this
    # benchmark. The extended MODEL index recovers the garbles by generalisation.
    raw = json.loads((DATA / "lexicons_v2.json").read_text(encoding="utf-8"))
    lex = raw["lexicons"]["lexicon"]
    added = 0
    for t in safe:
        if t not in lex:
            lex[t] = []                       # canonical only; model generalises variants
            added += 1
    raw["_comment"] = ("v2.2 = v2 + entity CANONICALS (no variants) added via "
                       "extend_canonicals. Deploy with PHONETIC=1: the model recovers "
                       "their garbled spellings from the extended index, no retraining.")
    args.out_lexicon.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  written: {args.out_lexicon.name}  (+{added} canonicals, no variants)")

    if args.save_index:
        ck = torch.load(CKPT, map_location="cpu")
        ck["canonicals"] = corr2.canonicals
        ck["canonical_embeddings"] = corr2.index.cpu()
        torch.save(ck, CKPT)
        print(f"  checkpoint index extended -> {CKPT.name} ({len(corr2.canonicals)} canonicals)")


if __name__ == "__main__":
    main()
