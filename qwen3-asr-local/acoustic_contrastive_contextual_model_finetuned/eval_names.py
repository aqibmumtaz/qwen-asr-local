#!/usr/bin/env python3
"""
Step 4 -- evaluate a trained adapter: name-recall metrics AND the general-WER
regression gate. STOP before Step 5/6 if the regression gate fails (see
plan §"Runtime Design" -- this is the safety check for entity-span words
specifically; non-entity words are protected structurally by
splice_inference.py, not by this gate, but the gate still matters for the
words the adapter IS trusted to touch).

*** NOT RUN -- needs a trained adapter, which needs a GPU. Written from the
plan spec's Evaluation section. Uses the same wrapper/build_prompt/build_model
helpers as train_lora.py (imported from it) rather than duplicating them. ***

Usage:
  python eval_names.py --adapter adapters/run1/phase3 --eval data/eval.jsonl --adversarial data/adversarial.jsonl
  python eval_names.py --adapter adapters/run1/phase3 --eval data/eval.jsonl --adversarial data/adversarial.jsonl --base-only
    (--base-only: skip adapter, evaluate base model alone -- for computing
    the "before" row in the results table)
"""
from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from train_lora import build_model, build_prompt, resume_lora, MODEL_ID


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def transcribe_one(wrapper, audio_path: str, context: str, device) -> str:
    """Single-example inference through the loaded wrapper -- mirrors
    qwen_asr's own _infer_asr_transformers path (build prompt, run
    processor+generate, decode) since Qwen3ASRModel.transcribe() doesn't
    expose a way to force a specific already-loaded adapter state cleanly
    for A/B (base vs adapter) comparison in one process."""
    import soundfile as sf

    audio, sr = sf.read(str(ROOT / audio_path if not Path(audio_path).is_absolute() else audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    prompt = build_prompt(wrapper, context)
    inputs = wrapper.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out_ids = wrapper.model.generate(**inputs, max_new_tokens=256)
    decoded = wrapper.processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    return decoded[0].strip()


def word_sim(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def contains_fuzzy(text: str, target: str, threshold: float = 0.85) -> bool:
    words = text.lower().split()
    return any(word_sim(w, target.lower()) >= threshold for w in words)


def evaluate(wrapper, device, eval_examples: list[dict], adversarial_examples: list[dict]):
    name_recall_no_ctx = [0, 0]     # [hit, total] -- eval examples w/ a name, context=""
    name_recall_with_ctx = [0, 0]   # same examples, context=correct name(s)
    context_fp = [0, 0]             # adversarial: [decoy_appeared, total]
    general_wer_matched = [0, 0]    # non-entity words only, eval set

    for ex in eval_examples:
        if ex["type"] != "positive":
            continue
        names = [n.strip() for n in ex["context"].split(",") if n.strip()]
        if not names:
            continue

        out_no_ctx = transcribe_one(wrapper, ex["audio_path"], "", device)
        out_with_ctx = transcribe_one(wrapper, ex["audio_path"], ex["context"], device)

        for name in names:
            name_recall_no_ctx[1] += 1
            if contains_fuzzy(out_no_ctx, name):
                name_recall_no_ctx[0] += 1
            name_recall_with_ctx[1] += 1
            if contains_fuzzy(out_with_ctx, name):
                name_recall_with_ctx[0] += 1

    for ex in adversarial_examples:
        decoy = ex["context"]
        out = transcribe_one(wrapper, ex["audio_path"], decoy, device)
        context_fp[1] += 1
        if contains_fuzzy(out, decoy):
            context_fp[0] += 1

    # general WER: negative eval examples (no names) -- entire word set is
    # "non-entity" by construction, so this whole comparison is the gate
    from test_accuracy import diff_words
    for ex in eval_examples:
        if ex["type"] != "negative":
            continue
        out = transcribe_one(wrapper, ex["audio_path"], "", device)
        gold = ex["target_hindi"]  # approximate -- vendor Hindi used as
                                     # reference here, not a true independent
                                     # gold transcript for this metric
        d = diff_words(gold, out)
        general_wer_matched[0] += d.matched
        general_wer_matched[1] += d.total

    def pct(pair):
        return round(100 * pair[0] / pair[1], 2) if pair[1] else None

    return {
        "name_recall_no_context": pct(name_recall_no_ctx),
        "name_recall_with_context": pct(name_recall_with_ctx),
        "context_fp_rate": pct(context_fp),
        "general_accuracy_negative_examples": pct(general_wer_matched),
        "_raw": {"name_recall_no_ctx": name_recall_no_ctx, "name_recall_with_ctx": name_recall_with_ctx,
                 "context_fp": context_fp, "general_wer": general_wer_matched},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval", default=str(HERE / "data" / "eval.jsonl"))
    ap.add_argument("--adversarial", default=str(HERE / "data" / "adversarial.jsonl"))
    ap.add_argument("--base-only", action="store_true")
    args = ap.parse_args()

    eval_examples = load_jsonl(Path(args.eval))
    adversarial_examples = load_jsonl(Path(args.adversarial))

    wrapper, device = build_model()
    if not args.base_only:
        if not args.adapter:
            print("ERROR: --adapter required unless --base-only", file=sys.stderr)
            sys.exit(1)
        wrapper = resume_lora(wrapper, args.adapter)
    wrapper.model.eval()

    label = "BASE MODEL (no adapter)" if args.base_only else f"ADAPTER: {args.adapter}"
    print(f"=== Evaluating {label} ===", flush=True)
    results = evaluate(wrapper, device, eval_examples, adversarial_examples)

    print()
    for k, v in results.items():
        if k == "_raw":
            continue
        print(f"  {k}: {v}%")
    print()
    print("Targets (from plan): name_recall_no_context 70%+, "
          "name_recall_with_context 85%+, context_fp_rate <5%, "
          "general_accuracy no degradation vs base-only run")
    print()
    print("REGRESSION GATE: compare general_accuracy_negative_examples against")
    print("a --base-only run on the same eval set. Do NOT proceed to")
    print("splice_inference.py / benchmark integration if this dropped.")


if __name__ == "__main__":
    main()
