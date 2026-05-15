#!/usr/bin/env python3
"""
Cross-backend ASR benchmark — Ubuntu + RTX 3090 + i7-14700KF target.

Runs each available backend on every audio sample, captures Hindi output +
timing + token-match against a chosen reference, writes a comparison table
+ CSV + JSON to benchmark_results/.

Backends tested (auto-detected, skipped if dependencies missing):
  1. HF vLLM            — qwen-asr[vllm] on CUDA      → app.py reference
  2. HF transformers    — qwen-asr on CUDA or CPU
  3. llama.cpp CUDA     — llama-mtmd-cli with -ngl 999
  4. llama.cpp CPU      — llama-mtmd-cli no GPU offload

Usage:
    python3 benchmark_results/benchmark_backends.py                # all samples, default ref=vllm
    python3 benchmark_results/benchmark_backends.py --reference llamacpp_cuda
    python3 benchmark_results/benchmark_backends.py --samples samples/sample_ur1.wav
    python3 benchmark_results/benchmark_backends.py --quants Q8_0,BF16

Run from project root or qwen3-asr-local/. Outputs go to:
    benchmark_results/backend_comparison.csv
    benchmark_results/backend_comparison.json
    benchmark_results/backend_comparison.md
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent          # benchmark_results/
PROJECT_DIR   = SCRIPT_DIR.parent                         # qwen3-asr-local/
TRANSCRIBE_SH = PROJECT_DIR / "transcribe.sh"
SAMPLES_DIR   = PROJECT_DIR / "samples"
LLAMA_MTMD    = PROJECT_DIR / "llama.cpp/build/bin/llama-mtmd-cli"
MODELS_DIR    = PROJECT_DIR / "models"

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
MMPROJ   = MODELS_DIR / "mmproj-Qwen3-ASR-1.7B-bf16-new.gguf"
ASR_PROMPT_TEMPLATE = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
    "language {lang}<asr_text>"
)


# ── Capability detection ─────────────────────────────────────────────────────
def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_vllm() -> bool:
    return importlib.util.find_spec("vllm") is not None


def has_qwen_asr() -> bool:
    return importlib.util.find_spec("qwen_asr") is not None


def has_llamacpp_binary() -> bool:
    return LLAMA_MTMD.is_file() and os.access(LLAMA_MTMD, os.X_OK)


def gguf_available(quant: str) -> Path | None:
    # Try -new naming first (current local convention), then plain
    for candidate in (
        MODELS_DIR / f"Qwen3-ASR-1.7B-{quant}-new.gguf",
        MODELS_DIR / f"Qwen3-ASR-1.7B-{quant}.gguf",
    ):
        if candidate.is_file():
            return candidate
    return None


# ── Token-match scoring (simple word-level for Devanagari/Latin) ─────────────
def token_match(ref: str, hyp: str) -> float:
    """Word-level match: fraction of reference words that appear in hyp."""
    if not ref:
        return 0.0
    ref_words = ref.split()
    hyp_words = set(hyp.split())
    if not ref_words:
        return 0.0
    return sum(1 for w in ref_words if w in hyp_words) / len(ref_words)


# ── Backend runners ──────────────────────────────────────────────────────────
_hf_model = None
_hf_model_kind = None  # "vllm" or "transformers"


def run_hf_vllm(audio: Path, language: str) -> tuple[str, float]:
    global _hf_model, _hf_model_kind
    from qwen_asr import Qwen3ASRModel
    if _hf_model is None or _hf_model_kind != "vllm":
        _hf_model = Qwen3ASRModel.LLM(
            model=MODEL_ID,
            gpu_memory_utilization=0.7,
            dtype="bfloat16",
            max_new_tokens=1024,
        )
        _hf_model_kind = "vllm"
    t0 = time.time()
    res = _hf_model.transcribe(audio=[str(audio)], language=[language])
    return res[0].text if res else "", time.time() - t0


def run_hf_transformers(audio: Path, language: str, device: str) -> tuple[str, float]:
    global _hf_model, _hf_model_kind
    import torch
    from qwen_asr import Qwen3ASRModel
    if _hf_model is None or _hf_model_kind != f"transformers_{device}":
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        _hf_model = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map=device,
            max_new_tokens=1024,
        )
        _hf_model_kind = f"transformers_{device}"
    t0 = time.time()
    res = _hf_model.transcribe(audio=[str(audio)], language=[language])
    return res[0].text if res else "", time.time() - t0


def run_llamacpp(audio: Path, language: str, quant: str, use_gpu: bool) -> tuple[str, float]:
    model = gguf_available(quant)
    if model is None:
        raise FileNotFoundError(f"GGUF not found for {quant}")
    cmd = [
        str(LLAMA_MTMD),
        "-m", str(model),
        "--mmproj", str(MMPROJ),
        "--image", str(audio),
        "-p", ASR_PROMPT_TEMPLATE.format(lang=language),
        "-n", "256",
        "--no-warmup",
    ]
    if use_gpu:
        cmd += ["-ngl", "999"]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    transcript = ""
    for line in result.stdout.splitlines():
        if "<asr_text>" in line:
            transcript = line.split("<asr_text>")[-1].strip()
            break
    return transcript, elapsed


# ── Audio duration helper (for RTF calc) ─────────────────────────────────────
def audio_duration(path: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.duration
    except Exception:
        return 0.0


# ── Backend registry ─────────────────────────────────────────────────────────
def build_backends(quants: list[str]) -> list[dict]:
    """Return list of available backends, each with name + runner + skip_reason."""
    cuda = has_cuda()
    backends = []

    # HF vLLM (GPU only)
    backends.append({
        "name": "hf_vllm_bf16",
        "runner": lambda a, l: run_hf_vllm(a, l),
        "skip_reason": (
            None if (has_qwen_asr() and has_vllm() and cuda)
            else "vllm not installed" if not has_vllm()
            else "qwen-asr not installed" if not has_qwen_asr()
            else "no CUDA device available"
        ),
    })

    # HF transformers — try GPU first, then CPU
    if has_qwen_asr():
        if cuda:
            backends.append({
                "name": "hf_transformers_cuda_bf16",
                "runner": lambda a, l: run_hf_transformers(a, l, "cuda:0"),
                "skip_reason": None,
            })
        backends.append({
            "name": "hf_transformers_cpu_fp32",
            "runner": lambda a, l: run_hf_transformers(a, l, "cpu"),
            "skip_reason": None,
        })
    else:
        backends.append({
            "name": "hf_transformers_*",
            "runner": None,
            "skip_reason": "qwen-asr not installed",
        })

    # llama.cpp — one entry per quant × {cuda, cpu}
    if has_llamacpp_binary():
        for quant in quants:
            for use_gpu, suffix in [(cuda, "cuda"), (False, "cpu")]:
                # Skip cuda variant if no CUDA; otherwise both variants
                if suffix == "cuda" and not cuda:
                    continue
                backends.append({
                    "name": f"llamacpp_{quant}_{suffix}",
                    "runner": (lambda a, l, q=quant, g=use_gpu: run_llamacpp(a, l, q, g)),
                    "skip_reason": (None if gguf_available(quant) else f"GGUF {quant} not found"),
                })
    else:
        backends.append({
            "name": "llamacpp_*",
            "runner": None,
            "skip_reason": "llama-mtmd-cli binary not built",
        })

    return backends


# ── Main benchmark loop ──────────────────────────────────────────────────────
def run_benchmark(samples: list[Path], language: str, quants: list[str],
                  reference: str) -> list[dict]:
    backends = build_backends(quants)

    print(f"\n{'=' * 78}")
    print(f"Backends detected ({sum(1 for b in backends if not b['skip_reason'])} runnable, "
          f"{sum(1 for b in backends if b['skip_reason'])} skipped):")
    for b in backends:
        flag = "✓" if not b["skip_reason"] else "✗"
        print(f"  {flag}  {b['name']:30}  {b['skip_reason'] or '(ready)'}")
    print(f"{'=' * 78}\n")

    runnable = [b for b in backends if not b["skip_reason"]]
    if not runnable:
        print("No backends available. Install qwen-asr / vllm or build llama.cpp.")
        sys.exit(1)

    if reference not in {b["name"] for b in runnable}:
        print(f"WARN: reference backend '{reference}' not available; "
              f"will use first runnable: {runnable[0]['name']}")
        reference = runnable[0]["name"]

    results = []
    for sample in samples:
        dur = audio_duration(sample)
        print(f"\n──── {sample.name} ({dur:.1f}s audio) ────")
        ref_text = None

        # Run all backends; reference goes first so we can compute match against it
        ordered = sorted(runnable, key=lambda b: (b["name"] != reference, b["name"]))
        for b in ordered:
            print(f"  {b['name']:30} … ", end="", flush=True)
            try:
                text, secs = b["runner"](sample, language)
            except Exception as e:
                print(f"FAIL: {e}")
                results.append({
                    "sample": sample.name,
                    "audio_sec": dur,
                    "backend": b["name"],
                    "text": "",
                    "wall_sec": 0,
                    "rtf": 0,
                    "match_vs_ref": 0,
                    "error": str(e),
                })
                continue

            if b["name"] == reference:
                ref_text = text
            match = token_match(ref_text, text) if ref_text else 1.0
            rtf = secs / dur if dur > 0 else 0

            print(f"{secs:5.1f}s  RTF={rtf:4.2f}  match={match*100:5.1f}%")
            results.append({
                "sample": sample.name,
                "audio_sec": dur,
                "backend": b["name"],
                "text": text,
                "wall_sec": round(secs, 2),
                "rtf": round(rtf, 3),
                "match_vs_ref": round(match * 100, 1),
                "error": None,
            })

    return results


# ── Output writers ───────────────────────────────────────────────────────────
def write_csv(results: list[dict], path: Path):
    import csv
    keys = ["sample", "audio_sec", "backend", "wall_sec", "rtf",
            "match_vs_ref", "text", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})


def write_json(results: list[dict], path: Path, meta: dict):
    payload = {"meta": meta, "results": results}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(results: list[dict], path: Path, meta: dict):
    lines = [
        "# ASR Backend Benchmark — Cross-backend comparison",
        "",
        f"**Run date:** {meta['timestamp']}",
        f"**Reference backend:** `{meta['reference']}`",
        f"**Language hint:** {meta['language']}",
        f"**Samples:** {meta['sample_count']}",
        "",
        "## Per-sample results",
        "",
    ]
    # Group by sample
    by_sample: dict[str, list[dict]] = {}
    for r in results:
        by_sample.setdefault(r["sample"], []).append(r)

    for sample, rows in by_sample.items():
        dur = rows[0]["audio_sec"]
        lines += [
            f"### `{sample}` ({dur:.1f}s)",
            "",
            "| Backend | Wall (s) | RTF | Match vs ref | Output (first 80 chars) |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            text = r["text"][:80] + ("…" if len(r["text"]) > 80 else "")
            if r["error"]:
                text = f"ERROR: {r['error'][:80]}"
            lines.append(
                f"| `{r['backend']}` | {r['wall_sec']} | {r['rtf']} | "
                f"{r['match_vs_ref']}% | {text} |"
            )
        lines.append("")

    # Aggregate table — avg RTF + avg match per backend
    backends = {r["backend"] for r in results}
    aggs = []
    for b in sorted(backends):
        rows = [r for r in results if r["backend"] == b and not r["error"]]
        if not rows:
            continue
        avg_rtf = sum(r["rtf"] for r in rows) / len(rows)
        avg_match = sum(r["match_vs_ref"] for r in rows) / len(rows)
        aggs.append({"backend": b, "n": len(rows),
                     "avg_rtf": avg_rtf, "avg_match": avg_match})
    aggs.sort(key=lambda x: -x["avg_match"])
    lines += [
        "## Aggregate (averaged across all samples)",
        "",
        "| Backend | N | Avg RTF | Avg Match % |",
        "|---|---|---|---|",
    ]
    for a in aggs:
        lines.append(f"| `{a['backend']}` | {a['n']} | {a['avg_rtf']:.2f} | "
                     f"{a['avg_match']:.1f}% |")
    lines += [
        "",
        "**Lower RTF = faster** (1.0 = real-time, 0.5 = 2× faster than real-time).",
        f"**Match % is word-level overlap with the reference (`{meta['reference']}`).**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--samples", nargs="*",
                   help="Specific audio files. Default: all samples/*.wav")
    p.add_argument("--language", default="English",
                   help="Language hint (default: English — empirically gives "
                        "better nukta-emission than 'Hindi' on Mac CPU; "
                        "see ard/word-level-confidence.md §6.1)")
    p.add_argument("--reference", default="hf_vllm_bf16",
                   help="Reference backend for match% scoring "
                        "(default: hf_vllm_bf16)")
    p.add_argument("--quants", default="Q8_0,BF16",
                   help="Comma-separated GGUF quants to test "
                        "(default: Q8_0,BF16)")
    args = p.parse_args()

    if args.samples:
        samples = [Path(s) for s in args.samples if Path(s).exists()]
    else:
        samples = sorted(SAMPLES_DIR.glob("*.wav"))
    if not samples:
        print(f"No audio found in {SAMPLES_DIR}")
        sys.exit(1)

    quants = [q.strip() for q in args.quants.split(",") if q.strip()]

    print(f"Samples ({len(samples)}):")
    for s in samples:
        print(f"  {s}")
    print(f"Quants: {quants}")
    print(f"Reference: {args.reference}")

    results = run_benchmark(samples, args.language, quants, args.reference)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference": args.reference,
        "language": args.language,
        "quants": quants,
        "sample_count": len(samples),
    }
    csv_path  = SCRIPT_DIR / "backend_comparison.csv"
    json_path = SCRIPT_DIR / "backend_comparison.json"
    md_path   = SCRIPT_DIR / "backend_comparison.md"

    write_csv(results,  csv_path)
    write_json(results, json_path, meta)
    write_md(results,   md_path,   meta)

    print(f"\nWrote:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
