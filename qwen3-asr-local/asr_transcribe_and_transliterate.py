#!/usr/bin/env python3
"""
ASR + transliteration with per-word confidence — native HuggingFace path.

Mirrors the architecture of the production app.py (vLLM streaming server)
adapted for a local CLI:
  - transformers backend (vLLM optional via BACKEND env)
  - thread-safe singleton model loader with readiness Event
  - smart dtype selection (cuda+SM>=80 → bf16, cuda<80 → fp16, cpu → fp32)
  - all knobs overridable via env vars
  - hallucination filter on output
  - structured logging with timestamps
  - startup config + system info dump

Per-word confidence extraction:
  - Monkey-patches model.model.generate to enable output_scores=True
  - Computes per-token logprobs from greedy logits
  - Aggregates sub-tokens to words by whitespace
  - Reports two scores per word:
      conf_min  = exp(min logprob)  — flags any uncertain sub-token (review-trigger)
      conf_geo  = exp(mean logprob) — geometric mean (sortable score)
  - Words with conf_min < LOW_CONF_THRESHOLD (default 0.5) are flagged

After ASR, the Hindi Devanagari text is fed to our transliteration pipeline:
  - Urdu Nastaliq via GokulNC HindustaniTransliterator
  - Roman Urdu via hindi_to_roman_urdu.sh (lexicon-corrected)

Usage:
    python3 asr_transcribe_and_transliterate.py                       # all samples
    python3 asr_transcribe_and_transliterate.py audio.wav             # one file
    python3 asr_transcribe_and_transliterate.py --conf-table audio.wav # full per-word table
    python3 asr_transcribe_and_transliterate.py --compare audio.wav   # HF vs llama.cpp
    python3 asr_transcribe_and_transliterate.py --language English audio.wav

Env vars (all optional):
    MODEL_ID            default: Qwen/Qwen3-ASR-1.7B
    TORCH_DEVICE        auto / cuda:0 / mps / cpu
    DTYPE               bfloat16 / float16 / float32 / auto
    MAX_NEW_TOKENS      default: 1024
    LANGUAGE            default: Hindi
    LOW_CONF_THRESHOLD  default: 0.5 (below this, word is flagged)
"""

import argparse
import logging
import math
import os
import sys
import threading
import time
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
warnings.filterwarnings("ignore")

import torch
from indo_arabic_transliteration.hindustani import HindustaniTransliterator
from qwen_asr import Qwen3ASRModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("qwen3-asr")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR             = Path(__file__).resolve().parent
TRANSCRIBE_SH          = SCRIPT_DIR / "transcribe.sh"          # llama.cpp path
HINDI_TO_ROMAN_URDU_SH = SCRIPT_DIR / "hindi_to_roman_urdu.sh"
SAMPLES_DIR            = SCRIPT_DIR / "samples"
OUT_DIR                = SCRIPT_DIR / "transcriptions"

# ── Config (env-overridable, mirrors app.py) ─────────────────────────────────
MODEL_ID               = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
MAX_NEW_TOKENS         = int(os.getenv("MAX_NEW_TOKENS", "1024"))
LANGUAGE               = os.getenv("LANGUAGE", "Hindi")
DEVICE_OVERRIDE        = os.getenv("TORCH_DEVICE")  # explicit override
DTYPE_OVERRIDE         = os.getenv("DTYPE")         # bfloat16 / float16 / float32 / auto

# Backend selection (vLLM = production/CUDA, transformers = dev/Mac fallback)
BACKEND = os.getenv("BACKEND", "auto")              # auto / vllm / transformers

# Hallucination filter (from app.py) — ASR sometimes emits these on silence/noise
HALLUCINATION_PHRASES = {
    "transcript", "transcription", "thank you", "thanks for watching",
    "you", "bye", "goodbye", "the end", "subtitle", "subtitles",
}


def _is_hallucination(text: str) -> bool:
    return text.lower().strip().rstrip(".!?,;:") in HALLUCINATION_PHRASES


# ── Device + dtype selection (mirrors app.py's _get_dtype) ───────────────────
def _pick_device() -> str:
    if DEVICE_OVERRIDE:
        return DEVICE_OVERRIDE
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        # MPS has a per-buffer cap (~2 GB) that 1.7B model's 3.2 GB warmup exceeds.
        # Forced CPU on Mac; override with TORCH_DEVICE=mps if running a smaller model.
        return "cpu"
    return "cpu"


def _pick_dtype(device: str):
    """Mirror app.py's _get_dtype() — string for vLLM, torch.dtype for transformers."""
    if DTYPE_OVERRIDE and DTYPE_OVERRIDE != "auto":
        return DTYPE_OVERRIDE  # string form for vLLM/transformers both accept
    # CUDA SM>=80 (Ampere+) → bf16; older CUDA → fp16; CPU → fp32
    if device.startswith("cuda") and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return "bfloat16" if cap[0] * 10 + cap[1] >= 80 else "half"
    if device == "mps":
        return "float16"
    return "float32"


def _dtype_to_torch(dtype_str: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16,
            "half":     torch.float16,
            "float16":  torch.float16,
            "float32":  torch.float32}[dtype_str]


def _pick_backend() -> str:
    """auto = vllm if importable + CUDA, else transformers."""
    if BACKEND != "auto":
        return BACKEND
    if not torch.cuda.is_available():
        return "transformers"
    import importlib.util
    if importlib.util.find_spec("vllm") is not None:
        return "vllm"
    return "transformers"


# ── Thread-safe singleton ASR loader (mirrors app.py's get_asr_model) ────────
_asr_model = None
_asr_lock = threading.Lock()
_model_ready = threading.Event()


def get_asr_model():
    """Mirrors app.py's get_asr_model() exactly when backend=vllm."""
    global _asr_model
    if _asr_model is None:
        with _asr_lock:
            if _asr_model is None:
                backend = _pick_backend()
                device  = _pick_device()
                dtype   = _pick_dtype(device)
                log.info(f"Loading {MODEL_ID}...")
                start = time.time()

                if backend == "vllm":
                    # ── EXACT app.py call signature ────────────────────
                    _asr_model = Qwen3ASRModel.LLM(
                        model=MODEL_ID,
                        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                        dtype=dtype,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    log.info(f"Model loaded in {time.time() - start:.1f}s "
                             f"(vLLM backend, dtype={dtype})")
                else:
                    # ── Transformers fallback (Mac / no CUDA) ──────────
                    log.warning("vLLM not available; falling back to transformers "
                                "backend (Qwen3ASRModel.from_pretrained).")
                    _asr_model = Qwen3ASRModel.from_pretrained(
                        MODEL_ID,
                        dtype=_dtype_to_torch(dtype),
                        device_map=device,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    log.info(f"Model loaded in {time.time() - start:.1f}s "
                             f"(transformers backend, device={device}, dtype={dtype})")

                _model_ready.set()
    return _asr_model


# ── Word-level confidence extraction ─────────────────────────────────────────
# Qwen3-ASR's high-level transcribe() discards token logprobs. We monkey-patch
# model.model.generate to capture output_scores=True, return_dict_in_generate=True
# on a single call, then aggregate per-token logprobs into per-word confidence.
# See ard/hindi-to-roman-urdu-design.md and the research report for details.

LOW_CONF_THRESHOLD = float(os.getenv("LOW_CONF_THRESHOLD", "0.5"))


@dataclass
class WordConf:
    """A word + its confidence (both min-token and geometric-mean variants)."""
    text:        str
    conf_min:    float   # min token prob — surfaces "any sub-token was unsure" → flag for review
    conf_geo:    float   # exp(mean(logprobs)) — well-calibrated joint prob → sortable score
    n_tokens:    int = 1

    @property
    def is_low(self) -> bool:
        return self.conf_min < LOW_CONF_THRESHOLD


def _aggregate_tokens_to_words(tokenizer,
                               gen_ids: list[int],
                               logprobs: list[float],
                               raw_tokens: list[str] | None = None,
                               ) -> list[WordConf]:
    """
    Group BPE sub-tokens into Devanagari/Latin words.

    Qwen3 uses GPT-style ByteLevel BPE where a leading space is encoded as
    the special character 'Ġ' (U+0120) at the start of a token. We use that
    as the word-boundary marker, then decode each group together so multi-byte
    Devanagari characters reassemble correctly.

    Special / EOS tokens (e.g. <|im_end|>) are excluded from words.
    """
    if raw_tokens is None:
        raw_tokens = tokenizer.convert_ids_to_tokens(gen_ids)

    # Find indices of special tokens to exclude
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    # Group token indices into words: a new word starts when raw token has 'Ġ' prefix
    groups: list[list[int]] = []
    current: list[int] = []
    for i, (tid, raw) in enumerate(zip(gen_ids, raw_tokens)):
        if tid in special_ids:
            # close current word, drop the special token itself
            if current:
                groups.append(current)
            current = []
            continue
        if raw.startswith("Ġ") and current:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)

    words: list[WordConf] = []
    for group in groups:
        word_ids = [gen_ids[i] for i in group]
        word_text = tokenizer.decode(word_ids, skip_special_tokens=False).strip()
        if not word_text:
            continue
        lps = [logprobs[i] for i in group]
        words.append(WordConf(
            text=word_text,
            conf_min=math.exp(min(lps)),
            conf_geo=math.exp(sum(lps) / len(lps)),
            n_tokens=len(lps),
        ))
    return words


def hf_asr(audio_path: Path,
           language: str | None = None,
           ) -> tuple[str, float]:
    """
    Native HF Qwen3-ASR — plain transcription, no confidence.
    Returns (transcript, elapsed_s). Uses the standard high-level
    Qwen3ASRModel.transcribe() call (no monkey-patching).
    """
    model = get_asr_model()
    lang = language or LANGUAGE

    t0 = time.time()
    results = model.transcribe(
        audio=[str(audio_path)],
        language=[lang] if lang else None,
    )
    elapsed = time.time() - t0

    text = results[0].text if results else ""
    if text and _is_hallucination(text):
        log.warning(f"Filtered hallucination: '{text}'")
        text = ""
    return text, elapsed


def hf_asr_with_confidence(audio_path: Path,
                           language: str | None = None,
                           ) -> tuple[str, float, list[WordConf]]:
    """
    Native HF Qwen3-ASR with per-word confidence extraction.
    Returns (transcript, elapsed_s, word_confidences).

    Mechanism: monkey-patch model.model.generate to enable output_scores so the
    high-level transcribe() still works but we capture the score tensor.
    The patch is local to this call (restored in finally) — no global state.

    Use this when you need to flag low-confidence words for human review.
    Use hf_asr() for the cheaper plain transcription path.

    Only works on the transformers backend. vLLM backend would need a different
    mechanism (SamplingParams.logprobs=N) — not yet implemented.
    """
    model = get_asr_model()
    lang = language or LANGUAGE

    # Patch the INNER thinker.generate (the LLM that emits tokens), not the
    # outer Qwen3ASR.generate — the outer one already hardcodes
    # return_dict_in_generate=True when it calls thinker.generate, so
    # adding it from outside causes a duplicate-kwarg TypeError.
    captured: dict = {"outputs": None}
    target = getattr(model.model, "thinker", None)
    if target is None or not hasattr(target, "generate"):
        log.warning("Inner thinker.generate not found; using plain transcribe.")
        text, elapsed = hf_asr(audio_path, language=language)
        return text, elapsed, []

    orig_generate = target.generate

    def patched_generate(*args, **kwargs):
        kwargs["output_scores"] = True
        # return_dict_in_generate is already passed by qwen-asr; do not set it
        # here, only ensure it's True (idempotent)
        kwargs.setdefault("return_dict_in_generate", True)
        result = orig_generate(*args, **kwargs)
        captured["outputs"] = result
        return result

    t0 = time.time()
    try:
        target.generate = patched_generate
        results = model.transcribe(
            audio=[str(audio_path)],
            language=[lang] if lang else None,
        )
    finally:
        target.generate = orig_generate
    elapsed = time.time() - t0

    text = results[0].text if results else ""
    if text and _is_hallucination(text):
        log.warning(f"Filtered hallucination: '{text}'")
        return "", elapsed, []

    # Pull token logprobs out of the captured outputs
    word_confs: list[WordConf] = []
    out = captured["outputs"]
    if out is not None and getattr(out, "scores", None):
        try:
            tok = model.processor.tokenizer
            # scores[i] = logits over vocab for generated token i, shape (1, vocab)
            # gen_ids are the LAST len(scores) tokens of sequences
            n_gen = len(out.scores)
            gen_ids = out.sequences[0, -n_gen:].tolist()
            logprobs = [
                torch.log_softmax(s[0].float(), dim=-1)[tid].item()
                for s, tid in zip(out.scores, gen_ids)
            ]
            word_confs = _aggregate_tokens_to_words(tok, gen_ids, logprobs)
        except Exception as e:
            log.warning(f"Could not extract token confidences: {e}")

    return text, elapsed, word_confs


def llamacpp_asr(audio_path: Path, language: str = LANGUAGE) -> tuple[str, float]:
    """llama.cpp path via transcribe.sh — for side-by-side comparison."""
    t0 = time.time()
    out = subprocess.run(
        ["bash", str(TRANSCRIBE_SH), "--language", language, str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip(), time.time() - t0


# ── Transliteration ──────────────────────────────────────────────────────────
_nastaliq_engine = HindustaniTransliterator()


def to_nastaliq(hindi: str) -> str:
    return _nastaliq_engine.transliterate_from_hindi_to_urdu(hindi) if hindi else ""


def to_roman_urdu(hindi: str) -> str:
    if not hindi:
        return ""
    out = subprocess.run(
        ["bash", str(HINDI_TO_ROMAN_URDU_SH), hindi],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.rstrip("\n")


# ── Pipeline orchestration ───────────────────────────────────────────────────
def _format_conf_inline(word_confs: list[WordConf]) -> str:
    """One line: each word with its min-confidence in parentheses, low-conf marked."""
    parts = []
    for wc in word_confs:
        flag = "*" if wc.is_low else ""
        parts.append(f"{wc.text}({wc.conf_min:.2f}{flag})")
    return " ".join(parts)


def _format_conf_table(word_confs: list[WordConf]) -> str:
    """Multi-line table: word | min | geo | n_tokens | flag."""
    if not word_confs:
        return "  (no per-word confidence captured)"
    rows = [f"  {'Word':<20} {'min':>6} {'geo':>6} {'tok':>4}  {'flag'}"]
    rows.append(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 4}  {'-' * 4}")
    for wc in word_confs:
        flag = "LOW" if wc.is_low else ""
        rows.append(f"  {wc.text[:20]:<20} {wc.conf_min:>6.3f} "
                    f"{wc.conf_geo:>6.3f} {wc.n_tokens:>4}  {flag}")
    return "\n".join(rows)


def process_one(audio: Path, language: str = LANGUAGE, compare: bool = False,
                show_conf_table: bool = False):
    print(f"\n{'═' * 72}")
    print(f"  {audio.name}")
    print(f"{'═' * 72}")

    hindi_hf, hf_t, word_confs = hf_asr_with_confidence(audio, language=language)
    if not hindi_hf:
        log.error(f"HF ASR returned empty for {audio.name}")
        return

    nastaliq_hf = to_nastaliq(hindi_hf)
    roman_hf    = to_roman_urdu(hindi_hf)

    n_low = sum(1 for w in word_confs if w.is_low)
    conf_summary = (
        f"  ({len(word_confs)} words, {n_low} flagged <{LOW_CONF_THRESHOLD:.2f})"
        if word_confs else ""
    )

    print(f"\n  ─── HF native (qwen-asr / transformers) — {hf_t:.1f}s ───")
    print(f"  Hindi      │ {hindi_hf}")
    print(f"  Nastaliq   │ {nastaliq_hf}")
    print(f"  Roman Urdu │ {roman_hf}")
    if word_confs:
        print(f"  Confidence │ {_format_conf_inline(word_confs)}")
        print(f"             {conf_summary}")
        if show_conf_table:
            print()
            print(_format_conf_table(word_confs))

    # Save full output + per-word confidence to disk
    out_file = OUT_DIR / f"{audio.stem}_post_processor.txt"
    OUT_DIR.mkdir(exist_ok=True)
    out_lines = [
        f"Hindi (ASR)          : {hindi_hf}",
        f"Urdu Nastaliq        : {nastaliq_hf}",
        f"Roman Urdu           : {roman_hf}",
        "",
    ]
    if word_confs:
        out_lines.append("Per-word confidence (min token prob, geo-mean prob):")
        out_lines.append(_format_conf_table(word_confs))
    out_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"  Saved      │ {out_file}")

    if compare:
        hindi_cpp, cpp_t = llamacpp_asr(audio, language=language)
        nastaliq_cpp     = to_nastaliq(hindi_cpp)
        roman_cpp        = to_roman_urdu(hindi_cpp)
        print(f"\n  ─── llama.cpp (transcribe.sh — no logprobs) — {cpp_t:.1f}s ───")
        print(f"  Hindi      │ {hindi_cpp}")
        print(f"  Nastaliq   │ {nastaliq_cpp}")
        print(f"  Roman Urdu │ {roman_cpp}")

        match = "✓ identical" if hindi_hf == hindi_cpp else "✗ differ"
        print(f"\n  ─── Comparison ───")
        print(f"  ASR text   │ {match}")
        speedup = cpp_t / hf_t if hf_t > 0 else 0
        faster_one = 'HF faster' if speedup > 1 else 'llama.cpp faster'
        print(f"  Timing     │ HF={hf_t:.1f}s  llama.cpp={cpp_t:.1f}s  "
              f"({faster_one} by {max(speedup, 1/speedup):.1f}×)")


def _log_system_info():
    """Log system info at startup (mirrors app.py)."""
    log.info("=" * 60)
    log.info("Qwen3-ASR Local CLI (transformers backend)")
    log.info("=" * 60)
    log.info(f"MODEL_ID        = {MODEL_ID}")
    log.info(f"MAX_NEW_TOKENS  = {MAX_NEW_TOKENS}")
    log.info(f"LANGUAGE        = {LANGUAGE}")
    log.info(f"Python          = {sys.version.split()[0]}")
    log.info(f"PyTorch         = {torch.__version__}")
    log.info(f"CUDA available  = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"CUDA version    = {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem = getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)
            log.info(f"GPU {i}           = {props.name} ({mem / (1024**3):.1f} GB)")
    log.info(f"MPS available   = {torch.backends.mps.is_available()}")
    device = _pick_device()
    log.info(f"Selected device = {device}")
    log.info(f"Selected dtype  = {_pick_dtype(device)}")
    log.info("=" * 60)


def main():
    global MAX_NEW_TOKENS  # must be at top of function before reference

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument("audio", nargs="*", help="Audio file(s); default: all samples/*.wav")
    parser.add_argument("--language", "-l", default=LANGUAGE,
                        help=f"Language hint (default: {LANGUAGE}). "
                             "Use 'None' or empty for auto-detect.")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Also run llama.cpp path and show side-by-side comparison")
    parser.add_argument("--conf-table", action="store_true",
                        help="Show full per-word confidence table (min/geo/n_tokens/flag)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"Max output tokens (default: {MAX_NEW_TOKENS}).")
    args = parser.parse_args()

    MAX_NEW_TOKENS = args.max_new_tokens
    lang = None if args.language.lower() in ("none", "") else args.language

    _log_system_info()

    targets = ([Path(a) for a in args.audio]
               if args.audio
               else sorted(SAMPLES_DIR.glob("*.wav")))

    if not targets:
        log.error(f"No audio files found in {SAMPLES_DIR}")
        sys.exit(1)

    log.info(f"Processing {len(targets)} file(s), language={lang or 'auto-detect'}, "
             f"compare={args.compare}")

    total_start = time.time()
    for audio in targets:
        if not audio.exists():
            log.warning(f"Skipping (not found): {audio}")
            continue
        process_one(audio, language=lang, compare=args.compare,
                    show_conf_table=args.conf_table)

    print(f"\n{'─' * 72}")
    log.info(f"Total wall time: {time.time() - total_start:.1f}s for {len(targets)} file(s)")


if __name__ == "__main__":
    main()
