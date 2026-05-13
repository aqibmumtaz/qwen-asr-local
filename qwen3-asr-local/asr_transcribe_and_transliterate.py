#!/usr/bin/env python3
"""
ASR + transliteration — native HuggingFace path (no llama.cpp).

Mirrors the architecture of the production app.py (vLLM streaming server) but
adapted for a local CLI run on Mac/Linux without GPU:
  - transformers backend (instead of vLLM, which needs CUDA)
  - thread-safe singleton model loader with readiness Event
  - smart dtype selection (cuda+SM>=80 → bf16, cuda<80 → fp16, cpu → fp32)
  - all knobs overridable via env vars (MODEL_ID, DEVICE, DTYPE, …)
  - hallucination filter on output
  - structured logging with timestamps
  - startup config + system info dump

After ASR, the Hindi Devanagari text is fed to our transliteration pipeline:
  - Urdu Nastaliq via GokulNC HindustaniTransliterator
  - Roman Urdu via hindi_to_roman_urdu.sh (lexicon-corrected)

Usage:
    python3 asr_transcribe_and_transliterate.py                       # all samples
    python3 asr_transcribe_and_transliterate.py audio.wav             # one file
    python3 asr_transcribe_and_transliterate.py --compare audio.wav   # HF vs llama.cpp
    python3 asr_transcribe_and_transliterate.py --language English audio.wav

Env vars (all optional):
    MODEL_ID           default: Qwen/Qwen3-ASR-1.7B
    TORCH_DEVICE       auto / cuda:0 / mps / cpu
    DTYPE              bfloat16 / float16 / float32 / auto
    MAX_NEW_TOKENS     default: 1024
    LANGUAGE           default: Hindi
"""

import argparse
import logging
import os
import sys
import threading
import time
import subprocess
import warnings
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


# ── ASR functions ────────────────────────────────────────────────────────────
def hf_asr(audio_path: Path, language: str | None = None) -> tuple[str, float]:
    """Native HF Qwen3-ASR. Returns (transcript, elapsed_s)."""
    model = get_asr_model()
    lang = language or LANGUAGE

    t0 = time.time()
    # List form matches app.py's batch transcribe signature
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
def process_one(audio: Path, language: str = LANGUAGE, compare: bool = False):
    print(f"\n{'═' * 72}")
    print(f"  {audio.name}")
    print(f"{'═' * 72}")

    hindi_hf, hf_t = hf_asr(audio, language=language)
    if not hindi_hf:
        log.error(f"HF ASR returned empty for {audio.name}")
        return

    nastaliq_hf = to_nastaliq(hindi_hf)
    roman_hf    = to_roman_urdu(hindi_hf)

    print(f"\n  ─── HF native (qwen-asr / transformers) — {hf_t:.1f}s ───")
    print(f"  Hindi      │ {hindi_hf}")
    print(f"  Nastaliq   │ {nastaliq_hf}")
    print(f"  Roman Urdu │ {roman_hf}")

    if compare:
        hindi_cpp, cpp_t = llamacpp_asr(audio, language=language)
        nastaliq_cpp     = to_nastaliq(hindi_cpp)
        roman_cpp        = to_roman_urdu(hindi_cpp)
        print(f"\n  ─── llama.cpp (transcribe.sh) — {cpp_t:.1f}s ───")
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
        process_one(audio, language=lang, compare=args.compare)

    print(f"\n{'─' * 72}")
    log.info(f"Total wall time: {time.time() - total_start:.1f}s for {len(targets)} file(s)")


if __name__ == "__main__":
    main()
