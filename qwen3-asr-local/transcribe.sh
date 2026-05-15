#!/bin/bash
# Pure Qwen3-ASR transcription — outputs Hindi/Devanagari text to stdout.
#
# Usage:
#   bash transcribe.sh [--language LANG] [--asr-model MODEL] audio.wav
#
# Examples:
#   bash transcribe.sh audio.wav                            # default English
#   bash transcribe.sh --language Persian audio.wav         # different language
#   bash transcribe.sh --language hi audio.wav              # ISO code accepted
#   bash transcribe.sh --asr-model Qwen3-ASR-1.7B-BF16-new audio.wav
#   HINDI=$(bash transcribe.sh audio.wav)                   # capture output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Defaults (matches transcribe.py / app.py conventions)
ASR_MODEL_NAME="Qwen3-ASR-1.7B-Q8_0-new"
ASR_LANGUAGE="${ASR_LANGUAGE:-English}"
AUDIO_FILE=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --asr-model)
            ASR_MODEL_NAME="$2"
            shift 2
            ;;
        --language|-l)
            ASR_LANGUAGE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,12p' "$0" >&2
            exit 0
            ;;
        *)
            AUDIO_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "$AUDIO_FILE" ]]; then
    echo "Error: audio file required" >&2
    echo "Usage: $0 [--language LANG] [--asr-model MODEL] audio.wav" >&2
    exit 1
fi

if [[ ! -f "$AUDIO_FILE" ]]; then
    echo "Error: file not found: $AUDIO_FILE" >&2
    exit 1
fi

ASR_MODEL="$SCRIPT_DIR/models/${ASR_MODEL_NAME}.gguf"
MMPROJ="$SCRIPT_DIR/models/mmproj-Qwen3-ASR-1.7B-bf16-new.gguf"
LLAMA_MTMD="$SCRIPT_DIR/llama.cpp/build/bin/llama-mtmd-cli"

# ISO code → full language name (matches transcribe.py LANG_MAP)
case "$ASR_LANGUAGE" in
    en) ASR_LANGUAGE="English" ;;
    hi) ASR_LANGUAGE="Hindi" ;;
    ar) ASR_LANGUAGE="Arabic" ;;
    fa) ASR_LANGUAGE="Persian" ;;
    zh) ASR_LANGUAGE="Chinese" ;;
    ja) ASR_LANGUAGE="Japanese" ;;
    ko) ASR_LANGUAGE="Korean" ;;
    de) ASR_LANGUAGE="German" ;;
    fr) ASR_LANGUAGE="French" ;;
    es) ASR_LANGUAGE="Spanish" ;;
    id) ASR_LANGUAGE="Indonesian" ;;
    ms) ASR_LANGUAGE="Malay" ;;
    th) ASR_LANGUAGE="Thai" ;;
    vi) ASR_LANGUAGE="Vietnamese" ;;
    ru) ASR_LANGUAGE="Russian" ;;
    tr) ASR_LANGUAGE="Turkish" ;;
    *) ;;  # leave as-is (assumed full name)
esac

# Run ASR — extract <asr_text>...</asr_text> from stdout, output transcript only.
"$LLAMA_MTMD" \
    -m "$ASR_MODEL" \
    --mmproj "$MMPROJ" \
    --image "$AUDIO_FILE" \
    -p "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage ${ASR_LANGUAGE}<asr_text>" \
    -n 256 --no-warmup \
    2>/dev/null \
    | grep '<asr_text>' \
    | sed 's/.*<asr_text>//' \
    | tr -d '\n'
echo  # trailing newline so output is a clean line
