#!/usr/bin/env bash
# transcribe.sh — transcribe audio file(s) with Qwen3-ASR (local runner)
#
# Usage:
#   bash transcribe.sh <audio_file>   # single file
#   bash transcribe.sh                # batch: all files in samples/
#
# Supported formats: wav, mp3, flac, ogg, m4a, aac, opus

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI="$SCRIPT_DIR/llama.cpp/build/bin/llama-mtmd-cli"
MODELS_DIR="$SCRIPT_DIR/models"
TEXT_MODEL="$MODELS_DIR/Qwen3-ASR-1.7B-Q8_0-new.gguf"
MMPROJ="$MODELS_DIR/mmproj-Qwen3-ASR-1.7B-bf16-new.gguf"

# ── Validate ──────────────────────────────────────────────────────────────────
if [ ! -f "$CLI" ]; then
    echo "ERROR: llama-mtmd-cli not found. Run: bash setup.sh"
    exit 1
fi

if [ ! -f "$TEXT_MODEL" ]; then
    echo "ERROR: Text model not found at $TEXT_MODEL"
    echo "       Run: bash setup.sh"
    exit 1
fi

if [ ! -f "$MMPROJ" ]; then
    echo "ERROR: Audio encoder not found at $MMPROJ"
    echo "       Run: bash setup.sh"
    exit 1
fi

TRANSCRIPT_DIR="$SCRIPT_DIR/transcriptions"
mkdir -p "$TRANSCRIPT_DIR"

transcribe_file() {
    local INPUT="$1"
    local AUDIO
    AUDIO="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"

    # Convert to WAV if needed
    local AUDIO_WAV="$AUDIO"
    if [[ "$AUDIO" != *.wav ]]; then
        AUDIO_WAV="/tmp/$(basename "${AUDIO%.*}").wav"
        ffmpeg -y -i "$AUDIO" -ar 16000 -ac 1 -f wav "$AUDIO_WAV" 2>/dev/null
    fi

    local OUT_FILE="$TRANSCRIPT_DIR/$(basename "${AUDIO%.*}").txt"

    local RESULT
    RESULT=$(set +e; "$CLI" \
        -m "$TEXT_MODEL" \
        --mmproj "$MMPROJ" \
        --image "$AUDIO_WAV" \
        -p "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n" \
        -n 256 \
        --no-warmup \
        2>&1 | sed -n 's/.*<asr_text>//p')

    echo "$(basename "$INPUT"): $RESULT"
    echo "$RESULT" > "$OUT_FILE"
}

# ── Batch or single ───────────────────────────────────────────────────────────
if [ -z "$1" ]; then
    # No argument: transcribe all files in samples/
    SAMPLES_DIR="$SCRIPT_DIR/samples"
    if [ ! -d "$SAMPLES_DIR" ] || [ -z "$(ls "$SAMPLES_DIR" 2>/dev/null)" ]; then
        echo "ERROR: No audio files found in $SAMPLES_DIR"
        exit 1
    fi
    echo "Transcribing all files in samples/ ..."
    echo ""
    rm -f "$TRANSCRIPT_DIR"/*.txt
    for f in "$SAMPLES_DIR"/*; do
        [ -f "$f" ] || continue
        case "${f##*.}" in wav|mp3|flac|ogg|m4a|aac|opus) ;; *) continue ;; esac
        transcribe_file "$f"
    done
    echo ""
    echo "Results saved to: $TRANSCRIPT_DIR/"
else
    if [ ! -f "$1" ]; then
        echo "ERROR: Audio file not found: $1"
        exit 1
    fi
    transcribe_file "$1"
fi
