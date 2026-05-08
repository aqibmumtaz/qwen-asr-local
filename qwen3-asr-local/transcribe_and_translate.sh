#!/bin/bash
# Transcribe audio (Qwen3-ASR) then translate to Urdu and English (Qwen3-LLM)
#
# Usage:
#   bash transcribe_and_translate.sh [--model MODEL_NAME] [audio.wav]
#   bash transcribe_and_translate.sh                                 # batch with default model (Q5_K_M)
#   bash transcribe_and_translate.sh --model Qwen3-8B-Q8_0           # batch with specified model
#   bash transcribe_and_translate.sh --model Qwen3-4B-Q4_K_M file.wav

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse arguments for model selection
TRANSLATOR_MODEL_NAME="Qwen3-8B-Q5_K_M"  # default
AUDIO_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            TRANSLATOR_MODEL_NAME="$2"
            shift 2
            ;;
        *)
            AUDIO_FILE="$1"
            shift
            ;;
    esac
done

ASR_MODEL="$SCRIPT_DIR/models/Qwen3-ASR-1.7B-Q8_0-new.gguf"
MMPROJ="$SCRIPT_DIR/models/mmproj-Qwen3-ASR-1.7B-bf16-new.gguf"
TRANSLATOR_MODEL="$SCRIPT_DIR/models/${TRANSLATOR_MODEL_NAME}.gguf"
LLAMA_MTMD="$SCRIPT_DIR/llama.cpp/build/bin/llama-mtmd-cli"
LLAMA_CLI="$SCRIPT_DIR/llama.cpp/build/bin/llama-cli"

TRANSCRIPT_DIR="$SCRIPT_DIR/transcriptions"
mkdir -p "$TRANSCRIPT_DIR"

if [[ ! -f "$TRANSLATOR_MODEL" ]]; then
    echo "Error: Translation model not found at $TRANSLATOR_MODEL"
    exit 1
fi

run_translation() {
    local PROMPT="$1"
    local INPUT_TEXT="$2"

    "$LLAMA_CLI" \
        -m "$TRANSLATOR_MODEL" \
        --temp 0.6 \
        --top-k 20 --top-p 0.95 \
        --presence-penalty 1.5 \
        -n 512 \
        --no-warmup \
        --single-turn \
        --simple-io \
        -sys "$PROMPT" \
        -p "$INPUT_TEXT" \
        2>/dev/null | awk '/^> /{found=1; next} /^\[ Prompt:/{found=0} /^Exiting/{found=0} found && NF{print}' | sed 's/<|im_end|>.*//' | tr -d '\n'
}

transcribe_and_translate_file() {
    local AUDIO="$1"
    local BASENAME
    BASENAME="$(basename "${AUDIO%.*}")"
    local OUT_FILE="$TRANSCRIPT_DIR/${BASENAME}_translated.txt"

    local asr_start asr_end
    asr_start=$(date +%s)
    local TRANSCRIPT
    TRANSCRIPT=$("$LLAMA_MTMD" \
        -m "$ASR_MODEL" \
        --mmproj "$MMPROJ" \
        --image "$AUDIO" \
        -p "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n" \
        -n 256 --no-warmup \
        2>/dev/null | grep '<asr_text>' | sed 's/.*<asr_text>//' | tr -d '\n')
    asr_end=$(date +%s)

    if [[ -z "$TRANSCRIPT" ]]; then
        echo "Error: ASR produced empty transcript for $AUDIO" >&2
        return 1
    fi

    local eng_start eng_end
    eng_start=$(date +%s)
    local ENGLISH
    ENGLISH=$(run_translation \
        "You are an expert English translator. The user will provide text in Hindi, Urdu, Roman Urdu, mixed language, or imperfect ASR English. Translate it into natural English. If the input is already English, lightly normalize obvious ASR mistakes while preserving meaning. Output only the English text, nothing else. /no_think" \
        "$TRANSCRIPT")
    eng_end=$(date +%s)

    if [[ -z "$ENGLISH" ]]; then
        echo "Error: English translation produced empty output for $AUDIO" >&2
        return 1
    fi

    local urdu_start urdu_end
    urdu_start=$(date +%s)
    local URDU
    URDU=$(run_translation \
        "You are an expert Urdu translator. The user will provide text in Hindi, Roman Urdu, or mixed language. Translate it to proper Urdu script. Output only the Urdu translation, nothing else. /no_think" \
        "$TRANSCRIPT")
    urdu_end=$(date +%s)

    if [[ -z "$URDU" ]]; then
        echo "Error: Urdu translation produced empty output for $AUDIO" >&2
        return 1
    fi

    printf "Transcript: %s\nEnglish: %s\nUrdu: %s\n" "$TRANSCRIPT" "$ENGLISH" "$URDU" > "$OUT_FILE"

    echo "File: $(basename "$AUDIO")"
    echo "Transcript: $TRANSCRIPT"
    echo "English: $ENGLISH"
    echo "Urdu: $URDU"
    echo "ASR_time: $((asr_end - asr_start))"
    echo "English_time: $((eng_end - eng_start))"
    echo "Urdu_time: $((urdu_end - urdu_start))"
}

if [[ -z "$AUDIO_FILE" ]]; then
    SAMPLES_DIR="$SCRIPT_DIR/samples"
    if [[ ! -d "$SAMPLES_DIR" ]] || [[ -z "$(ls "$SAMPLES_DIR" 2>/dev/null)" ]]; then
        echo "ERROR: No audio files found in $SAMPLES_DIR"
        exit 1
    fi
    for f in "$SAMPLES_DIR"/*; do
        [[ -f "$f" ]] || continue
        case "${f##*.}" in wav|mp3|flac|ogg|m4a|aac|opus) ;; *) continue ;; esac
        transcribe_and_translate_file "$f"
    done
else
    if [[ ! -f "$AUDIO_FILE" ]]; then
        echo "Error: Audio file not found: $AUDIO_FILE"
        exit 1
    fi
    transcribe_and_translate_file "$AUDIO_FILE"
fi
