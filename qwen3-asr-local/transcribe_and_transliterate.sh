#!/bin/bash
# Pipeline: Audio → Qwen3-ASR (Hindi) → Urdu Nastaliq + Roman Urdu
#
# Step 1: transcribe.sh                 — ASR to Hindi Devanagari
# Step 2: GokulNC HindustaniTransliterator — Hindi → Urdu Nastaliq
# Step 3: hindi_to_roman_urdu module    — Hindi → Roman Urdu
#
# Usage:
#   bash transcribe_and_transliterate.sh [--language LANG] [audio.wav]
#   bash transcribe_and_transliterate.sh                            # batch (all samples/)
#   bash transcribe_and_transliterate.sh --language Hindi audio.wav
#   bash transcribe_and_transliterate.sh --language hi audio.wav    # ISO code

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ASR_LANGUAGE="${ASR_LANGUAGE:-Hindi}"
ASR_MODEL_NAME="Qwen3-ASR-1.7B-Q8_0-new"
AUDIO_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --language|-l)
            ASR_LANGUAGE="$2"
            shift 2
            ;;
        --asr-model)
            ASR_MODEL_NAME="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,11p' "$0" >&2
            exit 0
            ;;
        *)
            AUDIO_FILE="$1"
            shift
            ;;
    esac
done

TRANSCRIBE_SH="$SCRIPT_DIR/transcribe.sh"
HINDI_TO_ROMAN_URDU_SH="$SCRIPT_DIR/hindi_to_roman_urdu.sh"
TRANSCRIPT_DIR="$SCRIPT_DIR/transcriptions"
mkdir -p "$TRANSCRIPT_DIR"

# Nastaliq is GokulNC (Python-only library); keep as inline Python heredoc.
# Roman Urdu is done via transliterate.sh (single shell entry point).
nastaliq_for() {
    local HINDI="$1"
    cd "$SCRIPT_DIR"
    python3 -c "
import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
from indo_arabic_transliteration.hindustani import HindustaniTransliterator
print(HindustaniTransliterator().transliterate_from_hindi_to_urdu(sys.argv[1]))
" "$HINDI"
}

process_file() {
    local AUDIO="$1"
    local BASENAME
    BASENAME="$(basename "${AUDIO%.*}")"
    local OUT_FILE="$TRANSCRIPT_DIR/${BASENAME}_post_processor.txt"

    echo
    echo "────────────────────────────────────────────────────────────"
    echo "File        : $(basename "$AUDIO")"

    # Step 1: ASR
    local asr_start asr_end HINDI
    asr_start=$(date +%s)
    HINDI=$(bash "$TRANSCRIBE_SH" \
        --language "$ASR_LANGUAGE" \
        --asr-model "$ASR_MODEL_NAME" \
        "$AUDIO" | tr -d '\n')
    asr_end=$(date +%s)

    if [[ -z "$HINDI" ]]; then
        echo "ERROR       : ASR returned empty transcript"
        return 1
    fi
    echo "Hindi       : $HINDI"

    # Step 2: Nastaliq (GokulNC via Python heredoc)
    local NASTALIQ
    NASTALIQ=$(nastaliq_for "$HINDI")

    # Step 3: Roman Urdu — call hindi_to_roman_urdu.sh (.sh calls .sh only)
    local ROMAN
    ROMAN=$(bash "$HINDI_TO_ROMAN_URDU_SH" "$HINDI")

    echo "Urdu        : $NASTALIQ"
    echo "Roman Urdu  : $ROMAN"
    echo "Timing      : ASR=$((asr_end - asr_start))s"

    {
        echo "Hindi (ASR)          : $HINDI"
        echo "Urdu Nastaliq        : $NASTALIQ"
        echo "Roman Urdu           : $ROMAN"
    } > "$OUT_FILE"
    echo "Saved       : $OUT_FILE"
}

echo "ASR language: $ASR_LANGUAGE"

if [[ -z "$AUDIO_FILE" ]]; then
    SAMPLES_DIR="$SCRIPT_DIR/samples"
    if [[ ! -d "$SAMPLES_DIR" ]] || [[ -z "$(ls "$SAMPLES_DIR" 2>/dev/null)" ]]; then
        echo "ERROR: No audio files found in $SAMPLES_DIR" >&2
        exit 1
    fi
    for f in "$SAMPLES_DIR"/*; do
        [[ -f "$f" ]] || continue
        case "${f##*.}" in wav|mp3|flac|ogg|m4a|aac|opus) ;; *) continue ;; esac
        process_file "$f"
    done
else
    if [[ ! -f "$AUDIO_FILE" ]]; then
        echo "Error: file not found: $AUDIO_FILE" >&2
        exit 1
    fi
    process_file "$AUDIO_FILE"
fi

echo
echo "────────────────────────────────────────────────────────────"
echo "Done. Output saved to transcriptions/"
