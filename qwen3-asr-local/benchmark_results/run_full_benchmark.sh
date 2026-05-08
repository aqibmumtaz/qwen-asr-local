#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$ROOT/benchmark_results"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/final_benchmark_sheet.csv"
FINAL_SHEET_BUILDER="$OUT_DIR/build_full_benchmark_sheet.py"

asr_decoders=("Qwen3-ASR-1.7B-Q8_0-new" "Qwen3-ASR-1.7B-BF16-new")
models=("Qwen3-8B-Q5_K_M" "Qwen3-8B-Q8_0" "Qwen3-8B-Q4_K_M" "Qwen3-8B-BF16" "Qwen3-4B-Q8_0" "Qwen3-4B-Q4_K_M" "Qwen3-4B-BF16")
audios=("sample_en1.wav" "sample_en2.wav" "sample_ur1.wav" "sample_ur2.wav")

printf '\xEF\xBB\xBFasr_model,model,audio,asr_seconds,urdu_seconds,inference_seconds,status,transcript,english_translation,urdu_translation\n' > "$CSV"

for asr in "${asr_decoders[@]}"; do
  for m in "${models[@]}"; do
    for a in "${audios[@]}"; do
      tmp=$(mktemp)
      if bash "$ROOT/transcribe_and_translate.sh" --asr-model "$asr" --model "$m" "$ROOT/samples/$a" > "$tmp" 2>/dev/null; then
        rc=0
      else
        rc=$?
      fi

      transcript=$(grep -m1 '^Transcript: ' "$tmp" | sed 's/^Transcript: //' || true)
      english=$(grep -m1 '^English: ' "$tmp" | sed 's/^English: //' || true)
      urdu=$(grep -m1 '^Urdu: ' "$tmp" | sed 's/^Urdu: //' || true)
      asr_secs=$(grep -m1 '^ASR_time: ' "$tmp" | sed 's/^ASR_time: //' || echo 0)
      eng_secs=$(grep -m1 '^English_time: ' "$tmp" | sed 's/^English_time: //' || echo 0)
      urdu_secs=$(grep -m1 '^Urdu_time: ' "$tmp" | sed 's/^Urdu_time: //' || echo 0)
      total_secs=$((asr_secs + urdu_secs))

      status="FAIL"
      if [[ "$rc" -eq 0 && -n "$transcript" && -n "$english" && -n "$urdu" ]]; then
        status="OK"
      fi

      esc_t=${transcript//\"/\"\"}
      esc_e=${english//\"/\"\"}
      esc_u=${urdu//\"/\"\"}
      echo "\"$asr\",\"$m\",\"$a\",\"$asr_secs\",\"$urdu_secs\",\"$total_secs\",\"$status\",\"$esc_t\",\"$esc_e\",\"$esc_u\"" >> "$CSV"

      cp "$tmp" "$OUT_DIR/run_${asr}_${m}_${a}.log"
      rm -f "$tmp"

      echo "$asr | $m | $a | asr=${asr_secs}s eng=${eng_secs}s urdu=${urdu_secs}s total=${total_secs}s | $status"
    done
  done
done

python3 "$FINAL_SHEET_BUILDER"
rm -f "$CSV"

echo "Benchmark complete. Output:"
echo "  - $OUT_DIR/final_benchmark_sheet.xlsx"
