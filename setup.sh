#!/usr/bin/env bash
# setup.sh — one-time build for Qwen3-ASR local inference
# Clones llama.cpp, builds llama-mtmd-cli, and converts GGUF models
#
# Usage: bash setup.sh
#
# Requirements:
#   - git, cmake (brew install cmake)
#   - python3 with: torch numpy sentencepiece transformers
#   - Xcode Command Line Tools (xcode-select --install)
#   - HuggingFace model cached: huggingface-cli download Qwen/Qwen3-ASR-1.7B

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
MODELS_DIR="$SCRIPT_DIR/models"

echo "=== Qwen3-ASR Local Setup ==="
echo ""

# ── 1. Check dependencies ─────────────────────────────────────────────────────
check_dep() {
    command -v "$1" &>/dev/null || {
        echo "ERROR: '$1' not found. $2"
        exit 1
    }
}

check_dep git     "Install Xcode Command Line Tools: xcode-select --install"
check_dep cmake   "Install via Homebrew: brew install cmake"
check_dep python3 "Install Python 3.10+"
echo "[OK] Dependencies: git, cmake, python3"

# ── 2. Clone llama.cpp ────────────────────────────────────────────────────────
echo ""
echo "=== Setting up llama.cpp ==="

if [ ! -d "$LLAMA_DIR/.git" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    echo "[OK] llama.cpp cloned"
else
    echo "[OK] llama.cpp already cloned"
fi

# ── 3. Build (CPU-only, no Metal) ─────────────────────────────────────────────
echo ""
echo "Building llama-mtmd-cli (CPU-only)..."

BUILD_DIR="$LLAMA_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=OFF \
    -DGGML_ACCELERATE=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DBUILD_SHARED_LIBS=ON \
    2>&1 | tail -5

cmake --build . --config Release --target llama-mtmd-cli -j$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)

echo "[OK] Build complete: $BUILD_DIR/bin/llama-mtmd-cli"

# ── 4. Find HuggingFace model ─────────────────────────────────────────────────
echo ""
echo "=== Converting GGUF models ==="

HF_MODEL=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots -maxdepth 1 -type d 2>/dev/null | tail -1)

if [ -z "$HF_MODEL" ] || [ ! -f "$HF_MODEL/config.json" ]; then
    echo "HuggingFace model not found. Downloading..."
    pip3 install -q "huggingface_hub>=0.20"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-1.7B')"
    HF_MODEL=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots -maxdepth 1 -type d | tail -1)
fi

echo "[OK] HF model: $HF_MODEL"

# ── 5. Convert GGUF files ─────────────────────────────────────────────────────
mkdir -p "$MODELS_DIR"
cd "$LLAMA_DIR"

pip3 install -q torch numpy sentencepiece transformers

TEXT_MODEL="$MODELS_DIR/Qwen3-ASR-1.7B-Q8_0-new.gguf"
MMPROJ="$MODELS_DIR/mmproj-Qwen3-ASR-1.7B-bf16-new.gguf"

if [ ! -f "$TEXT_MODEL" ]; then
    echo "Converting text decoder (Q8_0, ~2.2GB)..."
    python3 convert_hf_to_gguf.py "$HF_MODEL" --outtype q8_0 --outfile "$TEXT_MODEL"
    echo "[OK] Decoder: $TEXT_MODEL"
else
    echo "[OK] Decoder already exists"
fi

if [ ! -f "$MMPROJ" ]; then
    echo "Converting audio encoder (bf16, ~636MB)..."
    python3 convert_hf_to_gguf.py "$HF_MODEL" --mmproj --outtype bf16 --outfile "$MMPROJ"
    echo "[OK] Encoder: $MMPROJ"
else
    echo "[OK] Encoder already exists"
fi

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Models:"
echo "  Decoder: $TEXT_MODEL"
echo "  Encoder: $MMPROJ"
echo ""
echo "Usage:"
echo "  bash transcribe.sh <audio.wav>"
echo ""
echo "Example:"
echo "  bash transcribe.sh samples/sample_en1.wav"
