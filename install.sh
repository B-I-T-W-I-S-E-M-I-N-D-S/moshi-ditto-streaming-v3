#!/usr/bin/env bash
# =============================================================================
# install.sh — One-command dependency installer for the unified pipeline
# =============================================================================
# Targets: RunPod (CUDA 12.1, Python 3.10, Ubuntu 22.04)
# Run from the project root:
#   bash install.sh
# =============================================================================

set -e   # exit on any error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOLD="\033[1m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
RESET="\033[0m"

log() { echo -e "${CYAN}[install]${RESET} $*"; }
ok()  { echo -e "${GREEN}[install]${RESET} ✅  $*"; }

echo -e "${BOLD}"
echo "═══════════════════════════════════════════════════════════════"
echo "   Moshi + Bridge + Ditto — Unified Pipeline Installer"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${RESET}"

# ── 0. Detect CUDA version ────────────────────────────────────────────────────
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[\d.]+" | head -1)
    log "Detected CUDA: $CUDA_VERSION"
else
    log "nvcc not found; defaulting to CUDA 12.1 wheels"
    CUDA_VERSION="12.1"
fi

# Normalise to major.minor for wheel selection
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

if [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 4 ]]; then
    TORCH_CUDA_TAG="cu124"
    TORCH_VER="2.5.1"
elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
    TORCH_CUDA_TAG="cu121"
    TORCH_VER="2.5.1"
elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
    TORCH_CUDA_TAG="cu118"
    TORCH_VER="2.5.1"
else
    log "Unknown CUDA version — falling back to cu121"
    TORCH_CUDA_TAG="cu121"
    TORCH_VER="2.5.1"
fi

TORCHAUDIO_VER="2.5.1"
TORCHVISION_VER="0.20.1"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

# ── 1. PyTorch (matching CUDA) ────────────────────────────────────────────────
log "Step 1/7 — Installing PyTorch ${TORCH_VER}+${TORCH_CUDA_TAG} ..."
pip install --quiet \
    torch=="${TORCH_VER}" \
    torchaudio=="${TORCHAUDIO_VER}" \
    torchvision=="${TORCHVISION_VER}" \
    --index-url "${TORCH_INDEX}"
ok "PyTorch installed."

# ── 2. Moshi (local editable install) ────────────────────────────────────────
log "Step 2/7 — Installing Moshi from local source (moshi-inference/) ..."
pip install --quiet -e "${PROJECT_ROOT}/moshi-inference"
ok "Moshi installed."

# ── 3. Core audio / NLP dependencies ─────────────────────────────────────────
log "Step 3/7 — Installing core audio/NLP packages ..."
pip install --quiet \
    "transformers>=4.40.0" \
    "sentencepiece>=0.2.0,<0.3" \
    "huggingface_hub>=0.24,<1.0.0" \
    "safetensors>=0.4.0" \
    "sphn>=0.2.0,<0.3.0" \
    "einops>=0.7" \
    "pyyaml>=6.0" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "numpy>=1.26,<2.3" \
    "scipy>=1.10.0" \
    "tqdm>=4.48"
ok "Core audio/NLP packages installed."

# ── 4. pyworld (pitch / prosody; bridge module) ───────────────────────────────
log "Step 4/7 — Installing pyworld (prosody; may take a moment to compile) ..."
pip install --quiet "pyworld>=0.3.4"
ok "pyworld installed."

# ── 5. Computer-vision / Ditto dependencies ──────────────────────────────────
log "Step 5/7 — Installing CV / video packages for Ditto ..."
pip install --quiet \
    "opencv-python-headless>=4.8.0" \
    "imageio>=2.28.0" \
    "imageio-ffmpeg>=0.4.9" \
    "scikit-image>=0.21.0" \
    "scikit-learn>=1.3.0" \
    "Pillow>=10.0.0" \
    "onnxruntime-gpu" \
    "colored" \
    "polygraphy"
ok "CV/video packages installed."

# ── 6. TensorRT (Ditto uses TRT 8.6 compiled ONNX models) ───────────────────
log "Step 6/7 — Installing TensorRT 8.6 (for Ditto TRT inference) ..."
pip install --quiet \
    tensorrt==8.6.1 \
    tensorrt-bindings==8.6.1 \
    tensorrt-libs==8.6.1 \
    cuda-python \
    || log "⚠  TensorRT installation had warnings (may be OK if already installed)."
ok "TensorRT setup done."

# ── 7. Misc utilities ─────────────────────────────────────────────────────────
log "Step 7/8 — Installing miscellaneous utilities ..."
pip install --quiet \
    "tensorboard>=2.14.0" \
    "bitsandbytes>=0.45,<0.50.0" \
    "aiohttp>=3.10.5,<3.12" \
    "sounddevice==0.5"
ok "Miscellaneous utilities installed."

# ── 8. Streaming server dependencies ─────────────────────────────────────────
log "Step 8/8 — Installing streaming server dependencies (FastAPI / uvicorn) ..."
pip install --quiet \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    "websockets>=12.0"
ok "Streaming server dependencies installed."

# ── Check FFmpeg ──────────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    ok "FFmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    log "⚠  FFmpeg not found in PATH."
    log "   Install with:  apt-get update && apt-get install -y ffmpeg"
    log "   (Required for the final audio+video merge step)"
fi

# ── Checkpoint directory ──────────────────────────────────────────────────────
mkdir -p "${PROJECT_ROOT}/checkpoints"
log "Checkpoint directory: ${PROJECT_ROOT}/checkpoints"
log "Place your trained bridge model at: checkpoints/bridge_best.pt"

echo ""
echo -e "${BOLD}${GREEN}"
echo "═══════════════════════════════════════════════════════════════"
echo "   ✅  All dependencies installed successfully!"
echo ""
echo "   Next steps:"
echo "   1. Ensure bridge checkpoint: checkpoints/bridge_best.pt"
echo "   2. Ensure Ditto TRT models in:"
echo "      ditto-inference/checkpoints/ditto_trt_Ampere_Plus/"
echo "   3. Open MoshiDittoPipeline.ipynb and run all cells"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${RESET}"
