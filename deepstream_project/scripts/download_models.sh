#!/bin/bash
###############################################################################
# download_models.sh
# Downloads and sets up the required models for face detection + recognition.
#
# Models:
#   1. PGIE - NVIDIA TAO FaceDetectIR (ResNet18, pruned) from NGC
#   2. SGIE - MobileFaceNet ONNX (128D embedding) for face recognition
#
# Usage:
#   cd /workspace
#   bash scripts/download_models.sh
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
MODELS_DIR="${PROJECT_DIR}/models"

echo "=============================================="
echo "  Model Download & Setup Script"
echo "=============================================="
echo "  Project Dir: ${PROJECT_DIR}"
echo "  Models Dir:  ${MODELS_DIR}"
echo ""

# ─────────────────────────── Create Directories ─────────────────────────────
mkdir -p "${MODELS_DIR}/facedetect"
mkdir -p "${MODELS_DIR}/facenet"

# ─────────────────────────────────────────────────────────────────────────────
# 1. FACE DETECTION MODEL - NVIDIA TAO FaceDetectIR
# ─────────────────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "[1/2] Face Detection Model (TAO FaceDetectIR)"
echo "──────────────────────────────────────────────"

FACEDETECT_DIR="${MODELS_DIR}/facedetect"
FACEDETECT_ETLT="${FACEDETECT_DIR}/resnet18_facedetectir_pruned.etlt"

if [ -f "${FACEDETECT_ETLT}" ]; then
    echo "[INFO] FaceDetect model already exists. Skipping download."
else
    echo "[INFO] Downloading TAO FaceDetectIR model from NGC..."
    echo ""
    echo "  If ngc CLI is available:"
    echo "    ngc registry model download-version nvidia/tao/facedetectir:pruned_v1.0.1 \\"
    echo "      --dest ${FACEDETECT_DIR}"
    echo ""
    echo "  Alternatively, download manually from:"
    echo "    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facedetectir"
    echo ""

    # Attempt NGC download if CLI is available
    if command -v ngc &> /dev/null; then
        echo "[INFO] NGC CLI found. Attempting download..."
        ngc registry model download-version nvidia/tao/facedetectir:pruned_v1.0.1 \
            --dest "${FACEDETECT_DIR}" || {
            echo "[WARNING] NGC download failed. Please download manually."
        }
    else
        echo "[WARNING] NGC CLI not found."
        echo "  Install: pip install ngc-cli"
        echo "  Configure: ngc config set"
        echo ""
        echo "  Or download the model manually and place the .etlt file at:"
        echo "    ${FACEDETECT_ETLT}"
    fi
fi

# Create face detection labels file
FACEDETECT_LABELS="${FACEDETECT_DIR}/labels.txt"
if [ ! -f "${FACEDETECT_LABELS}" ]; then
    echo "face" > "${FACEDETECT_LABELS}"
    echo "[INFO] Created ${FACEDETECT_LABELS}"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. FACE EMBEDDING MODEL - MobileFaceNet (ONNX)
# ─────────────────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "[2/2] Face Embedding Model (MobileFaceNet)"
echo "──────────────────────────────────────────────"

FACENET_DIR="${MODELS_DIR}/facenet"
FACENET_ONNX="${FACENET_DIR}/mobilefacenet.onnx"

if [ -f "${FACENET_ONNX}" ]; then
    echo "[INFO] MobileFaceNet model already exists. Skipping download."
else
    echo "[INFO] MobileFaceNet ONNX model download."
    echo ""
    echo "  Download MobileFaceNet from one of these sources:"
    echo ""
    echo "  Option A - InsightFace model zoo (recommended):"
    echo "    https://github.com/deepinsight/insightface/tree/master/recognition"
    echo "    Look for: mobilefacenet or buffalo_l model pack"
    echo ""
    echo "  Option B - ONNX Model Zoo / HuggingFace:"
    echo "    Search for 'mobilefacenet onnx' models"
    echo ""
    echo "  Option C - Convert from PyTorch/MXNet:"
    echo "    python -c \"import torch; model = ...; torch.onnx.export(model, ...)\""
    echo ""
    echo "  Required specifications:"
    echo "    - Input:  NCHW format, shape [1, 3, 112, 112]"
    echo "    - Output: 128-dimensional embedding vector"
    echo "    - Format: ONNX (.onnx)"
    echo ""
    echo "  Place the model file at:"
    echo "    ${FACENET_ONNX}"
fi

# Create face embedding labels file (not used for classification, but required by nvinfer)
FACENET_LABELS="${FACENET_DIR}/labels.txt"
if [ ! -f "${FACENET_LABELS}" ]; then
    echo "embedding" > "${FACENET_LABELS}"
    echo "[INFO] Created ${FACENET_LABELS}"
fi

echo ""

# ─────────────────────────── Summary ────────────────────────────────────────
echo "=============================================="
echo "  Model Setup Summary"
echo "=============================================="
echo ""
echo "  Face Detection (PGIE):"
if [ -f "${FACEDETECT_ETLT}" ]; then
    echo "    ✅ ${FACEDETECT_ETLT}"
else
    echo "    ❌ MISSING: ${FACEDETECT_ETLT}"
fi
echo "    Labels: ${FACEDETECT_LABELS}"
echo ""
echo "  Face Embedding (SGIE):"
if [ -f "${FACENET_ONNX}" ]; then
    echo "    ✅ ${FACENET_ONNX}"
else
    echo "    ❌ MISSING: ${FACENET_ONNX}"
fi
echo "    Labels: ${FACENET_LABELS}"
echo ""
echo "  After downloading models, generate TensorRT engines"
echo "  by running the pipeline once (auto-conversion on first run)."
echo "=============================================="
