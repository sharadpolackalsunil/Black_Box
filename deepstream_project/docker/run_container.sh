#!/bin/bash
###############################################################################
# run_container.sh
# Launches the DeepStream 9.0 container for DGX Spark (ARM SBSA / Blackwell)
# with GPU, USB camera, display, and workspace volume access.
###############################################################################

set -euo pipefail

# ─────────────────────────── Configuration ───────────────────────────────────
CONTAINER_IMAGE="nvcr.io/nvidia/deepstream:9.0-triton-sbsa-dgx-spark"
CONTAINER_NAME="deepstream-face-recognition"
WORKSPACE_HOST="${HOME}/deepstream_project"
WORKSPACE_CONTAINER="/workspace"
CAMERA_DEVICE="/dev/video0"

# ─────────────────────────── Pre-flight Checks ──────────────────────────────
echo "=============================================="
echo "  DeepStream Face Recognition - Container Launcher"
echo "=============================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Container Toolkit is available
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "[WARNING] NVIDIA Container Toolkit may not be installed."
    echo "         Install it with: sudo apt-get install nvidia-container-toolkit"
fi

# Check if camera device exists
if [ ! -e "${CAMERA_DEVICE}" ]; then
    echo "[WARNING] Camera device ${CAMERA_DEVICE} not found."
    echo "         Plug in your USB camera before starting the pipeline."
    echo "         Listing available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "         No video devices found."
fi

# Check if workspace directory exists
if [ ! -d "${WORKSPACE_HOST}" ]; then
    echo "[INFO] Workspace directory ${WORKSPACE_HOST} does not exist. Creating..."
    mkdir -p "${WORKSPACE_HOST}"
fi

# ─────────────────────────── X11 Display Setup ──────────────────────────────
echo ""
echo "[INFO] Setting up X11 display forwarding..."
xhost +si:localuser:root 2>/dev/null || {
    echo "[WARNING] xhost command failed. Display forwarding may not work."
    echo "         Try running: xhost +si:localuser:root"
}

# ─────────────────────────── Stop Existing Container ────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] Stopping existing container '${CONTAINER_NAME}'..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null
fi

# ─────────────────────────── Launch Container ───────────────────────────────
echo ""
echo "[INFO] Launching DeepStream container..."
echo "  Image:     ${CONTAINER_IMAGE}"
echo "  Workspace: ${WORKSPACE_HOST} -> ${WORKSPACE_CONTAINER}"
echo "  Camera:    ${CAMERA_DEVICE}"
echo "  Display:   ${DISPLAY:-:0}"
echo ""

docker run -it \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    --privileged \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e CUDA_CACHE_DISABLE=0 \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${WORKSPACE_HOST}:${WORKSPACE_CONTAINER}" \
    --device "${CAMERA_DEVICE}:${CAMERA_DEVICE}" \
    -w "${WORKSPACE_CONTAINER}" \
    "${CONTAINER_IMAGE}" \
    /bin/bash

echo ""
echo "[INFO] Container exited."
