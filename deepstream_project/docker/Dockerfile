###############################################################################
# Dockerfile
# Custom DeepStream 9.0 container for Face Recognition on DGX Spark
# Extends the official Triton ARM SBSA image with Python dependencies.
###############################################################################

FROM nvcr.io/nvidia/deepstream:9.0-triton-sbsa-dgx-spark

# ─────────────────────────── System Dependencies ────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gst-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libgstreamer1.0-dev \
    libglib2.0-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────── Python Dependencies ────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ─────────────────────────── Working Directory ──────────────────────────────
WORKDIR /workspace

# ─────────────────────────── Entry Point ────────────────────────────────────
CMD ["/bin/bash"]
