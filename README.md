# 🎯 DeepStream Face Detection + Recognition System

A complete, containerized face detection and recognition system built with **NVIDIA DeepStream SDK 9.0**, running on **DGX Spark** (Blackwell GB10, ARM SBSA). The system uses a USB camera to detect faces in real-time, extract facial embeddings using MobileFaceNet, match identities against a local database, and display results with bounding boxes and name labels.

> 🛠️ **Looking for Setup Instructions?** 
> Please head over to the **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for a comprehensive, step-by-step tutorial on how to initialize Docker, link project files, and run the pipeline.

---

## 📋 Table of Contents

1. [What We Have Done](#-what-we-have-done)
2. [Architecture Overview](#-architecture-overview)
3. [Deep Dive: Components](#-deep-dive-components)
4. [File Structure](#-file-structure)
5. [Future Goals](#-future-goals)

---

## 🚀 What We Have Done

We have successfully engineered an edge-to-cloud capable, real-time Computer Vision pipeline directly leveraging the power of DGX Spark's Blackwell architecture via ARM SBSA containers. 

### Key Achievements:
- **Zero-Install Host Footprint:** The entire complex GStreamer/DeepStream environment is fully containerized. The host machine requires only standard Docker and NVIDIA drivers.
- **Two-Stage AI Inference Pipeline:** Implemented a robust Primary-Secondary inference design. 
  - *Stage 1:* NVIDIA TAO FaceDetectIR (ResNet18) acts as the high-speed Primary GIE (Face Detector).
  - *Stage 2:* MobileFaceNet executes as the Secondary GIE, extracting highly accurate 128-dimensional facial features from cropped bounding boxes.
- **Custom Hardware-Accelerated Tracking:** Integrated the `NvDCF` (Discriminative Correlation Filter) tracker combined with a Kalman Filter. This ensures faces maintain consistent IDs between frames, reducing the load on the AI inference engines.
- **Python-Native Application Logic:** Avoided restrictive bounding-box-only C++ constraints by attaching custom GStreamer pad probes inside Python (`pyds`). This probe intercepts the raw SGIE tensor memory, decodes it into a NumPy array, and runs a high-speed cosine similarity match against an offline face database.
- **Fully Local "Edge" Recognition:** Operates completely offline. Faces are registered to local `.npy` vector files ensuring extreme privacy and zero latency.

---

## 🧠 Architecture Overview

The system pipeline is entirely built as a Directed Acyclic Graph (DAG) utilizing GStreamer plugins powered by NVIDIA DeepStream hardware acceleration.

```text
┌─────────────────────────────────────────────────────────────────┐
│                    DGX Spark (Host System)                      │
│  [USB Camera]                                    [Display]      │
│       │                                              ▲          │
│       ▼                                              │          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  DeepStream 9.0 Docker Container                         │   │
│  │                                                          │   │
│  │  1. v4l2src (Ingests Camera Feed)                        │   │
│  │       │                                                  │   │
│  │       ▼                                                  │   │
│  │  2. nvstreammux (Batches frames into NVMM memory)        │   │
│  │       │                                                  │   │
│  │       ▼                                                  │   │
│  │  3. nvinfer [PGIE] -- FaceDetectIR (TensorRT FP16)       │   │
│  │       │               Outputs Bounding Boxes             │   │
│  │       ▼                                                  │   │
│  │  4. nvtracker -- NvDCF + Kalman Filter                   │   │
│  │       │          Assigns Persistent Object IDs           │   │
│  │       ▼                                                  │   │
│  │  5. nvinfer [SGIE] -- MobileFaceNet (TensorRT FP16)      │   │
│  │       │               Outputs 128D Tensor per face       │   │
│  │       ▼                                                  │   │
│  │  6. 🐍 PYTHON PROBE (Intercepts Pad Data)                │   │
│  │       ├─ Extract 128D Embedding                          │   │
│  │       ├─ Cosine Similarity vs /workspace/faces/*.npy     │   │
│  │       ├─ Apply Threshold (0.7)                           │   │
│  │       └─ Modify NvDsObjectMeta display text              │   │
│  │       │                                                  │   │
│  │       ▼                                                  │   │
│  │  7. nvdsosd (Hardware-accelerated rendering)             │   │
│  │       │         Draws BBoxes & Labels                    │   │
│  │       ▼                                                  │   │
│  │  8. nveglglessink (X11 output to host monitor)           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Recognition Workflow (Logic Level)
1. **Detection:** The PGIE spots a face and draws a box.
2. **Tracking:** The `nvtracker` remembers this box across multiple frames.
3. **Extraction:** The SGIE takes the isolated face box and generates a 128D numerical fingerprint (embedding).
4. **Matching:** The Python probe (`face_recognition_pipeline.py`) takes this live 128D array and compares it to all `.npy` arrays physically saved in the `faces/` directory using simple Cosine Similarity.
5. **Decision:** 
   - If the highest similarity score $>=$ `0.7`, the box turns Green and the `Name` is displayed.
   - If the highest score $<$ `0.7`, the box turns Red and reads `Unknown`.

---

## 🔍 Deep Dive: Components

### Model 1: Primary Inference (PGIE) - TAO FaceDetectIR
- **Architecture:** ResNet18 (Pruned)
- **Role:** Locate coordinates of human faces in a 640x480 frame.
- **Why we chose it:** It is pre-optimized by NVIDIA (TAO Toolkit) specifically for DeepStream, avoiding the complicated custom bounding-box parsers usually required by YOLO models.

### Model 2: Secondary Inference (SGIE) - MobileFaceNet
- **Architecture:** MobileFaceNet (ONNX)
- **Role:** Look specifically within the coordinates provided by the PGIE, and map the face's unique features to a 128-dimensional vector.
- **Why we chose it:** Highly optimized for edge hardware (ARM cores). It is lighter than default FaceNet but retains extremely high accuracy, fitting perfectly within real-time streaming constraints.

### The Match Engine: Cosine Similarity
We use a pure mathematics approach rather than a classifier model for identification. By normalizing the 128D vectors and computing the dot product (Cosine Similarity), we can instantly measure the distance between the "Live Face" and "Registered Profile". A threshold of `0.7` controls strictness.

---

## 📁 File Structure

```text
~/deepstream_project/
├── SETUP_GUIDE.md                      # Extensive visual Setup Instructions
├── README.md                           # Architecture and Overview (This File)
├── requirements.txt                    # Python library requirements
│
├── configs/
│   ├── pgie_facedetect_config.txt      # Face detection config (Stage 1)
│   ├── sgie_facenet_config.txt         # Embedding config (Stage 2)
│   ├── tracker_config.yml              # Tracking algorithm parameter tuning
│   └── deepstream_app_config.txt       # Sandbox CLI config file
│
├── docker/
│   ├── run_container.sh                # Hardcoded secure container launcher
│   └── Dockerfile                      # Dev container build definitions
│
├── models/
│   ├── facedetect/                     # PGIE .etlt models and labels
│   └── facenet/                        # SGIE .onnx models and labels
│
├── faces/                              # Local Database Directory
│   └── .gitkeep                        # Expects [name].npy files here
│
└── scripts/
    ├── face_recognition_pipeline.py    # MAIN EXECUTION SCRIPT (The Pipeline)
    ├── face_match.py                   # Math/Logic for Cosine Similarity
    ├── register_face.py                # CLI tool to snap faces to .npy 
    └── download_models.sh              # Automates AI model acquisition
```

---

## 🔮 Future Goals

This pipeline represents the foundational "Edge" implementation of the face recognition system. Future iterations are planned to expand this into a production-level enterprise application.

1. **📊 Cloud Database Integration:**
   - Migrate from local `.npy` flat files to a robust vector database (like Milvus or Qdrant) or a standard relational database (PostgreSQL).
   - Allows for 1-to-N matching over thousands of identities without slowing down the pipeline thread.

2. **📹 Multi-Camera Fleet Scaling:**
   - Modify the `nvstreammux` batching to handle 4, 8, or 16 concurrent RTSP IP-Camera streams simultaneously.
   - Utilize DGX Blackwell's full multi-processing capabilities.

3. **📋 Automated Attendance Logging & Events:**
   - Attach a Kafka message broker (`nvmsgconv` and `nvmsgbroker` DeepStream plugins) to the pipeline.
   - Fire asynchronous JSON payload events to a backend server whenever a recognized face is in frame, acting as an automated check-in/attendance logging system.

4. **🌐 Web Dashboard API:**
   - Build a lightweight FastAPI/Flask backend API.
   - Provide a GUI for administrators to upload photos for registration and monitor the live RTSP feed directly in a web browser.
