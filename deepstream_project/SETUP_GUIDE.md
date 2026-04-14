# 🛠️ DeepStream 9.0 Face Recognition Setup Guide

Welcome to the comprehensive, step-by-step setup guide for the DeepStream 9.0 Face Detection and Recognition System on the NVIDIA DGX Spark (ARM SBSA).

This guide is designed to be copy-paste friendly. Pay close attention to **Where to run** and **Which directory (`cd`)** for each block of commands.

---

## 🛑 Before You Begin: Understanding the Environments

You will be working in two distinct environments. Please keep track of which terminal you are using:

1. 💻 **Host Environment (DGX Spark):** This is your main Linux machine. You will download the project files and start the Docker container from here.
2. 🐳 **Container Environment (DeepStream Docker):** This is the isolated environment *inside* Docker where the actual DeepStream software, your models, and your Python scripts run.

---

## Part 1: Host System Preparation

Run these steps on your **💻 Host Environment (DGX Spark)**.

### 1. Verify NVIDIA Driver & Docker
Make sure your Blackwell GPU and Docker with the NVIDIA Container Toolkit are ready.

**Where to run:** 💻 Host Terminal (Any directory)

```bash
# Check GPU availability
nvidia-smi

# Check Docker installation
docker --version

# Check NVIDIA Container Toolkit
dpkg -l | grep nvidia-container-toolkit
```
*(If `nvidia-container-toolkit` is missing, run: `sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit` and restart docker with `sudo systemctl restart docker`)*

### 2. Create and Link Project Files
We need a dedicated folder on your host machine that will be shared (linked) into the Docker container. This way, any code changes you make outside are instantly available inside the container.

**Where to run:** 💻 Host Terminal
**Target Directory:** Your Home Directory (`~`)

```bash
# Navigate to your home directory
cd ~

# Create the main project folder and all necessary subfolders
mkdir -p ~/deepstream_project/{configs,models/facedetect,models/facenet,faces,scripts,docker}

# Verify the folders were created
ls -l ~/deepstream_project
```
*(Note: At this point, ensure all the project scripts like `face_recognition_pipeline.py`, bash scripts, and configs are placed inside these folders. If you cloned a git repo, you are already set).*

### 3. Allow Display Access for Docker
Your DeepStream pipeline needs to open a video window on your screen. You must explicitly allow your host's X11 server to accept connections from local users.

**Where to run:** 💻 Host Terminal (Any directory)

```bash
# Allow local connections to X11 display
xhost +si:localuser:root
```

---

## Part 2: Docker Initialization & Launch

Run these steps on your **💻 Host Environment (DGX Spark)**.

### 1. Pull the DeepStream Docker Image
We are using the specific Blackwell-optimized ARM SBSA tag for DeepStream 9.0.

**Where to run:** 💻 Host Terminal (Any directory)

```bash
# Pull the official DeepStream container for DGX Spark
docker pull nvcr.io/nvidia/deepstream:9.0-triton-sbsa-dgx-spark
```

### 2. Launch the Container
We will now start the container. This command links your USB camera (`/dev/video0`), shares your X11 display, gives full GPU access, and links your `~/deepstream_project` folder to `/workspace` inside the container.

**Where to run:** 💻 Host Terminal
**Target Directory:** `~/deepstream_project`

```bash
# Navigate to the project directory
cd ~/deepstream_project

# Launch the container interactively
docker run -it \
    --name deepstream-face-recognition \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e CUDA_CACHE_DISABLE=0 \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ~/deepstream_project:/workspace \
    --device /dev/video0:/dev/video0 \
    -w /workspace \
    nvcr.io/nvidia/deepstream:9.0-triton-sbsa-dgx-spark \
    /bin/bash
```

If successful, your terminal prompt will change to something like `root@<container-id>:/workspace#`. **You are now inside the Docker container.**

---

## Part 3: Container Setup & Dependency Installation

Run these steps in the **🐳 Container Environment**.

### 1. Install Required Python Packages
DeepStream 9.0 needs a few extra Python packages to handle the face embedding matching logic and camera access.

**Where to run:** 🐳 Container Terminal
**Target Directory:** `/workspace`

```bash
# Verify you are in the workspace
pwd

# Install required numerical, AI, and imaging libraries
pip3 install numpy scipy onnxruntime-gpu opencv-python-headless Pillow
```

### 2. Download the AI Models
You need the NVIDIA TAO FaceDetectIR model (for detecting faces) and the MobileFaceNet ONNX model (for extracting face embeddings). We have a script that automates this.

**Where to run:** 🐳 Container Terminal
**Target Directory:** `/workspace`

```bash
# Use the provided model download script
bash scripts/download_models.sh
```
*(If the NGC CLI is not set up inside your container, the script will provide manual download links. Be sure `resnet18_facedetectir_pruned.etlt` is in `models/facedetect/` and `mobilefacenet.onnx` is in `models/facenet/`)*.

---

## Part 4: Registering Faces & Running the System

Run these steps in the **🐳 Container Environment**.

### 1. Register Known Faces
The system requires a database of known faces (saved as `.npy` embedding files) to match against.

**Where to run:** 🐳 Container Terminal
**Target Directory:** `/workspace`

```bash
# Copy a photo of a person (e.g., 'sharad.jpg') into your ~/deepstream_project 
# on your host machine. Because of the volume link, it instantly appears in /workspace.

# Run the registration script (replace 'sharad.jpg' and 'sharad' with your file and name)
python3 scripts/register_face.py --image sharad.jpg --name sharad

# (Optional) Register using a live snapshot from your USB camera
python3 scripts/register_face.py --camera --name rahul
```

### 2. Start the Real-Time Recognition Pipeline
With models downloaded and faces registered, you can now launch the real-time AI pipeline.

**Where to run:** 🐳 Container Terminal
**Target Directory:** `/workspace`

```bash
# Run the pipeline with default settings
python3 scripts/face_recognition_pipeline.py

# To stop the pipeline, press:
# Ctrl + C
```

### Alternative Run Commands

You can customize the pipeline execution directly from the command line:

```bash
# Run with a stricter matching threshold
python3 scripts/face_recognition_pipeline.py --threshold 0.8

# Run with a 720p HD Camera (if your camera supports it)
python3 scripts/face_recognition_pipeline.py --width 1280 --height 720

# Run without displaying the video window (useful for debugging performance)
python3 scripts/face_recognition_pipeline.py --no-display
```

---

## 🎉 Congratulations!
You have successfully deployed a real-time, hardware-accelerated face recognition system. See the main `README.md` for architectural details and future project goals.
