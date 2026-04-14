#!/usr/bin/env python3
"""
register_face.py — Face Registration Utility

Takes a face image as input, extracts the face embedding using the same
MobileFaceNet ONNX model used by the DeepStream pipeline, and saves the
embedding as a .npy file in the local face database.

Usage (inside Docker container):
    # Register a face from an image
    python3 scripts/register_face.py --image photo.jpg --name sharad

    # Register with explicit model path
    python3 scripts/register_face.py --image photo.jpg --name rahul \
        --model /workspace/models/facenet/mobilefacenet.onnx

    # Register from webcam snapshot
    python3 scripts/register_face.py --camera --name sharad

Dependencies:
    pip install onnxruntime-gpu opencv-python-headless numpy
"""

import os
import sys
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV not installed. Run: pip install opencv-python-headless")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] ONNX Runtime not installed. Run: pip install onnxruntime-gpu")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL_PATH = "/workspace/models/facenet/mobilefacenet.onnx"
DEFAULT_FACES_DIR = "/workspace/faces"
FACE_INPUT_SIZE = (112, 112)  # MobileFaceNet input resolution
EMBEDDING_DIM = 128


# ═════════════════════════════════════════════════════════════════════════════
# Face Detection (Simple OpenCV-based for registration)
# ═════════════════════════════════════════════════════════════════════════════

def detect_face_opencv(image: np.ndarray) -> np.ndarray:
    """
    Detect the largest face in an image using OpenCV's DNN face detector
    or Haar cascade as fallback.

    Args:
        image: BGR image as numpy array.

    Returns:
        Cropped face region as numpy array, or None if no face found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try OpenCV's built-in Haar cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        print("[WARNING] No face detected in the image.")
        print("          Please provide a clear frontal face photo.")
        return None

    # Use the largest detected face
    if len(faces) > 1:
        print(f"[INFO] {len(faces)} faces detected. Using the largest one.")
        areas = [w * h for (x, y, w, h) in faces]
        largest_idx = np.argmax(areas)
        faces = [faces[largest_idx]]

    x, y, w, h = faces[0]

    # Add margin around the face (20%)
    margin = int(max(w, h) * 0.2)
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)

    face_crop = image[y1:y2, x1:x2]
    print(f"[INFO] Face detected at ({x}, {y}, {w}, {h}), "
          f"crop region: ({x1}, {y1}) → ({x2}, {y2})")

    return face_crop


# ═════════════════════════════════════════════════════════════════════════════
# Face Embedding Extraction
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_face(face_image: np.ndarray) -> np.ndarray:
    """
    Preprocess a face image for MobileFaceNet inference.

    Steps:
        1. Resize to 112x112
        2. Convert BGR → RGB
        3. Normalize pixel values: (pixel - 127.5) / 128.0
        4. Transpose to NCHW format
        5. Add batch dimension

    Args:
        face_image: BGR face crop as numpy array.

    Returns:
        Preprocessed tensor of shape (1, 3, 112, 112).
    """
    # Resize to model's expected input size
    face_resized = cv2.resize(face_image, FACE_INPUT_SIZE)

    # BGR → RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # Normalize: (pixel - 127.5) / 128.0 → range [-1, 1]
    face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0

    # HWC → CHW → NCHW
    face_transposed = np.transpose(face_normalized, (2, 0, 1))
    face_batch = np.expand_dims(face_transposed, axis=0)

    return face_batch


def extract_embedding(model_path: str, face_image: np.ndarray) -> np.ndarray:
    """
    Extract face embedding using MobileFaceNet ONNX model.

    Args:
        model_path:  Path to the MobileFaceNet ONNX model file.
        face_image:  BGR face crop as numpy array.

    Returns:
        1D embedding vector of shape (128,).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Download MobileFaceNet ONNX model first.\n"
            f"Run: bash scripts/download_models.sh"
        )

    # Preprocess the face image
    input_tensor = preprocess_face(face_image)

    # Create ONNX Runtime session
    print(f"[INFO] Loading model: {model_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    # Get input/output info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape

    print(f"[INFO] Model input:  {input_name} {input_shape}")
    print(f"[INFO] Model output: {output_name} {output_shape}")

    # Run inference
    print("[INFO] Running MobileFaceNet inference...")
    result = session.run([output_name], {input_name: input_tensor})
    embedding = result[0].flatten()

    # L2-normalize the embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    print(f"[INFO] Embedding extracted: shape={embedding.shape}, "
          f"norm={np.linalg.norm(embedding):.4f}")

    return embedding


# ═════════════════════════════════════════════════════════════════════════════
# Camera Capture
# ═════════════════════════════════════════════════════════════════════════════

def capture_from_camera(device: int = 0) -> np.ndarray:
    """
    Capture a single frame from a USB camera.

    Args:
        device: Camera device index (0 = /dev/video0).

    Returns:
        BGR image as numpy array.
    """
    print(f"[INFO] Opening camera /dev/video{device}...")
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera /dev/video{device}. "
            f"Ensure the camera is connected and accessible."
        )

    # Warm up the camera (skip first few frames)
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from camera.")

    print(f"[INFO] Captured frame: {frame.shape}")
    return frame


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Register a face in the local face database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register from image file
  python3 scripts/register_face.py --image photo.jpg --name sharad

  # Register from camera snapshot
  python3 scripts/register_face.py --camera --name rahul

  # Register with custom model and database paths
  python3 scripts/register_face.py --image face.png --name alice \\
      --model /path/to/mobilefacenet.onnx --faces-dir /path/to/faces
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str,
        help="Path to face image file"
    )
    input_group.add_argument(
        "--camera", action="store_true",
        help="Capture face from USB camera"
    )

    parser.add_argument(
        "--name", type=str, required=True,
        help="Name/identity to register (used as filename)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Path to MobileFaceNet ONNX model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--faces-dir", type=str, default=DEFAULT_FACES_DIR,
        help=f"Path to face database directory (default: {DEFAULT_FACES_DIR})"
    )
    parser.add_argument(
        "--camera-device", type=int, default=0,
        help="Camera device index (default: 0 = /dev/video0)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing embedding without confirmation"
    )

    return parser.parse_args()


def main():
    """Main entry point for face registration."""
    args = parse_args()

    print("=" * 60)
    print("  Face Registration Tool — MobileFaceNet (128D)")
    print("=" * 60)
    print(f"  Name:      {args.name}")
    print(f"  Model:     {args.model}")
    print(f"  Faces Dir: {args.faces_dir}")
    print("=" * 60)
    print()

    # ── Check for existing registration ──
    os.makedirs(args.faces_dir, exist_ok=True)
    output_path = os.path.join(args.faces_dir, f"{args.name}.npy")

    if os.path.exists(output_path) and not args.force:
        print(f"[WARNING] Embedding already exists: {output_path}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("[INFO] Registration cancelled.")
            return

    # ── Get input image ──
    if args.camera:
        image = capture_from_camera(args.camera_device)
    else:
        if not os.path.isfile(args.image):
            print(f"[ERROR] Image file not found: {args.image}")
            sys.exit(1)
        print(f"[INFO] Loading image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"[ERROR] Failed to read image: {args.image}")
            sys.exit(1)
        print(f"[INFO] Image loaded: {image.shape}")

    # ── Detect face ──
    print("\n[Step 1/3] Detecting face...")
    face_crop = detect_face_opencv(image)
    if face_crop is None:
        print("[ERROR] No face detected. Please try with a better image.")
        sys.exit(1)

    # ── Extract embedding ──
    print("\n[Step 2/3] Extracting face embedding...")
    embedding = extract_embedding(args.model, face_crop)

    # ── Save embedding ──
    print(f"\n[Step 3/3] Saving embedding...")
    np.save(output_path, embedding)
    print(f"[SUCCESS] Embedding saved to: {output_path}")
    print(f"          Name: {args.name}")
    print(f"          Dimensions: {embedding.shape[0]}D")
    print(f"          L2 Norm: {np.linalg.norm(embedding):.4f}")

    # ── Verification ──
    loaded = np.load(output_path)
    assert np.allclose(embedding, loaded), "Verification failed!"
    print(f"[VERIFIED] Embedding file is valid.\n")

    print("=" * 60)
    print(f"  ✅ '{args.name}' has been registered successfully!")
    print(f"     The pipeline will now recognize this person.")
    print("=" * 60)


if __name__ == "__main__":
    main()
