#!/usr/bin/env python3
"""
face_recognition_pipeline.py — DeepStream Face Detection + Recognition Pipeline

Custom GStreamer Python pipeline using NVIDIA DeepStream SDK 9.0
for real-time face detection, tracking, embedding extraction, and
identity matching on DGX Spark (ARM SBSA / Blackwell GPU).

Pipeline Architecture:
    v4l2src → videoconvert → capsfilter → nvvideoconvert → capsfilter →
    nvstreammux → nvinfer(PGIE) → nvtracker → nvinfer(SGIE) →
    nvvideoconvert → nvdsosd → nveglglessink

Usage (inside Docker container):
    cd /workspace
    python3 scripts/face_recognition_pipeline.py [options]

Options:
    --camera-device   Camera device index (default: 0 = /dev/video0)
    --width           Camera width (default: 640)
    --height          Camera height (default: 480)
    --threshold       Face matching threshold (default: 0.7)
    --faces-dir       Path to face embeddings database (default: /workspace/faces)
    --pgie-config     Path to PGIE config (default: /workspace/configs/pgie_facedetect_config.txt)
    --sgie-config     Path to SGIE config (default: /workspace/configs/sgie_facenet_config.txt)
    --tracker-config  Path to tracker config (default: /workspace/configs/tracker_config.yml)
    --no-display      Run without display (use fakesink)
"""

import sys
import os
import argparse
import ctypes
import numpy as np

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GLib

# DeepStream Python bindings
import pyds

# Local face matching module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_match import FaceDatabase


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════
PGIE_UNIQUE_ID = 1
SGIE_UNIQUE_ID = 2
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
EMBEDDING_DIM = 128  # MobileFaceNet output dimensions

# Global face database instance
face_db: FaceDatabase = None

# Frame counter for performance logging
frame_count = 0
face_count_total = 0


# ═════════════════════════════════════════════════════════════════════════════
# Probe Functions
# ═════════════════════════════════════════════════════════════════════════════

def sgie_src_pad_probe(pad, info, u_data):
    """
    Probe attached to the SGIE source pad.

    Extracts face embeddings from the SGIE tensor output metadata,
    performs cosine similarity matching against the face database,
    and updates the OSD display text with the matched identity.

    This is the CORE of the face recognition logic.
    """
    global face_db, frame_count, face_count_total

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_count += 1
        faces_in_frame = 0

        # ── Iterate over detected objects in this frame ──
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            faces_in_frame += 1
            face_count_total += 1
            identity = "Unknown"
            score = 0.0

            # ── Extract embedding from SGIE tensor output ──
            embedding = _extract_embedding_from_object(obj_meta)

            if embedding is not None:
                # ── Match against face database ──
                identity, score = face_db.find_best_match(embedding)

            # ── Update display label ──
            _set_object_label(obj_meta, identity, score)

            # ── Style bounding box based on match result ──
            _style_bbox(obj_meta, identity)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # ── Performance logging every 100 frames ──
        if frame_count % 100 == 0:
            print(f"[Pipeline] Frame #{frame_count} | "
                  f"Faces this frame: {faces_in_frame} | "
                  f"Total faces processed: {face_count_total}")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def _extract_embedding_from_object(obj_meta) -> np.ndarray:
    """
    Extract the face embedding vector from an object's SGIE tensor metadata.

    The SGIE must have output-tensor-meta=1 in its config for this to work.

    Args:
        obj_meta: NvDsObjectMeta from DeepStream.

    Returns:
        numpy array of shape (128,) or None if extraction fails.
    """
    # Traverse the object-level user metadata list
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break

        # Check if this metadata is a tensor output from our SGIE
        if (user_meta.base_meta.meta_type ==
                pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):

            tensor_meta = pyds.NvDsInferTensorMeta.cast(
                user_meta.user_meta_data
            )

            # Verify this tensor came from our SGIE
            if tensor_meta.unique_id == SGIE_UNIQUE_ID:
                # Extract the first output layer (the embedding)
                if tensor_meta.num_output_layers > 0:
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    num_elements = layer.inferDims.numElements

                    if num_elements > 0:
                        # Convert C buffer pointer to numpy array
                        ptr = ctypes.cast(
                            pyds.get_ptr(layer.buffer),
                            ctypes.POINTER(ctypes.c_float)
                        )
                        embedding = np.ctypeslib.as_array(
                            ptr, shape=(num_elements,)
                        ).copy()  # .copy() to own the memory

                        return embedding

        try:
            l_user = l_user.next
        except StopIteration:
            break

    return None


def _set_object_label(obj_meta, identity: str, score: float) -> None:
    """
    Set the display text for an object's OSD label.

    Args:
        obj_meta: NvDsObjectMeta from DeepStream.
        identity: Matched identity name or "Unknown".
        score:    Cosine similarity score.
    """
    txt_params = obj_meta.text_params

    if identity != "Unknown":
        txt_params.display_text = f"{identity} ({score:.2f})"
    else:
        txt_params.display_text = f"Unknown ({score:.2f})"

    # ── Text appearance ──
    txt_params.x_offset = max(0, int(obj_meta.rect_params.left))
    txt_params.y_offset = max(0, int(obj_meta.rect_params.top) - 25)

    # Font settings
    txt_params.font_params.font_name = "Serif"
    txt_params.font_params.font_size = 14

    # Text color: white
    txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

    # Background color: green for known, red for unknown
    if identity != "Unknown":
        txt_params.set_bg_clr = 1
        txt_params.text_bg_clr.set(0.0, 0.6, 0.0, 0.7)  # Green bg
    else:
        txt_params.set_bg_clr = 1
        txt_params.text_bg_clr.set(0.7, 0.0, 0.0, 0.7)  # Red bg


def _style_bbox(obj_meta, identity: str) -> None:
    """
    Style the bounding box based on recognition result.

    Args:
        obj_meta: NvDsObjectMeta from DeepStream.
        identity: Matched identity name or "Unknown".
    """
    rect_params = obj_meta.rect_params

    # Border width
    rect_params.border_width = 3

    if identity != "Unknown":
        # Green border for recognized faces
        rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
    else:
        # Red border for unknown faces
        rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)

    # Semi-transparent fill
    rect_params.has_bg_color = 0


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline Construction
# ═════════════════════════════════════════════════════════════════════════════

def create_pipeline(args) -> Gst.Pipeline:
    """
    Build the complete GStreamer pipeline for face recognition.

    Pipeline:
        v4l2src → videoconvert → capsfilter(NV12) → nvvideoconvert →
        capsfilter(NV12,memory:NVMM) → nvstreammux → nvinfer(PGIE) →
        nvtracker → nvinfer(SGIE) → nvvideoconvert → nvdsosd →
        nveglglessink/fakesink

    Args:
        args: Parsed command-line arguments.

    Returns:
        Gst.Pipeline instance.
    """
    print("=" * 60)
    print("  DeepStream Face Recognition Pipeline")
    print("=" * 60)
    print(f"  Camera:      /dev/video{args.camera_device}")
    print(f"  Resolution:  {args.width}x{args.height}")
    print(f"  Threshold:   {args.threshold}")
    print(f"  Faces DB:    {args.faces_dir}")
    print(f"  Display:     {'Disabled' if args.no_display else 'Enabled'}")
    print("=" * 60)
    print()

    # ── Initialize GStreamer ──
    Gst.init(None)
    pipeline = Gst.Pipeline()
    if not pipeline:
        raise RuntimeError("Failed to create GStreamer pipeline")

    # ═══════════════════ Source ═══════════════════════════════════════════
    print("[Pipeline] Creating source elements...")

    # V4L2 camera source
    source = _make_element("v4l2src", "usb-cam-source")
    source.set_property("device", f"/dev/video{args.camera_device}")

    # Video convert (CPU → GPU-friendly format)
    vidconv_src = _make_element("videoconvert", "src-videoconvert")

    # Caps filter to set format after videoconvert
    caps_src = _make_element("capsfilter", "src-capsfilter")
    caps_src.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw, format=NV12, width={args.width}, height={args.height}"
        )
    )

    # NV video convert — CPU to GPU memory
    nvvidconv_src = _make_element("nvvideoconvert", "src-nvvideoconvert")

    # Caps filter for NVMM memory
    caps_nvmm = _make_element("capsfilter", "nvmm-capsfilter")
    caps_nvmm.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), format=NV12, "
            f"width={args.width}, height={args.height}"
        )
    )

    # ═══════════════════ Stream Muxer ════════════════════════════════════
    print("[Pipeline] Creating stream muxer...")
    streammux = _make_element("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", args.width)
    streammux.set_property("height", args.height)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("live-source", 1)

    # ═══════════════════ Primary GIE (Face Detection) ═══════════════════
    print("[Pipeline] Creating Primary GIE (Face Detection)...")
    pgie = _make_element("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", args.pgie_config)

    # ═══════════════════ Tracker ════════════════════════════════════════
    print("[Pipeline] Creating Tracker...")
    tracker = _make_element("nvtracker", "tracker")
    tracker.set_property("tracker-width", args.width)
    tracker.set_property("tracker-height", args.height)
    tracker.set_property("ll-lib-file", TRACKER_LIB)
    tracker.set_property("ll-config-file", args.tracker_config)
    tracker.set_property("gpu-id", 0)

    # ═══════════════════ Secondary GIE (Face Embedding) ═════════════════
    print("[Pipeline] Creating Secondary GIE (MobileFaceNet Embedding)...")
    sgie = _make_element("nvinfer", "secondary-inference")
    sgie.set_property("config-file-path", args.sgie_config)

    # ═══════════════════ Video Convert (for OSD) ════════════════════════
    nvvidconv_osd = _make_element("nvvideoconvert", "osd-nvvideoconvert")

    # ═══════════════════ OSD (On-Screen Display) ════════════════════════
    print("[Pipeline] Creating OSD...")
    osd = _make_element("nvdsosd", "on-screen-display")
    osd.set_property("process-mode", 0)  # CPU mode for text
    osd.set_property("display-text", 1)

    # ═══════════════════ Sink ═══════════════════════════════════════════
    if args.no_display:
        print("[Pipeline] Creating fakesink (no display)...")
        sink = _make_element("fakesink", "video-sink")
        sink.set_property("sync", 0)
    else:
        print("[Pipeline] Creating EGL sink (display output)...")
        # Try nveglglessink first, fallback to nv3dsink
        try:
            sink = _make_element("nveglglessink", "video-sink")
        except RuntimeError:
            print("[Pipeline] nveglglessink not available, trying nv3dsink...")
            sink = _make_element("nv3dsink", "video-sink")
        sink.set_property("sync", 0)

    # ═══════════════════ Add Elements to Pipeline ═══════════════════════
    print("[Pipeline] Adding elements to pipeline...")
    for element in [source, vidconv_src, caps_src, nvvidconv_src, caps_nvmm,
                    streammux, pgie, tracker, sgie, nvvidconv_osd, osd, sink]:
        pipeline.add(element)

    # ═══════════════════ Link Elements ══════════════════════════════════
    print("[Pipeline] Linking elements...")

    # Source chain: v4l2src → videoconvert → caps → nvvideoconvert → caps(NVMM)
    _link(source, vidconv_src)
    _link(vidconv_src, caps_src)
    _link(caps_src, nvvidconv_src)
    _link(nvvidconv_src, caps_nvmm)

    # Connect to streammux sink pad
    sinkpad = streammux.request_pad_simple("sink_0")
    if not sinkpad:
        # Fallback for older GStreamer versions
        sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_nvmm.get_static_pad("src")
    srcpad.link(sinkpad)

    # Processing chain: streammux → PGIE → tracker → SGIE → convert → OSD → sink
    _link(streammux, pgie)
    _link(pgie, tracker)
    _link(tracker, sgie)
    _link(sgie, nvvidconv_osd)
    _link(nvvidconv_osd, osd)
    _link(osd, sink)

    # ═══════════════════ Attach Probe ══════════════════════════════════
    print("[Pipeline] Attaching face recognition probe to SGIE src pad...")
    sgie_srcpad = sgie.get_static_pad("src")
    if not sgie_srcpad:
        raise RuntimeError("Failed to get SGIE source pad")
    sgie_srcpad.add_probe(
        Gst.PadProbeType.BUFFER,
        sgie_src_pad_probe,
        0
    )

    print("[Pipeline] Pipeline construction complete!\n")
    return pipeline


def _make_element(element_type: str, name: str) -> Gst.Element:
    """Create a GStreamer element or raise an error."""
    element = Gst.ElementFactory.make(element_type, name)
    if not element:
        raise RuntimeError(
            f"Failed to create GStreamer element: {element_type} ({name}). "
            f"Ensure the DeepStream plugins are installed."
        )
    return element


def _link(src: Gst.Element, dst: Gst.Element) -> None:
    """Link two GStreamer elements or raise an error."""
    if not src.link(dst):
        raise RuntimeError(
            f"Failed to link {src.get_name()} → {dst.get_name()}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Bus Message Handler
# ═════════════════════════════════════════════════════════════════════════════

def bus_call(bus, message, loop):
    """Handle GStreamer bus messages."""
    t = message.type

    if t == Gst.MessageType.EOS:
        print("\n[Pipeline] End of stream reached.")
        loop.quit()

    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\n[Pipeline] ERROR: {err}")
        if debug:
            print(f"[Pipeline] Debug: {debug}")
        loop.quit()

    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"[Pipeline] WARNING: {err}")

    elif t == Gst.MessageType.STATE_CHANGED:
        if message.src == message.src.get_parent():
            old, new, pending = message.parse_state_changed()
            print(f"[Pipeline] State: {old.value_nick} → {new.value_nick}")

    return True


# ═════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepStream Face Detection + Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (camera 0, display on, threshold 0.7)
  python3 scripts/face_recognition_pipeline.py

  # Run with different camera and threshold
  python3 scripts/face_recognition_pipeline.py --camera-device 1 --threshold 0.8

  # Run without display (headless testing)
  python3 scripts/face_recognition_pipeline.py --no-display
        """
    )

    parser.add_argument(
        "--camera-device", type=int, default=0,
        help="Camera device index (0 = /dev/video0)"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Camera capture width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Camera capture height (default: 480)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Face similarity threshold (default: 0.7 for MobileFaceNet)"
    )
    parser.add_argument(
        "--faces-dir", type=str, default="/workspace/faces",
        help="Path to face embeddings database directory"
    )
    parser.add_argument(
        "--pgie-config", type=str,
        default="/workspace/configs/pgie_facedetect_config.txt",
        help="Path to PGIE config file"
    )
    parser.add_argument(
        "--sgie-config", type=str,
        default="/workspace/configs/sgie_facenet_config.txt",
        help="Path to SGIE config file"
    )
    parser.add_argument(
        "--tracker-config", type=str,
        default="/workspace/configs/tracker_config.yml",
        help="Path to tracker config file"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without display output (uses fakesink)"
    )

    return parser.parse_args()


def main():
    """Main entry point — build pipeline, load faces, run."""
    global face_db

    args = parse_args()

    # ── Load face database ──
    print("\n[FaceDB] Loading face database...")
    face_db = FaceDatabase(args.faces_dir, threshold=args.threshold)
    print(f"[FaceDB] {face_db}\n")

    # ── Build pipeline ──
    pipeline = create_pipeline(args)

    # ── Set up bus message handler ──
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # ── Start pipeline ──
    print("[Pipeline] Starting pipeline — press Ctrl+C to stop...\n")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[Pipeline] ERROR: Failed to start pipeline!")
        sys.exit(1)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.")
    finally:
        print("[Pipeline] Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        print(f"[Pipeline] Total frames processed: {frame_count}")
        print(f"[Pipeline] Total faces processed:  {face_count_total}")
        print("[Pipeline] Pipeline stopped. Goodbye!")


if __name__ == "__main__":
    main()
