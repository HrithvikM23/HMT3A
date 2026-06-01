from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import core.config as app_config
from core.backend_selection import BODY_BACKENDS, HAND_BACKENDS, LANDMARK_BACKENDS
from core.mediapipe_models import mediapipe_pose_model_names
from core.runtime_profiles import PROFILE_FASTEST, PROFILE_NAMES


@dataclass(frozen=True, slots=True)
class InputAssignment:
    label: str
    source: int | Path


def parse_color(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Colors must be provided as B,G,R.")

    try:
        color_values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color values must be integers.") from exc

    color = (color_values[0], color_values[1], color_values[2])
    if any(channel < 0 or channel > 255 for channel in color):
        raise argparse.ArgumentTypeError("Each color channel must be between 0 and 255.")
    return color


def parse_identity_hint(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Identity hints must look like 'person1=black,orange'.")
    label, colors_raw = value.split("=", 1)
    normalized_label = label.strip().lower()
    colors = tuple(color.strip().lower() for color in colors_raw.split(",") if color.strip())
    if not normalized_label or not colors:
        raise argparse.ArgumentTypeError("Identity hints must include a label and at least one color.")
    return normalized_label, colors


def parse_sync_offset(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Sync offsets must look like 'cam_1=3'.")
    label, offset_raw = value.split("=", 1)
    normalized_label = label.strip().upper()
    if not normalized_label:
        raise argparse.ArgumentTypeError("Sync offset label must not be empty.")
    try:
        offset = int(offset_raw.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Sync offset must be an integer frame count.") from exc
    return normalized_label, offset


def sanitize_label(label: str) -> str:
    return label.strip().lower()


def default_camera_label(index: int) -> str:
    return f"CAM_{index}"


def choose_video_gui(title: str = "Select Video File") -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ModuleNotFoundError:
        print("Error: tkinter is not available in this Python environment.")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def choose_camera_assignments_gui() -> list[InputAssignment]:
    print("How many cameras do you want to assign?")
    count_raw = input("Enter camera count [1-4]: ").strip()
    camera_count = int(count_raw) if count_raw.isdigit() else 1
    camera_count = max(1, min(4, camera_count))

    assignments: list[InputAssignment] = []
    for camera_index in range(camera_count):
        label = default_camera_label(camera_index)
        path = choose_video_gui(f"Select Video File for {label}")
        if not path:
            print(f"No file selected for {label}.")
            return []
        assignments.append(InputAssignment(label=label, source=Path(path)))

    return assignments


def resolve_sources(args: argparse.Namespace) -> list[InputAssignment]:
    if args.source is not None:
        assignments: list[InputAssignment] = []
        used_labels: set[str] = set()
        for source_index, raw_source in enumerate(args.source):
            source_text = raw_source.strip()
            label: str | None = None
            value_text = source_text
            if "=" in source_text:
                raw_label, raw_value = source_text.split("=", 1)
                label = sanitize_label(raw_label).upper()
                value_text = raw_value.strip()

            if label is None:
                label = default_camera_label(source_index)

            if label in used_labels:
                print(f"Error: duplicate source label: {label}")
                return []
            used_labels.add(label)

            if value_text.isdigit():
                assignments.append(InputAssignment(label=label, source=int(value_text)))
                continue

            path = Path(value_text)
            if not path.exists():
                print(f"Error: file not found: {path}")
                return []
            assignments.append(InputAssignment(label=label, source=path))

        return assignments

    print("Select input source:")
    print("  1. Webcam")
    print("  2. Video file(s)")
    choice = input("Enter choice [1/2]: ").strip()

    if choice == "1":
        idx = input("Webcam index: ").strip()
        return [InputAssignment(label=default_camera_label(0), source=int(idx) if idx.isdigit() else 0)]

    if choice == "2":
        return choose_camera_assignments_gui()

    print("Invalid choice.")
    return []


def explicit_option_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {
        option_string: action.dest
        for action in parser._actions
        for option_string in action.option_strings
        if action.dest != "help"
    }
    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit.add(dest)
    return explicit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pose and Hand Landmark Pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config file whose keys match CLI destination names. Explicit CLI flags override config values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources, outputs, runtime, and backend choices without opening video or running models.",
    )
    parser.add_argument(
        "--runtime-check",
        action="store_true",
        help="Check Kinara runtime dependencies and print a backend report without requiring an input source.",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=0,
        help="Stop after N processed frames and print run timing. 0 processes the full source.",
    )
    parser.add_argument(
        "--yolo-fast-preset",
        choices=("nano", "small", "medium", "large", "xlarge"),
        help="Convenience preset for legacy YOLO pose model size. Explicit --model still wins.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default=PROFILE_FASTEST,
        help="Runtime profile for app modes: fastest for realtime, mid for balanced, quality for offline renders.",
    )
    parser.add_argument(
        "--calibrate-cameras",
        action="store_true",
        help="Create a calibrated camera TOML from synchronized Charuco calibration videos.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Webcam index (e.g. 0) or path to a video file. If omitted, an interactive prompt runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output video path. The final file will still be stacked as '<name> rendered-N.ext'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where rendered/video sibling outputs should be written.",
    )
    parser.add_argument(
        "--output-basename",
        help="Base filename prefix used for rendered/fbx/json sibling outputs.",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        help="Output TOML path for --calibrate-cameras.",
    )
    parser.add_argument("--charuco-squares-x", type=int, default=7, help="Charuco board square count along X.")
    parser.add_argument("--charuco-squares-y", type=int, default=5, help="Charuco board square count along Y.")
    parser.add_argument("--charuco-square-size", type=float, default=1.0, help="Real square size in your chosen units.")
    parser.add_argument("--charuco-marker-scale", type=float, default=0.8, help="Marker length as a fraction of square size.")
    parser.add_argument("--charuco-marker-bits", type=int, default=4, choices=(4, 5, 6, 7), help="ArUco marker bit width used by the Charuco board.")
    parser.add_argument("--charuco-dict-size", type=int, default=50, choices=(50, 100, 250, 1000), help="ArUco dictionary size used by the Charuco board.")
    parser.add_argument(
        "--charuco-legacy-pattern",
        action="store_true",
        help="Use OpenCV's legacy ChArUco marker layout. Enable this for many online-generated boards, including Calib.io-style patterns.",
    )
    parser.add_argument(
        "--charuco-detection-strictness",
        choices=("strict", "balanced", "lenient"),
        default="balanced",
        help="Charuco detection strictness. Strict uses normal OpenCV detection; balanced retries low-resolution frames at 2x; lenient uses stronger 3x retry for compressed or distant boards.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Body model selector. In legacy YOLO mode use a YOLO pose filename/path. "
            f"In MediaPipe mode use one of: {', '.join(mediapipe_pose_model_names())}."
        ),
    )
    parser.add_argument(
        "--landmark-backend",
        choices=LANDMARK_BACKENDS,
        default="mediapipe",
        help="Select the landmark backend family: rtmpose, rtmpose-wholebody, mediapipe, hybrid, or yolo legacy.",
    )
    parser.add_argument(
        "--body-backend",
        choices=BODY_BACKENDS,
        help="Body backend. rtmpose-wholebody owns body+hands; rtmpose is body-only; yolo is legacy.",
    )
    parser.add_argument(
        "--hand-backend",
        choices=HAND_BACKENDS,
        help="Hand backend. onnx uses the local hand model; rtmpose-wholebody uses WholeBody hands.",
    )
    parser.add_argument(
        "--backend-fallbacks",
        action="store_true",
        help="Try the alternate backend when the selected body or hand backend misses a frame.",
    )
    parser.add_argument(
        "--hand-model-variant",
        choices=("low", "mid", "high", "max"),
        default="max",
        help="Hand model preset. Supported variants: low, mid, high, max.",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        help="Path to the ONNX hand model. Overrides the hand preset download.",
    )
    parser.add_argument(
        "--hand-input-name",
        default="images",
        help="Input tensor name for the hand model.",
    )
    parser.add_argument(
        "--body-input-size",
        type=int,
        default=640,
        help="Legacy YOLO body model image size.",
    )
    parser.add_argument(
        "--hand-input-size",
        type=int,
        default=640,
        help="Square input size for the hand model crop.",
    )
    parser.add_argument(
        "--processing-width",
        type=int,
        default=0,
        help="Optional downscaled inference width. 0 uses source resolution; 480 is good for fast preview.",
    )
    parser.add_argument(
        "--body-conf-threshold",
        type=float,
        default=0.30,
        help="Minimum confidence used to keep body landmarks.",
    )
    parser.add_argument(
        "--hand-det-threshold",
        type=float,
        default=0.15,
        help="Minimum hand detection score.",
    )
    parser.add_argument(
        "--hand-kp-threshold",
        type=float,
        default=0.20,
        help="Minimum hand keypoint confidence for drawing and live UDP output.",
    )
    parser.add_argument(
        "--hand-box-min-size",
        type=int,
        default=160,
        help="Minimum side length for the wrist-centered hand crop.",
    )
    parser.add_argument(
        "--hand-box-scale",
        type=float,
        default=2.0,
        help="Scale factor applied to the wrist-elbow based hand crop.",
    )
    parser.add_argument(
        "--body-iou-threshold",
        type=float,
        default=0.45,
        help="Legacy YOLO body NMS IoU threshold.",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=1,
        help="Maximum number of people to detect and track in a single view.",
    )
    parser.add_argument(
        "--identity",
        dest="identity_hints",
        action="append",
        type=parse_identity_hint,
        help="Optional clothing color hint like --identity person1=black,orange.",
    )
    parser.add_argument(
        "--person-box-scale",
        type=float,
        default=1.15,
        help="Expand each detected person box before running pose and hand inference.",
    )
    parser.add_argument(
        "--person-track-hold-frames",
        type=int,
        default=10,
        help="How many frames to keep a person track alive when detections are briefly missing.",
    )
    parser.add_argument(
        "--person-match-threshold",
        type=float,
        default=0.15,
        help="Minimum association score when matching a detected person to an existing track.",
    )
    parser.add_argument(
        "--person-cross-wrist-ratio",
        type=float,
        default=0.90,
        help="Hand ownership switch ratio. Lower values are stricter during crossings.",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        help="Optional JSON file with per-camera fusion calibration overrides.",
    )
    parser.add_argument(
        "--calibration-3d",
        type=Path,
        help="Optional calibrated camera TOML for real fused 3D triangulation.",
    )
    parser.add_argument(
        "--triangulate-3d",
        action="store_true",
        help="Use calibrated multi-camera triangulation in fused mode.",
    )
    parser.add_argument(
        "--triangulation-min-cameras",
        type=int,
        default=2,
        help="Minimum camera views required for a joint to be triangulated.",
    )
    parser.add_argument(
        "--triangulation-use-outlier-rejection",
        action="store_true",
        help="Drop bad camera views during triangulation when the calibration backend supports it.",
    )
    parser.add_argument(
        "--triangulation-max-cameras-to-drop",
        type=int,
        default=1,
        help="Maximum camera views to drop per point during outlier-rejection triangulation.",
    )
    parser.add_argument(
        "--triangulation-reprojection-error",
        type=float,
        default=0.01,
        help="Target reprojection error for outlier-rejection triangulation.",
    )
    parser.add_argument(
        "--triangulation-max-error",
        type=float,
        help="Drop triangulated joints above this reprojection error.",
    )
    parser.add_argument(
        "--triangulation-smoothing-alpha",
        type=float,
        default=0.65,
        help="EMA smoothing factor applied after 3D triangulation.",
    )
    parser.add_argument(
        "--sync-offset",
        dest="sync_offsets",
        action="append",
        type=parse_sync_offset,
        help="Frame offset per camera, e.g. --sync-offset CAM_1=3. Positive skips leading frames.",
    )
    parser.add_argument(
        "--fused-depth-scale",
        type=float,
        default=1.0,
        help="Depth scale multiplier used when estimating fused multi-camera joint depth.",
    )
    parser.add_argument(
        "--single-camera-depth",
        choices=("flat", "mediapipe"),
        default="flat",
        help="Single-camera export depth mode. flat is stable; mediapipe uses model-relative Z and can be noisy.",
    )
    parser.add_argument(
        "--no-auto-performance",
        action="store_true",
        help="Disable automatic GPU/FP16 and skip-frame performance choices.",
    )
    parser.add_argument(
        "--yolo-tracker",
        default="bytetrack.yaml",
        help="Legacy YOLO Ultralytics tracker config name for multi-person tracking.",
    )
    parser.add_argument(
        "--yolo-device",
        help="Optional Ultralytics device override such as 0, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--yolo-half",
        action="store_true",
        help="Request FP16 body inference on supported CUDA GPUs.",
    )
    parser.add_argument(
        "--no-yolo-fuse",
        action="store_true",
        help="Disable YOLO Conv+BatchNorm fusion at model startup.",
    )
    parser.add_argument(
        "--no-yolo-warmup",
        action="store_true",
        help="Disable the one-time YOLO warmup inference pass.",
    )
    parser.add_argument(
        "--no-yolo-person-class-filter",
        action="store_true",
        help="Disable YOLO class filtering. Pose models normally use class 0 for person.",
    )
    parser.add_argument(
        "--rtmpose-mode",
        choices=("lightweight", "balanced", "performance"),
        default="balanced",
        help="RTMPose model preset. lightweight is fastest; balanced is the default; performance is largest.",
    )
    parser.add_argument(
        "--rtmpose-backend",
        choices=("onnxruntime", "opencv"),
        default="onnxruntime",
        help="RTMPose inference backend. Use onnxruntime for RTX 50-series CUDA systems.",
    )
    parser.add_argument(
        "--rtmpose-device",
        default="cuda",
        help="RTMPose device, usually cuda for RTX 50-series or cpu as fallback.",
    )
    parser.add_argument(
        "--rtmpose-det-frequency",
        type=int,
        default=1,
        help="Run RTMPose person detection every N frames when tracking is enabled.",
    )
    parser.add_argument(
        "--no-rtmpose-tracking",
        action="store_true",
        help="Disable RTMPose tracker reuse between detection frames.",
    )
    parser.add_argument(
        "--body-detect-interval",
        type=int,
        default=1,
        help="Run body model every N frames and predict skipped frames. 1 means every frame.",
    )
    parser.add_argument(
        "--hand-detect-interval",
        type=int,
        default=1,
        help="Run hand model every N frames and translate hands on skipped frames. 1 means every frame.",
    )
    parser.add_argument(
        "--hand-crop-retries",
        type=int,
        default=3,
        help="Extra hand crop attempts after the primary crop. Lower is faster; higher is more robust.",
    )
    parser.add_argument(
        "--fps-log-interval",
        type=float,
        default=0.0,
        help="Print render throughput every N seconds. 0 disables FPS logging.",
    )
    parser.add_argument(
        "--no-fps-overlay",
        action="store_true",
        help="Disable the FPS tracker drawn into preview and output video frames.",
    )
    parser.add_argument(
        "--provider",
        dest="providers",
        action="append",
        help="ONNX Runtime provider priority for the hand model, e.g. --provider CUDAExecutionProvider --provider CPUExecutionProvider.",
    )
    parser.add_argument(
        "--osc-host",
        default=app_config.LiveUdpDefaults.HOST,
        help="Live UDP target host.",
    )
    parser.add_argument(
        "--osc-port",
        type=int,
        default=app_config.LiveUdpDefaults.PORT,
        help="Live UDP target port.",
    )
    parser.add_argument(
        "--osc-enabled",
        action="store_true",
        help="Enable live UDP sending.",
    )
    parser.add_argument(
        "--preview-title",
        default="Pose + Hand Landmarks",
        help="Window title for the live preview.",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=30.0,
        help="FPS to use when the source does not report one.",
    )
    parser.add_argument(
        "--output-fourcc",
        default="mp4v",
        help="FourCC codec for the output video writer.",
    )
    parser.add_argument("--body-line-color", type=parse_color, default=parse_color("255,0,0"), help="Body line color as B,G,R.")
    parser.add_argument("--body-point-color", type=parse_color, default=parse_color("0,255,0"), help="Body landmark color as B,G,R.")
    parser.add_argument("--hand-box-color", type=parse_color, default=parse_color("80,80,255"), help="Hand box color as B,G,R.")
    parser.add_argument("--hand-line-color", type=parse_color, default=parse_color("0,255,255"), help="Hand skeleton color as B,G,R.")
    parser.add_argument("--hand-point-color", type=parse_color, default=parse_color("0,165,255"), help="Hand keypoint color as B,G,R.")
    parser.add_argument("--body-line-thickness", type=int, default=2, help="Thickness of body skeleton lines.")
    parser.add_argument("--body-point-radius", type=int, default=4, help="Radius of body landmark points.")
    parser.add_argument("--hand-box-thickness", type=int, default=1, help="Thickness of the hand crop box.")
    parser.add_argument("--hand-line-thickness", type=int, default=2, help="Thickness of hand skeleton lines.")
    parser.add_argument("--hand-point-radius", type=int, default=3, help="Radius of hand landmark points.")
    parser.add_argument("--body-smoothing-alpha", type=float, default=0.65, help="EMA smoothing factor for body landmarks.")
    parser.add_argument("--hand-smoothing-alpha", type=float, default=0.55, help="EMA smoothing factor for hand landmarks.")
    parser.add_argument("--body-hold-frames", type=int, default=8, help="How many frames to keep the last valid body landmark.")
    parser.add_argument("--hand-hold-frames", type=int, default=6, help="How many frames to keep the last valid hand landmark.")
    parser.add_argument("--hold-confidence-decay", type=float, default=0.85, help="Confidence multiplier applied while reusing a held landmark.")
    parser.add_argument("--no-body-constraints", action="store_true", help="Disable soft body length constraints.")
    parser.add_argument("--body-length-smoothing-alpha", type=float, default=0.15, help="EMA factor used while learning body segment lengths.")
    parser.add_argument("--body-length-correction", type=float, default=0.35, help="How strongly each frame is pulled toward learned body segment lengths.")
    parser.add_argument("--no-export-cleanup", action="store_true", help="Disable offline export interpolation, spike cleanup, smoothing, and foot lock.")
    parser.add_argument("--export-cleanup-smoothing-alpha", type=float, default=0.55, help="EMA factor used by offline export smoothing.")
    parser.add_argument("--export-cleanup-max-velocity", type=float, default=220.0, help="Maximum per-frame joint movement before export cleanup treats a point as a spike.")
    parser.add_argument("--no-foot-lock", action="store_true", help="Disable export-time foot planting stabilization.")
    parser.add_argument("--foot-lock-velocity", type=float, default=8.0, help="Maximum per-frame foot movement considered planted during export cleanup.")
    parser.add_argument("--foot-lock-max-lift", type=float, default=16.0, help="Maximum distance from the detected floor for export foot locking.")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live OpenCV preview window.")
    return parser
