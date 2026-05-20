from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from mediapipe_models import mediapipe_pose_model_names, normalize_mediapipe_pose_model


@dataclass(frozen=True, slots=True)
class ModelSpec:
    source_url: str
    relative_path: Path
    input_size: int
    input_name: str
    input_dtype: str


DEFAULT_BODY_MODEL = "yolo11x-pose.pt"
BODY_MODEL_URLS: dict[str, str] = {
    "yolo11n-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n-pose.pt",
    "yolo11s-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s-pose.pt",
    "yolo11m-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11m-pose.pt",
    "yolo11l-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11l-pose.pt",
    "yolo11x-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11x-pose.pt",
}

HAND_MODEL_SPECS: dict[str, ModelSpec] = {
    "low": ModelSpec(
        source_url="https://huggingface.co/poptoz/yolo26-hand-pose-face-detection/resolve/main/models/yolo26_hand_pose_fp16.onnx",
        relative_path=Path("models") / "hand" / "yolo26_hand_pose_fp16.onnx",
        input_size=640,
        input_name="images",
        input_dtype="float32",
    ),
    "mid": ModelSpec(
        source_url="https://huggingface.co/poptoz/yolo26-hand-pose-face-detection/resolve/main/models/yolo26_hand_pose_fp16.onnx",
        relative_path=Path("models") / "hand" / "yolo26_hand_pose_fp16.onnx",
        input_size=640,
        input_name="images",
        input_dtype="float32",
    ),
    "high": ModelSpec(
        source_url="https://huggingface.co/poptoz/yolo26-hand-pose-face-detection/resolve/main/models/yolo26_hand_pose_fp32.onnx",
        relative_path=Path("models") / "hand" / "yolo26_hand_pose_fp32.onnx",
        input_size=640,
        input_name="images",
        input_dtype="float32",
    ),
    "max": ModelSpec(
        source_url="https://huggingface.co/poptoz/yolo26-hand-pose-face-detection/resolve/main/models/yolo26_hand_pose_fp32.onnx",
        relative_path=Path("models") / "hand" / "yolo26_hand_pose_fp32.onnx",
        input_size=640,
        input_name="images",
        input_dtype="float32",
    ),
}
MEDIAPIPE_ASSET_URL_PREFIX = "https://storage.googleapis.com/mediapipe-assets/"
MEDIAPIPE_POSE_MODEL_URLS: dict[str, str] = {
    model_name: f"{MEDIAPIPE_ASSET_URL_PREFIX}{model_name}"
    for model_name in mediapipe_pose_model_names()
}
MEDIAPIPE_HAND_ASSETS = (
    ("hand_landmark", "hand_landmark_full.tflite"),
    ("hand_landmark", "hand_landmark_lite.tflite"),
    ("palm_detection", "palm_detection_full.tflite"),
    ("palm_detection", "palm_detection_lite.tflite"),
)


def _download_to_path(source_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with urlopen(source_url) as response, temp_path.open("wb") as output_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
        temp_path.replace(destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def ensure_model_file(project_root: Path, spec: ModelSpec) -> Path:
    destination = project_root / spec.relative_path
    if destination.exists():
        return destination

    _download_to_path(spec.source_url, destination)
    return destination


def ensure_body_model_file(project_root: Path, model_name_or_path: str) -> Path:
    candidate_path = Path(model_name_or_path)
    if candidate_path.is_absolute() or candidate_path.parent != Path("."):
        return candidate_path

    destination = project_root / "models" / "body" / candidate_path.name
    if destination.exists():
        return destination

    source_url = BODY_MODEL_URLS.get(candidate_path.name)
    if source_url is None:
        return destination

    _download_to_path(source_url, destination)
    return destination


def _installed_mediapipe_asset_path(module_name: str, asset_name: str) -> Path | None:
    try:
        import mediapipe as mp
    except ModuleNotFoundError:
        return None

    package_root = Path(mp.__file__).resolve().parent
    return package_root / "modules" / module_name / asset_name


def _copy_if_present(source: Path | None, destination: Path) -> bool:
    if source is None or not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def ensure_mediapipe_pose_model_file(project_root: Path, model_name_or_path: str | Path | None) -> Path:
    model_name = normalize_mediapipe_pose_model(model_name_or_path)
    if model_name not in MEDIAPIPE_POSE_MODEL_URLS:
        accepted = ", ".join(mediapipe_pose_model_names())
        raise ValueError(f"unsupported MediaPipe pose model '{model_name}'. Accepted values: {accepted}")

    destination = project_root / "models" / "body" / model_name
    if destination.exists():
        return destination

    if _copy_if_present(_installed_mediapipe_asset_path("pose_landmark", model_name), destination):
        return destination

    _download_to_path(MEDIAPIPE_POSE_MODEL_URLS[model_name], destination)
    return destination


def ensure_mediapipe_hand_asset_files(project_root: Path) -> tuple[Path, ...]:
    staged_paths: list[Path] = []
    for module_name, asset_name in MEDIAPIPE_HAND_ASSETS:
        destination = project_root / "models" / "hand" / "mediapipe" / asset_name
        if not destination.exists():
            _copy_if_present(_installed_mediapipe_asset_path(module_name, asset_name), destination)
        if destination.exists():
            staged_paths.append(destination)
    return tuple(staged_paths)
