from __future__ import annotations

from pathlib import Path


DEFAULT_MEDIAPIPE_POSE_MODEL = "pose_landmark_full.tflite"

MEDIAPIPE_POSE_MODEL_COMPLEXITIES = {
    "pose_landmark_lite.tflite": 0,
    "pose_landmark_full.tflite": 1,
    "pose_landmark_heavy.tflite": 2,
}


def mediapipe_pose_model_names() -> tuple[str, ...]:
    return tuple(MEDIAPIPE_POSE_MODEL_COMPLEXITIES)


def normalize_mediapipe_pose_model(value: str | Path | None) -> str:
    if value is None:
        return DEFAULT_MEDIAPIPE_POSE_MODEL
    return Path(str(value)).name


def is_mediapipe_pose_model(value: str | Path | None) -> bool:
    return normalize_mediapipe_pose_model(value) in MEDIAPIPE_POSE_MODEL_COMPLEXITIES


def mediapipe_pose_model_complexity(model_name: str | Path | None) -> int:
    normalized_name = normalize_mediapipe_pose_model(model_name)
    return MEDIAPIPE_POSE_MODEL_COMPLEXITIES[normalized_name]

