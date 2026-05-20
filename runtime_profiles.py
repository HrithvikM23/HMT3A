from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PROFILE_FASTEST = "fastest"
PROFILE_MID = "mid"
PROFILE_QUALITY = "quality"
PROFILE_NAMES = (PROFILE_FASTEST, PROFILE_MID, PROFILE_QUALITY)


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    description: str
    settings: dict[str, Any]


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    PROFILE_FASTEST: RuntimeProfile(
        name=PROFILE_FASTEST,
        description="Fast stable mode for live preview and webcams.",
        settings={
            "model": "yolo11n-pose.pt",
            "hand_model_variant": "low",
            "body_input_size": 640,
            "processing_width": 640,
            "yolo_half": True,
            "body_conf_threshold": 0.30,
            "hand_det_threshold": 0.18,
            "hand_kp_threshold": 0.22,
            "hand_crop_retries": 1,
            "body_detect_interval": 1,
            "hand_detect_interval": 2,
            "backend_fallbacks": False,
            "body_smoothing_alpha": 0.60,
            "hand_smoothing_alpha": 0.50,
            "body_hold_frames": 6,
            "hand_hold_frames": 8,
            "hold_confidence_decay": 0.88,
            "export_cleanup_smoothing_alpha": 0.50,
            "export_cleanup_max_velocity": 240.0,
        },
    ),
    PROFILE_MID: RuntimeProfile(
        name=PROFILE_MID,
        description="Balanced mode for stable preview without the heaviest model settings.",
        settings={
            "model": "yolo11m-pose.pt",
            "hand_model_variant": "mid",
            "body_input_size": 768,
            "processing_width": 720,
            "yolo_half": True,
            "body_conf_threshold": 0.30,
            "hand_det_threshold": 0.18,
            "hand_kp_threshold": 0.22,
            "hand_crop_retries": 1,
            "body_detect_interval": 1,
            "hand_detect_interval": 2,
            "body_smoothing_alpha": 0.60,
            "hand_smoothing_alpha": 0.50,
            "body_hold_frames": 6,
            "hand_hold_frames": 8,
            "hold_confidence_decay": 0.88,
            "export_cleanup_smoothing_alpha": 0.50,
            "export_cleanup_max_velocity": 240.0,
        },
    ),
    PROFILE_QUALITY: RuntimeProfile(
        name=PROFILE_QUALITY,
        description="Quality-first mode for offline renders.",
        settings={
            "model": "yolo11x-pose.pt",
            "hand_model_variant": "max",
            "body_input_size": 960,
            "yolo_half": False,
            "body_conf_threshold": 0.30,
            "hand_det_threshold": 0.15,
            "hand_kp_threshold": 0.20,
            "hand_crop_retries": 3,
            "body_detect_interval": 1,
            "hand_detect_interval": 1,
            "body_smoothing_alpha": 0.65,
            "hand_smoothing_alpha": 0.55,
            "body_hold_frames": 8,
            "hand_hold_frames": 6,
            "hold_confidence_decay": 0.85,
            "export_cleanup_smoothing_alpha": 0.55,
            "export_cleanup_max_velocity": 220.0,
        },
    ),
}


PROFILE_CONTROLLED_ARGS = frozenset(
    {
        "model",
        "hand_model_variant",
        "body_input_size",
        "processing_width",
        "yolo_half",
        "body_conf_threshold",
        "hand_det_threshold",
        "hand_kp_threshold",
        "hand_crop_retries",
        "body_detect_interval",
        "hand_detect_interval",
        "backend_fallbacks",
        "body_smoothing_alpha",
        "hand_smoothing_alpha",
        "body_hold_frames",
        "hand_hold_frames",
        "hold_confidence_decay",
        "export_cleanup_smoothing_alpha",
        "export_cleanup_max_velocity",
    }
)


def apply_runtime_profile(args: Any, explicit_dests: set[str]) -> None:
    profile_name = getattr(args, "profile", PROFILE_QUALITY)
    profile = RUNTIME_PROFILES[profile_name]
    for dest, value in profile.settings.items():
        if dest not in explicit_dests:
            setattr(args, dest, value)
