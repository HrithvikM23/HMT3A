from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

class LiveUdpDefaults:
    HOST = "127.0.0.1"
    PORT = 9000
    ENABLED = False

@dataclass(slots=True)
class PipelineConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    body_model_path: str | Path | None = None
    hand_model_path: Path | None = None
    profile: str = "quality"
    body_backend: str = "yolo"
    hand_backend: str = "onnx"
    enable_backend_fallbacks: bool = False
    body_model_variant: str = "yolo11x-pose.pt"
    hand_model_variant: str = "max"
    video_path: int | Path = 0
    output_path: Path | None = None
    output_directory: Path | None = None
    output_basename: str | None = None
    hand_input_name: str = "images"
    hand_input_dtype: str = "float32"
    body_input_size: int = 960
    hand_input_size: int = 640
    body_conf_threshold: float = 0.30
    body_iou_threshold: float = 0.45
    hand_det_threshold: float = 0.15
    hand_kp_threshold: float = 0.20
    hand_box_min_size: int = 160
    hand_box_scale: float = 2.0
    hand_box_forward_shift: float = 0.35
    hand_wrist_max_offset_scale: float = 0.65
    hand_min_valid_points: int = 8
    hand_default_confidence: float = 0.55
    hand_default_scale: float = 0.85
    max_people: int = 1
    person_box_scale: float = 1.15
    person_track_hold_frames: int = 10
    identity_hints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    identity_min_score: float = 0.05
    person_match_threshold: float = 0.15
    person_cross_wrist_ratio: float = 0.90
    camera_calibration_path: Path | None = None
    calibration_3d_path: Path | None = None
    enable_3d_triangulation: bool = False
    triangulation_min_cameras: int = 2
    triangulation_use_outlier_rejection: bool = False
    triangulation_max_cameras_to_drop: int = 1
    triangulation_reprojection_error: float = 0.01
    triangulation_max_error: float | None = None
    triangulation_smoothing_alpha: float = 0.65
    sync_offsets: dict[str, int] = field(default_factory=dict)
    fused_depth_scale: float = 1.0
    auto_performance_enabled: bool = True
    yolo_tracker: str = "botsort.yaml"
    yolo_device: str | None = None
    yolo_half: bool = False
    body_detect_interval: int = 1
    hand_detect_interval: int = 1
    hand_crop_retries: int = 3
    body_constraints_enabled: bool = True
    body_length_smoothing_alpha: float = 0.15
    body_length_correction: float = 0.35
    export_cleanup_enabled: bool = True
    export_cleanup_smoothing_alpha: float = 0.55
    export_cleanup_max_velocity: float = 220.0
    export_foot_lock_enabled: bool = True
    export_foot_lock_velocity: float = 8.0
    export_foot_lock_max_lift: float = 16.0
    fps_log_interval: float = 0.0
    enable_preview: bool = True
    provider_names: tuple[str, ...] = ("CUDAExecutionProvider",)
    preview_window_title: str = "Pose + Hand Landmarks"
    osc_host: str = LiveUdpDefaults.HOST
    osc_port: int = LiveUdpDefaults.PORT
    osc_enabled: bool = LiveUdpDefaults.ENABLED
    fallback_fps: float = 30.0
    output_fourcc: str = "mp4v"
    body_line_color: tuple[int, int, int] = (255, 0, 0)
    body_point_color: tuple[int, int, int] = (0, 255, 0)
    hand_box_color: tuple[int, int, int] = (80, 80, 255)
    hand_line_color: tuple[int, int, int] = (0, 255, 255)
    hand_point_color: tuple[int, int, int] = (0, 165, 255)
    body_line_thickness: int = 2
    body_point_radius: int = 4
    hand_box_thickness: int = 1
    hand_line_thickness: int = 2
    hand_point_radius: int = 3
    body_smoothing_alpha: float = 0.65
    hand_smoothing_alpha: float = 0.55
    body_hold_frames: int = 8
    hand_hold_frames: int = 6
    hold_confidence_decay: float = 0.85
    run_index: int = field(init=False)
    video_extension: str = field(init=False)
    rendered_output_path: Path = field(init=False)
    fbx_output_path: Path = field(init=False)
    json_output_path: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.body_model_path is not None and isinstance(self.body_model_path, Path):
            self.body_model_path = Path(self.body_model_path)

        if self.hand_model_path is not None:
            self.hand_model_path = Path(self.hand_model_path)

        if self.output_directory is None:
            output_directory = self.project_root / "outputs"
        else:
            output_directory = Path(self.output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory = output_directory
        resolved_output_directory = self.output_directory

        if self.output_basename is None:
            if isinstance(self.video_path, Path):
                base_name = self.video_path.stem
            else:
                base_name = f"webcam_{self.video_path}"
        else:
            base_name = self.output_basename.strip()

        if self.output_path is not None:
            requested_output_path = Path(self.output_path)
            self.output_directory = requested_output_path.parent
            self.output_directory.mkdir(parents=True, exist_ok=True)
            resolved_output_directory = self.output_directory
            base_name = requested_output_path.stem
            self.video_extension = requested_output_path.suffix or ".mp4"
        else:
            self.video_extension = ".mp4"

        self.run_index = self._next_run_index(base_name)
        self.rendered_output_path = resolved_output_directory / f"{base_name} rendered-{self.run_index}{self.video_extension}"
        self.fbx_output_path = resolved_output_directory / f"{base_name} fbx-{self.run_index}.fbx"
        self.json_output_path = resolved_output_directory / f"{base_name} json-{self.run_index}.json"
        self.output_path = self.rendered_output_path
        self.output_fourcc = self.output_fourcc[:4].ljust(4)

    def _next_run_index(self, base_name: str) -> int:
        assert self.output_directory is not None
        output_directory = self.output_directory
        run_index = 1
        while True:
            sibling_paths = (
                output_directory / f"{base_name} rendered-{run_index}{self.video_extension}",
                output_directory / f"{base_name} fbx-{run_index}.fbx",
                output_directory / f"{base_name} json-{run_index}.json",
            )
            if not any(path.exists() for path in sibling_paths):
                return run_index
            run_index += 1


BODY_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11),
    (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

BODY_KEYPOINTS = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

WRIST_TO_ELBOW = {9: 7, 10: 8}
