from __future__ import annotations

import argparse
from pathlib import Path

from core.backend_selection import needs_mediapipe, needs_onnx_hand, needs_yolo_body, resolve_backend_selection
from core.cli import InputAssignment, sanitize_label
from core.config import PipelineConfig
from core.mediapipe_models import (
    DEFAULT_MEDIAPIPE_POSE_MODEL,
    is_mediapipe_pose_model,
    mediapipe_pose_model_names,
    normalize_mediapipe_pose_model,
)
from core.runtime_report import build_runtime_report, runtime_report_lines
from utils.logging import log_error, log_info
from utils.model_assets import (
    DEFAULT_BODY_MODEL,
    HAND_MODEL_SPECS,
    ensure_body_model_file,
    ensure_mediapipe_hand_asset_files,
    ensure_mediapipe_hand_task_file,
    ensure_mediapipe_pose_model_file,
    ensure_mediapipe_pose_task_file,
    ensure_model_file,
)

DEFAULT_PROVIDER_NAMES = ("CUDAExecutionProvider",)


def resolve_body_model_settings(
    requested_model: str | Path | None,
    body_backend: str,
) -> tuple[str | Path | None, str, str]:
    if body_backend == "mediapipe":
        mediapipe_model = (
            normalize_mediapipe_pose_model(requested_model)
            if is_mediapipe_pose_model(requested_model)
            else DEFAULT_MEDIAPIPE_POSE_MODEL
        )
        return None, DEFAULT_BODY_MODEL, mediapipe_model

    return requested_model, str(requested_model or DEFAULT_BODY_MODEL), DEFAULT_MEDIAPIPE_POSE_MODEL


def prepare_model_assets(config: PipelineConfig) -> None:
    if needs_mediapipe(config.body_backend, config.hand_backend, config.enable_backend_fallbacks):
        if config.body_backend == "mediapipe" or config.enable_backend_fallbacks:
            config.mediapipe_pose_model_path = ensure_mediapipe_pose_model_file(
                config.project_root,
                config.mediapipe_pose_model,
            )
            if config.mediapipe_delegate == "gpu":
                config.mediapipe_pose_task_path = ensure_mediapipe_pose_task_file(config.project_root)
        if config.hand_backend == "mediapipe" or config.enable_backend_fallbacks:
            ensure_mediapipe_hand_asset_files(config.project_root)
            if config.mediapipe_delegate == "gpu":
                config.mediapipe_hand_task_path = ensure_mediapipe_hand_task_file(config.project_root)

    if needs_yolo_body(config.body_backend, config.enable_backend_fallbacks) and config.body_model_path is None:
        config.body_model_path = config.body_model_variant or DEFAULT_BODY_MODEL
    if needs_yolo_body(config.body_backend, config.enable_backend_fallbacks):
        config.body_model_path = ensure_body_model_file(config.project_root, str(config.body_model_path))

    if not needs_onnx_hand(config.hand_backend, config.enable_backend_fallbacks):
        return

    hand_spec = HAND_MODEL_SPECS[config.hand_model_variant]
    if config.hand_model_path is None:
        log_info(f"Preparing hand model preset '{config.hand_model_variant}'")
        config.hand_model_path = ensure_model_file(config.project_root, hand_spec)
        config.hand_input_size = hand_spec.input_size
        config.hand_input_name = hand_spec.input_name
        config.hand_input_dtype = hand_spec.input_dtype


def resolve_output_basename(base_name: str | None, source: int | Path, label: str, multi_input: bool) -> str | None:
    if base_name is not None:
        cleaned = base_name.strip()
        return f"{cleaned}_{sanitize_label(label)}" if multi_input else cleaned

    stem = source.stem if isinstance(source, Path) else f"webcam_{source}"
    return f"{stem}_{sanitize_label(label)}" if multi_input else stem


def resolve_output_path(output_path: Path | None, label: str, multi_input: bool) -> Path | None:
    if output_path is None:
        return None
    if not multi_input:
        return output_path
    return output_path.with_name(f"{output_path.stem}_{sanitize_label(label)}{output_path.suffix or '.mp4'}")


def select_reference_assignment(assignments: list[InputAssignment]) -> InputAssignment:
    return assignments[0]


def resolve_fused_output_basename(base_name: str | None, assignments: list[InputAssignment]) -> str | None:
    if base_name is not None:
        return f"{base_name.strip()}_fused"

    reference_assignment = select_reference_assignment(assignments)
    if isinstance(reference_assignment.source, Path):
        return f"{reference_assignment.source.stem}_fused"
    return f"webcam_{reference_assignment.source}_fused"


def resolve_fused_output_path(output_path: Path | None) -> Path | None:
    if output_path is None:
        return None
    return output_path.with_name(f"{output_path.stem}_fused{output_path.suffix or '.mp4'}")


def validate_config(config: PipelineConfig) -> bool:
    if config.body_backend not in {"yolo", "mediapipe", "rtmpose", "rtmpose-wholebody"}:
        print(f"Error: invalid body backend: {config.body_backend}")
        return False
    if config.hand_backend not in {"onnx", "mediapipe", "rtmpose-wholebody"}:
        print(f"Error: invalid hand backend: {config.hand_backend}")
        return False
    if (config.body_backend == "rtmpose-wholebody") != (config.hand_backend == "rtmpose-wholebody"):
        print("Error: RTMPose WholeBody must own both body and hand backends.")
        return False
    if config.body_backend == "mediapipe" and config.mediapipe_pose_model not in mediapipe_pose_model_names():
        accepted = ", ".join(mediapipe_pose_model_names())
        print(f"Error: invalid MediaPipe pose model: {config.mediapipe_pose_model}. Accepted values: {accepted}")
        return False
    if config.mediapipe_delegate not in {"cpu", "gpu"}:
        print(f"Error: invalid MediaPipe delegate: {config.mediapipe_delegate}")
        return False
    if config.rtmpose_mode not in {"lightweight", "balanced", "performance"}:
        print(f"Error: invalid RTMPose mode: {config.rtmpose_mode}")
        return False
    if config.rtmpose_backend not in {"onnxruntime", "opencv"}:
        print(f"Error: invalid RTMPose backend: {config.rtmpose_backend}")
        return False
    missing_paths = [
        path
        for path in (
            config.hand_model_path,
            config.mediapipe_pose_model_path,
            config.mediapipe_pose_task_path,
            config.mediapipe_hand_task_path,
        )
        if path is not None and not Path(path).exists()
    ]
    if missing_paths:
        for path in missing_paths:
            print(f"Error: model file not found: {path}")
        return False
    if config.osc_port < 1 or config.osc_port > 65535:
        print(f"Error: invalid OSC port: {config.osc_port}")
        return False
    if config.fallback_fps <= 0:
        print(f"Error: fallback FPS must be positive: {config.fallback_fps}")
        return False
    if len(config.output_fourcc) < 4:
        print(f"Error: output FourCC must have at least 4 characters: {config.output_fourcc}")
        return False
    if config.output_basename is not None and not config.output_basename.strip():
        print("Error: output basename must not be empty.")
        return False

    bounded_float_fields = {
        "body_smoothing_alpha": config.body_smoothing_alpha,
        "hand_smoothing_alpha": config.hand_smoothing_alpha,
        "hold_confidence_decay": config.hold_confidence_decay,
        "body_conf_threshold": config.body_conf_threshold,
        "body_iou_threshold": config.body_iou_threshold,
        "hand_det_threshold": config.hand_det_threshold,
        "hand_kp_threshold": config.hand_kp_threshold,
        "identity_min_score": config.identity_min_score,
        "person_cross_wrist_ratio": config.person_cross_wrist_ratio,
        "triangulation_smoothing_alpha": config.triangulation_smoothing_alpha,
        "body_length_smoothing_alpha": config.body_length_smoothing_alpha,
        "body_length_correction": config.body_length_correction,
        "export_cleanup_smoothing_alpha": config.export_cleanup_smoothing_alpha,
    }
    for field_name, value in bounded_float_fields.items():
        if value <= 0 or value > 1:
            print(f"Error: {field_name} must be in the range (0, 1]: {value}")
            return False

    positive_int_fields = {
        "body_input_size": config.body_input_size,
        "hand_input_size": config.hand_input_size,
        "hand_box_min_size": config.hand_box_min_size,
        "body_line_thickness": config.body_line_thickness,
        "body_point_radius": config.body_point_radius,
        "hand_box_thickness": config.hand_box_thickness,
        "hand_line_thickness": config.hand_line_thickness,
        "hand_point_radius": config.hand_point_radius,
        "body_hold_frames": config.body_hold_frames,
        "hand_hold_frames": config.hand_hold_frames,
        "max_people": config.max_people,
        "person_track_hold_frames": config.person_track_hold_frames,
        "body_detect_interval": config.body_detect_interval,
        "hand_detect_interval": config.hand_detect_interval,
        "rtmpose_det_frequency": config.rtmpose_det_frequency,
    }
    for field_name, value in positive_int_fields.items():
        if value <= 0:
            print(f"Error: {field_name} must be positive: {value}")
            return False
    if config.processing_width < 0:
        print(f"Error: processing_width must be zero or greater: {config.processing_width}")
        return False
    if config.person_box_scale <= 0:
        print(f"Error: person_box_scale must be positive: {config.person_box_scale}")
        return False
    if config.person_match_threshold <= 0:
        print(f"Error: person_match_threshold must be positive: {config.person_match_threshold}")
        return False
    if config.fused_depth_scale <= 0:
        print(f"Error: fused_depth_scale must be positive: {config.fused_depth_scale}")
        return False
    if config.single_camera_depth_mode not in {"flat", "mediapipe"}:
        print(f"Error: invalid single_camera_depth_mode: {config.single_camera_depth_mode}")
        return False
    if config.enable_3d_triangulation and config.calibration_3d_path is None:
        print("Error: --triangulate-3d requires --calibration-3d.")
        return False
    if config.calibration_3d_path is not None and not Path(config.calibration_3d_path).exists():
        print(f"Error: 3D calibration file not found: {config.calibration_3d_path}")
        return False
    if config.triangulation_min_cameras < 2:
        print(f"Error: triangulation_min_cameras must be at least 2: {config.triangulation_min_cameras}")
        return False
    if config.triangulation_max_cameras_to_drop < 0:
        print(
            "Error: triangulation_max_cameras_to_drop must be zero or greater: "
            f"{config.triangulation_max_cameras_to_drop}"
        )
        return False
    if config.triangulation_reprojection_error <= 0:
        print(f"Error: triangulation_reprojection_error must be positive: {config.triangulation_reprojection_error}")
        return False
    if config.triangulation_max_error is not None and config.triangulation_max_error <= 0:
        print(f"Error: triangulation_max_error must be positive: {config.triangulation_max_error}")
        return False
    if config.hand_crop_retries < 0:
        print(f"Error: hand_crop_retries must be zero or greater: {config.hand_crop_retries}")
        return False
    if config.fps_log_interval < 0:
        print(f"Error: fps_log_interval must be zero or greater: {config.fps_log_interval}")
        return False
    if config.benchmark_frames < 0:
        print(f"Error: benchmark_frames must be zero or greater: {config.benchmark_frames}")
        return False
    if config.export_cleanup_max_velocity <= 0:
        print(f"Error: export_cleanup_max_velocity must be positive: {config.export_cleanup_max_velocity}")
        return False
    if config.export_foot_lock_velocity <= 0:
        print(f"Error: export_foot_lock_velocity must be positive: {config.export_foot_lock_velocity}")
        return False
    if config.export_foot_lock_max_lift < 0:
        print(f"Error: export_foot_lock_max_lift must be zero or greater: {config.export_foot_lock_max_lift}")
        return False
    return True


def validate_dry_run_config(config: PipelineConfig) -> bool:
    if not validate_config(config):
        return False
    assert config.output_directory is not None
    if not config.output_directory.exists():
        log_error(f"Output directory does not exist: {config.output_directory}")
        return False
    if not config.output_directory.is_dir():
        log_error(f"Output path is not a directory: {config.output_directory}")
        return False
    probe_path = config.output_directory / ".kinara_write_test"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        log_error(f"Output directory is not writable: {config.output_directory} ({exc})")
        return False
    return True


def prepare_runtime_config(config: PipelineConfig, *, prepare_assets: bool = True) -> bool:
    apply_auto_performance(config)
    if prepare_assets:
        try:
            prepare_model_assets(config)
        except Exception as exc:
            log_error(f"Failed to prepare model assets: {exc}")
            return False
    if not validate_dry_run_config(config):
        return False
    report = build_runtime_report(config)
    for line in runtime_report_lines(report):
        log_info(line)
    return True


def apply_auto_performance(config: PipelineConfig) -> None:
    if not config.auto_performance_enabled:
        return
    if config.body_backend != "yolo" and not config.enable_backend_fallbacks:
        return
    if config.yolo_device:
        return
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        config.yolo_device = "0"
        if config.profile in {"fastest", "mid"}:
            config.yolo_half = True


def _build_pipeline_config(
    args: argparse.Namespace,
    *,
    video_path: int | Path,
    output_path: Path | None,
    output_basename: str | None,
    preview_window_title: str,
) -> PipelineConfig:
    calibration_3d_path = args.calibration_3d
    body_backend, hand_backend, enable_backend_fallbacks = resolve_backend_selection(args)
    body_model_path, body_model_variant, mediapipe_pose_model = resolve_body_model_settings(args.model, body_backend)
    sync_offsets = args.sync_offsets or []
    if isinstance(sync_offsets, dict):
        sync_offset_map = {str(label).upper(): int(offset) for label, offset in sync_offsets.items()}
    else:
        sync_offset_map = {label.upper(): offset for label, offset in sync_offsets}
    identity_hints = args.identity_hints or []
    if isinstance(identity_hints, dict):
        identity_hint_map = {
            str(label): tuple(colors if isinstance(colors, list | tuple) else [str(colors)])
            for label, colors in identity_hints.items()
        }
    else:
        identity_hint_map = dict(identity_hints)
    return PipelineConfig(
        video_path=video_path,
        output_path=output_path,
        output_directory=args.output_dir,
        output_basename=output_basename,
        profile=args.profile,
        body_backend=body_backend,
        hand_backend=hand_backend,
        enable_backend_fallbacks=enable_backend_fallbacks,
        body_model_path=body_model_path,
        hand_model_path=args.hand_model,
        body_model_variant=body_model_variant,
        mediapipe_pose_model=mediapipe_pose_model,
        mediapipe_delegate=args.mediapipe_delegate,
        hand_model_variant=args.hand_model_variant,
        hand_input_name=args.hand_input_name,
        hand_input_dtype="float32",
        body_input_size=args.body_input_size,
        hand_input_size=args.hand_input_size,
        processing_width=args.processing_width,
        body_conf_threshold=args.body_conf_threshold,
        body_iou_threshold=args.body_iou_threshold,
        hand_det_threshold=args.hand_det_threshold,
        hand_kp_threshold=args.hand_kp_threshold,
        hand_box_min_size=args.hand_box_min_size,
        hand_box_scale=args.hand_box_scale,
        enable_preview=not args.no_preview,
        provider_names=tuple(args.providers) if args.providers else DEFAULT_PROVIDER_NAMES,
        preview_window_title=preview_window_title,
        osc_host=args.osc_host,
        osc_port=args.osc_port,
        osc_enabled=args.osc_enabled,
        fallback_fps=args.fallback_fps,
        output_fourcc=args.output_fourcc,
        body_line_color=args.body_line_color,
        body_point_color=args.body_point_color,
        hand_box_color=args.hand_box_color,
        hand_line_color=args.hand_line_color,
        hand_point_color=args.hand_point_color,
        body_line_thickness=args.body_line_thickness,
        body_point_radius=args.body_point_radius,
        hand_box_thickness=args.hand_box_thickness,
        hand_line_thickness=args.hand_line_thickness,
        hand_point_radius=args.hand_point_radius,
        body_smoothing_alpha=args.body_smoothing_alpha,
        hand_smoothing_alpha=args.hand_smoothing_alpha,
        body_hold_frames=args.body_hold_frames,
        hand_hold_frames=args.hand_hold_frames,
        hold_confidence_decay=args.hold_confidence_decay,
        max_people=args.max_people,
        person_box_scale=args.person_box_scale,
        person_track_hold_frames=args.person_track_hold_frames,
        person_match_threshold=args.person_match_threshold,
        person_cross_wrist_ratio=args.person_cross_wrist_ratio,
        camera_calibration_path=args.camera_calibration,
        calibration_3d_path=calibration_3d_path,
        enable_3d_triangulation=args.triangulate_3d,
        triangulation_min_cameras=args.triangulation_min_cameras,
        triangulation_use_outlier_rejection=args.triangulation_use_outlier_rejection,
        triangulation_max_cameras_to_drop=args.triangulation_max_cameras_to_drop,
        triangulation_reprojection_error=args.triangulation_reprojection_error,
        triangulation_max_error=args.triangulation_max_error,
        triangulation_smoothing_alpha=args.triangulation_smoothing_alpha,
        sync_offsets=sync_offset_map,
        fused_depth_scale=args.fused_depth_scale,
        single_camera_depth_mode=args.single_camera_depth,
        auto_performance_enabled=not args.no_auto_performance,
        yolo_tracker=args.yolo_tracker,
        yolo_device=args.yolo_device,
        yolo_half=args.yolo_half,
        yolo_fuse=not args.no_yolo_fuse,
        yolo_warmup=not args.no_yolo_warmup,
        yolo_person_class_filter=not args.no_yolo_person_class_filter,
        rtmpose_mode=args.rtmpose_mode,
        rtmpose_backend=args.rtmpose_backend,
        rtmpose_device=args.rtmpose_device,
        rtmpose_det_frequency=args.rtmpose_det_frequency,
        rtmpose_tracking=not args.no_rtmpose_tracking,
        body_detect_interval=args.body_detect_interval,
        hand_detect_interval=args.hand_detect_interval,
        hand_crop_retries=args.hand_crop_retries,
        body_constraints_enabled=not args.no_body_constraints,
        body_length_smoothing_alpha=args.body_length_smoothing_alpha,
        body_length_correction=args.body_length_correction,
        export_cleanup_enabled=not args.no_export_cleanup,
        export_cleanup_smoothing_alpha=args.export_cleanup_smoothing_alpha,
        export_cleanup_max_velocity=args.export_cleanup_max_velocity,
        export_foot_lock_enabled=not args.no_foot_lock,
        export_foot_lock_velocity=args.foot_lock_velocity,
        export_foot_lock_max_lift=args.foot_lock_max_lift,
        fps_log_interval=args.fps_log_interval,
        fps_overlay_enabled=not args.no_fps_overlay,
        benchmark_frames=args.benchmark_frames,
        identity_hints=identity_hint_map,
    )


def build_config_for_assignment(args: argparse.Namespace, assignment: InputAssignment, multi_input: bool) -> PipelineConfig:
    return _build_pipeline_config(
        args,
        video_path=assignment.source,
        output_path=resolve_output_path(args.output, assignment.label, multi_input),
        output_basename=resolve_output_basename(args.output_basename, assignment.source, assignment.label, multi_input),
        preview_window_title=f"{args.preview_title} - {assignment.label}" if multi_input else args.preview_title,
    )


def build_fused_config(args: argparse.Namespace, assignments: list[InputAssignment]) -> PipelineConfig:
    reference_assignment = select_reference_assignment(assignments)
    return _build_pipeline_config(
        args,
        video_path=reference_assignment.source,
        output_path=resolve_fused_output_path(args.output),
        output_basename=resolve_fused_output_basename(args.output_basename, assignments),
        preview_window_title=f"{args.preview_title} - FUSED",
    )
