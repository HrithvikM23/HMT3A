from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import MethodType

from core.cli import InputAssignment
from utils.logging import log_info


def calibration_available() -> bool:
    try:
        _load_calibration_classes()
    except (ModuleNotFoundError, RuntimeError):
        return False
    return True


def calibrate_cameras(
    assignments: list[InputAssignment],
    output_path: Path,
    *,
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_scale: float,
    marker_bits: int = 4,
    dict_size: int = 50,
    legacy_pattern: bool = False,
    detection_strictness: str = "balanced",
) -> Path:
    if len(assignments) < 2:
        raise ValueError("Camera calibration needs at least two video sources.")
    if any(isinstance(assignment.source, int) for assignment in assignments):
        raise ValueError("Camera calibration needs recorded video files, not live webcam indices.")

    camera_group_class, board_class = _load_calibration_classes()
    labels = [assignment.label for assignment in assignments]
    videos = [[str(assignment.source)] for assignment in assignments]
    board = board_class(
        squares_x,
        squares_y,
        square_length=square_size,
        marker_length=square_size * marker_scale,
        marker_bits=marker_bits,
        dict_size=dict_size,
    )
    _configure_charuco_legacy_pattern(board, legacy_pattern)
    _enable_low_resolution_charuco_detection(board, detection_strictness=detection_strictness)
    camera_group = camera_group_class.from_names(labels)
    all_rows = camera_group.get_rows_videos(videos, board, verbose=True)
    _validate_detected_rows(all_rows, labels, marker_bits=marker_bits, dict_size=dict_size)
    camera_group.set_camera_sizes_videos(videos)
    error = camera_group.calibrate_rows(all_rows, board, verbose=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera_group.dump(str(output_path))
    _write_calibration_report(
        output_path,
        labels=labels,
        squares_x=squares_x,
        squares_y=squares_y,
        square_size=square_size,
        marker_scale=marker_scale,
        marker_bits=marker_bits,
        dict_size=dict_size,
        legacy_pattern=legacy_pattern,
        detection_strictness=detection_strictness,
        error=error,
        all_rows=all_rows,
    )
    return output_path


def _write_calibration_report(
    output_path: Path,
    *,
    labels: list[str],
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_scale: float,
    marker_bits: int,
    dict_size: int,
    legacy_pattern: bool,
    detection_strictness: str,
    error: object,
    all_rows: object,
) -> None:
    report_path = output_path.with_suffix(".quality.json")
    report = {
        "camera_labels": labels,
        "camera_count": len(labels),
        "board": {
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_size": square_size,
            "marker_scale": marker_scale,
            "marker_bits": marker_bits,
            "dict_size": dict_size,
            "legacy_pattern": legacy_pattern,
            "detection_strictness": detection_strictness,
        },
        "mean_reprojection_error": _finite_float(error),
        "detected_rows": _row_count(all_rows),
        "advice": _quality_advice(_finite_float(error)),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _row_count(all_rows: object) -> int | None:
    try:
        return sum(len(rows) for rows in all_rows)
    except TypeError:
        return None


def _validate_detected_rows(
    all_rows: list[list[dict]],
    labels: list[str],
    *,
    marker_bits: int,
    dict_size: int,
) -> None:
    row_counts = [len(rows) for rows in all_rows]
    if not any(row_counts):
        raise ValueError(
            "no Charuco boards were detected in any calibration video. "
            f"Checked dictionary DICT_{marker_bits}X{marker_bits}_{dict_size}. "
            "Make sure the app board settings match the printed board, keep the whole board sharp and large "
            "in the frame, avoid glare/blur, and record several tilted positions visible to every camera."
        )
    missing = [label for label, count in zip(labels, row_counts, strict=True) if count == 0]
    if missing:
        raise ValueError(
            "no Charuco boards were detected for camera(s): "
            + ", ".join(missing)
            + f". Checked dictionary DICT_{marker_bits}X{marker_bits}_{dict_size}."
        )


def _quality_advice(error: float | None) -> list[str]:
    if error is None:
        return ["Calibration saved, but no numeric reprojection error was reported by the backend."]
    if error <= 1.0:
        return ["Calibration quality looks strong."]
    if error <= 3.0:
        return ["Calibration is usable, but more sharp board coverage can improve triangulation."]
    return [
        "Calibration error is high.",
        "Record a sharper board video with more shared camera views, less blur, and more near/far/tilted board poses.",
    ]


def _configure_charuco_legacy_pattern(board: object, enabled: bool) -> None:
    if not enabled:
        log_info("Calibration ChArUco legacy pattern: disabled.")
        return
    charuco_board = getattr(board, "board", None)
    if charuco_board is None or not hasattr(charuco_board, "setLegacyPattern"):
        log_info("Calibration ChArUco legacy pattern requested, but this OpenCV runtime does not expose it.")
        return
    charuco_board.setLegacyPattern(True)
    log_info("Calibration ChArUco legacy pattern: enabled.")


def _enable_low_resolution_charuco_detection(board: object, *, detection_strictness: str) -> None:
    settings = {
        "strict": None,
        "balanced": {"scale": 2.0, "minimum_markers": 8, "sharpen": False},
        "lenient": {"scale": 3.0, "minimum_markers": 12, "sharpen": True},
    }.get(detection_strictness)
    if settings is None:
        log_info("Calibration marker detector strictness: strict.")
        return

    scale = settings["scale"]
    minimum_markers = settings["minimum_markers"]
    sharpen = settings["sharpen"]
    original_detect_markers = board.detect_markers

    def detect_markers_with_retry(self, image, camera=None, refine=True):
        corners, ids = original_detect_markers(image, camera=camera, refine=refine)
        if ids is not None and len(ids) >= minimum_markers:
            return corners, ids

        try:
            import cv2
        except ModuleNotFoundError:
            return corners, ids

        enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if sharpen:
            blur = cv2.GaussianBlur(enlarged, (0, 0), 1.0)
            enlarged = cv2.addWeighted(enlarged, 1.6, blur, -0.6, 0)
        scaled_corners, scaled_ids = original_detect_markers(enlarged, camera=camera, refine=refine)
        if scaled_ids is None or len(scaled_ids) <= (0 if ids is None else len(ids)):
            return corners, ids

        corrected_corners = [marker_corners / scale for marker_corners in scaled_corners]
        return corrected_corners, scaled_ids

    board.detect_markers = MethodType(detect_markers_with_retry, board)
    log_info(
        f"Calibration marker detector strictness: {detection_strictness}; "
        f"retrying weak frames at {scale:.1f}x."
    )


def _load_calibration_classes():
    try:
        import cv2

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV ArUco support is not installed; install opencv-contrib-python.")
        package_prefix = "".join(("ani", "pose", "lib"))
        boards_module = importlib.import_module(f"{package_prefix}.boards")
        cameras_module = importlib.import_module(f"{package_prefix}.cameras")
    except ModuleNotFoundError:
        raise
    return cameras_module.CameraGroup, boards_module.CharucoBoard
