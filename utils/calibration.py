from __future__ import annotations

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
    retry_scale: float | None = None,
    minimum_markers: int | None = None,
    retry_sharpen: bool = False,
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
    detector_settings = _enable_low_resolution_charuco_detection(
        board,
        detection_strictness=detection_strictness,
        retry_scale=retry_scale,
        minimum_markers=minimum_markers,
        retry_sharpen=retry_sharpen,
    )
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
        detector_settings=detector_settings,
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
    detector_settings: dict[str, object],
    error: object,
    all_rows: object,
) -> None:
    mean_err = error
    per_cam_errs = None
    if isinstance(error, (tuple, list)) and len(error) == 2:
        mean_err = error[0]
        per_cam_errs = error[1]

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
            "detection_retry": detector_settings,
        },
        "mean_reprojection_error": _finite_float(mean_err),
        "detected_rows": _row_count(all_rows),
        "advice": _quality_advice(_finite_float(mean_err), per_cam_errs, labels),
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
    row_counts = [sum(1 for row in rows if _detection_count(row.get("corners")) > 0) for rows in all_rows]
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


def _quality_advice(error: float | None, per_camera_errors: list[float] | None = None, labels: list[str] | None = None) -> list[str]:
    if per_camera_errors and labels and len(per_camera_errors) == len(labels):
        import statistics
        try:
            median = statistics.median(per_camera_errors)
            for label, cam_err in zip(labels, per_camera_errors):
                if cam_err > 2 * median:
                    print(f"[calibration] WARNING: Camera '{label}' has high reprojection error ({cam_err:.3f}) vs median ({median:.3f}). Consider recalibrating this camera.")
        except Exception:
            pass

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


def _detection_count(values: object) -> int:
    if values is None:
        return 0
    try:
        return len(values)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _enable_low_resolution_charuco_detection(
    board: object,
    *,
    detection_strictness: str,
    retry_scale: float | None = None,
    minimum_markers: int | None = None,
    retry_sharpen: bool = False,
) -> dict[str, object]:
    settings = {
        "strict": None,
        "balanced": {"scale": 2.0, "minimum_markers": 8, "sharpen": False},
        "lenient": {"scale": 3.0, "minimum_markers": 12, "sharpen": True},
    }.get(detection_strictness)
    if settings is None and retry_scale is None and minimum_markers is None and not retry_sharpen:
        log_info("Calibration marker detector strictness: strict.")
        return {"enabled": False, "strictness": detection_strictness}

    if settings is None:
        settings = {"scale": 2.0, "minimum_markers": 8, "sharpen": False}

    scale = retry_scale if retry_scale is not None else settings["scale"]
    if scale <= 1.0 or scale > 5.0:
        raise ValueError("--charuco-retry-scale must be greater than 1.0 and no more than 5.0.")

    marker_floor = minimum_markers if minimum_markers is not None else settings["minimum_markers"]
    if marker_floor < 1:
        raise ValueError("--charuco-min-markers must be at least 1.")

    sharpen = bool(settings["sharpen"] or retry_sharpen)
    original_detect_markers = board.detect_markers
    original_detect_image = board.detect_image

    def retry_image(image):
        try:
            import cv2
        except ModuleNotFoundError:
            return None

        enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if sharpen:
            blur = cv2.GaussianBlur(enlarged, (0, 0), 1.0)
            enlarged = cv2.addWeighted(enlarged, 1.6, blur, -0.6, 0)
        return enlarged

    def detect_markers_with_retry(self, image, camera=None, refine=True):
        corners, ids = original_detect_markers(image, camera=camera, refine=refine)
        original_count = _detection_count(ids)
        if original_count >= marker_floor:
            return corners, ids

        enlarged = retry_image(image)
        if enlarged is None:
            return corners, ids

        scaled_corners, scaled_ids = original_detect_markers(enlarged, camera=camera, refine=refine)
        if _detection_count(scaled_ids) <= original_count:
            return corners, ids

        corrected_corners = [marker_corners / scale for marker_corners in scaled_corners]
        return corrected_corners, scaled_ids

    def detect_image_with_retry(self, image, camera=None):
        corners, ids = original_detect_image(image, camera=camera)
        if _detection_count(corners) > 0:
            return corners, ids

        try:
            import cv2
        except ModuleNotFoundError:
            return corners, ids

        enlarged = retry_image(image)
        if enlarged is None:
            return corners, ids

        scaled_marker_corners, scaled_marker_ids = original_detect_markers(enlarged, camera=camera, refine=True)
        if _detection_count(scaled_marker_corners) == 0:
            return corners, ids

        if len(enlarged.shape) == 3:
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        else:
            gray = enlarged

        _, scaled_corners, scaled_ids = cv2.aruco.interpolateCornersCharuco(
            scaled_marker_corners,
            scaled_marker_ids,
            gray,
            self.board,
        )
        if _detection_count(scaled_corners) == 0:
            return corners, ids

        corrected_corners = scaled_corners / scale
        if (
            getattr(self, "manually_verify", False)
            and not self.manually_verify_board_detection(image, corrected_corners, scaled_ids)
        ):
            return corners, ids

        return corrected_corners, scaled_ids

    board.detect_markers = MethodType(detect_markers_with_retry, board)
    board.detect_image = MethodType(detect_image_with_retry, board)
    log_info(
        f"Calibration marker detector strictness: {detection_strictness}; "
        f"retrying weak frames at {scale:.1f}x, marker floor {marker_floor}, "
        f"sharpen {'on' if sharpen else 'off'}."
    )
    return {
        "enabled": True,
        "strictness": detection_strictness,
        "scale": scale,
        "minimum_markers": marker_floor,
        "sharpen": sharpen,
    }


def _load_calibration_classes():
    try:
        import cv2
        from aniposelib.boards import CharucoBoard
        from aniposelib.cameras import CameraGroup

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV ArUco support is not installed; install opencv-contrib-python.")
    except ModuleNotFoundError:
        raise
    return CameraGroup, CharucoBoard
