from __future__ import annotations

from pathlib import Path
import importlib
import json

from core.cli import InputAssignment


def calibration_available() -> bool:
    try:
        _load_calibration_classes()
    except ModuleNotFoundError:
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
        marker_bits=4,
        dict_size=50,
    )
    camera_group = camera_group_class.from_names(labels)
    error, all_rows = camera_group.calibrate_videos(videos, board)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera_group.dump(str(output_path))
    _write_calibration_report(
        output_path,
        labels=labels,
        squares_x=squares_x,
        squares_y=squares_y,
        square_size=square_size,
        marker_scale=marker_scale,
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


def _load_calibration_classes():
    try:
        package_prefix = "".join(("ani", "pose", "lib"))
        boards_module = importlib.import_module(f"{package_prefix}.boards")
        cameras_module = importlib.import_module(f"{package_prefix}.cameras")
    except ModuleNotFoundError:
        raise
    return cameras_module.CameraGroup, boards_module.CharucoBoard
