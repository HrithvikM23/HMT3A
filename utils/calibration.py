from __future__ import annotations

from pathlib import Path
import importlib

from cli import InputAssignment


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
    camera_group.calibrate_videos(videos, board)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera_group.dump(str(output_path))
    return output_path


def _load_calibration_classes():
    try:
        package_prefix = "".join(("ani", "pose", "lib"))
        boards_module = importlib.import_module(f"{package_prefix}.boards")
        cameras_module = importlib.import_module(f"{package_prefix}.cameras")
    except ModuleNotFoundError:
        raise
    return cameras_module.CameraGroup, boards_module.CharucoBoard
