from __future__ import annotations

import math
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.skeleton import BODY_FOOT_NAME_TO_INDEX, BODY_NAME_TO_INDEX, HAND_NAME_TO_INDEX, JointMap, Point


@dataclass(frozen=True, slots=True)
class TriangulationResult:
    joint_overrides_by_frame: list[dict[str, dict[str, float]]]
    camera_labels: list[str]
    joint_names: list[str]
    mean_reprojection_error: float | None
    triangulated_point_count: int


BODY_TRIANGULATION_JOINTS = tuple(BODY_NAME_TO_INDEX) + tuple(BODY_FOOT_NAME_TO_INDEX)
HAND_TRIANGULATION_JOINTS = tuple(
    (f"Left{name}", "left", index)
    for name, index in HAND_NAME_TO_INDEX.items()
) + tuple(
    (f"Right{name}", "right", index)
    for name, index in HAND_NAME_TO_INDEX.items()
)
TRIANGULATION_JOINT_NAMES = BODY_TRIANGULATION_JOINTS + tuple(name for name, _, _ in HAND_TRIANGULATION_JOINTS)


def calibrated_backend_available() -> bool:
    try:
        _load_camera_group_class()
    except ModuleNotFoundError:
        return False
    return True


def triangulate_observation_frames(
    calibration_path: Path,
    observation_frames: list[dict[str, Any]],
    *,
    body_threshold: float,
    hand_threshold: float,
    minimum_cameras: int,
    use_outlier_rejection: bool,
    maximum_cameras_to_drop: int,
    target_reprojection_error: float,
    max_reprojection_error: float | None = None,
    smoothing_alpha: float = 1.0,
) -> TriangulationResult:
    camera_group = _load_camera_group(calibration_path)
    camera_labels = _matching_camera_labels(camera_group, observation_frames)
    if len(camera_labels) < 2:
        raise ValueError(
            "3D triangulation needs at least two calibration camera names that match the fused source labels."
        )

    points_2d, confidences = _build_points_2d(
        observation_frames,
        camera_labels,
        body_threshold=body_threshold,
        hand_threshold=hand_threshold,
    )
    flat_points = points_2d.reshape(len(camera_labels), -1, 2)
    if use_outlier_rejection and hasattr(camera_group, "triangulate_using_outlier_rejection"):
        points_3d_flat, _ = camera_group.triangulate_using_outlier_rejection(
            flat_points,
            progress=False,
            minimum_cameras_for_triangulation=minimum_cameras,
            maximum_cameras_to_drop=maximum_cameras_to_drop,
            target_reprojection_error=target_reprojection_error,
        )
    else:
        points_3d_flat = camera_group.triangulate(
            flat_points,
            progress=False,
            minimum_cameras_for_triangulation=minimum_cameras,
        )

    reprojection_error, reprojection_errors = _reprojection_errors(camera_group, points_3d_flat, flat_points)
    points_3d = points_3d_flat.reshape(len(observation_frames), len(TRIANGULATION_JOINT_NAMES), 3)
    if max_reprojection_error is not None and reprojection_errors is not None:
        bad_points = reprojection_errors.reshape(len(observation_frames), len(TRIANGULATION_JOINT_NAMES)) > max_reprojection_error
        points_3d[bad_points] = np.nan
    points_3d = _smooth_points_3d(points_3d, confidences, smoothing_alpha)
    return _build_result(points_3d, confidences, camera_labels, reprojection_error)


def apply_triangulated_overrides(
    motion_frames: list[dict[str, object]],
    triangulation_result: TriangulationResult,
) -> list[dict[str, object]]:
    updated_frames: list[dict[str, object]] = []
    for frame, overrides in zip(motion_frames, triangulation_result.joint_overrides_by_frame):
        joints = frame.get("joints")
        if not isinstance(joints, dict):
            updated_frames.append(frame)
            continue

        updated_joints = {name: dict(value) for name, value in joints.items()}
        for joint_name, override in overrides.items():
            if joint_name not in updated_joints:
                continue
            updated_joints[joint_name] = {
                "x": float(override["x"]),
                "y": float(override["y"]),
                "z": float(override["z"]),
                "confidence": max(float(updated_joints[joint_name].get("confidence", 0.0)), float(override["confidence"])),
            }

        updated_frame = dict(frame)
        updated_frame["joints"] = updated_joints
        updated_frames.append(updated_frame)

    return updated_frames


def triangulation_metadata(result: TriangulationResult) -> dict[str, object]:
    return {
        "enabled": True,
        "camera_labels": result.camera_labels,
        "joint_count": len(result.joint_names),
        "triangulated_point_count": result.triangulated_point_count,
        "mean_reprojection_error": result.mean_reprojection_error,
    }


def _load_camera_group_class():
    try:
        module = importlib.import_module("".join(("ani", "pose", "lib.cameras")))
    except ModuleNotFoundError:
        raise
    return module.CameraGroup


def _load_camera_group(calibration_path: Path):
    camera_group_class = _load_camera_group_class()
    return camera_group_class.load(str(calibration_path))


def _matching_camera_labels(camera_group, observation_frames: list[dict[str, Any]]) -> list[str]:
    if not observation_frames:
        return []
    available_labels = set(observation_frames[0].get("camera_bodies", {}))
    available_by_upper = {label.upper(): label for label in available_labels}
    camera_names = list(camera_group.get_names())
    matched_camera_names = [name for name in camera_names if name.upper() in available_by_upper]
    matched_labels = [available_by_upper[name.upper()] for name in matched_camera_names]
    if len(matched_camera_names) == len(camera_names):
        return matched_labels
    if hasattr(camera_group, "subset_cameras_names"):
        subset_group = camera_group.subset_cameras_names(matched_camera_names)
        camera_group.cameras = subset_group.cameras
    return matched_labels


def _build_points_2d(
    observation_frames: list[dict[str, Any]],
    camera_labels: list[str],
    *,
    body_threshold: float,
    hand_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    points_2d = np.full(
        (len(camera_labels), len(observation_frames), len(TRIANGULATION_JOINT_NAMES), 2),
        np.nan,
        dtype=np.float64,
    )
    confidences = np.zeros(
        (len(observation_frames), len(TRIANGULATION_JOINT_NAMES)),
        dtype=np.float64,
    )

    for frame_index, frame in enumerate(observation_frames):
        camera_bodies = frame.get("camera_bodies", {})
        camera_hands = frame.get("camera_hands", {})
        for camera_index, label in enumerate(camera_labels):
            body_points = camera_bodies.get(label)
            if body_points is not None:
                _fill_body_points(points_2d, confidences, camera_index, frame_index, body_points, body_threshold)
            hands_by_side = camera_hands.get(label, {})
            if isinstance(hands_by_side, dict):
                _fill_hand_points(points_2d, confidences, camera_index, frame_index, hands_by_side, hand_threshold)

    return points_2d, confidences


def _fill_body_points(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    camera_index: int,
    frame_index: int,
    body_points: list[Point],
    threshold: float,
) -> None:
    for joint_offset, joint_name in enumerate(BODY_TRIANGULATION_JOINTS):
        point_index = BODY_NAME_TO_INDEX.get(joint_name, BODY_FOOT_NAME_TO_INDEX.get(joint_name))
        if point_index is None or len(body_points) <= point_index:
            continue
        point = body_points[point_index]
        _fill_point(points_2d, confidences, camera_index, frame_index, joint_offset, point, threshold)


def _fill_hand_points(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    camera_index: int,
    frame_index: int,
    hands_by_side: dict[str, dict],
    threshold: float,
) -> None:
    start_offset = len(BODY_TRIANGULATION_JOINTS)
    for hand_offset, (_, side, point_index) in enumerate(HAND_TRIANGULATION_JOINTS):
        hand_payload = hands_by_side.get(side)
        if hand_payload is None:
            continue
        points = hand_payload.get("points")
        if not points or len(points) <= point_index:
            continue
        _fill_point(
            points_2d,
            confidences,
            camera_index,
            frame_index,
            start_offset + hand_offset,
            points[point_index],
            threshold,
        )


def _fill_point(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    camera_index: int,
    frame_index: int,
    joint_offset: int,
    point: Point,
    threshold: float,
) -> None:
    x, y, confidence = point
    if confidence <= threshold:
        return
    points_2d[camera_index, frame_index, joint_offset] = (float(x), float(y))
    confidences[frame_index, joint_offset] = max(confidences[frame_index, joint_offset], float(confidence))


def _reprojection_errors(camera_group, points_3d_flat: np.ndarray, flat_points: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    if not hasattr(camera_group, "reprojection_error"):
        return None, None
    reprojection_error = np.asarray(camera_group.reprojection_error(points_3d_flat, flat_points, mean=True))
    finite_error = reprojection_error[np.isfinite(reprojection_error)]
    if finite_error.size == 0:
        return None, reprojection_error
    return float(np.mean(finite_error)), reprojection_error


def _smooth_points_3d(points_3d: np.ndarray, confidences: np.ndarray, smoothing_alpha: float) -> np.ndarray:
    alpha = max(0.0, min(float(smoothing_alpha), 1.0))
    if alpha >= 1.0:
        return points_3d
    smoothed = points_3d.copy()
    for joint_index in range(points_3d.shape[1]):
        previous: np.ndarray | None = None
        for frame_index in range(points_3d.shape[0]):
            current = points_3d[frame_index, joint_index]
            if not np.all(np.isfinite(current)):
                continue
            if confidences[frame_index, joint_index] <= 0:
                continue
            if previous is None:
                previous = current.copy()
                continue
            blended = previous * (1.0 - alpha) + current * alpha
            smoothed[frame_index, joint_index] = blended
            previous = blended
    return smoothed


def _build_result(
    points_3d: np.ndarray,
    confidences: np.ndarray,
    camera_labels: list[str],
    mean_reprojection_error: float | None,
) -> TriangulationResult:
    overrides_by_frame: list[dict[str, dict[str, float]]] = []
    triangulated_count = 0
    for frame_index in range(points_3d.shape[0]):
        frame_overrides: dict[str, dict[str, float]] = {}
        for joint_index, joint_name in enumerate(TRIANGULATION_JOINT_NAMES):
            x, y, z = points_3d[frame_index, joint_index]
            if not all(math.isfinite(value) for value in (x, y, z)):
                continue
            frame_overrides[joint_name] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "confidence": float(confidences[frame_index, joint_index]),
            }
            triangulated_count += 1
        overrides_by_frame.append(frame_overrides)

    return TriangulationResult(
        joint_overrides_by_frame=overrides_by_frame,
        camera_labels=camera_labels,
        joint_names=list(TRIANGULATION_JOINT_NAMES),
        mean_reprojection_error=mean_reprojection_error,
        triangulated_point_count=triangulated_count,
    )
