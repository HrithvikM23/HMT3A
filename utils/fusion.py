from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from utils.payloads import HandPayload

BODY_TORSO_POINTS = (5, 6, 11, 12)
BODY_JOINT_NAME_TO_INDEX = {
    "LeftShoulder": 5,
    "RightShoulder": 6,
    "LeftElbow": 7,
    "RightElbow": 8,
    "LeftWrist": 9,
    "RightWrist": 10,
    "LeftHip": 11,
    "RightHip": 12,
    "LeftKnee": 13,
    "RightKnee": 14,
    "LeftAnkle": 15,
    "RightAnkle": 16,
}
HAND_JOINT_NAME_TO_INDEX = {
    "LeftThumb1": ("left", 1),
    "LeftThumb2": ("left", 2),
    "LeftThumb3": ("left", 3),
    "LeftThumb4": ("left", 4),
    "LeftIndex1": ("left", 5),
    "LeftIndex2": ("left", 6),
    "LeftIndex3": ("left", 7),
    "LeftIndex4": ("left", 8),
    "LeftMiddle1": ("left", 9),
    "LeftMiddle2": ("left", 10),
    "LeftMiddle3": ("left", 11),
    "LeftMiddle4": ("left", 12),
    "LeftRing1": ("left", 13),
    "LeftRing2": ("left", 14),
    "LeftRing3": ("left", 15),
    "LeftRing4": ("left", 16),
    "LeftPinky1": ("left", 17),
    "LeftPinky2": ("left", 18),
    "LeftPinky3": ("left", 19),
    "LeftPinky4": ("left", 20),
    "RightThumb1": ("right", 1),
    "RightThumb2": ("right", 2),
    "RightThumb3": ("right", 3),
    "RightThumb4": ("right", 4),
    "RightIndex1": ("right", 5),
    "RightIndex2": ("right", 6),
    "RightIndex3": ("right", 7),
    "RightIndex4": ("right", 8),
    "RightMiddle1": ("right", 9),
    "RightMiddle2": ("right", 10),
    "RightMiddle3": ("right", 11),
    "RightMiddle4": ("right", 12),
    "RightRing1": ("right", 13),
    "RightRing2": ("right", 14),
    "RightRing3": ("right", 15),
    "RightRing4": ("right", 16),
    "RightPinky1": ("right", 17),
    "RightPinky2": ("right", 18),
    "RightPinky3": ("right", 19),
    "RightPinky4": ("right", 20),
}
DEFAULT_CAMERA_WEIGHT = 1.0
DEFAULT_CAMERA_CALIBRATION = {"depth_sign": 0.0, "depth_scale": 0.0, "weight": DEFAULT_CAMERA_WEIGHT}


@dataclass(frozen=True, slots=True)
class FrameReference:
    origin_x: float
    origin_y: float
    scale: float


def load_camera_calibrations(path: Path | None) -> dict[str, dict[str, float]]:
    calibrations: dict[str, dict[str, float]] = {}
    if path is None:
        return calibrations

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration file must contain a JSON object keyed by camera label.")

    for label, values in payload.items():
        if not isinstance(label, str) or not isinstance(values, dict):
            continue
        target = calibrations.setdefault(label.upper(), dict(DEFAULT_CAMERA_CALIBRATION))


@dataclass(frozen=True, slots=True)
class FrameReference:
    origin_x: float
    origin_y: float
    scale: float


def load_camera_calibrations(path: Path | None) -> dict[str, dict[str, float]]:
    calibrations: dict[str, dict[str, float]] = {}
    if path is None:
        return calibrations

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration file must contain a JSON object keyed by camera label.")

    for label, values in payload.items():
        if not isinstance(label, str) or not isinstance(values, dict):
            continue
        target = calibrations.setdefault(label.upper(), dict(DEFAULT_CAMERA_CALIBRATION))
        for key in ("depth_sign", "depth_scale", "weight"):
            if key in values:
                target[key] = float(values[key])
    return calibrations


def _get_calibration_entry(label: str, calibrations: Mapping[str, Mapping[str, float]] | None = None) -> Mapping[str, float]:
    if not calibrations:
        return {}
    if label in calibrations:
        return calibrations[label]
    if label.upper() in calibrations:
        return calibrations[label.upper()]
    if label.lower() in calibrations:
        return calibrations[label.lower()]
    return {}


def _camera_weight(label: str, calibrations: Mapping[str, Mapping[str, float]] | None = None) -> float:
    calibration = _get_calibration_entry(label, calibrations)
    weight = float(calibration.get("weight", DEFAULT_CAMERA_WEIGHT))
    return max(0.0, weight)


def compute_body_reference(points, threshold: float) -> FrameReference | None:
    confident = [points[index] for index in BODY_TORSO_POINTS if points[index][2] > threshold]
    if len(confident) < 2:
        return None

    xs = [point[0] for point in confident]
    ys = [point[1] for point in confident]
    origin_x = sum(xs) / len(xs)
    origin_y = sum(ys) / len(ys)
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return FrameReference(origin_x=origin_x, origin_y=origin_y, scale=scale)


def compute_hand_reference(hand_payload, threshold: float) -> FrameReference | None:
    if hand_payload is None:
        return None

    points = hand_payload["points"]
    confident = [point for point in points if point[2] > threshold]
    if len(confident) < 2:
        return None

    wrist_x, wrist_y, _ = points[0]
    x1, y1, x2, y2 = hand_payload["box"]
    scale = max(x2 - x1, y2 - y1, 1)
    return FrameReference(origin_x=float(wrist_x), origin_y=float(wrist_y), scale=float(scale))


def project_points(points, source_reference: FrameReference, target_reference: FrameReference):
    projected = []
    for x, y, conf in points:
        local_x = (x - source_reference.origin_x) / source_reference.scale
        local_y = (y - source_reference.origin_y) / source_reference.scale
        projected_x = int(round(target_reference.origin_x + local_x * target_reference.scale))
        projected_y = int(round(target_reference.origin_y + local_y * target_reference.scale))
        projected.append((projected_x, projected_y, float(conf)))
    return projected


def _project_box(box, source_reference: FrameReference, target_reference: FrameReference):
    projected_corners = project_points(
        [(box[0], box[1], 1.0), (box[2], box[3], 1.0)],
        source_reference,
        target_reference,
    )
    x1, y1, _ = projected_corners[0]
    x2, y2, _ = projected_corners[1]
    return x1, y1, x2, y2


def _choose_reference(camera_points_by_label, threshold: float, reference_label: str, reference_builder):
    preferred = camera_points_by_label.get(reference_label)
    if preferred is not None:
        reference = reference_builder(preferred, threshold)
        if reference is not None:
            return preferred, reference

    for payload in camera_points_by_label.values():
        reference = reference_builder(payload, threshold)
        if reference is not None:
            return payload, reference
    return None, None


def _prepare_body_sources(
    camera_bodies: dict[str, list[tuple[int, int, float]]],
    threshold: float,
    calibrations: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, tuple[list[tuple[int, int, float]], FrameReference, float]]:
    prepared: dict[str, tuple[list[tuple[int, int, float]], FrameReference, float]] = {}
    for label, points in camera_bodies.items():
        reference = compute_body_reference(points, threshold)
        if reference is None:
            continue
        prepared[label] = (points, reference, _camera_weight(label, calibrations))
    return prepared


def _prepare_hand_sources(
    camera_hands: Mapping[str, HandPayload],
    threshold: float,
    calibrations: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, tuple[HandPayload, FrameReference, float]]:
    prepared: dict[str, tuple[HandPayload, FrameReference, float]] = {}
    for label, hand_payload in camera_hands.items():
        reference = compute_hand_reference(hand_payload, threshold)
        if reference is None:
            continue
        prepared[label] = (hand_payload, reference, _camera_weight(label, calibrations))
    return prepared


def fuse_body_views(
    camera_bodies: dict[str, list[tuple[int, int, float]]],
    threshold: float,
    reference_label: str = "CAM_0",
    calibrations: Mapping[str, Mapping[str, float]] | None = None,
):
    prepared_sources = _prepare_body_sources(camera_bodies, threshold, calibrations)
    if not prepared_sources:
        return None
    reference_source = prepared_sources.get(reference_label)
    if reference_source is None:
        reference_points, reference_frame, _ = next(iter(prepared_sources.values()))
    else:
        reference_points, reference_frame, _ = reference_source

    projected_by_label: dict[str, list[tuple[int, int, float]]] = {}
    for label, (points, source_reference, _) in prepared_sources.items():
        projected_by_label[label] = project_points(points, source_reference, reference_frame)

    fused_points = []
    point_count = len(reference_points)
    for point_index in range(point_count):
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        best_conf = 0.0

        for label, points in projected_by_label.items():
            x, y, conf = points[point_index]
            if conf <= 0:
                continue
            _, _, view_weight = prepared_sources[label]
            weight = (conf ** 2) * view_weight
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight
            best_conf = max(best_conf, conf)

        if total_weight <= 0:
            fused_points.append(reference_points[point_index])
            continue

        fused_points.append(
            (
                int(round(weighted_x / total_weight)),
                int(round(weighted_y / total_weight)),
                float(best_conf),
            )
        )

    return fused_points


def fuse_hand_views(
    camera_hands: Mapping[str, HandPayload],
    threshold: float,
    reference_label: str = "CAM_0",
    calibrations: Mapping[str, Mapping[str, float]] | None = None,
):
    prepared_sources = _prepare_hand_sources(camera_hands, threshold, calibrations)
    if not prepared_sources:
        return None
    reference_source = prepared_sources.get(reference_label)
    if reference_source is None:
        reference_hand, reference_frame, _ = next(iter(prepared_sources.values()))
    else:
        reference_hand, reference_frame, _ = reference_source

    projected_by_label: dict[str, HandPayload] = {}
    for label, (hand_payload, source_reference, _) in prepared_sources.items():
        projected_points = project_points(hand_payload["points"], source_reference, reference_frame)
        projected_box = _project_box(hand_payload["box"], source_reference, reference_frame)
        projected_by_label[label] = {"points": projected_points, "box": projected_box}

    fused_points = []
    point_count = len(reference_hand["points"])
    for point_index in range(point_count):
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        best_conf = 0.0

        for label, hand_payload in projected_by_label.items():
            x, y, conf = hand_payload["points"][point_index]
            if conf <= 0:
                continue
            _, _, view_weight = prepared_sources[label]
            weight = (conf ** 2) * view_weight
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight
            best_conf = max(best_conf, conf)

        if total_weight <= 0:
            fused_points.append(reference_hand["points"][point_index])
            continue

        fused_points.append(
            (
                int(round(weighted_x / total_weight)),
                int(round(weighted_y / total_weight)),
                float(best_conf),
            )
        )

    wrist_x, wrist_y, _ = fused_points[0]
    scale = int(round(reference_frame.scale))
    half = max(scale // 2, 1)
    fused_box = (
        int(round(wrist_x - half)),
        int(round(wrist_y - half)),
        int(round(wrist_x + half)),
        int(round(wrist_y + half)),
    )
    return {"points": fused_points, "box": fused_box}


def _estimate_body_joint_depths(
    camera_bodies: dict[str, list[tuple[int, int, float]]],
    threshold: float,
    calibrations: dict[str, dict[str, float]],
    depth_scale: float,
) -> dict[str, float]:
    prepared_sources: list[tuple[list[tuple[int, int, float]], FrameReference, float, float, float]] = []
    for label, points in camera_bodies.items():
        calibration = _get_calibration_entry(label, calibrations)
        depth_sign = float(calibration.get("depth_sign", 0.0))
        if depth_sign == 0.0:
            continue
        reference = compute_body_reference(points, threshold)
        if reference is None:
            continue
        prepared_sources.append(
            (
                points,
                reference,
                depth_sign,
                float(calibration.get("depth_scale", 1.0)),
                _camera_weight(label, calibrations),
            )
        )

    joint_depths: dict[str, float] = {}
    for joint_name, point_index in BODY_JOINT_NAME_TO_INDEX.items():
        weighted_depth = 0.0
        total_weight = 0.0
        for points, reference, depth_sign, depth_view_scale, view_weight in prepared_sources:
            point = points[point_index]
            if point[2] <= threshold:
                continue
            local_x = (point[0] - reference.origin_x) / max(reference.scale, 1.0)
            weight = (point[2] ** 2) * view_weight
            weighted_depth += local_x * depth_sign * depth_view_scale * depth_scale * reference.scale * weight
            total_weight += weight
        joint_depths[joint_name] = 0.0 if total_weight <= 0.0 else weighted_depth / total_weight
    return joint_depths


def _estimate_hand_joint_depths(
    camera_hands: Mapping[str, Mapping[str, HandPayload]],
    threshold: float,
    calibrations: dict[str, dict[str, float]],
    depth_scale: float,
) -> dict[str, float]:
    prepared_sources_by_side: dict[str, list[tuple[HandPayload, FrameReference, float, float, float]]] = {
        "left": [],
        "right": [],
    }
    for label, hands_by_side in camera_hands.items():
        calibration = _get_calibration_entry(label, calibrations)
        depth_sign = float(calibration.get("depth_sign", 0.0))
        if depth_sign == 0.0:
            continue
        depth_view_scale = float(calibration.get("depth_scale", 1.0))
        view_weight = _camera_weight(label, calibrations)
        for side, hand_payload in hands_by_side.items():
            reference = compute_hand_reference(hand_payload, threshold)
            if reference is None:
                continue
            prepared_sources_by_side.setdefault(side, []).append(
                (hand_payload, reference, depth_sign, depth_view_scale, view_weight)
            )

    joint_depths: dict[str, float] = {}
    for joint_name, (side, point_index) in HAND_JOINT_NAME_TO_INDEX.items():
        weighted_depth = 0.0
        total_weight = 0.0
        for hand_payload, reference, depth_sign, depth_view_scale, view_weight in prepared_sources_by_side.get(side, []):
            point = hand_payload["points"][point_index]
            if point[2] <= threshold:
                continue
            local_x = (point[0] - reference.origin_x) / max(reference.scale, 1.0)
            weight = (point[2] ** 2) * view_weight
            weighted_depth += local_x * depth_sign * depth_view_scale * depth_scale * reference.scale * weight
            total_weight += weight
        joint_depths[joint_name] = 0.0 if total_weight <= 0.0 else weighted_depth / total_weight
    return joint_depths


def estimate_joint_depths(
    camera_bodies: dict[str, list[tuple[int, int, float]]],
    camera_hands: Mapping[str, Mapping[str, HandPayload]],
    body_threshold: float,
    hand_threshold: float,
    calibrations: dict[str, dict[str, float]],
    depth_scale: float = 1.0,
) -> dict[str, float]:
    joint_depths = _estimate_body_joint_depths(
        camera_bodies=camera_bodies,
        threshold=body_threshold,
        calibrations=calibrations,
        depth_scale=depth_scale,
    )
    joint_depths.update(
        _estimate_hand_joint_depths(
            camera_hands=camera_hands,
            threshold=hand_threshold,
            calibrations=calibrations,
            depth_scale=depth_scale,
        )
    )
    if "HipsRoot" not in joint_depths:
        joint_depths["HipsRoot"] = 0.0
    if "Chest" not in joint_depths:
        joint_depths["Chest"] = 0.0
    return joint_depths
