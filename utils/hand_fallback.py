from __future__ import annotations

import math
from collections import Counter

Point = tuple[int, int, float]
HAND_REJECTION_COUNTS: Counter[str] = Counter()

DEFAULT_HAND_TEMPLATE = {
    0: (0.00, 0.00),
    1: (-0.12, 0.16),
    2: (-0.28, 0.30),
    3: (-0.42, 0.48),
    4: (-0.54, 0.68),
    5: (-0.20, 0.34),
    6: (-0.22, 0.58),
    7: (-0.24, 0.82),
    8: (-0.26, 1.04),
    9: (0.00, 0.40),
    10: (0.00, 0.66),
    11: (0.00, 0.92),
    12: (0.00, 1.16),
    13: (0.20, 0.38),
    14: (0.22, 0.62),
    15: (0.24, 0.86),
    16: (0.26, 1.08),
    17: (0.38, 0.28),
    18: (0.48, 0.46),
    19: (0.56, 0.64),
    20: (0.62, 0.82),
}


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-6:
        return (0.0, -1.0)
    return (vector[0] / length, vector[1] / length)


def _point_xy(point: Point | tuple[int, int, float]) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def anchor_hand_to_wrist(hand_points: list[Point], wrist_point: tuple[int, int, float]) -> list[Point]:
    if not hand_points:
        return hand_points

    target_x, target_y = wrist_point[0], wrist_point[1]
    source_x, source_y = hand_points[0][0], hand_points[0][1]
    delta_x = target_x - source_x
    delta_y = target_y - source_y
    return [
        (int(round(x + delta_x)), int(round(y + delta_y)), float(conf))
        for x, y, conf in hand_points
    ]


def is_hand_detection_valid(
    hand_points: list[Point] | None,
    wrist_point: tuple[int, int, float],
    elbow_point: tuple[int, int, float],
    config,
) -> bool:
    if hand_points is None or len(hand_points) != 21:
        record_hand_rejection("missing_points")
        return False

    wrist_xy = _point_xy(wrist_point)
    elbow_xy = _point_xy(elbow_point)
    hand_wrist_xy = _point_xy(hand_points[0])
    forearm_len = max(math.hypot(wrist_xy[0] - elbow_xy[0], wrist_xy[1] - elbow_xy[1]), 1.0)

    wrist_offset = math.hypot(hand_wrist_xy[0] - wrist_xy[0], hand_wrist_xy[1] - wrist_xy[1])
    max_wrist_offset = max(config.hand_box_min_size * 0.38, forearm_len * config.hand_wrist_max_offset_scale * 1.15)
    if wrist_offset > max_wrist_offset:
        record_hand_rejection("wrist_offset")
        return False

    valid_points = sum(point[2] > config.hand_kp_threshold * 0.5 for point in hand_points)
    if valid_points < config.hand_min_valid_points:
        record_hand_rejection("valid_points")
        return False

    max_radial = max(math.hypot(point[0] - wrist_xy[0], point[1] - wrist_xy[1]) for point in hand_points[1:])
    if max_radial > max(forearm_len * 2.75, config.hand_box_min_size * 1.40):
        record_hand_rejection("radial_distance")
        return False

    palm_indices = (5, 9, 13, 17)
    palm_distances = [math.hypot(hand_points[index][0] - wrist_xy[0], hand_points[index][1] - wrist_xy[1]) for index in palm_indices]
    average_palm_distance = sum(palm_distances) / len(palm_distances)
    if average_palm_distance < forearm_len * 0.035 or average_palm_distance > max(forearm_len * 1.85, config.hand_box_min_size * 0.55):
        record_hand_rejection("palm_distance")
        return False

    return True


def record_hand_rejection(reason: str) -> None:
    HAND_REJECTION_COUNTS[reason] += 1


def pop_hand_rejection_telemetry() -> dict[str, int]:
    if not HAND_REJECTION_COUNTS:
        return {}
    snapshot = dict(HAND_REJECTION_COUNTS)
    HAND_REJECTION_COUNTS.clear()
    return snapshot


def has_usable_hand_detection(hand_points: list[Point] | None, config) -> bool:
    if hand_points is None or len(hand_points) != 21:
        return False
    minimum_points = max(5, config.hand_min_valid_points // 2)
    return sum(point[2] > config.hand_kp_threshold * 0.25 for point in hand_points) >= minimum_points


def generate_default_hand(
    wrist_point: tuple[int, int, float],
    elbow_point: tuple[int, int, float],
    side: str,
    config,
) -> list[Point]:
    wrist_x, wrist_y, _ = wrist_point
    elbow_x, elbow_y, _ = elbow_point
    forward_axis = _normalize((wrist_x - elbow_x, wrist_y - elbow_y))
    lateral_axis = (-forward_axis[1], forward_axis[0])
    if side == "left":
        lateral_axis = (-lateral_axis[0], -lateral_axis[1])

    forearm_len = max(math.hypot(wrist_x - elbow_x, wrist_y - elbow_y), 1.0)
    hand_scale = max(forearm_len * config.hand_default_scale, 18.0)
    confidence = max(config.hand_default_confidence, config.hand_kp_threshold + 0.05)

    default_points: list[Point] = []
    for point_index in range(21):
        lateral, forward = DEFAULT_HAND_TEMPLATE[point_index]
        point_x = wrist_x + lateral_axis[0] * lateral * hand_scale + forward_axis[0] * forward * hand_scale
        point_y = wrist_y + lateral_axis[1] * lateral * hand_scale + forward_axis[1] * forward * hand_scale
        default_points.append((int(round(point_x)), int(round(point_y)), float(confidence)))

    return default_points
