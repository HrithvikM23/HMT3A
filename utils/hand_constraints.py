from __future__ import annotations

import math

Point = tuple[int, int, float]

THUMB_CHAIN = (0, 1, 2, 3, 4)
FINGER_CHAINS = (
    ("index", (0, 5, 6, 7, 8)),
    ("middle", (0, 9, 10, 11, 12)),
    ("ring", (0, 13, 14, 15, 16)),
    ("pinky", (0, 17, 18, 19, 20)),
)

MAX_CHAIN_BEND_DEGREES = {
    "thumb": (65.0, 55.0, 50.0),
    "index": (75.0, 95.0, 80.0),
    "middle": (75.0, 100.0, 85.0),
    "ring": (75.0, 95.0, 80.0),
    "pinky": (80.0, 95.0, 80.0),
}

MAX_RADIAL_MULTIPLIER = {
    0: 0.0,
    1: 0.70,
    2: 0.95,
    3: 1.20,
    4: 1.40,
    5: 0.75,
    6: 1.00,
    7: 1.25,
    8: 1.55,
    9: 0.80,
    10: 1.10,
    11: 1.35,
    12: 1.65,
    13: 0.80,
    14: 1.08,
    15: 1.30,
    16: 1.58,
    17: 0.78,
    18: 1.00,
    19: 1.20,
    20: 1.45,
}

BONE_LENGTH_MULTIPLIER = {
    (0, 1): (0.18, 0.55),
    (1, 2): (0.15, 0.42),
    (2, 3): (0.12, 0.35),
    (3, 4): (0.10, 0.30),
    (0, 5): (0.22, 0.60),
    (5, 6): (0.14, 0.42),
    (6, 7): (0.12, 0.34),
    (7, 8): (0.10, 0.28),
    (0, 9): (0.24, 0.66),
    (9, 10): (0.15, 0.44),
    (10, 11): (0.13, 0.36),
    (11, 12): (0.11, 0.30),
    (0, 13): (0.22, 0.62),
    (13, 14): (0.14, 0.42),
    (14, 15): (0.12, 0.34),
    (15, 16): (0.10, 0.28),
    (0, 17): (0.20, 0.58),
    (17, 18): (0.13, 0.38),
    (18, 19): (0.11, 0.30),
    (19, 20): (0.09, 0.24),
}


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-6:
        return (1.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def _rotate(vector: tuple[float, float], angle_radians: float) -> tuple[float, float]:
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)
    return (
        vector[0] * cos_theta - vector[1] * sin_theta,
        vector[0] * sin_theta + vector[1] * cos_theta,
    )


def _signed_angle(parent_vector: tuple[float, float], child_vector: tuple[float, float]) -> float:
    cross = parent_vector[0] * child_vector[1] - parent_vector[1] * child_vector[0]
    dot = parent_vector[0] * child_vector[0] + parent_vector[1] * child_vector[1]
    return math.atan2(cross, dot)


def _point_xy(point: Point) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _make_point(x: float, y: float, confidence: float) -> Point:
    return int(round(x)), int(round(y)), float(confidence)


def _base_hand_scale(points: list[Point]) -> float:
    wrist_xy = _point_xy(points[0])
    anchors = [_point_xy(points[index]) for index in (5, 9, 13, 17)]
    distances = [_distance(wrist_xy, anchor_xy) for anchor_xy in anchors]
    scale = sum(distances) / max(len(distances), 1)
    return max(scale, 12.0)


def _clamp_distance(
    origin_xy: tuple[float, float],
    target_xy: tuple[float, float],
    min_distance: float,
    max_distance: float,
) -> tuple[float, float]:
    direction = (target_xy[0] - origin_xy[0], target_xy[1] - origin_xy[1])
    length = math.hypot(direction[0], direction[1])
    if length <= 1e-6:
        return origin_xy[0] + max_distance, origin_xy[1]

    clamped_length = max(min_distance, min(max_distance, length))
    unit = (direction[0] / length, direction[1] / length)
    return (
        origin_xy[0] + unit[0] * clamped_length,
        origin_xy[1] + unit[1] * clamped_length,
    )


def _enforce_radial_limits(points: list[Point], base_scale: float) -> None:
    wrist_xy = _point_xy(points[0])
    for point_index, multiplier in MAX_RADIAL_MULTIPLIER.items():
        if point_index == 0:
            continue
        max_distance = base_scale * multiplier
        point_xy = _point_xy(points[point_index])
        clamped_x, clamped_y = _clamp_distance(wrist_xy, point_xy, 0.0, max_distance)
        points[point_index] = _make_point(clamped_x, clamped_y, points[point_index][2])


def _enforce_bone_lengths(points: list[Point], base_scale: float) -> None:
    for (parent_index, child_index), (min_multiplier, max_multiplier) in BONE_LENGTH_MULTIPLIER.items():
        parent_xy = _point_xy(points[parent_index])
        child_xy = _point_xy(points[child_index])
        min_distance = base_scale * min_multiplier
        max_distance = base_scale * max_multiplier
        clamped_x, clamped_y = _clamp_distance(parent_xy, child_xy, min_distance, max_distance)
        points[child_index] = _make_point(clamped_x, clamped_y, points[child_index][2])


def _enforce_chain_bend(points: list[Point], chain_name: str, chain_indices: tuple[int, int, int, int, int]) -> None:
    bend_limits = MAX_CHAIN_BEND_DEGREES[chain_name]

    for segment_offset, max_bend_degrees in enumerate(bend_limits, start=2):
        anchor_index = chain_indices[segment_offset - 2]
        parent_index = chain_indices[segment_offset - 1]
        child_index = chain_indices[segment_offset]

        anchor_xy = _point_xy(points[anchor_index])
        parent_xy = _point_xy(points[parent_index])
        child_xy = _point_xy(points[child_index])
        child_conf = points[child_index][2]

        parent_vector = (parent_xy[0] - anchor_xy[0], parent_xy[1] - anchor_xy[1])
        child_vector = (child_xy[0] - parent_xy[0], child_xy[1] - parent_xy[1])
        parent_unit = _normalize(parent_vector)
        child_unit = _normalize(child_vector)
        child_length = _distance(parent_xy, child_xy)

        angle = _signed_angle(parent_unit, child_unit)
        max_bend = math.radians(max_bend_degrees)
        clamped_angle = max(-max_bend, min(max_bend, angle))

        if abs(clamped_angle - angle) <= 1e-6:
            continue

        constrained_direction = _rotate(parent_unit, clamped_angle)
        constrained_x = parent_xy[0] + constrained_direction[0] * child_length
        constrained_y = parent_xy[1] + constrained_direction[1] * child_length
        points[child_index] = _make_point(constrained_x, constrained_y, child_conf)


def _project_local(
    wrist_xy: tuple[float, float],
    point_xy: tuple[float, float],
    lateral_axis: tuple[float, float],
    forward_axis: tuple[float, float],
) -> tuple[float, float]:
    delta_x = point_xy[0] - wrist_xy[0]
    delta_y = point_xy[1] - wrist_xy[1]
    return (
        delta_x * lateral_axis[0] + delta_y * lateral_axis[1],
        delta_x * forward_axis[0] + delta_y * forward_axis[1],
    )


def _unproject_local(
    wrist_xy: tuple[float, float],
    lateral_value: float,
    forward_value: float,
    lateral_axis: tuple[float, float],
    forward_axis: tuple[float, float],
) -> tuple[float, float]:
    return (
        wrist_xy[0] + lateral_value * lateral_axis[0] + forward_value * forward_axis[0],
        wrist_xy[1] + lateral_value * lateral_axis[1] + forward_value * forward_axis[1],
    )


def _enforce_finger_lanes(points: list[Point]) -> None:
    wrist_xy = _point_xy(points[0])
    index_root_xy = _point_xy(points[5])
    pinky_root_xy = _point_xy(points[17])
    lateral_axis = _normalize((pinky_root_xy[0] - index_root_xy[0], pinky_root_xy[1] - index_root_xy[1]))
    forward_axis = (-lateral_axis[1], lateral_axis[0])

    root_values = {
        "index": _project_local(wrist_xy, _point_xy(points[5]), lateral_axis, forward_axis)[0],
        "middle": _project_local(wrist_xy, _point_xy(points[9]), lateral_axis, forward_axis)[0],
        "ring": _project_local(wrist_xy, _point_xy(points[13]), lateral_axis, forward_axis)[0],
        "pinky": _project_local(wrist_xy, _point_xy(points[17]), lateral_axis, forward_axis)[0],
    }

    boundaries = {
        "index": (root_values["index"] - abs(root_values["middle"] - root_values["index"]), (root_values["index"] + root_values["middle"]) * 0.5),
        "middle": ((root_values["index"] + root_values["middle"]) * 0.5, (root_values["middle"] + root_values["ring"]) * 0.5),
        "ring": ((root_values["middle"] + root_values["ring"]) * 0.5, (root_values["ring"] + root_values["pinky"]) * 0.5),
        "pinky": ((root_values["ring"] + root_values["pinky"]) * 0.5, root_values["pinky"] + abs(root_values["pinky"] - root_values["ring"])),
    }

    for chain_name, chain_indices in FINGER_CHAINS:
        lower, upper = boundaries[chain_name]
        for point_index in chain_indices[2:]:
            point_xy = _point_xy(points[point_index])
            lateral_value, forward_value = _project_local(wrist_xy, point_xy, lateral_axis, forward_axis)
            clamped_lateral = max(lower, min(upper, lateral_value))
            if abs(clamped_lateral - lateral_value) <= 1e-6:
                continue
            constrained_x, constrained_y = _unproject_local(
                wrist_xy,
                clamped_lateral,
                forward_value,
                lateral_axis,
                forward_axis,
            )
            points[point_index] = _make_point(constrained_x, constrained_y, points[point_index][2])


def enforce_hand_constraints(hand_points: list[Point]) -> list[Point]:
    if len(hand_points) != 21:
        return hand_points

    constrained = list(hand_points)
    base_scale = _base_hand_scale(constrained)
    _enforce_radial_limits(constrained, base_scale)
    _enforce_bone_lengths(constrained, base_scale)
    _enforce_chain_bend(constrained, "thumb", THUMB_CHAIN)
    for chain_name, chain_indices in FINGER_CHAINS:
        _enforce_chain_bend(constrained, chain_name, chain_indices)
    _enforce_finger_lanes(constrained)
    _enforce_radial_limits(constrained, base_scale)
    _enforce_bone_lengths(constrained, base_scale)
    _enforce_chain_bend(constrained, "thumb", THUMB_CHAIN)
    for chain_name, chain_indices in FINGER_CHAINS:
        _enforce_chain_bend(constrained, chain_name, chain_indices)
    return constrained
