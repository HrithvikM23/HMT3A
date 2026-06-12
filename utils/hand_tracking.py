from __future__ import annotations

import math

from utils.hand_fallback import is_hand_detection_valid
from utils.prediction import translate_box, translate_points

Point = tuple[int, int, float]
Box = tuple[int, int, int, int]

PALM_INDICES = (0, 5, 9, 13, 17)


def average_confidence(points: list[Point]) -> float:
    if not points:
        return 0.0
    return sum(float(point[2]) for point in points) / float(len(points))


def valid_point_count(points: list[Point], threshold: float) -> int:
    return sum(1 for point in points if float(point[2]) > threshold)


def hand_motion_delta(
    previous_wrist: Point | None,
    wrist_point: Point,
    previous_elbow: Point | None = None,
    elbow_point: Point | None = None,
) -> tuple[float, float]:
    if previous_wrist is None:
        return 0.0, 0.0

    wrist_dx = float(wrist_point[0] - previous_wrist[0])
    wrist_dy = float(wrist_point[1] - previous_wrist[1])
    if previous_elbow is None or elbow_point is None:
        return wrist_dx, wrist_dy

    elbow_dx = float(elbow_point[0] - previous_elbow[0])
    elbow_dy = float(elbow_point[1] - previous_elbow[1])
    return (wrist_dx * 0.75) + (elbow_dx * 0.25), (wrist_dy * 0.75) + (elbow_dy * 0.25)


def predict_hand_payload(
    previous_payload: dict | None,
    previous_wrist: Point | None,
    wrist_point: Point,
    confidence_decay: float,
    previous_elbow: Point | None = None,
    elbow_point: Point | None = None,
):
    if previous_payload is None or previous_wrist is None:
        return None

    offset_x, offset_y = hand_motion_delta(previous_wrist, wrist_point, previous_elbow, elbow_point)
    depths = previous_payload.get("depths")
    return (
        translate_box(previous_payload["box"], offset_x, offset_y),
        translate_points(previous_payload["points"], offset_x, offset_y, confidence_decay),
        depths,
    )


def temporal_distance(points: list[Point], previous_points: list[Point] | None) -> float:
    if previous_points is None or len(previous_points) != len(points):
        return 0.0

    distances = [
        math.hypot(float(point[0] - previous[0]), float(point[1] - previous[1]))
        for point, previous in zip(points, previous_points, strict=True)
        if point[2] > 0.0 and previous[2] > 0.0
    ]
    if not distances:
        return 0.0
    return sum(distances) / float(len(distances))


def palm_spread(points: list[Point]) -> float:
    if len(points) != 21:
        return 0.0

    wrist = points[0]
    distances = [
        math.hypot(float(points[index][0] - wrist[0]), float(points[index][1] - wrist[1]))
        for index in PALM_INDICES[1:]
    ]
    return sum(distances) / float(len(distances))


def hand_detection_score(
    points: list[Point],
    wrist_point: Point,
    elbow_point: Point,
    config,
    previous_points: list[Point] | None = None,
) -> float:
    if len(points) != 21:
        return -1_000.0

    wrist_distance = math.hypot(float(points[0][0] - wrist_point[0]), float(points[0][1] - wrist_point[1]))
    forearm_len = max(math.hypot(float(wrist_point[0] - elbow_point[0]), float(wrist_point[1] - elbow_point[1])), 1.0)
    valid_points = valid_point_count(points, config.hand_kp_threshold * 0.5)
    confidence = average_confidence(points)
    spread = palm_spread(points)
    temporal = temporal_distance(points, previous_points)

    score = float(valid_points) * 2.0
    score += confidence * 20.0
    score -= min(wrist_distance / max(forearm_len, 1.0), 2.0) * 8.0
    score -= min(temporal / max(forearm_len, 1.0), 3.0) * 5.0

    if spread < forearm_len * 0.06:
        score -= 10.0
    if spread > max(forearm_len * 1.8, config.hand_box_min_size * 0.7):
        score -= 8.0
    if is_hand_detection_valid(points, wrist_point, elbow_point, config):
        score += 25.0
    return score


def blend_with_prediction(
    detected_points: list[Point],
    predicted_points: list[Point] | None,
    config,
) -> list[Point]:
    if predicted_points is None or len(predicted_points) != len(detected_points):
        return detected_points

    blended: list[Point] = []
    for detected, predicted in zip(detected_points, predicted_points, strict=True):
        dx = detected[0] - predicted[0]
        dy = detected[1] - predicted[1]
        distance = math.hypot(float(dx), float(dy))
        detection_weight = 0.72 if distance < config.hand_box_min_size * 0.45 else 0.48
        x = int(round((detected[0] * detection_weight) + (predicted[0] * (1.0 - detection_weight))))
        y = int(round((detected[1] * detection_weight) + (predicted[1] * (1.0 - detection_weight))))
        confidence = max(float(detected[2]), float(predicted[2]) * config.hold_confidence_decay)
        blended.append((x, y, confidence))
    return blended
