from __future__ import annotations

import math

from utils.skeleton import Point


def derive_foot_points(
    knee_point: Point,
    ankle_point: Point,
) -> tuple[Point, Point]:
    dx = float(ankle_point[0] - knee_point[0])
    dy = float(ankle_point[1] - knee_point[1])
    shin_length = math.hypot(dx, dy)
    if shin_length <= 1e-6:
        unit_x, unit_y = 0.0, 1.0
        shin_length = 24.0
    else:
        unit_x = dx / shin_length
        unit_y = dy / shin_length

    foot_length = max(shin_length * 0.35, 12.0)
    toe_length = max(shin_length * 0.25, 10.0)
    foot_x = float(ankle_point[0]) + unit_x * foot_length
    foot_y = float(ankle_point[1]) + unit_y * foot_length
    toe_x = foot_x + unit_x * toe_length
    toe_y = foot_y + unit_y * toe_length
    confidence = min(float(knee_point[2]), float(ankle_point[2]))

    return (
        (int(round(foot_x)), int(round(foot_y)), confidence),
        (int(round(toe_x)), int(round(toe_y)), confidence),
    )
