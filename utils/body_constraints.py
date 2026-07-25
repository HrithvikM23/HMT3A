from __future__ import annotations

import math

from utils.skeleton import Point

BODY_SEGMENT_GROUPS = {
    "upper_arm": ((5, 7), (6, 8)),
    "forearm": ((7, 9), (8, 10)),
    "thigh": ((11, 13), (12, 14)),
    "shin": ((13, 15), (14, 16)),
}


def _distance(a: Point, b: Point) -> float:
    return math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))


def _blend(previous: float, current: float, alpha: float) -> float:
    return previous * (1.0 - alpha) + current * alpha


def _make_point(x: float, y: float, confidence: float) -> Point:
    return int(round(x)), int(round(y)), float(confidence)


class BodyKinematicConstraints:
    def __init__(self, config) -> None:
        self.config = config
        self._target_lengths: dict[tuple[int, int], float] = {}

    def apply(self, points: list[Point]) -> list[Point]:
        if not self.config.body_constraints_enabled or len(points) < 17:
            return points

        constrained = list(points)
        self._update_target_lengths(constrained)
        self._mirror_segment_targets()
        self._apply_length_targets(constrained)
        return constrained

    def _update_target_lengths(self, points: list[Point]) -> None:
        alpha = self.config.body_length_smoothing_alpha
        threshold = self.config.body_conf_threshold
        for segment_pairs in BODY_SEGMENT_GROUPS.values():
            for parent_index, child_index in segment_pairs:
                parent = points[parent_index]
                child = points[child_index]
                if parent[2] <= threshold or child[2] <= threshold:
                    continue
                length = _distance(parent, child)
                if length <= 2.0:
                    continue
                key = (parent_index, child_index)
                previous = self._target_lengths.get(key)
                self._target_lengths[key] = length if previous is None else _blend(previous, length, alpha)

    def _mirror_segment_targets(self) -> None:
        # 2D mirroring is disabled because perspective foreshortening makes it counterproductive
        return

    def _apply_length_targets(self, points: list[Point]) -> None:
        correction = self.config.body_length_correction
        threshold = self.config.body_conf_threshold
        for segment_pairs in BODY_SEGMENT_GROUPS.values():
            for parent_index, child_index in segment_pairs:
                key = (parent_index, child_index)
                target_length = self._target_lengths.get(key)
                if target_length is None:
                    continue

                parent = points[parent_index]
                child = points[child_index]
                if parent[2] <= threshold or child[2] <= threshold:
                    continue

                dx = float(child[0] - parent[0])
                dy = float(child[1] - parent[1])
                current_length = math.hypot(dx, dy)
                if current_length <= 1e-6:
                    continue

                corrected_length = _blend(current_length, target_length, correction)
                unit_x = dx / current_length
                unit_y = dy / current_length
                points[child_index] = _make_point(
                    float(parent[0]) + unit_x * corrected_length,
                    float(parent[1]) + unit_y * corrected_length,
                    child[2],
                )
