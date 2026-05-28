from __future__ import annotations

Point = tuple[int, int, float]


class LandmarkSmoother:
    def __init__(self, config):
        self.config = config
        self._body_state: list[Point] | None = None
        self._body_missing_counts: list[int] = []
        self._hand_state: dict[str, list[Point]] = {}
        self._hand_missing_counts: dict[str, list[int]] = {}

    def smooth_body(self, points):
        smoothed = self._smooth_points(
            points=points,
            previous_points=self._body_state,
            missing_counts=self._body_missing_counts,
            threshold=self.config.body_conf_threshold,
            alpha=self.config.body_smoothing_alpha,
            hold_frames=self.config.body_hold_frames,
        )
        self._body_state = None if smoothed is None else smoothed
        if smoothed is None:
            self._body_missing_counts = []
        elif not self._body_missing_counts:
            self._body_missing_counts = [0] * len(smoothed)
        return smoothed

    def smooth_hand(self, side, points):
        smoothed = self._smooth_points(
            points=points,
            previous_points=self._hand_state.get(side),
            missing_counts=self._hand_missing_counts.get(side, []),
            threshold=self.config.hand_kp_threshold,
            alpha=self.config.hand_smoothing_alpha,
            hold_frames=self.config.hand_hold_frames,
        )
        if smoothed is None:
            self._hand_state.pop(side, None)
            self._hand_missing_counts.pop(side, None)
            return None

        self._hand_state[side] = smoothed
        if side not in self._hand_missing_counts:
            self._hand_missing_counts[side] = [0] * len(smoothed)
        return smoothed

    def _smooth_points(
        self,
        points: list[Point] | None,
        previous_points: list[Point] | None,
        missing_counts: list[int],
        threshold: float,
        alpha: float,
        hold_frames: int,
    ) -> list[Point] | None:
        if points is None and previous_points is None:
            return None

        if points is None:
            assert previous_points is not None
            points = [(0, 0, 0.0) for _ in previous_points]

        if previous_points is None or len(previous_points) != len(points):
            missing_counts[:] = [0] * len(points)
        elif len(missing_counts) != len(points):
            missing_counts[:] = [0] * len(points)

        smoothed_points: list[Point] = []
        valid_points_found = False

        for index, point in enumerate(points):
            px, py, conf = point
            prev_point = None
            if previous_points is not None and index < len(previous_points):
                prev_point = previous_points[index]

            if conf > threshold:
                if prev_point is not None:
                    prev_x, prev_y, _ = prev_point
                    px = int(round(alpha * px + (1.0 - alpha) * prev_x))
                    py = int(round(alpha * py + (1.0 - alpha) * prev_y))
                smoothed_point = (px, py, float(conf))
                missing_counts[index] = 0
                smoothed_points.append(smoothed_point)
                valid_points_found = True
                continue

            if prev_point is not None and missing_counts[index] < hold_frames:
                held_x, held_y, held_conf = prev_point
                held_conf *= self.config.hold_confidence_decay
                smoothed_point = (held_x, held_y, float(held_conf))
                missing_counts[index] += 1
                smoothed_points.append(smoothed_point)
                valid_points_found = True
                continue

            missing_counts[index] = hold_frames
            smoothed_points.append((int(px), int(py), float(conf)))

        if not valid_points_found:
            return None

        return smoothed_points
