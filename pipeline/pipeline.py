from __future__ import annotations

import cv2

from config import BODY_EDGES, BODY_KEYPOINTS, HAND_EDGES, WRIST_TO_ELBOW
from utils.body_geometry import derive_foot_points
from utils.hand_constraints import enforce_hand_constraints
from utils.hand_fallback import anchor_hand_to_wrist, generate_default_hand, has_usable_hand_detection, is_hand_detection_valid
from utils.normalize import build_hand_box
from utils.prediction import predict_points, translate_box, translate_points


class PoseHandPipeline:
    def __init__(self, config, runner, smoother, osc_sender):
        self.config = config
        self.runner = runner
        self.smoother = smoother
        self.osc_sender = osc_sender
        self._frame_index = 0
        self._hand_frame_index = 0
        self._previous_body_points = None
        self._last_body_points = None
        self._last_hand_by_side = {}
        self._last_wrist_by_side = {}

    def process_frame(self, frame):
        body_points, hands_by_side = self.detect_pose(frame)
        self.render_pose(frame, body_points, hands_by_side)
        return frame

    def detect_pose(self, frame):
        if self._should_run_body_model():
            body_points = self.smoother.smooth_body(self.runner.detect_body(frame))
        else:
            body_points = predict_points(
                self._last_body_points,
                self._previous_body_points,
                self.config.hold_confidence_decay,
            )
        if body_points is None:
            body_points = [(0, 0, 0.0) for _ in range(17)]
        self._previous_body_points = self._last_body_points
        self._last_body_points = body_points
        hands_by_side = self.detect_hands(frame, body_points)
        self._frame_index += 1
        return body_points, hands_by_side

    def detect_hands(self, frame, body_points):
        frame_height, frame_width = frame.shape[:2]
        hands_by_side = {}
        run_hand_model = self._should_run_hand_model()

        for wrist_idx, elbow_idx in WRIST_TO_ELBOW.items():
            wrist_point = body_points[wrist_idx]
            elbow_point = body_points[elbow_idx]

            if wrist_point[2] <= self.config.body_conf_threshold or elbow_point[2] <= self.config.body_conf_threshold:
                continue

            side = "left" if wrist_idx == 9 else "right"
            box = build_hand_box(
                wrist_point,
                elbow_point,
                frame_width,
                frame_height,
                self.config.hand_box_min_size,
                self.config.hand_box_scale,
                self.config.hand_box_forward_shift,
            )

            raw_hand_points = None
            hand_box = box
            if run_hand_model:
                raw_hand_points = self._detect_best_hand(frame, wrist_point, elbow_point, box)
            else:
                predicted_hand = self._predict_hand(side, wrist_point)
                if predicted_hand is not None:
                    hand_box, raw_hand_points = predicted_hand

            hand_points = self.smoother.smooth_hand(side, raw_hand_points)
            if hand_points is not None:
                hand_points = enforce_hand_constraints(hand_points)
                if not is_hand_detection_valid(hand_points, wrist_point, elbow_point, self.config):
                    hand_points = None

            if hand_points is None:
                hand_points = generate_default_hand(wrist_point, elbow_point, side, self.config)
                hand_points = enforce_hand_constraints(hand_points)

            hands_by_side[side] = {"box": hand_box, "points": hand_points}
            self._last_hand_by_side[side] = hands_by_side[side]
            self._last_wrist_by_side[side] = wrist_point

        self._hand_frame_index += 1
        return hands_by_side

    def _should_run_body_model(self) -> bool:
        return self._last_body_points is None or self._frame_index % self.config.body_detect_interval == 0

    def _should_run_hand_model(self) -> bool:
        return self._hand_frame_index % self.config.hand_detect_interval == 0

    def _predict_hand(self, side, wrist_point):
        previous_payload = self._last_hand_by_side.get(side)
        previous_wrist = self._last_wrist_by_side.get(side)
        if previous_payload is None or previous_wrist is None:
            return None

        offset_x = wrist_point[0] - previous_wrist[0]
        offset_y = wrist_point[1] - previous_wrist[1]
        return (
            translate_box(previous_payload["box"], offset_x, offset_y),
            translate_points(
                previous_payload["points"],
                offset_x,
                offset_y,
                self.config.hold_confidence_decay,
            ),
        )

    def _detect_best_hand(self, frame, wrist_point, elbow_point, primary_box):
        frame_height, frame_width = frame.shape[:2]
        candidate_boxes = self._hand_candidate_boxes(
            wrist_point,
            elbow_point,
            frame_width,
            frame_height,
            primary_box,
        )

        best_points = None
        best_score = -1.0
        for box in candidate_boxes:
            raw_points = self.runner.detect_hand(frame, box)
            if not has_usable_hand_detection(raw_points, self.config):
                continue

            anchored_points = anchor_hand_to_wrist(raw_points, wrist_point)
            constrained_points = enforce_hand_constraints(anchored_points)
            valid_points = sum(point[2] > self.config.hand_kp_threshold * 0.5 for point in constrained_points)
            average_confidence = sum(point[2] for point in constrained_points) / len(constrained_points)
            score = valid_points + average_confidence
            if is_hand_detection_valid(constrained_points, wrist_point, elbow_point, self.config):
                score += 10.0
            if score > best_score:
                best_score = score
                best_points = constrained_points

        return best_points

    def _hand_candidate_boxes(self, wrist_point, elbow_point, frame_width, frame_height, primary_box):
        boxes = [primary_box]
        retry_specs = ((2.4, 0.15), (2.8, 0.35), (3.2, 0.05))
        for scale_multiplier, forward_shift in retry_specs[: self.config.hand_crop_retries]:
            boxes.append(
                build_hand_box(
                    wrist_point,
                    elbow_point,
                    frame_width,
                    frame_height,
                    self.config.hand_box_min_size,
                    self.config.hand_box_scale * scale_multiplier / 2.0,
                    forward_shift,
                )
            )

        unique_boxes = []
        seen = set()
        for box in boxes:
            if box in seen:
                continue
            seen.add(box)
            unique_boxes.append(box)
        return unique_boxes

    def render_pose(self, frame, body_points, hands_by_side, send_osc: bool = True) -> None:
        if body_points is None:
            body_points = [(0, 0, 0.0) for _ in range(17)]
        self._draw_body(frame, body_points)
        self._draw_hands(frame, hands_by_side)
        if send_osc:
            self.osc_sender.send_pose(body_points, hands_by_side)

    def _draw_body(self, frame, body_points) -> None:
        for start_idx, end_idx in BODY_EDGES:
            x1, y1, c1 = body_points[start_idx]
            x2, y2, c2 = body_points[end_idx]
            if c1 > self.config.body_conf_threshold and c2 > self.config.body_conf_threshold:
                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    self.config.body_line_color,
                    self.config.body_line_thickness,
                )

        for idx, (px, py, conf) in enumerate(body_points):
            if idx in BODY_KEYPOINTS and conf > self.config.body_conf_threshold:
                cv2.circle(frame, (px, py), self.config.body_point_radius, self.config.body_point_color, -1)

        self._draw_feet(frame, body_points)

    def _draw_feet(self, frame, body_points) -> None:
        for knee_idx, ankle_idx in ((13, 15), (14, 16)):
            knee_point = body_points[knee_idx]
            ankle_point = body_points[ankle_idx]
            if knee_point[2] <= self.config.body_conf_threshold or ankle_point[2] <= self.config.body_conf_threshold:
                continue

            foot_point, toe_point = derive_foot_points(knee_point, ankle_point)
            ankle_xy = (ankle_point[0], ankle_point[1])
            foot_xy = (foot_point[0], foot_point[1])
            toe_xy = (toe_point[0], toe_point[1])
            cv2.line(frame, ankle_xy, foot_xy, self.config.body_line_color, self.config.body_line_thickness)
            cv2.line(frame, foot_xy, toe_xy, self.config.body_line_color, self.config.body_line_thickness)
            cv2.circle(frame, foot_xy, self.config.body_point_radius, self.config.body_point_color, -1)
            cv2.circle(frame, toe_xy, self.config.body_point_radius, self.config.body_point_color, -1)

    def _draw_hands(self, frame, hands_by_side) -> None:
        for hand_payload in hands_by_side.values():
            x1, y1, x2, y2 = hand_payload["box"]
            hand_points = hand_payload["points"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.hand_box_color, self.config.hand_box_thickness)

            for start_idx, end_idx in HAND_EDGES:
                x1p, y1p, c1 = hand_points[start_idx]
                x2p, y2p, c2 = hand_points[end_idx]
                if c1 > self.config.hand_kp_threshold and c2 > self.config.hand_kp_threshold:
                    cv2.line(
                        frame,
                        (x1p, y1p),
                        (x2p, y2p),
                        self.config.hand_line_color,
                        self.config.hand_line_thickness,
                    )

            for px, py, conf in hand_points:
                if conf > self.config.hand_kp_threshold:
                    cv2.circle(frame, (px, py), self.config.hand_point_radius, self.config.hand_point_color, -1)
