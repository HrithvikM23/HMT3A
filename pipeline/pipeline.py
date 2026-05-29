from __future__ import annotations

import cv2

from core.config import BODY_EDGES, BODY_KEYPOINTS, HAND_EDGES, WRIST_TO_ELBOW
from utils.body_constraints import BodyKinematicConstraints
from utils.body_geometry import derive_foot_points
from utils.hand_constraints import enforce_hand_constraints
from utils.hand_fallback import anchor_hand_to_wrist, generate_default_hand, has_usable_hand_detection, is_hand_detection_valid
from utils.hand_tracking import blend_with_prediction, hand_detection_score, predict_hand_payload
from utils.normalize import build_hand_box
from utils.payloads import HandPayload
from utils.prediction import predict_points
from utils.skeleton import HAND_NAME_TO_INDEX


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
        self._last_elbow_by_side = {}
        self._body_constraints = BodyKinematicConstraints(config)
        self.last_joint_depths: dict[str, float] = {}
        self._processing_size_logged = False

    def process_frame(self, frame):
        body_points, hands_by_side = self.detect_pose(frame)
        self.render_pose(frame, body_points, hands_by_side)
        return frame

    def detect_pose(self, frame):
        inference_frame, output_scale_x, output_scale_y = self._build_inference_frame(frame)
        if self._should_run_body_model():
            raw_body_points = self.runner.detect_body(inference_frame)
            body_points = self.smoother.smooth_body(self._scale_points(raw_body_points, output_scale_x, output_scale_y))
            detected_depths = self._scale_depths(
                getattr(self.runner, "last_body_depths", {}) or {},
                (output_scale_x + output_scale_y) * 0.5,
            )
        else:
            body_points = predict_points(
                self._last_body_points,
                self._previous_body_points,
                self.config.hold_confidence_decay,
            )
            detected_depths = dict(self.last_joint_depths)
        if body_points is None:
            body_points = [(0, 0, 0.0) for _ in range(17)]
        body_points = self._body_constraints.apply(body_points)
        self._previous_body_points = self._last_body_points
        self._last_body_points = body_points
        self.last_joint_depths = detected_depths
        hands_by_side = self.detect_hands(
            frame,
            body_points,
            output_scale=(1.0, 1.0),
        )
        self._frame_index += 1
        return body_points, hands_by_side

    def _build_inference_frame(self, frame):
        target_width = int(getattr(self.config, "processing_width", 0) or 0)
        frame_height, frame_width = frame.shape[:2]
        if target_width <= 0 or frame_width <= target_width:
            self._log_processing_size(frame_width, frame_height, frame_width, frame_height, target_width)
            return frame, 1.0, 1.0
        target_height = max(1, int(round(frame_height * (target_width / float(frame_width)))))
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        self._log_processing_size(frame_width, frame_height, target_width, target_height, target_width)
        return resized, frame_width / float(target_width), frame_height / float(target_height)

    def _log_processing_size(
        self,
        frame_width: int,
        frame_height: int,
        inference_width: int,
        inference_height: int,
        target_width: int,
    ) -> None:
        if self._processing_size_logged or target_width <= 0:
            return
        self._processing_size_logged = True
        if inference_width == frame_width and inference_height == frame_height:
            print(
                f"[processing] source {frame_width}x{frame_height}; "
                f"--processing-width {target_width} does not downscale this source"
            )
            return
        scale_x = frame_width / float(inference_width)
        scale_y = frame_height / float(inference_height)
        print(
            f"[processing] source {frame_width}x{frame_height} -> inference {inference_width}x{inference_height} "
            f"(scale {scale_x:.2f}x, {scale_y:.2f}x)"
        )

    @staticmethod
    def _scale_point(point, scale_x: float, scale_y: float):
        return (int(round(point[0] * scale_x)), int(round(point[1] * scale_y)), point[2])

    @classmethod
    def _scale_points(cls, points, scale_x: float, scale_y: float):
        if points is None:
            return None
        return [cls._scale_point(point, scale_x, scale_y) for point in points]

    @staticmethod
    def _scale_box(box, scale_x: float, scale_y: float):
        x1, y1, x2, y2 = box
        return (
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        )

    @staticmethod
    def _scale_depths(depths, scale: float):
        return {name: float(depth) * scale for name, depth in dict(depths).items()}

    def detect_hands(self, frame, body_points, output_scale=(1.0, 1.0)) -> dict[str, HandPayload]:
        frame_height, frame_width = frame.shape[:2]
        hands_by_side: dict[str, HandPayload] = {}
        run_hand_model = self._should_run_hand_model()
        output_scale_x, output_scale_y = output_scale

        for wrist_idx, elbow_idx in WRIST_TO_ELBOW.items():
            wrist_point = body_points[wrist_idx]
            elbow_point = body_points[elbow_idx]
            wrist_output = self._scale_point(wrist_point, output_scale_x, output_scale_y)
            elbow_output = self._scale_point(elbow_point, output_scale_x, output_scale_y)

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
            raw_hand_depths = None
            raw_hand_in_inference_space = True
            hand_box = box
            predicted_hand = self._predict_hand(side, wrist_output, elbow_output)
            if run_hand_model:
                detected_hand = self._detect_best_hand(frame, side, wrist_point, elbow_point, box, output_scale)
                if detected_hand is not None:
                    hand_box, raw_hand_points, raw_hand_depths = detected_hand
                if raw_hand_points is None:
                    if predicted_hand is not None:
                        hand_box, raw_hand_points, raw_hand_depths = predicted_hand
                        raw_hand_in_inference_space = False
            else:
                if predicted_hand is not None:
                    hand_box, raw_hand_points, raw_hand_depths = predicted_hand
                    raw_hand_in_inference_space = False

            if raw_hand_in_inference_space and raw_hand_points is not None and (output_scale_x != 1.0 or output_scale_y != 1.0):
                raw_hand_points = self._scale_points(raw_hand_points, output_scale_x, output_scale_y)
                hand_box = self._scale_box(hand_box, output_scale_x, output_scale_y)
                if raw_hand_depths is not None:
                    raw_hand_depths = [float(depth) * ((output_scale_x + output_scale_y) * 0.5) for depth in raw_hand_depths]

            if raw_hand_points is not None and raw_hand_in_inference_space and predicted_hand is not None:
                raw_hand_points = blend_with_prediction(raw_hand_points, predicted_hand[1], self.config)

            hand_points = self.smoother.smooth_hand(side, raw_hand_points)
            hand_depths = raw_hand_depths
            if hand_points is not None:
                hand_points = enforce_hand_constraints(hand_points)
                if not is_hand_detection_valid(hand_points, wrist_output, elbow_output, self.config):
                    hand_points = None
                    hand_depths = None

            if hand_points is None:
                hand_box = self._scale_box(box, output_scale_x, output_scale_y)
                hand_points = generate_default_hand(wrist_output, elbow_output, side, self.config)
                hand_points = enforce_hand_constraints(hand_points)
                hand_depths = None

            payload: HandPayload = {"box": hand_box, "points": hand_points}
            if hand_depths is not None and len(hand_depths) == len(hand_points):
                payload["depths"] = hand_depths
                self._store_hand_depths(side, hand_depths)
            hands_by_side[side] = payload
            self._last_hand_by_side[side] = hands_by_side[side]
            self._last_wrist_by_side[side] = wrist_output
            self._last_elbow_by_side[side] = elbow_output

        self._hand_frame_index += 1
        return hands_by_side

    def _store_hand_depths(self, side, hand_depths) -> None:
        side_label = "Left" if side == "left" else "Right"
        wrist_depth = self.last_joint_depths.get(f"{side_label}Wrist", 0.0)
        base_depth = float(hand_depths[0]) if hand_depths else 0.0
        for suffix, index in HAND_NAME_TO_INDEX.items():
            if index >= len(hand_depths):
                continue
            self.last_joint_depths[f"{side_label}{suffix}"] = wrist_depth + float(hand_depths[index]) - base_depth

    def _should_run_body_model(self) -> bool:
        return self._last_body_points is None or self._frame_index % self.config.body_detect_interval == 0

    def _should_run_hand_model(self) -> bool:
        return self._hand_frame_index % self.config.hand_detect_interval == 0

    def _predict_hand(self, side, wrist_point, elbow_point=None):
        return predict_hand_payload(
            self._last_hand_by_side.get(side),
            self._last_wrist_by_side.get(side),
            wrist_point,
            self.config.hold_confidence_decay,
            self._last_elbow_by_side.get(side),
            elbow_point,
        )

    def _detect_best_hand(self, frame, side, wrist_point, elbow_point, primary_box, output_scale=(1.0, 1.0)):
        frame_height, frame_width = frame.shape[:2]
        candidate_boxes = self._hand_candidate_boxes(
            wrist_point,
            elbow_point,
            frame_width,
            frame_height,
            primary_box,
            output_scale,
            side=side,
        )

        best_points = None
        best_depths = None
        best_box = None
        best_score = -1.0
        previous_points = self._last_hand_points_in_inference_space(side, output_scale)
        for box in candidate_boxes:
            raw_points = self.runner.detect_hand(frame, box)
            raw_depths = getattr(self.runner, "last_hand_depths", None)
            if not has_usable_hand_detection(raw_points, self.config):
                continue

            anchored_points = anchor_hand_to_wrist(raw_points, wrist_point)
            constrained_points = enforce_hand_constraints(anchored_points)
            score = hand_detection_score(constrained_points, wrist_point, elbow_point, self.config, previous_points)
            if score > best_score:
                best_score = score
                best_box = box
                best_points = constrained_points
                best_depths = raw_depths

        if best_points is None:
            return None
        return best_box or primary_box, best_points, best_depths

    def _hand_candidate_boxes(self, wrist_point, elbow_point, frame_width, frame_height, primary_box, output_scale=(1.0, 1.0), side=None):
        if self.config.hand_backend == "mediapipe" and not self.config.enable_backend_fallbacks:
            return [primary_box]

        boxes = [primary_box]
        previous_box = None if side is None else self._last_hand_box_in_inference_space(side, output_scale)
        if previous_box is not None:
            boxes.append(self._clamp_box(previous_box, frame_width, frame_height))
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

    def _last_hand_points_in_inference_space(self, side, output_scale):
        previous_payload = self._last_hand_by_side.get(side)
        if previous_payload is None:
            return None
        scale_x, scale_y = output_scale
        return self._scale_points(previous_payload["points"], 1.0 / scale_x, 1.0 / scale_y)

    def _last_hand_box_in_inference_space(self, side, output_scale):
        previous_payload = self._last_hand_by_side.get(side)
        previous_wrist = self._last_wrist_by_side.get(side)
        if previous_payload is None or previous_wrist is None:
            return None

        scale_x, scale_y = output_scale
        wrist_point = (
            int(round(previous_wrist[0] / scale_x)),
            int(round(previous_wrist[1] / scale_y)),
            previous_wrist[2],
        )
        predicted = self._predict_hand(side, previous_wrist, self._last_elbow_by_side.get(side))
        box = previous_payload["box"] if predicted is None else predicted[0]
        inference_box = self._scale_box(box, 1.0 / scale_x, 1.0 / scale_y)
        x1, y1, x2, y2 = inference_box
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        delta_x = wrist_point[0] - ((x1 + x2) * 0.5)
        delta_y = wrist_point[1] - ((y1 + y2) * 0.5)
        return (
            int(round(center_x + delta_x - (x2 - x1) * 0.5)),
            int(round(center_y + delta_y - (y2 - y1) * 0.5)),
            int(round(center_x + delta_x + (x2 - x1) * 0.5)),
            int(round(center_y + delta_y + (y2 - y1) * 0.5)),
        )

    @staticmethod
    def _clamp_box(box, frame_width, frame_height):
        x1, y1, x2, y2 = box
        return (
            max(0, min(frame_width, x1)),
            max(0, min(frame_height, y1)),
            max(0, min(frame_width, x2)),
            max(0, min(frame_height, y2)),
        )

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
        for knee_idx, ankle_idx, foot_idx, toe_idx in ((13, 15, 17, 19), (14, 16, 18, 20)):
            knee_point = body_points[knee_idx]
            ankle_point = body_points[ankle_idx]
            if knee_point[2] <= self.config.body_conf_threshold or ankle_point[2] <= self.config.body_conf_threshold:
                continue

            if len(body_points) > toe_idx and body_points[foot_idx][2] > self.config.body_conf_threshold and body_points[toe_idx][2] > self.config.body_conf_threshold:
                foot_point, toe_point = body_points[foot_idx], body_points[toe_idx]
            else:
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
