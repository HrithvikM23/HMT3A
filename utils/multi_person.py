from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from network.osc_sender import OSCSender
from pipeline.pipeline import PoseHandPipeline
from utils.color_profile import ColorProfile, color_profile_similarity
from utils.hand_constraints import enforce_hand_constraints
from utils.hand_fallback import generate_default_hand
from utils.payloads import HandPayload
from utils.smoothing import LandmarkSmoother

Point = tuple[int, int, float]
Box = tuple[int, int, int, int]

COLOR_HSV_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "black": ((0, 0, 0), (180, 255, 60)),
    "white": ((0, 0, 180), (180, 50, 255)),
    "gray": ((0, 0, 80), (180, 40, 190)),
    "silver": ((0, 0, 120), (180, 45, 235)),
    "red": ((0, 90, 60), (10, 255, 255)),
    "orange": ((10, 90, 60), (25, 255, 255)),
    "yellow": ((25, 90, 60), (35, 255, 255)),
    "green": ((35, 60, 50), (85, 255, 255)),
    "blue": ((90, 70, 50), (135, 255, 255)),
    "purple": ((135, 70, 50), (165, 255, 255)),
    "pink": ((165, 60, 60), (179, 255, 255)),
    "brown": ((5, 80, 30), (25, 255, 160)),
}
COLOR_NAMES = tuple(COLOR_HSV_RANGES)
COLOR_HSV_BOUNDS = tuple(
    (
        color_name,
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8),
    )
    for color_name, (lower, upper) in COLOR_HSV_RANGES.items()
)


@dataclass(slots=True)
class PersonDetection:
    track_id: int | None
    box: Box
    body_points: list[Point]
    score: float
    color_scores: dict[str, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass(slots=True)
class PersonTrack:
    id: int
    box: Box
    pipeline: PoseHandPipeline
    tracker_id: int | None = None
    missed_frames: int = 0
    label: str | None = None
    body_points: list[Point] = field(default_factory=list)
    hands_by_side: dict[str, HandPayload] = field(default_factory=dict)
    joint_depths: dict[str, float] = field(default_factory=dict)
    velocity: tuple[float, float] = (0.0, 0.0)
    color_signature: ColorProfile = field(default_factory=dict)
    detection_score: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
    area_b = max((bx2 - bx1) * (by2 - by1), 1)
    return inter_area / float(area_a + area_b - inter_area)


def _center_distance_score(box_a: Box, box_b: Box) -> float:
    ax = (box_a[0] + box_a[2]) * 0.5
    ay = (box_a[1] + box_a[3]) * 0.5
    bx = (box_b[0] + box_b[2]) * 0.5
    by = (box_b[1] + box_b[3]) * 0.5
    distance = math.hypot(ax - bx, ay - by)
    scale = max(box_a[2] - box_a[0], box_a[3] - box_a[1], box_b[2] - box_b[0], box_b[3] - box_b[1], 1)
    return max(0.0, 1.0 - distance / (scale * 2.0))


def _size_similarity_score(box_a: Box, box_b: Box) -> float:
    area_a = max((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]), 1)
    area_b = max((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]), 1)
    smaller = min(area_a, area_b)
    larger = max(area_a, area_b)
    return float(smaller) / float(larger)


def _expand_box(box: Box, frame_width: int, frame_height: int, scale: float) -> Box:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    expanded_w = width * scale
    expanded_h = height * scale
    nx1 = max(0, int(round(cx - expanded_w * 0.5)))
    ny1 = max(0, int(round(cy - expanded_h * 0.5)))
    nx2 = min(frame_width, int(round(cx + expanded_w * 0.5)))
    ny2 = min(frame_height, int(round(cy + expanded_h * 0.5)))
    return nx1, ny1, nx2, ny2


def _torso_crop(frame, box: Box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    tx1 = x1 + int(width * 0.2)
    tx2 = x2 - int(width * 0.2)
    ty1 = y1 + int(height * 0.12)
    ty2 = y1 + int(height * 0.60)
    return frame[max(0, ty1):max(0, ty2), max(0, tx1):max(0, tx2)]


def _color_scores(frame, box: Box) -> dict[str, float]:
    crop = _torso_crop(frame, box)
    if crop.size == 0:
        return {}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixel_count = max(hsv.shape[0] * hsv.shape[1], 1)
    scores: dict[str, float] = {}
    for color_name, lower_np, upper_np in COLOR_HSV_BOUNDS:
        mask = cv2.inRange(hsv, lower_np, upper_np)
        scores[color_name] = float(cv2.countNonZero(mask)) / float(pixel_count)
    return scores


def _blend_color_scores(previous: ColorProfile, current: ColorProfile, alpha: float = 0.45) -> ColorProfile:
    if not previous:
        return dict(current)
    blended: ColorProfile = {}
    retain_previous_weight = 1.0 - alpha
    for key in COLOR_NAMES:
        blended_value = previous.get(key, 0.0) * retain_previous_weight + current.get(key, 0.0) * alpha
        if blended_value > 0.0:
            blended[key] = blended_value
    return blended


def _translate_body_points(points: list[Point], offset_x: int, offset_y: int) -> list[Point]:
    return [(x + offset_x, y + offset_y, conf) for x, y, conf in points]


def _smooth_and_constrain_body(track: PersonTrack, body_points: list[Point]) -> list[Point]:
    smoothed_points = track.pipeline.smoother.smooth_body(body_points)
    if not smoothed_points:
        return body_points
    return track.pipeline._body_constraints.apply(smoothed_points)


def _build_inference_frame(frame, processing_width: int):
    frame_height, frame_width = frame.shape[:2]
    if processing_width <= 0 or frame_width <= processing_width:
        return frame, 1.0, 1.0
    target_height = max(1, int(round(frame_height * (processing_width / float(frame_width)))))
    resized = cv2.resize(frame, (processing_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, frame_width / float(processing_width), frame_height / float(target_height)


def _scale_box(box: Box, scale_x: float, scale_y: float) -> Box:
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )


def _scale_body_points(points: list[Point], scale_x: float, scale_y: float) -> list[Point]:
    return [(int(round(x * scale_x)), int(round(y * scale_y)), conf) for x, y, conf in points]


class MultiPersonTracker:
    def __init__(self, config, runner):
        self.config = config
        self.runner = runner
        self._tracks: list[PersonTrack] = []
        self._next_id = 1
        self._frame_index = 0
        self._processing_size_logged = False

    def update(self, frame) -> list[PersonTrack]:
        if self._should_run_person_model():
            detections = self._detect_people(frame)
            self._associate_tracks(frame, detections)
        else:
            self._predict_tracks(frame)
        self._resolve_cross_person_hands()
        self._frame_index += 1
        return [track for track in self._tracks if track.missed_frames == 0 and track.body_points]

    def _should_run_person_model(self) -> bool:
        return not self._tracks or self._frame_index % self.config.body_detect_interval == 0

    def _predict_tracks(self, frame) -> None:
        updated_tracks: list[PersonTrack] = []
        for track in self._tracks:
            if track.missed_frames != 0:
                track.missed_frames += 1
                if track.missed_frames <= self.config.person_track_hold_frames:
                    updated_tracks.append(track)
                continue

            vx, vy = track.velocity
            offset_x = int(round(vx))
            offset_y = int(round(vy))
            x1, y1, x2, y2 = track.box
            track.box = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            track.body_points = _smooth_and_constrain_body(
                track,
                _translate_body_points(track.body_points, offset_x, offset_y),
            )
            track.hands_by_side = track.pipeline.detect_hands(frame, track.body_points)
            track.joint_depths = dict(track.pipeline.last_joint_depths)
            track.detection_score *= self.config.hold_confidence_decay
            updated_tracks.append(track)

        self._tracks = updated_tracks[: self.config.max_people]

    def _detect_people(self, frame) -> list[PersonDetection]:
        inference_frame, output_scale_x, output_scale_y = self._build_inference_frame(frame)
        raw_detections = self.runner.detect_bodies(inference_frame, max_people=self.config.max_people, track=True)
        frame_height, frame_width = frame.shape[:2]
        detections: list[PersonDetection] = []
        for detection in raw_detections:
            scaled_box = _scale_box(detection["box"], output_scale_x, output_scale_y)
            expanded_box = _expand_box(scaled_box, frame_width, frame_height, self.config.person_box_scale)
            detections.append(
                PersonDetection(
                    track_id=detection["id"],
                    box=expanded_box,
                    body_points=_scale_body_points(detection["body_points"], output_scale_x, output_scale_y),
                    score=float(detection["score"]),
                    color_scores=_color_scores(frame, expanded_box),
                )
            )
        return detections

    def _build_inference_frame(self, frame):
        target_width = int(getattr(self.config, "processing_width", 0) or 0)
        inference_frame, output_scale_x, output_scale_y = _build_inference_frame(frame, target_width)
        self._log_processing_size(frame, inference_frame, target_width)
        return inference_frame, output_scale_x, output_scale_y

    def _log_processing_size(self, frame, inference_frame, target_width: int) -> None:
        if self._processing_size_logged or target_width <= 0:
            return
        self._processing_size_logged = True
        frame_height, frame_width = frame.shape[:2]
        inference_height, inference_width = inference_frame.shape[:2]
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

    def _associate_tracks(self, frame, detections: list[PersonDetection]) -> None:
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        # Fast path: preserve stable tracker-id matches without falling back to
        # the more expensive all-pairs scoring pass.
        detection_index_by_tracker_id = {
            detection.track_id: detection_index
            for detection_index, detection in enumerate(detections)
            if detection.track_id is not None
        }
        for track_index, track in enumerate(self._tracks):
            if track.tracker_id is None:
                continue
            detection_index = detection_index_by_tracker_id.get(track.tracker_id)
            if detection_index is None or detection_index in used_detections:
                continue
            assignments[track_index] = detection_index
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        unmatched_track_indices = [
            track_index
            for track_index in range(len(self._tracks))
            if track_index not in used_tracks
        ]
        unmatched_detection_indices = [
            detection_index
            for detection_index in range(len(detections))
            if detection_index not in used_detections
        ]

        scored_pairs: list[tuple[float, int, int]] = []
        for detection_index in unmatched_detection_indices:
            detection = detections[detection_index]
            for track_index in unmatched_track_indices:
                track = self._tracks[track_index]
                score = self._match_score(track, detection)
                if score > self.config.person_match_threshold:
                    scored_pairs.append((score, track_index, detection_index))

        for _, track_index, detection_index in sorted(scored_pairs, reverse=True):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            assignments[track_index] = detection_index
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        updated_tracks: list[PersonTrack] = []
        for track_index, track in enumerate(self._tracks):
            detection_index = assignments.get(track_index)
            if detection_index is None:
                track.missed_frames += 1
                if track.missed_frames <= self.config.person_track_hold_frames:
                    updated_tracks.append(track)
                continue

            detection = detections[detection_index]
            self._update_track_from_detection(track, detection, frame)
            updated_tracks.append(track)

        assigned_labels = {track.label for track in updated_tracks if track.label}
        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            track = self._create_track()
            track.label = self._best_identity_label(detection, assigned_labels)
            self._update_track_from_detection(track, detection, frame)
            updated_tracks.append(track)
            if track.label:
                assigned_labels.add(track.label)

        self._tracks = self._enforce_unique_labels(updated_tracks[: self.config.max_people])

    def _create_track(self) -> PersonTrack:
        track = PersonTrack(
            id=self._next_id,
            box=(0, 0, 0, 0),
            pipeline=PoseHandPipeline(
                self.config,
                self.runner,
                LandmarkSmoother(self.config),
                OSCSender(enabled=False),
            ),
        )
        self._next_id += 1
        return track

    def _match_score(self, track: PersonTrack, detection: PersonDetection) -> float:
        if detection.track_id is not None and detection.track_id == track.tracker_id:
            return 10.0
        predicted_box = self._predict_track_box(track)
        score = (
            _iou(predicted_box, detection.box) * 0.40
            + _center_distance_score(predicted_box, detection.box) * 0.20
            + _size_similarity_score(predicted_box, detection.box) * 0.10
            + color_profile_similarity(track.color_signature, detection.color_scores) * 0.30
        )
        if track.label:
            score += self._identity_score(track.label, detection.color_scores) * 0.5
        return score

    def _best_identity_label(self, detection: PersonDetection, assigned_labels: set[str]) -> str | None:
        best_label: str | None = None
        best_score = 0.0
        for label in self.config.identity_hints:
            if label in assigned_labels:
                continue
            score = self._identity_score(label, detection.color_scores)
            if score > best_score:
                best_score = score
                best_label = label
        if best_score < self.config.identity_min_score:
            return None
        return best_label

    def _identity_score(self, label: str, color_scores: dict[str, float]) -> float:
        colors = self.config.identity_hints.get(label, ())
        if not colors:
            return 0.0
        return sum(color_scores.get(color, 0.0) for color in colors) / float(len(colors))

    def _update_track_from_detection(self, track: PersonTrack, detection: PersonDetection, frame) -> None:
        previous_center = track.center
        body_points = _smooth_and_constrain_body(track, detection.body_points)
        hands_by_side = track.pipeline.detect_hands(frame, body_points)
        current_center = ((detection.box[0] + detection.box[2]) * 0.5, (detection.box[1] + detection.box[3]) * 0.5)
        track.velocity = (
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )
        track.box = detection.box
        track.tracker_id = detection.track_id
        track.missed_frames = 0
        track.body_points = body_points
        track.hands_by_side = hands_by_side
        track.joint_depths = dict(track.pipeline.last_joint_depths)
        track.color_signature = _blend_color_scores(track.color_signature, detection.color_scores)
        track.detection_score = detection.score
        self._refresh_track_label(track, detection)

    def _enforce_unique_labels(self, tracks: list[PersonTrack]) -> list[PersonTrack]:
        label_groups: dict[str, list[PersonTrack]] = {}
        for track in tracks:
            if track.label:
                label_groups.setdefault(track.label, []).append(track)

        for label, grouped_tracks in label_groups.items():
            if len(grouped_tracks) <= 1:
                continue
            grouped_tracks.sort(
                key=lambda track: (
                    self._identity_score(label, track.color_signature),
                    track.detection_score,
                ),
                reverse=True,
            )
            for duplicate_track in grouped_tracks[1:]:
                duplicate_track.label = None
        return tracks

    def _predict_track_box(self, track: PersonTrack) -> Box:
        vx, vy = track.velocity
        x1, y1, x2, y2 = track.box
        return (
            int(round(x1 + vx)),
            int(round(y1 + vy)),
            int(round(x2 + vx)),
            int(round(y2 + vy)),
        )

    def _refresh_track_label(self, track: PersonTrack, detection: PersonDetection) -> None:
        if not self.config.identity_hints:
            return
        candidate_scores = {
            label: self._identity_score(label, detection.color_scores)
            for label in self.config.identity_hints
        }
        best_label = max(candidate_scores, key=lambda label: candidate_scores[label], default=None)
        if best_label is None:
            return
        best_score = candidate_scores[best_label]
        if best_score < self.config.identity_min_score:
            return
        if track.label is None:
            track.label = best_label
            return
        current_score = candidate_scores.get(track.label, 0.0)
        if best_label != track.label and best_score > current_score + 0.05:
            track.label = best_label

    def _hand_owner_score(self, track: PersonTrack, side: str, hand_payload: dict) -> float:
        side_indices = {
            "left": (9, 7),
            "right": (10, 8),
        }
        wrist_index, elbow_index = side_indices[side]
        if len(track.body_points) <= wrist_index:
            return -1.0

        wrist = track.body_points[wrist_index]
        elbow = track.body_points[elbow_index]
        if wrist[2] <= self.config.body_conf_threshold or elbow[2] <= self.config.body_conf_threshold:
            return -1.0

        hand_wrist = hand_payload["points"][0]
        hx = (hand_payload["box"][0] + hand_payload["box"][2]) * 0.5
        hy = (hand_payload["box"][1] + hand_payload["box"][3]) * 0.5
        wrist_distance = math.hypot(hand_wrist[0] - wrist[0], hand_wrist[1] - wrist[1])
        elbow_distance = math.hypot(hand_wrist[0] - elbow[0], hand_wrist[1] - elbow[1])
        body_scale = max(track.box[2] - track.box[0], track.box[3] - track.box[1], 1)
        inside_box = 1.0 if track.box[0] <= hx <= track.box[2] and track.box[1] <= hy <= track.box[3] else 0.0
        return (
            max(0.0, 1.0 - wrist_distance / float(body_scale)) * 0.55
            + max(0.0, 1.0 - elbow_distance / float(body_scale * 1.35)) * 0.20
            + _iou(track.box, hand_payload["box"]) * 0.15
            + inside_box * 0.10
        )

    def _resolve_cross_person_hands(self) -> None:
        active_tracks = [track for track in self._tracks if track.missed_frames == 0 and len(track.body_points) >= 11]
        for track in active_tracks:
            for side, hand_payload in list(track.hands_by_side.items()):
                wrist_index, elbow_index = {"left": (9, 7), "right": (10, 8)}[side]
                owner_score = self._hand_owner_score(track, side, hand_payload)
                if owner_score < 0:
                    continue

                closest_track = track
                best_score = owner_score

                for other_track in active_tracks:
                    if other_track.id == track.id or len(other_track.body_points) <= wrist_index:
                        continue
                    other_score = self._hand_owner_score(other_track, side, hand_payload)
                    if other_score > best_score:
                        best_score = other_score
                        closest_track = other_track

                if closest_track.id == track.id:
                    continue

                if best_score <= owner_score / max(self.config.person_cross_wrist_ratio, 1e-6):
                    continue

                owner_wrist = track.body_points[wrist_index]
                owner_elbow = track.body_points[elbow_index]
                fallback_hand = generate_default_hand(owner_wrist, owner_elbow, side, self.config)
                track.hands_by_side[side] = {
                    "box": hand_payload["box"],
                    "points": enforce_hand_constraints(fallback_hand),
                }
