from __future__ import annotations

import cv2
import math
import numpy as np
import shutil
from pathlib import Path
from typing import Any, TypedDict, cast

from backend_selection import needs_mediapipe, needs_onnx_hand, needs_yolo_body
from mediapipe_models import (
    DEFAULT_MEDIAPIPE_POSE_MODEL,
    mediapipe_pose_model_complexity,
)
from utils.skeleton import BODY_FOOT_NAME_TO_INDEX, BODY_NAME_TO_INDEX

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


def _cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


class BodyDetection(TypedDict):
    id: int | None
    score: float
    box: tuple[int, int, int, int]
    body_points: list[tuple[int, int, float]]


def _normalize_provider_name(value: str) -> str:
    # Some Windows shells can inject control characters into argv values.
    return "".join(character for character in value.strip() if character.isprintable())


def _resolve_provider_names(config) -> list[str]:
    if ort is None:
        return []
    available_providers = {
        _normalize_provider_name(provider_name): provider_name
        for provider_name in ort.get_available_providers()
    }

    requested_providers: list[str] = []
    for provider_name in config.provider_names:
        normalized_name = _normalize_provider_name(str(provider_name))
        if not normalized_name:
            continue
        resolved_name = available_providers.get(normalized_name)
        if resolved_name is not None and resolved_name not in requested_providers:
            requested_providers.append(resolved_name)

    if requested_providers:
        return requested_providers

    if "CPUExecutionProvider" in available_providers:
        return [available_providers["CPUExecutionProvider"]]

    return list(available_providers.values())


def _mediapipe_pose_asset_exists(mp: Any, model_name: str) -> bool:
    package_root = Path(mp.__file__).resolve().parent
    model_path = package_root / "modules" / "pose_landmark" / model_name
    return model_path.exists()


def _sync_mediapipe_pose_asset(mp: Any, model_name: str, source_path: Path | None) -> None:
    if _mediapipe_pose_asset_exists(mp, model_name):
        return
    if source_path is None or not source_path.exists():
        return
    package_root = Path(mp.__file__).resolve().parent
    destination = package_root / "modules" / "pose_landmark" / model_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)


class ONNXPoseHandRunner:
    def __init__(self, config):
        self.config = config
        self.body_model = None
        self.hand_session = None
        self.last_body_depths: dict[str, float] = {}
        self.last_hand_depths: list[float] | None = None
        self._uses_yolo_body = needs_yolo_body(config.body_backend, config.enable_backend_fallbacks)
        self._uses_onnx_hand = needs_onnx_hand(config.hand_backend, config.enable_backend_fallbacks)
        self._uses_mediapipe = needs_mediapipe(config.body_backend, config.hand_backend, config.enable_backend_fallbacks)

        if self._uses_yolo_body:
            if YOLO is None:
                raise ModuleNotFoundError("ultralytics is not installed. Install `ultralytics` or use --body-backend mediapipe.")
            self.body_model = YOLO(str(config.body_model_path))

        if self._uses_onnx_hand:
            if ort is None:
                raise ModuleNotFoundError("onnxruntime is not installed. Install ONNX Runtime or use --hand-backend mediapipe.")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.hand_session = ort.InferenceSession(
                str(config.hand_model_path),
                sess_options=session_options,
                providers=_resolve_provider_names(config),
            )
        device_name = "" if config.yolo_device is None else str(config.yolo_device).lower()
        self._use_yolo_half = bool(config.yolo_half and device_name != "cpu" and _cuda_available())
        self._mp_pose = None
        self._mp_hands = None
        self._mp_available = False
        if self._uses_mediapipe:
            self._setup_mediapipe()

    def _setup_mediapipe(self) -> None:
        try:
            import mediapipe as mp
        except ModuleNotFoundError:
            if self.config.body_backend == "mediapipe" or self.config.hand_backend == "mediapipe":
                print("Warning: MediaPipe is not installed; selected MediaPipe backend will return empty detections.")
            return

        self._mp_available = True
        if self.config.body_backend == "mediapipe" or self.config.enable_backend_fallbacks:
            self._mp_pose = self._create_mediapipe_pose(mp)
        if self.config.hand_backend == "mediapipe" or self.config.enable_backend_fallbacks:
            mp_solutions = cast(Any, mp.solutions)
            self._mp_hands = mp_solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=self.config.hand_det_threshold,
                min_tracking_confidence=self.config.hand_det_threshold,
            )

    def _create_mediapipe_pose(self, mp: Any):
        requested_model = self.config.mediapipe_pose_model
        selected_model = requested_model
        model_path = self.config.mediapipe_pose_model_path
        _sync_mediapipe_pose_asset(mp, selected_model, model_path)
        if selected_model != DEFAULT_MEDIAPIPE_POSE_MODEL and not _mediapipe_pose_asset_exists(mp, selected_model):
            print(
                "Warning: MediaPipe pose model "
                f"{selected_model} is not installed in this Python environment; "
                f"using {DEFAULT_MEDIAPIPE_POSE_MODEL}."
            )
            selected_model = DEFAULT_MEDIAPIPE_POSE_MODEL
            self.config.mediapipe_pose_model = selected_model

        try:
            return mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=mediapipe_pose_model_complexity(selected_model),
                smooth_landmarks=False,
                enable_segmentation=False,
                min_detection_confidence=self.config.body_conf_threshold,
                min_tracking_confidence=self.config.body_conf_threshold,
            )
        except Exception as exc:
            if selected_model == DEFAULT_MEDIAPIPE_POSE_MODEL:
                raise
            print(
                "Warning: MediaPipe pose model "
                f"{selected_model} failed to initialize ({exc}); "
                f"using {DEFAULT_MEDIAPIPE_POSE_MODEL}."
            )
            self.config.mediapipe_pose_model = DEFAULT_MEDIAPIPE_POSE_MODEL
            return mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=mediapipe_pose_model_complexity(DEFAULT_MEDIAPIPE_POSE_MODEL),
                smooth_landmarks=False,
                enable_segmentation=False,
                min_detection_confidence=self.config.body_conf_threshold,
                min_tracking_confidence=self.config.body_conf_threshold,
            )

    def detect_body(self, frame_bgr):
        self.last_body_depths = {}
        if self.config.body_backend == "mediapipe":
            body_points = self._detect_body_mediapipe(frame_bgr)
            if body_points is not None:
                return body_points
            if not self.config.enable_backend_fallbacks:
                return [(0, 0, 0.0) for _ in range(21)]

        detections = self.detect_bodies(frame_bgr, max_people=1, track=False)
        if not detections:
            return [(0, 0, 0.0) for _ in range(17)]
        return cast(list[tuple[int, int, float]], detections[0]["body_points"])

    def _detect_body_mediapipe(self, frame_bgr):
        if not self._mp_available or self._mp_pose is None:
            return None

        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._mp_pose.process(frame_rgb)
        landmarks = None if result.pose_landmarks is None else result.pose_landmarks.landmark
        if not landmarks:
            return None

        mapping = (
            0,
            2,
            5,
            7,
            8,
            11,
            12,
            13,
            14,
            15,
            16,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        body_points = []
        for mp_index in mapping:
            landmark = landmarks[mp_index]
            px = int(round(float(landmark.x) * frame_width))
            py = int(round(float(landmark.y) * frame_height))
            confidence = float(getattr(landmark, "visibility", 1.0))
            body_points.append((px, py, confidence))
        world_landmarks = None if result.pose_world_landmarks is None else result.pose_world_landmarks.landmark
        self.last_body_depths = self._mediapipe_body_depths(
            image_landmarks=landmarks,
            world_landmarks=world_landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return body_points

    @staticmethod
    def _distance2(first: Any, second: Any, width: int, height: int) -> float:
        return math.hypot(
            (float(first.x) - float(second.x)) * float(width),
            (float(first.y) - float(second.y)) * float(height),
        )

    @staticmethod
    def _distance3(first: Any, second: Any) -> float:
        dx = float(first.x) - float(second.x)
        dy = float(first.y) - float(second.y)
        dz = float(first.z) - float(second.z)
        return math.sqrt((dx * dx) + (dy * dy) + (dz * dz))

    def _mediapipe_body_depths(
        self,
        image_landmarks: Any,
        world_landmarks: Any,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, float]:
        if not world_landmarks:
            return {}

        name_to_mp_index = {
            "LeftShoulder": 11,
            "RightShoulder": 12,
            "LeftElbow": 13,
            "RightElbow": 14,
            "LeftWrist": 15,
            "RightWrist": 16,
            "LeftHip": 23,
            "RightHip": 24,
            "LeftKnee": 25,
            "RightKnee": 26,
            "LeftAnkle": 27,
            "RightAnkle": 28,
            "LeftFoot": 29,
            "RightFoot": 30,
            "LeftToeBase": 31,
            "RightToeBase": 32,
        }
        scale_candidates = []
        pixel_candidates = []
        for first_index, second_index in ((11, 12), (23, 24), (11, 23), (12, 24)):
            pixel_distance = self._distance2(image_landmarks[first_index], image_landmarks[second_index], frame_width, frame_height)
            world_distance = self._distance3(world_landmarks[first_index], world_landmarks[second_index])
            if pixel_distance > 1e-6 and world_distance > 1e-6:
                scale_candidates.append(pixel_distance / world_distance)
                pixel_candidates.append(pixel_distance)
        if not scale_candidates:
            return {}

        world_to_pixel_scale = float(np.median(np.asarray(scale_candidates, dtype=np.float64)))
        pixel_reference = max(float(np.median(np.asarray(pixel_candidates, dtype=np.float64))), 48.0)
        max_depth = max(pixel_reference * 0.9, 64.0)
        root_z = (float(world_landmarks[23].z) + float(world_landmarks[24].z)) * 0.5

        depths = {
            name: float(np.clip(-(float(world_landmarks[mp_index].z) - root_z) * world_to_pixel_scale, -max_depth, max_depth))
            for name, mp_index in name_to_mp_index.items()
            if name in BODY_NAME_TO_INDEX or name in BODY_FOOT_NAME_TO_INDEX
        }
        if "LeftHip" in depths and "RightHip" in depths:
            depths["HipsRoot"] = (depths["LeftHip"] + depths["RightHip"]) * 0.5
        if "LeftShoulder" in depths and "RightShoulder" in depths:
            chest_depth = (depths["LeftShoulder"] + depths["RightShoulder"]) * 0.5
            depths["Chest"] = chest_depth
            depths["Neck"] = chest_depth
            depths["Head"] = chest_depth
        return depths

    @staticmethod
    def _to_numpy(value: Any, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
        if value is None:
            return np.empty(shape, dtype=dtype)
        tensor_like = cast(Any, value)
        if hasattr(tensor_like, "cpu"):
            tensor_like = tensor_like.cpu()
        if hasattr(tensor_like, "numpy"):
            tensor_like = tensor_like.numpy()
        return np.asarray(tensor_like, dtype=dtype)

    def detect_bodies(self, frame_bgr, max_people: int, track: bool):
        if track:
            assert self.body_model is not None
            results = self.body_model.track(
                frame_bgr,
                conf=self.config.body_conf_threshold,
                iou=self.config.body_iou_threshold,
                imgsz=self.config.body_input_size,
                max_det=max_people,
                persist=True,
                verbose=False,
                tracker=self.config.yolo_tracker,
                device=self.config.yolo_device,
                half=self._use_yolo_half,
            )
        else:
            assert self.body_model is not None
            results = self.body_model.predict(
                frame_bgr,
                conf=self.config.body_conf_threshold,
                iou=self.config.body_iou_threshold,
                imgsz=self.config.body_input_size,
                max_det=max_people,
                verbose=False,
                device=self.config.yolo_device,
                half=self._use_yolo_half,
            )

        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return []

        boxes_xyxy = self._to_numpy(result.boxes.xyxy, (0, 4), np.dtype(np.float32))
        boxes_conf = self._to_numpy(result.boxes.conf, (0,), np.dtype(np.float32))
        boxes_id = None if result.boxes.id is None else self._to_numpy(result.boxes.id, (0,), np.dtype(np.float32))
        keypoints_data = self._to_numpy(result.keypoints.data, (0, 17, 3), np.dtype(np.float32))

        detections: list[BodyDetection] = []
        for index in range(min(len(boxes_xyxy), len(keypoints_data))):
            box = boxes_xyxy[index]
            keypoints = keypoints_data[index]
            detection_id = None if boxes_id is None else int(float(boxes_id[index]))
            detection_score = float(boxes_conf[index])
            body_points: list[tuple[int, int, float]] = []
            for point in keypoints:
                point_x = float(point[0])
                point_y = float(point[1])
                point_conf = float(point[2])
                body_points.append((int(round(point_x)), int(round(point_y)), point_conf))
            detections.append(
                {
                    "id": detection_id,
                    "score": detection_score,
                    "box": (
                        int(round(float(box[0]))),
                        int(round(float(box[1]))),
                        int(round(float(box[2]))),
                        int(round(float(box[3]))),
                    ),
                    "body_points": body_points,
                }
            )
        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections[:max_people]

    def detect_hand(self, frame_bgr, box):
        self.last_hand_depths = None
        if self.config.hand_backend == "mediapipe":
            hand_points = self._detect_hand_mediapipe(frame_bgr, box)
            if hand_points is not None or not self.config.enable_backend_fallbacks:
                return hand_points

        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            crop_rgb,
            (self.config.hand_input_size, self.config.hand_input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        hand_input = resized.astype(np.float32) / 255.0
        hand_input = np.transpose(hand_input, (2, 0, 1))
        hand_input = np.expand_dims(hand_input, axis=0)

        hand_session = self.hand_session
        if hand_session is None:
            return None
        outputs = hand_session.run(None, {self.config.hand_input_name: hand_input})
        detections = np.asarray(outputs[0], dtype=np.float32)[0]

        best = detections[np.argmax(detections[:, 4])]
        if float(best[4]) <= self.config.hand_det_threshold:
            return None

        crop_w = x2 - x1
        crop_h = y2 - y1
        points = []
        for i in range(21):
            base = 6 + i * 3
            x = float(best[base])
            y = float(best[base + 1])
            conf = float(best[base + 2])

            px = x1 + int((x / self.config.hand_input_size) * crop_w)
            py = y1 + int((y / self.config.hand_input_size) * crop_h)
            points.append((px, py, conf))

        return points

    def _detect_hand_mediapipe(self, frame_bgr, box):
        if not self._mp_available or self._mp_hands is None:
            return None

        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_height, crop_width = crop.shape[:2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_rgb.flags.writeable = False
        result = self._mp_hands.process(crop_rgb)
        landmarks = None if not result.multi_hand_landmarks else result.multi_hand_landmarks[0].landmark
        if not landmarks:
            return None
        world_landmarks = None if not result.multi_hand_world_landmarks else result.multi_hand_world_landmarks[0].landmark

        points = []
        for landmark in landmarks:
            px = x1 + int(round(float(landmark.x) * crop_width))
            py = y1 + int(round(float(landmark.y) * crop_height))
            relative_z = float(getattr(landmark, "z", 0.0))
            confidence = 1.0 - min(abs(relative_z), 0.85)
            points.append((px, py, confidence))
        self.last_hand_depths = self._mediapipe_hand_depths(landmarks, world_landmarks, crop_width, crop_height)
        return points

    def _mediapipe_hand_depths(self, image_landmarks: Any, world_landmarks: Any, crop_width: int, crop_height: int) -> list[float] | None:
        if not world_landmarks:
            return None

        scale_candidates = []
        pixel_candidates = []
        for first_index, second_index in ((0, 5), (0, 9), (0, 17), (5, 9), (9, 13), (13, 17)):
            pixel_distance = self._distance2(image_landmarks[first_index], image_landmarks[second_index], crop_width, crop_height)
            world_distance = self._distance3(world_landmarks[first_index], world_landmarks[second_index])
            if pixel_distance > 1e-6 and world_distance > 1e-6:
                scale_candidates.append(pixel_distance / world_distance)
                pixel_candidates.append(pixel_distance)
        if not scale_candidates:
            return None

        world_to_pixel_scale = float(np.median(np.asarray(scale_candidates, dtype=np.float64)))
        pixel_reference = max(float(np.median(np.asarray(pixel_candidates, dtype=np.float64))), 12.0)
        max_depth = max(pixel_reference * 0.8, 16.0)
        root_z = float(world_landmarks[0].z)
        return [
            float(np.clip(-(float(landmark.z) - root_z) * world_to_pixel_scale, -max_depth, max_depth))
            for landmark in world_landmarks
        ]
