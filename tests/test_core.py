from __future__ import annotations

import io
import os
import tempfile
import unittest
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np

from core.cli import InputAssignment, build_parser, explicit_option_dests, resolve_sources
from core.config import PipelineConfig
from inference.rtmpose import ONNXPoseHandRunner
from runners.fused_alignment import align_people_across_cameras
from core.runtime_config import build_config_for_assignment, prepare_model_assets, select_reference_assignment
from core.runtime_profiles import apply_runtime_profile
from utils.body_geometry import derive_foot_points
from utils.fusion import DEFAULT_CAMERA_CALIBRATION, fuse_body_views, load_camera_calibrations
from utils.fps import FpsMeter, draw_fps_overlay
from utils.normalize import build_hand_box
from utils.color_profile import color_profile_similarity
from utils.exports import build_joint_map, export_motion_json
from utils.hand_fallback import anchor_hand_to_wrist, has_usable_hand_detection, is_hand_detection_valid
from utils.hand_tracking import hand_detection_score, predict_hand_payload
from utils.motion_cleanup import cleanup_motion_frames
from utils.prediction import predict_points, translate_points
from utils.preview_stream import PreviewFrameSink
from utils.skeleton import JointMap
from utils.triangulation import apply_triangulated_overrides, triangulate_observation_frames
from pipeline.pipeline import PoseHandPipeline


def _body_points() -> list[tuple[int, int, float]]:
    points = [(0, 0, 0.0) for _ in range(17)]
    points[5] = (100, 100, 0.9)
    points[6] = (140, 100, 0.9)
    points[7] = (90, 140, 0.8)
    points[8] = (150, 140, 0.8)
    points[9] = (80, 180, 0.7)
    points[10] = (160, 180, 0.7)
    points[11] = (105, 200, 0.9)
    points[12] = (135, 200, 0.9)
    points[13] = (105, 260, 0.8)
    points[14] = (135, 260, 0.8)
    points[15] = (105, 320, 0.7)
    points[16] = (135, 320, 0.7)
    return points


class CoreLogicTests(unittest.TestCase):
    def test_cli_defaults_build_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--source", "0", "--no-preview"])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.video_path, 0)
        self.assertEqual(config.body_backend, "mediapipe")
        self.assertEqual(config.hand_backend, "mediapipe")
        self.assertEqual(config.provider_names, ("CUDAExecutionProvider",))
        self.assertFalse(config.enable_preview)
        self.assertEqual(config.max_people, 1)
        self.assertEqual(config.body_detect_interval, 1)

    def test_cli_speed_knobs_build_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--source", "0",
            "--body-detect-interval", "2",
            "--hand-detect-interval", "3",
            "--hand-crop-retries", "1",
            "--fps-log-interval", "0.5",
        ])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.body_detect_interval, 2)
        self.assertEqual(config.hand_detect_interval, 3)
        self.assertEqual(config.hand_crop_retries, 1)
        self.assertEqual(config.fps_log_interval, 0.5)

    def test_cli_single_camera_depth_mode_build_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--source", "0",
            "--single-camera-depth", "mediapipe",
        ])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.single_camera_depth_mode, "mediapipe")

    def test_cli_processing_width_build_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--source", "0", "--processing-width", "480"])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.processing_width, 480)

    def test_runtime_profile_applies_without_overriding_explicit_knobs(self) -> None:
        parser = build_parser()
        argv = [
            "--source", "0",
            "--landmark-backend", "yolo",
            "--profile", "fastest",
            "--body-detect-interval", "1",
        ]
        explicit = explicit_option_dests(parser, argv)
        args = parser.parse_args(argv)
        apply_runtime_profile(args, explicit)
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.profile, "fastest")
        self.assertEqual(config.body_model_variant, "yolo11n-pose.pt")
        self.assertEqual(config.hand_model_variant, "low")
        self.assertEqual(config.body_input_size, 640)
        self.assertEqual(config.processing_width, 640)
        self.assertTrue(config.yolo_half)
        self.assertEqual(config.body_detect_interval, 1)
        self.assertEqual(config.hand_detect_interval, 2)

    def test_fastest_profile_keeps_stable_detection_cadence(self) -> None:
        parser = build_parser()
        argv = ["--source", "0", "--profile", "fastest"]
        args = parser.parse_args(argv)
        apply_runtime_profile(args, explicit_option_dests(parser, argv))
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.processing_width, 640)
        self.assertEqual(config.body_detect_interval, 1)
        self.assertEqual(config.hand_detect_interval, 2)
        self.assertEqual(config.hand_crop_retries, 1)
        self.assertTrue(config.body_constraints_enabled)
        self.assertFalse(config.enable_backend_fallbacks)

    def test_cli_landmark_backend_build_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--source", "0", "--landmark-backend", "mediapipe"])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.body_backend, "mediapipe")
        self.assertEqual(config.hand_backend, "mediapipe")
        self.assertEqual(config.mediapipe_pose_model, "pose_landmark_full.tflite")

    def test_profile_yolo_model_does_not_pollute_mediapipe_backend(self) -> None:
        parser = build_parser()
        argv = ["--source", "0", "--landmark-backend", "mediapipe"]
        args = parser.parse_args(argv)
        apply_runtime_profile(args, explicit_option_dests(parser, argv))
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.body_backend, "mediapipe")
        self.assertIsNone(config.body_model_path)
        self.assertEqual(config.body_model_variant, "yolo11x-pose.pt")
        self.assertEqual(config.mediapipe_pose_model, "pose_landmark_full.tflite")

    def test_cli_mediapipe_model_name_build_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--source", "0",
            "--landmark-backend", "mediapipe",
            "--model", "pose_landmark_heavy.tflite",
        ])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.body_backend, "mediapipe")
        self.assertIsNone(config.body_model_path)
        self.assertEqual(config.body_model_variant, "yolo11x-pose.pt")
        self.assertEqual(config.mediapipe_pose_model, "pose_landmark_heavy.tflite")

    def test_cli_split_backends_build_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--source", "0",
            "--body-backend", "yolo",
            "--hand-backend", "mediapipe",
        ])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.body_backend, "yolo")
        self.assertEqual(config.hand_backend, "mediapipe")
        self.assertFalse(config.enable_backend_fallbacks)

    def test_hybrid_shortcut_enables_backend_fallbacks(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--source", "0", "--landmark-backend", "hybrid"])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.body_backend, "mediapipe")
        self.assertEqual(config.hand_backend, "mediapipe")
        self.assertTrue(config.enable_backend_fallbacks)

    def test_prepare_model_assets_skips_unused_models(self) -> None:
        config = PipelineConfig(body_backend="mediapipe", hand_backend="mediapipe")

        prepare_model_assets(config)

        self.assertIsNone(config.body_model_path)
        self.assertIsNone(config.hand_model_path)
        self.assertIsNotNone(config.mediapipe_pose_model_path)
        assert config.mediapipe_pose_model_path is not None
        self.assertEqual(config.mediapipe_pose_model_path.name, "pose_landmark_full.tflite")
        self.assertTrue(config.mediapipe_pose_model_path.exists())

    def test_mediapipe_hand_backend_uses_single_crop(self) -> None:
        config = PipelineConfig(hand_backend="mediapipe", hand_crop_retries=3)
        pipeline = PoseHandPipeline(config, SimpleNamespace(), SimpleNamespace(), SimpleNamespace())

        boxes = pipeline._hand_candidate_boxes(
            wrist_point=(50, 50, 0.9),
            elbow_point=(50, 80, 0.9),
            frame_width=100,
            frame_height=100,
            primary_box=(20, 20, 80, 80),
        )

        self.assertEqual(boxes, [(20, 20, 80, 80)])

    def test_processing_width_downscales_inference_frame(self) -> None:
        config = PipelineConfig(processing_width=50)
        pipeline = PoseHandPipeline(config, SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        with redirect_stdout(io.StringIO()):
            resized, scale_x, scale_y = pipeline._build_inference_frame(frame)

        self.assertEqual(resized.shape[:2], (25, 50))
        self.assertEqual(scale_x, 4.0)
        self.assertEqual(scale_y, 4.0)

    def test_processing_width_keeps_hand_detection_on_source_frame(self) -> None:
        class Runner:
            last_body_depths = {}
            last_hand_depths = None

            def __init__(self) -> None:
                self.body_frame_shape: tuple[int, int, int] = (0, 0, 0)
                self.hand_frame_shapes = []

            def detect_body(self, frame):
                self.body_frame_shape = (int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2]))
                points = [(0, 0, 0.0) for _ in range(17)]
                points[5] = (25, 25, 0.9)
                points[6] = (35, 25, 0.9)
                points[7] = (22, 35, 0.9)
                points[8] = (38, 35, 0.9)
                points[9] = (20, 45, 0.9)
                points[10] = (40, 45, 0.9)
                points[11] = (26, 50, 0.9)
                points[12] = (34, 50, 0.9)
                points[13] = (26, 65, 0.9)
                points[14] = (34, 65, 0.9)
                points[15] = (26, 80, 0.9)
                points[16] = (34, 80, 0.9)
                return points

            def detect_hand(self, frame, box):
                self.hand_frame_shapes.append(frame.shape)
                return None

        runner = Runner()
        smoother = SimpleNamespace(
            smooth_body=lambda points: points,
            smooth_hand=lambda _side, points: points,
        )
        osc_sender = SimpleNamespace(send_pose=lambda _body, _hands: None)
        config = PipelineConfig(processing_width=50, hand_crop_retries=0)
        pipeline = PoseHandPipeline(config, runner, smoother, osc_sender)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        with redirect_stdout(io.StringIO()) as stdout:
            body_points, hands_by_side = pipeline.detect_pose(frame)

        self.assertIn("source 200x100 -> inference 50x25", stdout.getvalue())
        self.assertEqual(runner.body_frame_shape[:2], (25, 50))
        self.assertTrue(runner.hand_frame_shapes)
        self.assertTrue(all(shape[:2] == (100, 200) for shape in runner.hand_frame_shapes))
        self.assertEqual(body_points[5][0], 100)
        self.assertIn("left", hands_by_side)

    def test_mediapipe_body_depth_uses_world_landmarks(self) -> None:
        runner = ONNXPoseHandRunner.__new__(ONNXPoseHandRunner)
        image_landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(33)]
        world_landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(33)]
        image_landmarks[11] = SimpleNamespace(x=0.4, y=0.3, z=0.0)
        image_landmarks[12] = SimpleNamespace(x=0.6, y=0.3, z=0.0)
        image_landmarks[23] = SimpleNamespace(x=0.45, y=0.6, z=0.0)
        image_landmarks[24] = SimpleNamespace(x=0.55, y=0.6, z=0.0)
        world_landmarks[11] = SimpleNamespace(x=-0.2, y=0.4, z=0.0)
        world_landmarks[12] = SimpleNamespace(x=0.2, y=0.4, z=0.0)
        world_landmarks[23] = SimpleNamespace(x=-0.15, y=0.0, z=0.0)
        world_landmarks[24] = SimpleNamespace(x=0.15, y=0.0, z=0.0)
        world_landmarks[15] = SimpleNamespace(x=-0.45, y=0.2, z=-0.2)

        depths = runner._mediapipe_body_depths(image_landmarks, world_landmarks, 1000, 1000)

        self.assertGreater(depths["LeftWrist"], 0.0)
        self.assertLess(depths["LeftWrist"], 200.0)

    def test_fps_overlay_draws_on_frame(self) -> None:
        frame = np.zeros((80, 220, 3), dtype=np.uint8)
        meter = FpsMeter("test", interval_seconds=0.0)
        meter.current_fps = 59.5
        meter.average_fps = 58.0

        draw_fps_overlay(frame, meter, enabled=True)

        self.assertGreater(int(frame.sum()), 0)

    def test_unlabeled_multi_sources_get_camera_labels(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--source", "0", "--source", "1"])

        assignments = resolve_sources(args)

        self.assertEqual([assignment.label for assignment in assignments], ["CAM_0", "CAM_1"])

    def test_cli_triangulation_knobs_build_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration_path = Path(tmp_dir) / "calibration.toml"
            calibration_path.write_text("", encoding="utf-8")
            parser = build_parser()
            args = parser.parse_args([
                "--source", "CAM_0=0",
                "--source", "CAM_1=1",
                "--triangulate-3d",
                "--calibration-3d", str(calibration_path),
                "--triangulation-min-cameras", "2",
                "--triangulation-use-outlier-rejection",
                "--triangulation-max-error", "5.0",
                "--triangulation-smoothing-alpha", "0.25",
                "--sync-offset", "CAM_0=0",
                "--sync-offset", "CAM_1=3",
            ])
            config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), True)

        self.assertTrue(config.enable_3d_triangulation)
        self.assertEqual(config.calibration_3d_path, calibration_path)
        self.assertEqual(config.triangulation_min_cameras, 2)
        self.assertTrue(config.triangulation_use_outlier_rejection)
        self.assertEqual(config.triangulation_max_error, 5.0)
        self.assertEqual(config.triangulation_smoothing_alpha, 0.25)
        self.assertEqual(config.sync_offsets, {"CAM_0": 0, "CAM_1": 3})

    def test_reference_assignment_uses_first_source(self) -> None:
        assignments = [InputAssignment("CAM_1", 1), InputAssignment("CAM_0", 0)]

        self.assertIs(select_reference_assignment(assignments), assignments[0])

    def test_pipeline_config_run_index_avoids_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            (output_dir / "clip rendered-1.mp4").write_text("", encoding="utf-8")
            config = PipelineConfig(
                project_root=output_dir,
                video_path=output_dir / "clip.mp4",
                output_directory=output_dir,
                output_basename="clip",
            )

            self.assertEqual(config.run_index, 2)
            self.assertEqual(config.rendered_output_path.name, "clip rendered-2.mp4")

    def test_build_joint_map_derives_expected_core_joints(self) -> None:
        joints = build_joint_map(_body_points(), {})

        self.assertIn("HipsRoot", joints)
        self.assertIn("Chest", joints)
        self.assertIn("LeftWrist", joints)
        self.assertGreater(joints["HipsRoot"]["confidence"], 0.0)
        self.assertEqual(joints["LeftWrist"]["x"], 80.0)
        self.assertEqual(joints["LeftWrist"]["y"], -180.0)

    def test_derive_foot_points_extends_beyond_ankle(self) -> None:
        foot_point, toe_point = derive_foot_points((100, 260, 0.8), (100, 320, 0.7))

        self.assertGreater(foot_point[1], 320)
        self.assertGreater(toe_point[1], foot_point[1])
        self.assertEqual(foot_point[2], 0.7)

    def test_build_joint_map_prefers_real_extra_foot_points(self) -> None:
        points = _body_points()
        points.extend([
            (91, 333, 0.95),
            (149, 333, 0.95),
            (82, 330, 0.90),
            (158, 330, 0.90),
        ])

        joints = build_joint_map(points, {})

        self.assertEqual(joints["LeftFoot"]["x"], 91.0)
        self.assertEqual(joints["LeftFoot"]["y"], -333.0)
        self.assertEqual(joints["LeftToeBase"]["x"], 82.0)
        self.assertEqual(joints["RightToeBase"]["x"], 158.0)

    def test_build_joint_map_keeps_single_view_flat_by_default(self) -> None:
        joints = build_joint_map(_body_points(), {})

        self.assertEqual(joints["LeftToeBase"]["z"], 0.0)
        self.assertEqual(joints["LeftWrist"]["z"], 0.0)

    def test_build_joint_map_uses_supplied_dynamic_depths(self) -> None:
        joints = build_joint_map(
            _body_points(),
            {},
            joint_depths={"LeftWrist": 123.0, "RightWrist": -45.0},
        )

        self.assertEqual(joints["LeftWrist"]["z"], 123.0)
        self.assertEqual(joints["RightWrist"]["z"], -45.0)

    def test_export_cleanup_interpolates_missing_joint(self) -> None:
        config = PipelineConfig()
        frames = [
            {"frame_index": 0, "joints": {"LeftWrist": {"x": 1.0, "y": 0.0, "z": 0.0, "confidence": 0.9}}},
            {"frame_index": 1, "joints": {"LeftWrist": {"x": 0.0, "y": 0.0, "z": 0.0, "confidence": 0.0}}},
            {"frame_index": 2, "joints": {"LeftWrist": {"x": 20.0, "y": 0.0, "z": 0.0, "confidence": 0.9}}},
        ]

        cleaned = cleanup_motion_frames(frames, config)

        cleaned_joints = cast(JointMap, cleaned[1]["joints"])
        self.assertGreater(cleaned_joints["LeftWrist"]["confidence"], 0.0)
        self.assertGreater(cleaned_joints["LeftWrist"]["x"], 0.0)

    def test_prediction_projects_points_with_confidence_decay(self) -> None:
        previous = [(8, 10, 0.9), (20, 30, 0.5)]
        current = [(10, 14, 0.8), (25, 33, 0.4)]

        predicted = predict_points(current, previous, confidence_decay=0.5)

        self.assertEqual(predicted, [(12, 18, 0.4), (30, 36, 0.2)])
        self.assertEqual(translate_points(current, 2, -3, confidence_decay=0.5), [(12, 11, 0.4), (27, 30, 0.2)])

    def test_color_profile_similarity_is_normalized_overlap(self) -> None:
        score = color_profile_similarity({"orange": 0.8, "black": 0.2}, {"orange": 0.4, "blue": 0.6})
        self.assertAlmostEqual(score, 0.25)
        self.assertEqual(color_profile_similarity({}, {"orange": 1.0}), 0.0)

    def test_align_people_across_cameras_prefers_labels_then_color(self) -> None:
        front = SimpleNamespace(
            label="person1",
            id=1,
            center=(10.0, 0.0),
            color_signature={"orange": 0.9},
        )
        left_labeled = SimpleNamespace(
            label="person1",
            id=4,
            center=(100.0, 0.0),
            color_signature={"blue": 0.9},
        )
        front_unlabeled = SimpleNamespace(
            label=None,
            id=2,
            center=(50.0, 0.0),
            color_signature={"blue": 0.9},
        )
        left_unlabeled = SimpleNamespace(
            label=None,
            id=5,
            center=(120.0, 0.0),
            color_signature={"blue": 0.8},
        )

        grouped = align_people_across_cameras(
            {"CAM_0": [front, front_unlabeled], "CAM_1": [left_unlabeled, left_labeled]},
            "CAM_0",
        )

        self.assertIs(grouped["person1"]["CAM_1"], left_labeled)
        self.assertIs(grouped["person2"]["CAM_1"], left_unlabeled)

    def test_build_hand_box_is_clamped_to_frame(self) -> None:
        box = build_hand_box(
            wrist_point=(5, 5, 0.9),
            elbow_point=(20, 20, 0.9),
            frame_width=100,
            frame_height=80,
            min_box_size=40,
            scale=2.0,
            forward_shift=0.35,
        )

        x1, y1, x2, y2 = box
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 100)
        self.assertLessEqual(y2, 80)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_anchor_then_validate_realistic_hand_detection(self) -> None:
        config = PipelineConfig()
        wrist = (100, 100, 0.9)
        elbow = (100, 150, 0.9)
        raw_hand = [(x + 15, y + 10, conf) for x, y, conf in [
            (100, 100, 0.9), (94, 86, 0.8), (90, 72, 0.8), (86, 58, 0.7), (82, 44, 0.7),
            (96, 82, 0.8), (94, 64, 0.8), (92, 46, 0.7), (90, 30, 0.7),
            (104, 80, 0.8), (104, 60, 0.8), (104, 42, 0.7), (104, 24, 0.7),
            (112, 82, 0.8), (114, 64, 0.8), (116, 48, 0.7), (118, 34, 0.7),
            (120, 88, 0.8), (126, 74, 0.8), (132, 60, 0.7), (138, 48, 0.7),
        ]]

        self.assertTrue(has_usable_hand_detection(raw_hand, config))
        anchored = anchor_hand_to_wrist(raw_hand, wrist)
        self.assertTrue(is_hand_detection_valid(anchored, wrist, elbow, config))

    def test_hand_detection_score_prefers_temporally_consistent_hand(self) -> None:
        config = PipelineConfig()
        wrist = (100, 100, 0.9)
        elbow = (100, 150, 0.9)
        stable = [
            (100, 100, 0.9), (94, 86, 0.8), (90, 72, 0.8), (86, 58, 0.7), (82, 44, 0.7),
            (96, 82, 0.8), (94, 64, 0.8), (92, 46, 0.7), (90, 30, 0.7),
            (104, 80, 0.8), (104, 60, 0.8), (104, 42, 0.7), (104, 24, 0.7),
            (112, 82, 0.8), (114, 64, 0.8), (116, 48, 0.7), (118, 34, 0.7),
            (120, 88, 0.8), (126, 74, 0.8), (132, 60, 0.7), (138, 48, 0.7),
        ]
        jumpy = [(x + 90, y + 70, conf) for x, y, conf in stable]

        stable_score = hand_detection_score(stable, wrist, elbow, config, previous_points=stable)
        jumpy_score = hand_detection_score(jumpy, wrist, elbow, config, previous_points=stable)

        self.assertGreater(stable_score, jumpy_score)

    def test_predict_hand_payload_uses_wrist_and_elbow_motion(self) -> None:
        previous_payload = {
            "box": (90, 90, 150, 150),
            "points": [(100, 100, 0.8)] * 21,
        }

        predicted = predict_hand_payload(
            previous_payload,
            previous_wrist=(100, 100, 0.9),
            wrist_point=(110, 112, 0.9),
            confidence_decay=0.5,
            previous_elbow=(100, 150, 0.9),
            elbow_point=(106, 158, 0.9),
        )

        self.assertIsNotNone(predicted)
        assert predicted is not None
        box, points, _ = predicted
        self.assertEqual(box, (99, 101, 159, 161))
        self.assertEqual(points[0], (109, 111, 0.4))

    def test_preview_frame_sink_writes_latest_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            preview_path = Path(tmp_dir) / "preview.jpg"
            old_path = os.environ.get("KINARA_PREVIEW_FRAME")
            old_interval = os.environ.get("KINARA_PREVIEW_INTERVAL")
            os.environ["KINARA_PREVIEW_FRAME"] = str(preview_path)
            os.environ["KINARA_PREVIEW_INTERVAL"] = "1"
            try:
                sink = PreviewFrameSink()
                frame = np.zeros((12, 16, 3), dtype=np.uint8)
                frame[:, :, 1] = 255
                sink.write(frame, 0)
            finally:
                if old_path is None:
                    os.environ.pop("KINARA_PREVIEW_FRAME", None)
                else:
                    os.environ["KINARA_PREVIEW_FRAME"] = old_path
                if old_interval is None:
                    os.environ.pop("KINARA_PREVIEW_INTERVAL", None)
                else:
                    os.environ["KINARA_PREVIEW_INTERVAL"] = old_interval

            frame_files = sorted(Path(tmp_dir).glob("preview_*.jpg"))
            self.assertEqual(len(frame_files), 1)
            self.assertGreater(frame_files[0].stat().st_size, 0)

    def test_preview_frame_sink_ignores_locked_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            preview_path = Path(tmp_dir) / "preview.jpg"
            old_path = os.environ.get("KINARA_PREVIEW_FRAME")
            old_interval = os.environ.get("KINARA_PREVIEW_INTERVAL")
            os.environ["KINARA_PREVIEW_FRAME"] = str(preview_path)
            os.environ["KINARA_PREVIEW_INTERVAL"] = "1"
            try:
                sink = PreviewFrameSink()
                frame = np.zeros((12, 16, 3), dtype=np.uint8)
                with patch.object(Path, "replace", side_effect=PermissionError("locked")):
                    sink.write(frame, 0)
            finally:
                if old_path is None:
                    os.environ.pop("KINARA_PREVIEW_FRAME", None)
                else:
                    os.environ["KINARA_PREVIEW_FRAME"] = old_path
                if old_interval is None:
                    os.environ.pop("KINARA_PREVIEW_INTERVAL", None)
                else:
                    os.environ["KINARA_PREVIEW_INTERVAL"] = old_interval

            self.assertEqual(list(Path(tmp_dir).glob("*.tmp.jpg")), [])

    def test_load_camera_calibrations_merges_uppercase_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "calibration.json"
            path.write_text(json.dumps({"cam_1": {"depth_scale": 2.5}}), encoding="utf-8")

            calibrations = load_camera_calibrations(path)

        self.assertEqual(calibrations["CAM_1"]["depth_scale"], 2.5)
        self.assertEqual(calibrations["CAM_1"]["depth_sign"], DEFAULT_CAMERA_CALIBRATION["depth_sign"])

    def test_fuse_body_views_keeps_reference_shape(self) -> None:
        front = _body_points()
        left = [(x + 10, y, conf) for x, y, conf in front]

        fused = fuse_body_views({"CAM_0": front, "CAM_1": left}, threshold=0.1)

        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertEqual(len(fused), len(front))
        self.assertGreater(fused[5][2], 0.0)

    def test_triangulation_applies_world_overrides(self) -> None:
        import utils.triangulation as triangulation

        class FakeCameraGroup:
            def get_names(self):
                return ["CAM_0", "CAM_1"]

            def triangulate(self, points, progress=False, minimum_cameras_for_triangulation=2):
                self.received_points = points
                out = np.full((points.shape[1], 3), np.nan, dtype=np.float64)
                good = ~np.isnan(points[:, :, 0])
                for point_index in range(points.shape[1]):
                    if np.count_nonzero(good[:, point_index]) >= minimum_cameras_for_triangulation:
                        mean_xy = np.nanmean(points[:, point_index, :], axis=0)
                        out[point_index] = [mean_xy[0], mean_xy[1], 42.0]
                return out

            def reprojection_error(self, points_3d, points_2d, mean=True):
                return np.zeros(points_3d.shape[0], dtype=np.float64)

        fake_group = FakeCameraGroup()
        original_loader = triangulation._load_camera_group
        triangulation._load_camera_group = lambda _: fake_group
        try:
            result = triangulate_observation_frames(
                Path("calibration.toml"),
                [{
                    "camera_bodies": {
                        "CAM_0": _body_points(),
                        "CAM_1": [(x + 10, y, conf) for x, y, conf in _body_points()],
                    },
                    "camera_hands": {},
                }],
                body_threshold=0.1,
                hand_threshold=0.1,
                minimum_cameras=2,
                use_outlier_rejection=False,
                maximum_cameras_to_drop=1,
                target_reprojection_error=0.01,
            )
        finally:
            triangulation._load_camera_group = original_loader

        frame = {"frame_index": 0, "joints": build_joint_map(_body_points(), {})}
        updated_frames = apply_triangulated_overrides([frame], result)
        updated_joints = cast(JointMap, updated_frames[0]["joints"])
        left_shoulder = updated_joints["LeftShoulder"]

        self.assertEqual(left_shoulder["x"], 105.0)
        self.assertEqual(left_shoulder["z"], 42.0)
        self.assertGreater(result.triangulated_point_count, 0)

    def test_export_motion_json_writes_metadata_and_frames(self) -> None:
        frame = {"frame_index": 0, "joints": build_joint_map(_body_points(), {})}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "motion.json"
            export_motion_json(path, fps=30.0, frames=[frame], metadata={"mode": "test"})
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["format"], "kinara-motion-json-v1")
        self.assertEqual(payload["frame_count"], 1)
        self.assertEqual(payload["metadata"]["mode"], "test")
        self.assertIn("skeleton", payload["metadata"])


if __name__ == "__main__":
    unittest.main()
