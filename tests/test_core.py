from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np

from core.cli import InputAssignment, build_parser, explicit_option_dests, resolve_sources
from core.config import PipelineConfig
from core.config_file import load_config_defaults
from core.runtime_config import (
    build_config_for_assignment,
    prepare_model_assets,
    select_reference_assignment,
    validate_config,
)
from core.runtime_profiles import apply_runtime_profile
from inference.rtmpose import ONNXPoseHandRunner
from pipeline.pipeline import PoseHandPipeline
from runners.fused_alignment import align_people_across_cameras
from utils.body_geometry import derive_foot_points
from utils.color_profile import color_profile_similarity
from utils.exports import build_joint_map, export_motion_json
from utils.fps import FpsMeter, draw_fps_overlay
from utils.fusion import DEFAULT_CAMERA_CALIBRATION, fuse_body_views, load_camera_calibrations
from utils.hand_fallback import anchor_hand_to_wrist, has_usable_hand_detection, is_hand_detection_valid
from utils.hand_tracking import hand_detection_score, predict_hand_payload
from utils.motion_cleanup import cleanup_motion_frames
from utils.normalize import build_hand_box
from utils.prediction import predict_points, translate_points
from utils.preview_stream import PreviewFrameSink
from utils.skeleton import JointMap
from utils.triangulation import apply_triangulated_overrides, triangulate_observation_frames


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
    def test_module_help_smoke(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "kinara", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--source", completed.stdout)
        self.assertIn("--landmark-backend", completed.stdout)

    def test_cli_entrypoint_dispatches_single_assignment(self) -> None:
        from app import main as app_main

        argv = ["kinara", "--source", "0", "--no-preview", "--output-basename", "smoke"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(app_main, "ensure_runtime_ready") as ensure_runtime_ready,
            patch("runners.single.run_assignment") as run_assignment,
        ):
            app_main.main()

        ensure_runtime_ready.assert_called_once()
        run_assignment.assert_called_once()
        config = run_assignment.call_args.args[0]
        self.assertEqual(config.video_path, 0)
        self.assertFalse(config.enable_preview)
        self.assertEqual(config.output_basename, "smoke")

    def test_config_file_defaults_can_be_overridden(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "kinara.json"
            config_path.write_text(
                json.dumps(
                    {
                        "source": ["0"],
                        "output_basename": "from_config",
                        "benchmark_frames": 5,
                        "no_preview": True,
                    }
                ),
                encoding="utf-8",
            )
            config_dests = load_config_defaults(parser, config_path)
            args = parser.parse_args(["--config", str(config_path), "--output-basename", "from_cli"])

        self.assertIn("benchmark_frames", config_dests)
        self.assertEqual(args.source, ["0"])
        self.assertEqual(args.output_basename, "from_cli")
        self.assertEqual(args.benchmark_frames, 5)
        self.assertTrue(args.no_preview)

    def test_dry_run_does_not_dispatch_runner(self) -> None:
        from app import main as app_main

        argv = ["kinara", "--source", "0", "--dry-run", "--no-preview", "--output-basename", "dry"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(app_main, "ensure_runtime_ready") as ensure_runtime_ready,
            patch("runners.single.run_assignment") as run_assignment,
        ):
            app_main.main()

        ensure_runtime_ready.assert_called_once()
        self.assertTrue(ensure_runtime_ready.call_args.kwargs["check_only"])
        run_assignment.assert_not_called()

    def test_skip_runtime_check_bypasses_bootstrap(self) -> None:
        from app import main as app_main

        argv = ["kinara", "--source", "0", "--skip-runtime-check", "--no-preview", "--output-basename", "skip"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(app_main, "ensure_runtime_ready") as ensure_runtime_ready,
            patch("runners.single.run_assignment") as run_assignment,
        ):
            app_main.main()

        ensure_runtime_ready.assert_not_called()
        run_assignment.assert_called_once()

    def test_app_startup_prepares_vendor_path_even_when_bootstrap_is_skipped(self) -> None:
        import app.main as app_main

        self.assertIn(str(app_main.PROJECT_ROOT / ".vendor_py311"), sys.path)
        self.assertEqual(os.environ.get("XDG_CACHE_HOME"), str(app_main.PROJECT_ROOT / ".kinara_runtime" / "cache"))

    def test_safe_text_io_accepts_missing_windowed_stream(self) -> None:
        from utils.logging import SafeTextIO, configure_run_log

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = configure_run_log(Path(tmp_dir) / "run.txt")
            stream = SafeTextIO(None)

            written = stream.write("worker message")
            stream.flush()

            self.assertEqual(written, 0)
            self.assertIn("worker message", log_path.read_text(encoding="utf-8"))

    def test_pipeline_config_allocates_metadata_path(self) -> None:
        config = PipelineConfig(output_basename="metadata_test")

        self.assertTrue(config.metadata_output_path.name.startswith("metadata_test metadata-"))
        self.assertTrue(config.metadata_output_path.name.endswith(".json"))

    def test_runtime_report_relativizes_project_paths(self) -> None:
        from core.runtime_report import build_runtime_report

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "Kinara"
            project_root.mkdir()
            source_path = Path.home() / "Downloads" / "local_clip.mp4"
            config = PipelineConfig(
                project_root=project_root,
                video_path=source_path,
                output_directory=project_root / "outputs",
                output_basename="local",
            )

            report = build_runtime_report(config)

        report_text = json.dumps(report)
        self.assertEqual(report["outputs"]["metadata"].split("/")[0], "<PROJECT_ROOT>")
        self.assertIn("<PROJECT_ROOT>", report_text)
        self.assertNotIn(str(project_root), report_text)

    def test_run_metadata_keeps_local_source_path(self) -> None:
        from utils.run_metadata import build_run_metadata

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "Kinara"
            project_root.mkdir()
            source_path = Path.home() / "Downloads" / "local_clip.mp4"
            config = PipelineConfig(
                project_root=project_root,
                video_path=source_path,
                output_directory=project_root / "outputs",
                output_basename="local",
            )

            metadata = build_run_metadata(config, mode="test", fps=30.0, frame_count=1)

        metadata_text = json.dumps(metadata)
        self.assertEqual(metadata["source"], str(source_path).replace("\\", "/"))
        self.assertIn("local_clip.mp4", metadata_text)
        self.assertIn("<PROJECT_ROOT>", metadata_text)
        self.assertNotIn(str(project_root), metadata_text)

    def test_installer_python_does_not_default_to_blender_python(self) -> None:
        import utils.bootstrap_packages as bootstrap_packages

        with (
            patch.dict(os.environ, {"KINARA_PYTHON": ""}, clear=False),
            patch.object(bootstrap_packages.sys, "_base_executable", "", create=True),
        ):
            installer = bootstrap_packages.installer_python()

        self.assertEqual(Path(installer).resolve(), Path(sys.executable).resolve())

    def test_installer_python_accepts_python_directory(self) -> None:
        import utils.bootstrap_packages as bootstrap_packages

        python_dir = str(Path(sys.executable).parent)
        with patch.dict(os.environ, {"KINARA_PYTHON": python_dir}, clear=False):
            installer = bootstrap_packages.installer_python()

        self.assertEqual(Path(installer).resolve(), Path(sys.executable).resolve())

    def test_installer_python_discovers_path_python(self) -> None:
        import utils.bootstrap_packages as bootstrap_packages

        original_python = sys.executable
        with (
            patch.dict(os.environ, {"KINARA_PYTHON": ""}, clear=False),
            patch.object(bootstrap_packages.sys, "_base_executable", "", create=True),
            patch.object(bootstrap_packages.sys, "executable", "Kinara.exe"),
            patch.object(bootstrap_packages, "_common_python_candidates", return_value=[original_python]),
        ):
            installer = bootstrap_packages.installer_python()

        self.assertEqual(Path(installer).resolve(), Path(original_python).resolve())

    def test_calibration_mode_requests_calibration_runtime_modules(self) -> None:
        from utils.bootstrap_dependencies import _selected_runtime_modules

        modules = _selected_runtime_modules(["--calibrate-cameras", "--source", "CAM_0=a.mp4", "--source", "CAM_1=b.mp4"])

        self.assertIn("aniposelib", modules)
        self.assertIn("cv2", modules)

    def test_calibration_install_plan_uses_contrib_opencv_and_numpy_pin(self) -> None:
        from utils.bootstrap_packages import resolve_install_plan
        from utils.bootstrap_state import ModuleStatus, RuntimeReport

        plan = resolve_install_plan(
            [
                ModuleStatus("aniposelib", False),
                ModuleStatus("cv2", False),
                ModuleStatus("numpy", False),
            ],
            RuntimeReport(),
        )

        self.assertIn("aniposelib>=0.7,<0.8", plan)
        self.assertIn("numpy>=1.26,<2.0", plan)
        self.assertIn("opencv-contrib-python>=4.9,<4.12", plan)
        self.assertIn("protobuf>=4.25.3,<5", plan)

    def test_calibration_empty_detections_raise_actionable_error(self) -> None:
        from utils.calibration import _validate_detected_rows

        with self.assertRaisesRegex(ValueError, "no Charuco boards were detected"):
            _validate_detected_rows([[], []], ["CAM_0", "CAM_1"], marker_bits=4, dict_size=250)

    def test_calibration_empty_corner_rows_raise_actionable_error(self) -> None:
        from utils.calibration import _validate_detected_rows

        rows = [[{"corners": np.float64([]), "ids": np.float64([])}], []]

        with self.assertRaisesRegex(ValueError, "no Charuco boards were detected"):
            _validate_detected_rows(rows, ["CAM_0", "CAM_1"], marker_bits=4, dict_size=250)

    def test_cli_charuco_retry_overrides_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--source", "a.mp4",
            "--source", "b.mp4",
            "--calibrate-cameras",
            "--charuco-retry-scale", "3.5",
            "--charuco-min-markers", "6",
            "--charuco-retry-sharpen",
        ])

        self.assertEqual(args.charuco_retry_scale, 3.5)
        self.assertEqual(args.charuco_min_markers, 6)
        self.assertTrue(args.charuco_retry_sharpen)

    def test_low_resolution_charuco_detection_retries_empty_marker_list(self) -> None:
        from utils.calibration import _enable_low_resolution_charuco_detection

        class FakeBoard:
            def __init__(self) -> None:
                self.detected_shapes = []
                self.board = object()

            def detect_markers(self, image, camera=None, refine=True):
                self.detected_shapes.append(image.shape)
                if image.shape[0] < 20:
                    return [], []
                return [np.array([[[6.0, 8.0]]], dtype=np.float32)], np.array([[1]], dtype=np.int32)

            def detect_image(self, image, camera=None):
                return np.float64([]), np.float64([])

        board = FakeBoard()

        _enable_low_resolution_charuco_detection(board, detection_strictness="balanced")
        corners, ids = board.detect_markers(np.zeros((10, 12), dtype=np.uint8))

        self.assertEqual([shape[:2] for shape in board.detected_shapes], [(10, 12), (20, 24)])
        self.assertEqual(ids.tolist(), [[1]])
        self.assertEqual(corners[0].tolist(), [[[3.0, 4.0]]])

    def test_low_resolution_charuco_detection_uses_retry_overrides(self) -> None:
        from utils.calibration import _enable_low_resolution_charuco_detection

        class FakeBoard:
            def detect_markers(self, image, camera=None, refine=True):
                return [], []

            def detect_image(self, image, camera=None):
                return np.float64([]), np.float64([])

        settings = _enable_low_resolution_charuco_detection(
            FakeBoard(),
            detection_strictness="strict",
            retry_scale=2.5,
            minimum_markers=4,
            retry_sharpen=True,
        )

        self.assertEqual(
            settings,
            {
                "enabled": True,
                "strictness": "strict",
                "scale": 2.5,
                "minimum_markers": 4,
                "sharpen": True,
            },
        )

    def test_low_resolution_charuco_detection_retries_corner_interpolation(self) -> None:
        from utils.calibration import _enable_low_resolution_charuco_detection

        class FakeBoard:
            def __init__(self) -> None:
                self.board = object()
                self.manually_verify = False
                self.marker_shapes = []

            def detect_markers(self, image, camera=None, refine=True):
                self.marker_shapes.append(image.shape)
                return [np.array([[[10.0, 14.0]]], dtype=np.float32)], np.array([[2]], dtype=np.int32)

            def detect_image(self, image, camera=None):
                return np.float64([]), np.float64([])

        board = FakeBoard()
        scaled_corners = np.array([[[20.0, 24.0]]], dtype=np.float32)
        scaled_ids = np.array([[3]], dtype=np.int32)

        with patch("cv2.aruco.interpolateCornersCharuco", return_value=(1, scaled_corners, scaled_ids)):
            _enable_low_resolution_charuco_detection(board, detection_strictness="balanced")
            corners, ids = board.detect_image(np.zeros((10, 12, 3), dtype=np.uint8))

        self.assertEqual(corners.tolist(), [[[10.0, 12.0]]])
        self.assertEqual(ids.tolist(), [[3]])
        self.assertEqual(board.marker_shapes[-1][:2], (20, 24))

    def test_mediapipe_install_plan_pins_protobuf(self) -> None:
        from utils.bootstrap_packages import resolve_install_plan
        from utils.bootstrap_state import ModuleStatus, RuntimeReport

        plan = resolve_install_plan([ModuleStatus("mediapipe", False)], RuntimeReport())

        self.assertIn("mediapipe==0.10.21", plan)
        self.assertIn("protobuf>=4.25.3,<5", plan)
        self.assertIn("jax==0.7.1", plan)
        self.assertIn("jaxlib==0.7.1", plan)

    def test_mediapipe_install_plan_uses_single_opencv_flavor(self) -> None:
        from utils.bootstrap_packages import resolve_install_plan
        from utils.bootstrap_state import ModuleStatus, RuntimeReport

        plan = resolve_install_plan(
            [
                ModuleStatus("cv2", False),
                ModuleStatus("mediapipe", False),
            ],
            RuntimeReport(),
        )

        self.assertIn("opencv-contrib-python>=4.9,<4.12", plan)
        self.assertNotIn("opencv-python", plan)
        self.assertIn("mediapipe==0.10.21", plan)
        self.assertEqual(plan.count("opencv-contrib-python>=4.9,<4.12"), 1)

    def test_plain_cv2_install_plan_uses_bounded_opencv_and_numpy(self) -> None:
        from utils.bootstrap_packages import resolve_install_plan
        from utils.bootstrap_state import ModuleStatus, RuntimeReport

        plan = resolve_install_plan([ModuleStatus("cv2", False), ModuleStatus("numpy", False)], RuntimeReport())

        self.assertIn("numpy>=1.26,<2.0", plan)
        self.assertIn("opencv-python>=4.9,<4.12", plan)
        self.assertNotIn("opencv-python", plan)
        self.assertNotIn("numpy", plan)

    def test_rtmlib_install_plan_pins_numpy_and_opencv(self) -> None:
        from utils.bootstrap_packages import resolve_install_plan
        from utils.bootstrap_state import ModuleStatus, RuntimeReport

        plan = resolve_install_plan([ModuleStatus("rtmlib", False)], RuntimeReport())

        self.assertIn("numpy>=1.26,<2.0", plan)
        self.assertIn("opencv-contrib-python>=4.9,<4.12", plan)
        self.assertIn("rtmlib", plan)
        self.assertIn("protobuf>=4.25.3,<5", plan)

    def test_installed_mediapipe_repair_plan_pins_new_protobuf(self) -> None:
        import utils.bootstrap_packages as bootstrap_packages
        from utils.bootstrap_packages import resolve_install_plan
        from utils.bootstrap_state import ModuleStatus, RuntimeReport

        with patch.object(bootstrap_packages, "distribution_version", return_value="7.35.0"):
            plan = resolve_install_plan([ModuleStatus("mediapipe", True)], RuntimeReport())

        self.assertIn("protobuf>=4.25.3,<5", plan)

    def test_install_packages_hides_user_site_packages_from_pip(self) -> None:
        import utils.bootstrap_packages as bootstrap_packages

        captured_envs = []

        def fake_run(*_args, **kwargs):
            captured_envs.append(kwargs.get("env", {}))
            return subprocess.CompletedProcess(_args, 0)

        with (
            patch.object(bootstrap_packages, "installer_python", return_value=sys.executable),
            patch.object(bootstrap_packages.subprocess, "run", side_effect=fake_run),
            patch.object(bootstrap_packages, "prune_vendor_distributions"),
            patch.object(bootstrap_packages, "prepend_pythonpath"),
            patch.object(bootstrap_packages, "prepend_sys_path"),
        ):
            bootstrap_packages.install_packages(["example-package"])

        self.assertTrue(captured_envs)
        self.assertTrue(all(env.get("PYTHONNOUSERSITE") == "1" for env in captured_envs))

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

    def test_cli_rtmpose_backend_build_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--source",
            "0",
            "--landmark-backend",
            "rtmpose",
            "--rtmpose-mode",
            "lightweight",
            "--rtmpose-device",
            "cuda",
            "--no-preview",
        ])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.body_backend, "rtmpose")
        self.assertEqual(config.hand_backend, "onnx")
        self.assertEqual(config.rtmpose_mode, "lightweight")
        self.assertEqual(config.rtmpose_device, "cuda")

    def test_cli_rtmpose_wholebody_locks_body_and_hand_backends(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--source",
            "0",
            "--landmark-backend",
            "rtmpose-wholebody",
            "--body-backend",
            "yolo",
            "--hand-backend",
            "onnx",
            "--no-preview",
        ])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.body_backend, "rtmpose-wholebody")
        self.assertEqual(config.hand_backend, "rtmpose-wholebody")

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

    def test_cli_parallel_knobs_build_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--source", "0",
            "--parallel-workers", "0",
            "--parallel-chunk-seconds", "0",
            "--parallel-overlap-seconds", "0",
        ])
        config = build_config_for_assignment(args, InputAssignment("CAM_0", 0), False)

        self.assertEqual(config.parallel_workers, 0)
        self.assertEqual(config.parallel_chunk_seconds, 0.0)
        self.assertEqual(config.parallel_overlap_seconds, 0.0)

    def test_parallel_auto_resolves_single_worker_for_short_sources(self) -> None:
        from runners.parallel_single import resolve_parallel_workers

        config = PipelineConfig(parallel_workers=0)

        self.assertEqual(resolve_parallel_workers(config, total_frames=30, fps=30.0), 1)

    def test_parallel_chunk_specs_include_warmup_overlap(self) -> None:
        from runners.parallel_single import build_chunk_specs

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = PipelineConfig(
                output_directory=Path(tmp_dir),
                parallel_chunk_seconds=5.0,
                parallel_overlap_seconds=0.5,
            )
            specs = build_chunk_specs(config, total_frames=360, fps=30.0, chunk_root=Path(tmp_dir))

        self.assertEqual(len(specs), 3)
        self.assertEqual(specs[0].start_frame, 0)
        self.assertEqual(specs[0].warmup_start_frame, 0)
        self.assertEqual(specs[1].start_frame, 150)
        self.assertEqual(specs[1].warmup_start_frame, 135)

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

    def test_mediapipe_allows_multi_person_runner_path(self) -> None:
        config = PipelineConfig(body_backend="mediapipe", hand_backend="mediapipe", max_people=2)

        self.assertTrue(validate_config(config))

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
        self.assertNotEqual(joints["LeftElbow"]["z"], 0.0)

    def test_hand_candidate_retries_extend_past_three(self) -> None:
        short_retry_pipeline = PoseHandPipeline(
            PipelineConfig(hand_backend="onnx", hand_crop_retries=3),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        )
        long_retry_pipeline = PoseHandPipeline(
            PipelineConfig(hand_backend="onnx", hand_crop_retries=5),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        )

        short_retry_boxes = short_retry_pipeline._hand_candidate_boxes(
            (320, 240, 0.9),
            (270, 320, 0.9),
            1280,
            720,
            (280, 200, 360, 280),
        )
        long_retry_boxes = long_retry_pipeline._hand_candidate_boxes(
            (320, 240, 0.9),
            (270, 320, 0.9),
            1280,
            720,
            (280, 200, 360, 280),
        )

        self.assertGreater(len(long_retry_boxes), len(short_retry_boxes))
        self.assertGreater(len(long_retry_boxes), 4)

    def test_nested_relative_body_model_path_must_exist(self) -> None:
        from utils.model_assets import ensure_body_model_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                ensure_body_model_file(Path(tmp_dir), "custom/my_model.pt")

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

    def test_multi_person_track_smooths_and_constrains_body_points(self) -> None:
        from utils.multi_person import PersonDetection, PersonTrack, _smooth_and_constrain_body

        raw_points = _body_points()
        smoothed_points = [(x + 1, y + 1, conf) for x, y, conf in raw_points]
        constrained_points = [(x + 2, y + 2, conf) for x, y, conf in smoothed_points]

        class FakeSmoother:
            def smooth_body(self, points):
                self.received_points = points
                return smoothed_points

        class FakeConstraints:
            def apply(self, points):
                self.received_points = points
                return constrained_points

        pipeline = SimpleNamespace(
            smoother=FakeSmoother(),
            _body_constraints=FakeConstraints(),
            detect_hands=lambda frame, points: {},
            last_joint_depths={},
        )
        track = PersonTrack(id=1, box=(0, 0, 10, 10), pipeline=pipeline)

        body_points = _smooth_and_constrain_body(track, raw_points)

        self.assertIs(body_points, constrained_points)
        self.assertIs(pipeline.smoother.received_points, raw_points)
        self.assertIs(pipeline._body_constraints.received_points, smoothed_points)

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

    def test_triangulation_rejects_extra_calibration_cameras_without_subset_support(self) -> None:
        import utils.triangulation as triangulation

        class FakeCameraGroup:
            def get_names(self):
                return ["CAM_0", "CAM_1"]

        original_loader = triangulation._load_camera_group
        triangulation._load_camera_group = lambda _: FakeCameraGroup()
        try:
            with self.assertRaisesRegex(ValueError, "cannot subset cameras"):
                triangulate_observation_frames(
                    Path("calibration.toml"),
                    [{
                        "camera_bodies": {
                            "CAM_0": _body_points(),
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

    def test_freemocap_style_output_writes_calibrated_arrays(self) -> None:
        from utils.triangulation import TriangulationResult, export_freemocap_style_output

        points_3d = np.array([[[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]]], dtype=np.float64)
        confidences = np.array([[0.9, 0.0]], dtype=np.float64)
        result = TriangulationResult(
            joint_overrides_by_frame=[],
            camera_labels=["CAM_0", "CAM_1"],
            joint_names=["A", "B"],
            points_2d_xy=np.zeros((2, 1, 2, 2), dtype=np.float64),
            points_3d_xyz=points_3d,
            confidences=confidences,
            reprojection_error=np.array([[0.1, np.nan]], dtype=np.float64),
            full_reprojection_error=np.zeros((2, 1, 2), dtype=np.float64),
            mean_reprojection_error=0.5,
            triangulated_point_count=1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = export_freemocap_style_output(Path(tmp_dir), result)
            saved = np.load(paths["skeleton_3d_npy"])
            csv_text = Path(paths["skeleton_3d_csv"]).read_text(encoding="utf-8")
            self.assertTrue(Path(paths["raw_2d_npy"]).exists())
            self.assertTrue(Path(paths["reprojection_error_npy"]).exists())
            self.assertTrue(Path(paths["full_reprojection_error_npy"]).exists())

        self.assertEqual(saved.shape, (1, 2, 3))
        self.assertIn("frame,tracked_point,x,y,z,confidence", csv_text)
        self.assertIn("0,A,1.00000000,2.00000000,3.00000000,0.90000000", csv_text)

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
