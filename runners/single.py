from __future__ import annotations

import time

import cv2

from camera.capture import VideoCaptureSession
from core.config import PipelineConfig
from core.runtime_config import prepare_runtime_config
from inference.rtmpose import ONNXPoseHandRunner
from network.osc_sender import OSCSender
from pipeline.pipeline import PoseHandPipeline
from runners.common import export_motion_bundle, print_saved_paths
from runners.multi_person import run_multi_person_assignment
from runners.parallel_single import run_parallel_assignment
from utils.exports import build_joint_map
from utils.fps import FpsMeter, draw_fps_overlay
from utils.logging import log_info
from utils.preview_stream import PreviewFrameSink
from utils.run_metadata import build_run_metadata, write_run_metadata
from utils.smoothing import LandmarkSmoother


def run_assignment(config: PipelineConfig) -> bool:
    if config.max_people > 1:
        return run_multi_person_assignment(config)
    if not prepare_runtime_config(config):
        return False
    if run_parallel_assignment(config):
        return True

    assert config.output_path is not None
    session = VideoCaptureSession(
        config.video_path,
        config.output_path,
        fallback_fps=config.fallback_fps,
        output_fourcc=config.output_fourcc,
    )
    runner = ONNXPoseHandRunner(config)
    smoother = LandmarkSmoother(config)
    osc_sender = OSCSender(config.osc_host, config.osc_port, config.osc_enabled)
    pipeline = PoseHandPipeline(config, runner, smoother, osc_sender)
    motion_frames: list[dict[str, object]] = []
    frame_index = 0
    fps_meter = FpsMeter("single", config.fps_log_interval)
    preview_sink = PreviewFrameSink()
    started_at = time.perf_counter()

    try:
        while True:
            if config.benchmark_frames and frame_index >= config.benchmark_frames:
                break
            ok, frame = session.read()
            if not ok or frame is None:
                break

            body_points, hands_by_side = pipeline.detect_pose(frame)
            joint_depths = pipeline.last_joint_depths if config.single_camera_depth_mode == "mediapipe" else None
            joints = build_joint_map(body_points, hands_by_side, joint_depths=joint_depths)
            pipeline.render_pose(frame, body_points, hands_by_side, send_osc=False)
            osc_sender.send_pose(
                body_points,
                hands_by_side,
                joints=joints,
                metadata={
                    "frame_index": frame_index,
                    "mode": "single",
                    "profile": config.profile,
                    "body_backend": config.body_backend,
                    "hand_backend": config.hand_backend,
                    "mediapipe_pose_model": config.mediapipe_pose_model,
                    "source": str(config.video_path),
                },
            )
            motion_frames.append({"frame_index": frame_index, "people": [{"id": 1, "joints": joints}]})
            fps_meter.tick(frame_index)
            draw_fps_overlay(frame, fps_meter, config.fps_overlay_enabled)
            preview_sink.write(frame, frame_index)
            frame_index += 1
    finally:
        session.close()
        osc_sender.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    export_metadata = {
        "mediapipe_pose_model": config.mediapipe_pose_model,
        "source": str(config.video_path),
        "body_model_variant": config.body_model_variant,
        "hand_model_variant": config.hand_model_variant,
    }
    export_motion_bundle(
        config,
        fps=session.fps,
        frames=motion_frames,
        metadata=export_metadata,
    )
    write_run_metadata(
        config.metadata_output_path,
        build_run_metadata(
            config,
            mode="single",
            fps=session.fps,
            frame_count=len(motion_frames),
            extra=export_metadata,
        ),
    )
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    log_info(f"Processed {len(motion_frames)} frames in {elapsed:.2f}s ({len(motion_frames) / elapsed:.2f} FPS)")
    print_saved_paths(config.output_path, config.json_output_path, config.fbx_output_path, config.metadata_output_path)
    return True
