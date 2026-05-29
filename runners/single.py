from __future__ import annotations

import cv2

from camera.capture import VideoCaptureSession
from core.config import PipelineConfig
from inference.rtmpose import ONNXPoseHandRunner
from network.osc_sender import OSCSender
from pipeline.pipeline import PoseHandPipeline
from core.runtime_config import prepare_runtime_config
from runners.common import export_motion_bundle, print_saved_paths
from runners.multi_person import run_multi_person_assignment
from utils.exports import build_joint_map
from utils.fps import FpsMeter, draw_fps_overlay
from utils.preview_stream import PreviewFrameSink
from utils.smoothing import LandmarkSmoother


def run_assignment(config: PipelineConfig) -> None:
    if config.max_people > 1:
        run_multi_person_assignment(config)
        return
    if not prepare_runtime_config(config):
        return

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

    try:
        while True:
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
            motion_frames.append({"frame_index": frame_index, "joints": joints})
            fps_meter.tick(frame_index)
            draw_fps_overlay(frame, fps_meter, config.fps_overlay_enabled)
            preview_sink.write(frame, frame_index)
            frame_index += 1
            session.write(frame)

            if config.enable_preview:
                cv2.imshow(config.preview_window_title, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        session.close()
        osc_sender.close()
        cv2.destroyAllWindows()

    export_motion_bundle(
        config,
        fps=session.fps,
        frames=motion_frames,
        metadata={
            "mode": "single",
            "profile": config.profile,
            "body_backend": config.body_backend,
            "hand_backend": config.hand_backend,
            "mediapipe_pose_model": config.mediapipe_pose_model,
            "source": str(config.video_path),
            "body_model_variant": config.body_model_variant,
            "hand_model_variant": config.hand_model_variant,
        },
    )
    print_saved_paths(config.output_path, config.json_output_path, config.fbx_output_path)
