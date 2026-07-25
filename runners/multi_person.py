from __future__ import annotations

import time

import cv2

from camera.capture import VideoCaptureSession
from core.config import PipelineConfig
from core.runtime_config import prepare_runtime_config
from inference.rtmpose import ONNXPoseHandRunner
from network.osc_sender import OSCSender
from runners.common import build_person_payload, draw_person_overlay, print_saved_paths
from utils.exports import export_multi_person_fbx_bundle, export_multi_person_json
from utils.fps import FpsMeter, draw_fps_overlay
from utils.logging import log_info
from utils.motion_cleanup import cleanup_multi_person_frames
from utils.multi_person import MultiPersonTracker
from utils.payloads import PersonPayload
from utils.preview_stream import PreviewFrameSink
from utils.run_metadata import build_run_metadata, write_run_metadata


def run_multi_person_assignment(config: PipelineConfig) -> bool:
    if not prepare_runtime_config(config):
        return False

    assert config.output_path is not None
    session = VideoCaptureSession(
        config.video_path,
        config.output_path,
        fallback_fps=config.fallback_fps,
        output_fourcc=config.output_fourcc,
    )
    runner = ONNXPoseHandRunner(config)
    osc_sender = OSCSender(config.osc_host, config.osc_port, config.osc_enabled)
    tracker = MultiPersonTracker(config, runner)
    motion_frames: list[dict[str, object]] = []
    frame_index = 0
    fps_meter = FpsMeter("multi_person", config.fps_log_interval)
    preview_sink = PreviewFrameSink()
    started_at = time.perf_counter()

    try:
        while True:
            if config.benchmark_frames and frame_index >= config.benchmark_frames:
                break
            ok, frame = session.read()
            if not ok or frame is None:
                break

            payload_people: list[PersonPayload] = []
            for track in tracker.update(frame):
                track.pipeline.render_pose(frame, track.body_points, track.hands_by_side, send_osc=False)
                label = track.label or f"person{track.id}"
                draw_person_overlay(frame, label, track.box, track.detection_score)
                joint_depths = track.joint_depths if config.single_camera_depth_mode == "mediapipe" else None
                payload_people.append(
                    build_person_payload(
                        person_id=track.id,
                        label=label,
                        box=track.box,
                        score=track.detection_score,
                        body_points=track.body_points,
                        hands_by_side=track.hands_by_side,
                        joint_depths=joint_depths,
                        camera_views=["CAM_0"],
                    )
                )

            osc_sender.send_people(
                payload_people,
                metadata={
                    "frame_index": frame_index,
                    "mode": "multi_person",
                    "profile": config.profile,
                    "body_backend": config.body_backend,
                    "hand_backend": config.hand_backend,
                    "mediapipe_pose_model": config.mediapipe_pose_model,
                    "source": str(config.video_path),
                },
            )
            motion_frames.append({"frame_index": frame_index, "people": payload_people})
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
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    cleaned_motion_frames = cleanup_multi_person_frames(motion_frames, config)
    export_metadata = {
        "mode": "multi_person",
        "profile": config.profile,
        "body_backend": config.body_backend,
        "hand_backend": config.hand_backend,
        "mediapipe_pose_model": config.mediapipe_pose_model,
        "source": str(config.video_path),
        "max_people": config.max_people,
        "identity_hints": {key: list(value) for key, value in config.identity_hints.items()},
    }
    export_multi_person_json(
        config.json_output_path,
        fps=session.fps,
        frames=cleaned_motion_frames,
        metadata=export_metadata,
    )
    exported_fbx_paths = export_multi_person_fbx_bundle(config.fbx_output_path, session.fps, cleaned_motion_frames)
    write_run_metadata(
        config.metadata_output_path,
        build_run_metadata(
            config,
            mode="multi_person",
            fps=session.fps,
            frame_count=len(cleaned_motion_frames),
            extra=export_metadata,
        ),
    )
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    log_info(f"Processed {len(cleaned_motion_frames)} frames in {elapsed:.2f}s ({len(cleaned_motion_frames) / elapsed:.2f} FPS)")
    print_saved_paths(config.output_path, config.json_output_path, *exported_fbx_paths, config.metadata_output_path)
    return True
