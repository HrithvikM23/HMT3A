from __future__ import annotations

import cv2

from camera.capture import VideoCaptureSession
from config import PipelineConfig
from inference.rtmpose import ONNXPoseHandRunner
from network.osc_sender import OSCSender
from runtime_config import prepare_runtime_config
from runners.common import build_person_payload, draw_person_overlay, print_saved_paths
from utils.exports import export_multi_person_fbx_bundle, export_multi_person_json
from utils.fps import FpsMeter
from utils.multi_person import MultiPersonTracker


def run_multi_person_assignment(config: PipelineConfig) -> None:
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
    osc_sender = OSCSender(config.osc_host, config.osc_port, config.osc_enabled)
    tracker = MultiPersonTracker(config, runner)
    motion_frames: list[dict[str, object]] = []
    frame_index = 0
    fps_meter = FpsMeter("multi_person", config.fps_log_interval)

    try:
        while True:
            ok, frame = session.read()
            if not ok or frame is None:
                break

            payload_people: list[dict[str, object]] = []
            for track in tracker.update(frame):
                track.pipeline.render_pose(frame, track.body_points, track.hands_by_side, send_osc=False)
                label = track.label or f"person{track.id}"
                draw_person_overlay(frame, label, track.box, track.detection_score)
                payload_people.append(
                    build_person_payload(
                        person_id=track.id,
                        label=label,
                        box=track.box,
                        score=track.detection_score,
                        body_points=track.body_points,
                        hands_by_side=track.hands_by_side,
                        camera_views=["FRONT"],
                    )
                )

            osc_sender.send_people(
                payload_people,
                metadata={
                    "frame_index": frame_index,
                    "mode": "multi_person",
                    "source": str(config.video_path),
                },
            )
            motion_frames.append({"frame_index": frame_index, "people": payload_people})
            fps_meter.tick(frame_index)
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

    export_multi_person_json(
        config.json_output_path,
        fps=session.fps,
        frames=motion_frames,
        metadata={
            "mode": "multi_person",
            "source": str(config.video_path),
            "max_people": config.max_people,
            "identity_hints": {key: list(value) for key, value in config.identity_hints.items()},
        },
    )
    exported_fbx_paths = export_multi_person_fbx_bundle(config.fbx_output_path, session.fps, motion_frames)
    print_saved_paths(config.output_path, config.json_output_path, *exported_fbx_paths)
