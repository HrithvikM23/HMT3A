from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2

from camera.capture import VideoInputSource, VideoOutputWriter
from core.cli import InputAssignment
from core.config import PipelineConfig
from core.runtime_config import build_fused_config, prepare_runtime_config
from inference.rtmpose import ONNXPoseHandRunner
from network.osc_sender import OSCSender
from pipeline.pipeline import PoseHandPipeline
from runners.common import (
    box_from_body_points,
    build_person_payload,
    draw_person_overlay,
    export_motion_bundle,
    fuse_smoothed_hands,
    print_saved_paths,
)
from runners.fused_alignment import align_people_across_cameras
from utils.exports import build_joint_map, export_multi_person_fbx_bundle, export_multi_person_json
from utils.fps import FpsMeter, draw_fps_overlay
from utils.fusion import estimate_joint_depths, fuse_body_views, load_camera_calibrations
from utils.logging import log_info, log_warning
from utils.motion_cleanup import cleanup_multi_person_frames
from utils.multi_person import MultiPersonTracker
from utils.payloads import HandPayload, PersonPayload
from utils.preview_stream import PreviewFrameSink
from utils.run_metadata import build_run_metadata, write_run_metadata
from utils.smoothing import LandmarkSmoother
from utils.triangulation import (
    apply_triangulated_overrides,
    calibrated_backend_available,
    export_freemocap_style_output,
    triangulate_observation_frames,
    triangulation_metadata,
)


def _build_fused_metadata(
    mode: str,
    assignments: list[InputAssignment],
    config: PipelineConfig,
) -> dict[str, object]:
    return {
        "mode": mode,
        "profile": config.profile,
        "body_backend": config.body_backend,
        "hand_backend": config.hand_backend,
        "mediapipe_pose_model": config.mediapipe_pose_model,
        "camera_labels": [assignment.label for assignment in assignments],
        "sources": {assignment.label: str(assignment.source) for assignment in assignments},
        "body_model_variant": config.body_model_variant,
        "hand_model_variant": config.hand_model_variant,
        "camera_calibration_path": None if config.camera_calibration_path is None else str(config.camera_calibration_path),
        "calibration_3d_path": None if config.calibration_3d_path is None else str(config.calibration_3d_path),
        "triangulation_3d": {"enabled": False},
    }


def run_fused_assignments(
    assignments: list[InputAssignment],
    args: argparse.Namespace,
) -> bool:
    try:
        config = build_fused_config(args, assignments)
    except OSError as exc:
        print(f"Error: could not prepare output paths: {exc}")
        return False
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return False
    if not prepare_runtime_config(config):
        return False

    try:
        calibrations = load_camera_calibrations(config.camera_calibration_path)
    except Exception as exc:
        print(f"Error: failed to load camera calibration: {exc}")
        return False

    sources: dict[str, VideoInputSource] = {}
    osc_sender = OSCSender(config.osc_host, config.osc_port, config.osc_enabled)
    writer: VideoOutputWriter | None = None
    finished = False
    motion_frames: list[dict[str, object]] = []
    triangulation_frames: list[dict[str, object]] = []
    frame_index = 0
    export_fps = config.fallback_fps
    fps_meter = FpsMeter("fused", config.fps_log_interval)
    preview_sinks: dict[str, PreviewFrameSink] = {}
    for cam_index, assignment in enumerate(assignments):
        preview_sinks[assignment.label] = PreviewFrameSink(worker_index=cam_index)
    started_at = time.perf_counter()
    try:
        min_offset = min((config.sync_offsets.get(assignment.label.upper(), 0) for assignment in assignments), default=0)
        for assignment in assignments:
            sources[assignment.label] = VideoInputSource(assignment.source, fallback_fps=config.fallback_fps)
            sources[assignment.label].skip_frames(config.sync_offsets.get(assignment.label.upper(), 0) - min_offset)

        reference_label = next(iter(sources))
        reference_source = sources[reference_label]
        export_fps = reference_source.fps
        assert config.output_path is not None
        writer = VideoOutputWriter(
            config.output_path,
            frame_width=reference_source.frame_width,
            frame_height=reference_source.frame_height,
            fps=reference_source.fps,
            output_fourcc=config.output_fourcc,
        )

        runner = ONNXPoseHandRunner(config)
        single_view_pipelines = {
            label: PoseHandPipeline(config, runner, LandmarkSmoother(config), OSCSender(enabled=False))
            for label in sources
        }
        multi_person_trackers = {
            label: MultiPersonTracker(config, runner)
            for label in sources
        }
        fused_renderers: dict[str, PoseHandPipeline] = {}

        while True:
            if config.benchmark_frames and frame_index >= config.benchmark_frames:
                break
            frames_by_label: dict[str, Any] = {}
            # NOTE: Lockstep reading assumes identical frame rates across all cameras. Frame rate drift between cameras (e.g., 29.97 vs 30.0 FPS) will cause progressive desync over long captures.
            for label, source in sources.items():
                ok, frame = source.read()
                if not ok or frame is None:
                    finished = True
                    break
                frames_by_label[label] = frame
            if finished:
                break

            if frame_index > 0 and frame_index % 1000 == 0:
                ref_frame_idx = sources[reference_label].cap.get(cv2.CAP_PROP_POS_FRAMES)
                for label, source in sources.items():
                    if label == reference_label:
                        continue
                    curr_frame_idx = source.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if abs(curr_frame_idx - ref_frame_idx) > 1:
                        log_warning(f"Frame drift detected: {label} is at frame {curr_frame_idx}, reference {reference_label} is at {ref_frame_idx}")

            canvas = frames_by_label[reference_label].copy()
            if config.max_people > 1:
                _run_fused_multi_person_frame(
                    config,
                    runner,
                    frames_by_label,
                    reference_label,
                    multi_person_trackers,
                    fused_renderers,
                    canvas,
                    calibrations,
                    osc_sender,
                    motion_frames,
                    triangulation_frames,
                    frame_index,
                    preview_sinks,
                )
            else:
                _run_fused_single_frame(
                    config,
                    runner,
                    frames_by_label,
                    reference_label,
                    single_view_pipelines,
                    fused_renderers,
                    canvas,
                    calibrations,
                    osc_sender,
                    motion_frames,
                    triangulation_frames,
                    frame_index,
                    preview_sinks,
                )

            frame_index += 1
            fps_meter.tick(frame_index - 1)
            draw_fps_overlay(canvas, fps_meter, config.fps_overlay_enabled)
            # Write fused canvas to reference camera's preview sink
            if reference_label in preview_sinks:
                preview_sinks[reference_label].write(canvas, frame_index - 1)
            writer.write(canvas)

            if config.enable_preview:
                cv2.imshow(config.preview_window_title, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        for source in sources.values():
            source.close()
        if writer is not None:
            writer.close()
        osc_sender.close()
        for sink in preview_sinks.values():
            if hasattr(sink, 'close'):
                sink.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    if config.max_people > 1:
        metadata = _build_fused_metadata("fused_multi_person", assignments, config)
        if config.enable_3d_triangulation:
            motion_frames, metadata = _apply_multi_person_triangulation(
                config,
                motion_frames,
                triangulation_frames,
                metadata,
            )
        cleaned_motion_frames = cleanup_multi_person_frames(motion_frames, config)
        export_multi_person_json(
            config.json_output_path,
            fps=export_fps,
            frames=cleaned_motion_frames,
            metadata=metadata,
        )
        exported_fbx_paths = export_multi_person_fbx_bundle(config.fbx_output_path, export_fps, cleaned_motion_frames)
        write_run_metadata(
            config.metadata_output_path,
            build_run_metadata(
                config,
                mode="fused_multi_person",
                fps=export_fps,
                frame_count=len(cleaned_motion_frames),
                extra=metadata,
            ),
        )
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        log_info(
            f"Processed {len(cleaned_motion_frames)} fused frames in {elapsed:.2f}s "
            f"({len(cleaned_motion_frames) / elapsed:.2f} FPS)"
        )
        print_saved_paths(config.output_path, config.json_output_path, *exported_fbx_paths, config.metadata_output_path)
        return True

    metadata = _build_fused_metadata("fused", assignments, config)
    if config.enable_3d_triangulation:
        try:
            motion_frames, metadata = _apply_calibrated_triangulation(config, motion_frames, triangulation_frames, metadata)
        except RuntimeError as exc:
            print(f"Error: calibrated 3D export failed: {exc}")
            return False

    export_motion_bundle(
        config,
        fps=export_fps,
        frames=motion_frames,
        metadata=metadata,
    )
    write_run_metadata(
        config.metadata_output_path,
        build_run_metadata(
            config,
            mode="fused",
            fps=export_fps,
            frame_count=len(motion_frames),
            extra=metadata,
        ),
    )
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    log_info(f"Processed {len(motion_frames)} fused frames in {elapsed:.2f}s ({len(motion_frames) / elapsed:.2f} FPS)")
    print_saved_paths(config.output_path, config.json_output_path, config.fbx_output_path, config.metadata_output_path)
    return True


def _apply_calibrated_triangulation(
    config: PipelineConfig,
    motion_frames: list[dict[str, object]],
    triangulation_frames: list[dict[str, object]],
    metadata: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if config.calibration_3d_path is None:
        raise RuntimeError("--triangulate-3d requires --calibration-3d.")
    if not calibrated_backend_available():
        raise RuntimeError("calibrated 3D backend is not installed.")

    try:
        result = triangulate_observation_frames(
            config.calibration_3d_path,
            triangulation_frames,
            body_threshold=config.body_conf_threshold,
            hand_threshold=config.hand_kp_threshold,
            minimum_cameras=config.triangulation_min_cameras,
            use_outlier_rejection=config.triangulation_use_outlier_rejection,
            maximum_cameras_to_drop=config.triangulation_max_cameras_to_drop,
            target_reprojection_error=config.triangulation_reprojection_error,
            max_reprojection_error=config.triangulation_max_error,
            smoothing_alpha=config.triangulation_smoothing_alpha,
        )
    except Exception as exc:
        raise RuntimeError(f"triangulation failed: {exc}") from exc

    updated_metadata = dict(metadata)
    updated_metadata["triangulation_3d"] = triangulation_metadata(result)
    freemocap_output_root = _freemocap_output_root(config)
    updated_metadata["freemocap_style_output"] = export_freemocap_style_output(freemocap_output_root, result)
    return apply_triangulated_overrides(motion_frames, result), updated_metadata


def _freemocap_output_root(config: PipelineConfig) -> Path:
    assert config.output_path is not None
    return config.output_path.with_name(f"{config.output_path.stem} freemocap")


def _apply_multi_person_triangulation(
    config: PipelineConfig,
    motion_frames: list[dict[str, object]],
    triangulation_frames: list[dict[str, object]],
    metadata: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if config.calibration_3d_path is None or not calibrated_backend_available():
        return motion_frames, metadata

    people_observations: dict[str, list[dict[str, object]]] = {}
    for frame in triangulation_frames:
        people = frame.get("people")
        if not isinstance(people, list):
            continue
        for person in people:
            if not isinstance(person, dict):
                continue
            label = str(person["label"])
            people_observations.setdefault(label, []).append(person)

    updated_frames = motion_frames
    triangulation_people: dict[str, object] = {}
    for label, observations in people_observations.items():
        try:
            result = triangulate_observation_frames(
                config.calibration_3d_path,
                observations,
                body_threshold=config.body_conf_threshold,
                hand_threshold=config.hand_kp_threshold,
                minimum_cameras=config.triangulation_min_cameras,
                use_outlier_rejection=config.triangulation_use_outlier_rejection,
                maximum_cameras_to_drop=config.triangulation_max_cameras_to_drop,
                target_reprojection_error=config.triangulation_reprojection_error,
                max_reprojection_error=config.triangulation_max_error,
                smoothing_alpha=config.triangulation_smoothing_alpha,
            )
        except Exception as exc:
            triangulation_people[label] = {"enabled": False, "error": str(exc)}
            continue
        updated_frames = _apply_person_overrides(updated_frames, label, result)
        triangulation_people[label] = triangulation_metadata(result)

    updated_metadata = dict(metadata)
    updated_metadata["triangulation_3d"] = {
        "enabled": True,
        "people": triangulation_people,
    }
    return updated_frames, updated_metadata


def _apply_person_overrides(
    motion_frames: list[dict[str, object]],
    label: str,
    result,
) -> list[dict[str, object]]:
    updated_frames: list[dict[str, object]] = []
    override_frames = result.joint_overrides_by_frame
    for frame_index, frame in enumerate(motion_frames):
        people = frame.get("people")
        if not isinstance(people, list) or frame_index >= len(override_frames):
            updated_frames.append(frame)
            continue
        overrides = override_frames[frame_index]
        updated_people: list[dict[str, object]] = []
        for person in people:
            if not isinstance(person, dict) or person.get("label") != label:
                updated_people.append(person)
                continue
            joints = person.get("joints")
            if not isinstance(joints, dict):
                updated_people.append(person)
                continue
            updated_joints = {name: dict(value) for name, value in joints.items()}
            for joint_name, override in overrides.items():
                if joint_name in updated_joints:
                    updated_joints[joint_name] = dict(override)
            updated_person = dict(person)
            updated_person["joints"] = updated_joints
            updated_people.append(updated_person)
        updated_frame = dict(frame)
        updated_frame["people"] = updated_people
        updated_frames.append(updated_frame)
    return updated_frames


def _run_fused_multi_person_frame(
    config: PipelineConfig,
    runner: ONNXPoseHandRunner,
    frames_by_label: dict[str, Any],
    reference_label: str,
    trackers: dict[str, MultiPersonTracker],
    renderers: dict[str, PoseHandPipeline],
    canvas,
    calibrations: dict[str, dict[str, float]],
    osc_sender: OSCSender,
    motion_frames: list[dict[str, object]],
    triangulation_frames: list[dict[str, object]],
    frame_index: int,
    preview_sinks: dict[str, PreviewFrameSink] | None = None,
) -> None:
    camera_tracks = {
        label: trackers[label].update(frame)
        for label, frame in frames_by_label.items()
    }

    # Write per-camera preview frames with per-camera track overlays
    if preview_sinks:
        for label, tracks in camera_tracks.items():
            if label == reference_label or label not in preview_sinks:
                continue
            cam_frame = frames_by_label[label].copy()
            for track in tracks:
                track.pipeline.render_pose(cam_frame, track.body_points, track.hands_by_side, send_osc=False)
                track_label = track.label or f"person{track.id}"
                draw_person_overlay(cam_frame, track_label, track.box, track.detection_score)
            preview_sinks[label].write(cam_frame, frame_index)

    grouped_people = align_people_across_cameras(camera_tracks, reference_label)
    payload_people: list[PersonPayload] = []

    for person_index, (person_key, views) in enumerate(grouped_people.items(), start=1):
        camera_bodies = {
            label: track.body_points
            for label, track in views.items()
            if track.body_points
        }
        camera_hands = {
            label: track.hands_by_side
            for label, track in views.items()
        }
        if not camera_bodies:
            continue

        renderer = renderers.setdefault(
            person_key,
            PoseHandPipeline(config, runner, LandmarkSmoother(config), OSCSender(enabled=False)),
        )
        fused_body = fuse_body_views(
            camera_bodies,
            config.body_conf_threshold,
            reference_label=reference_label,
            calibrations=calibrations,
        )
        if fused_body is not None:
            fused_body = renderer.smoother.smooth_body(fused_body)
        if fused_body is None:
            continue

        fused_hands = fuse_smoothed_hands(renderer, camera_hands, config, reference_label, calibrations=calibrations)
        renderer.render_pose(canvas, fused_body, fused_hands, send_osc=False)

        label = next((track.label for track in views.values() if track.label), person_key)
        box = box_from_body_points(fused_body, config.body_conf_threshold)
        best_score = max(track.detection_score for track in views.values())
        if box is not None:
            draw_person_overlay(canvas, label, box, best_score)

        joint_depths = estimate_joint_depths(
            camera_bodies=camera_bodies,
            camera_hands=camera_hands,
            body_threshold=config.body_conf_threshold,
            hand_threshold=config.hand_kp_threshold,
            calibrations=calibrations,
            depth_scale=config.fused_depth_scale,
        )
        payload_people.append(
            build_person_payload(
                person_id=person_index,
                label=label,
                box=box,
                score=best_score,
                body_points=fused_body,
                hands_by_side=fused_hands,
                joint_depths=joint_depths,
                camera_views=sorted(views),
            )
        )

    osc_sender.send_people(
        payload_people,
        metadata={
            "frame_index": frame_index,
            "mode": "fused_multi_person",
            "profile": config.profile,
            "body_backend": config.body_backend,
            "hand_backend": config.hand_backend,
            "mediapipe_pose_model": config.mediapipe_pose_model,
            "camera_labels": list(frames_by_label),
        },
    )
    motion_frames.append({"frame_index": frame_index, "people": payload_people})
    triangulation_frames.append(
        {
            "frame_index": frame_index,
            "people": [
                {
                    "label": next((track.label for track in views.values() if track.label), person_key),
                    "camera_bodies": {
                        label: track.body_points
                        for label, track in views.items()
                        if track.body_points
                    },
                    "camera_hands": {
                        label: track.hands_by_side
                        for label, track in views.items()
                    },
                }
                for person_key, views in grouped_people.items()
            ],
        }
    )


def _run_fused_single_frame(
    config: PipelineConfig,
    runner: ONNXPoseHandRunner,
    frames_by_label: dict[str, Any],
    reference_label: str,
    pipelines: dict[str, PoseHandPipeline],
    renderers: dict[str, PoseHandPipeline],
    canvas,
    calibrations: dict[str, dict[str, float]],
    osc_sender: OSCSender,
    motion_frames: list[dict[str, object]],
    triangulation_frames: list[dict[str, object]],
    frame_index: int,
    preview_sinks: dict[str, PreviewFrameSink] | None = None,
) -> None:
    camera_bodies: dict[str, list[tuple[int, int, float]]] = {}
    camera_hands: dict[str, dict[str, HandPayload]] = {}
    for label, frame in frames_by_label.items():
        body_points, hands_by_side = pipelines[label].detect_pose(frame)
        camera_bodies[label] = body_points
        camera_hands[label] = hands_by_side

        # Write per-camera preview frames with per-camera detection overlays
        if preview_sinks and label != reference_label and label in preview_sinks:
            cam_frame = frame.copy()
            pipelines[label].render_pose(cam_frame, body_points, hands_by_side, send_osc=False)
            preview_sinks[label].write(cam_frame, frame_index)

    renderer = renderers.setdefault(
        "single",
        PoseHandPipeline(config, runner, LandmarkSmoother(config), OSCSender(enabled=False)),
    )
    fused_body = fuse_body_views(
        camera_bodies,
        config.body_conf_threshold,
        reference_label=reference_label,
        calibrations=calibrations,
    )
    if fused_body is not None:
        fused_body = renderer.smoother.smooth_body(fused_body)
    if fused_body is None:
        fused_body = [(0, 0, 0.0) for _ in range(17)]

    fused_hands = fuse_smoothed_hands(renderer, camera_hands, config, reference_label, calibrations=calibrations)
    joint_depths = estimate_joint_depths(
        camera_bodies=camera_bodies,
        camera_hands=camera_hands,
        body_threshold=config.body_conf_threshold,
        hand_threshold=config.hand_kp_threshold,
        calibrations=calibrations,
        depth_scale=config.fused_depth_scale,
    )
    joints = build_joint_map(fused_body, fused_hands, joint_depths=joint_depths)
    renderer.render_pose(canvas, fused_body, fused_hands, send_osc=False)
    osc_sender.send_pose(
        fused_body,
        fused_hands,
        joints=joints,
        metadata={
            "frame_index": frame_index,
            "mode": "fused",
            "profile": config.profile,
            "body_backend": config.body_backend,
            "hand_backend": config.hand_backend,
            "mediapipe_pose_model": config.mediapipe_pose_model,
            "camera_labels": list(frames_by_label),
        },
    )
    motion_frames.append({"frame_index": frame_index, "joints": joints})
    triangulation_frames.append(
        {
            "frame_index": frame_index,
            "camera_bodies": camera_bodies,
            "camera_hands": camera_hands,
        }
    )
