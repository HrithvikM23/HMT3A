from __future__ import annotations

import argparse
from typing import Any

import cv2

from camera.capture import VideoInputSource, VideoOutputWriter
from cli import InputAssignment
from config import PipelineConfig
from inference.rtmpose import ONNXPoseHandRunner
from network.osc_sender import OSCSender
from pipeline.pipeline import PoseHandPipeline
from runtime_config import build_fused_config, prepare_runtime_config
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
from utils.fps import FpsMeter
from utils.fusion import estimate_joint_depths, fuse_body_views, load_camera_calibrations
from utils.motion_cleanup import cleanup_multi_person_frames
from utils.multi_person import MultiPersonTracker
from utils.smoothing import LandmarkSmoother
from utils.triangulation import (
    calibrated_backend_available,
    apply_triangulated_overrides,
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
) -> None:
    config = build_fused_config(args, assignments)
    if not prepare_runtime_config(config):
        return

    try:
        calibrations = load_camera_calibrations(config.camera_calibration_path)
    except Exception as exc:
        print(f"Error: failed to load camera calibration: {exc}")
        return

    sources: dict[str, VideoInputSource] = {}
    osc_sender = OSCSender(config.osc_host, config.osc_port, config.osc_enabled)
    writer: VideoOutputWriter | None = None
    finished = False
    motion_frames: list[dict[str, object]] = []
    triangulation_frames: list[dict[str, object]] = []
    frame_index = 0
    export_fps = config.fallback_fps
    fps_meter = FpsMeter("fused", config.fps_log_interval)
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
            frames_by_label: dict[str, Any] = {}
            for label, source in sources.items():
                ok, frame = source.read()
                if not ok or frame is None:
                    finished = True
                    break
                frames_by_label[label] = frame
            if finished:
                break

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
                )

            frame_index += 1
            fps_meter.tick(frame_index - 1)
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
        cv2.destroyAllWindows()

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
        print_saved_paths(config.output_path, config.json_output_path, *exported_fbx_paths)
        return

    metadata = _build_fused_metadata("fused", assignments, config)
    if config.enable_3d_triangulation:
        motion_frames, metadata = _apply_calibrated_triangulation(config, motion_frames, triangulation_frames, metadata)

    export_motion_bundle(
        config,
        fps=export_fps,
        frames=motion_frames,
        metadata=metadata,
    )
    print_saved_paths(config.output_path, config.json_output_path, config.fbx_output_path)


def _apply_calibrated_triangulation(
    config: PipelineConfig,
    motion_frames: list[dict[str, object]],
    triangulation_frames: list[dict[str, object]],
    metadata: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if config.calibration_3d_path is None:
        return motion_frames, metadata
    if not calibrated_backend_available():
        print("Warning: calibrated 3D backend is not installed; keeping heuristic fused depth.")
        return motion_frames, metadata

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
        print(f"Warning: calibrated 3D triangulation failed; keeping heuristic fused depth. Details: {exc}")
        return motion_frames, metadata

    updated_metadata = dict(metadata)
    updated_metadata["triangulation_3d"] = triangulation_metadata(result)
    return apply_triangulated_overrides(motion_frames, result), updated_metadata


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
        for person in frame.get("people", []):
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
) -> None:
    camera_tracks = {
        label: trackers[label].update(frame)
        for label, frame in frames_by_label.items()
    }
    grouped_people = align_people_across_cameras(camera_tracks, reference_label)
    payload_people: list[dict[str, object]] = []

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
        fused_body = fuse_body_views(camera_bodies, config.body_conf_threshold, reference_label=reference_label)
        if fused_body is not None:
            fused_body = renderer.smoother.smooth_body(fused_body)
        if fused_body is None:
            continue

        fused_hands = fuse_smoothed_hands(renderer, camera_hands, config, reference_label)
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
) -> None:
    camera_bodies: dict[str, list[tuple[int, int, float]]] = {}
    camera_hands: dict[str, dict[str, dict]] = {}
    for label, frame in frames_by_label.items():
        body_points, hands_by_side = pipelines[label].detect_pose(frame)
        camera_bodies[label] = body_points
        camera_hands[label] = hands_by_side

    renderer = renderers.setdefault(
        "single",
        PoseHandPipeline(config, runner, LandmarkSmoother(config), OSCSender(enabled=False)),
    )
    fused_body = fuse_body_views(camera_bodies, config.body_conf_threshold, reference_label=reference_label)
    if fused_body is not None:
        fused_body = renderer.smoother.smooth_body(fused_body)
    if fused_body is None:
        fused_body = [(0, 0, 0.0) for _ in range(17)]

    fused_hands = fuse_smoothed_hands(renderer, camera_hands, config, reference_label)
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
