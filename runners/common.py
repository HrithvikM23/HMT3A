from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2

from config import PipelineConfig
from pipeline.pipeline import PoseHandPipeline
from utils.exports import build_joint_map, export_motion_fbx, export_motion_json, _normalize_export_frames
from utils.fusion import fuse_hand_views
from utils.motion_cleanup import cleanup_motion_frames
from utils.payloads import HandPayload, PersonPayload
from utils.skeleton import Point


def export_motion_bundle(
    config: PipelineConfig,
    fps: float,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    if not frames:
        return
    frames = cleanup_motion_frames(frames, config)
    normalized_frames = _normalize_export_frames(frames)
    export_motion_json(config.json_output_path, fps, normalized_frames, metadata, frames_are_normalized=True)
    export_motion_fbx(config.fbx_output_path, fps, normalized_frames, frames_are_normalized=True)


def build_person_payload(
    person_id: int,
    label: str,
    body_points: list[Point],
    hands_by_side: Mapping[str, HandPayload],
    joint_depths: dict[str, float] | None = None,
    box: tuple[int, int, int, int] | None = None,
    score: float | None = None,
    camera_views: list[str] | None = None,
) -> PersonPayload:
    return {
        "id": person_id,
        "label": label,
        "box": box,
        "score": score,
        "camera_views": camera_views or [],
        "body_points": body_points,
        "hands_by_side": dict(hands_by_side),
        "joint_depths": joint_depths or {},
        "joints": build_joint_map(body_points, hands_by_side, joint_depths=joint_depths),
    }


def draw_person_overlay(frame, track_label: str, box: tuple[int, int, int, int], score: float | None = None) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
    label_text = track_label if score is None else f"{track_label} {score:.2f}"
    cv2.putText(frame, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def box_from_body_points(points: list[Point], threshold: float) -> tuple[int, int, int, int] | None:
    confident_points = [(x, y) for x, y, conf in points if conf > threshold]
    if not confident_points:
        return None
    xs = [point[0] for point in confident_points]
    ys = [point[1] for point in confident_points]
    return min(xs), min(ys), max(xs), max(ys)


def fuse_smoothed_hands(
    renderer: PoseHandPipeline,
    camera_hands: Mapping[str, Mapping[str, HandPayload]],
    config: PipelineConfig,
    reference_label: str,
) -> dict[str, HandPayload]:
    fused_hands: dict[str, HandPayload] = {}
    for side in ("left", "right"):
        side_views = {
            label: hands_by_side[side]
            for label, hands_by_side in camera_hands.items()
            if side in hands_by_side
        }
        fused_hand = fuse_hand_views(side_views, config.hand_kp_threshold, reference_label=reference_label)
        if fused_hand is None:
            continue
        smoothed_points = renderer.smoother.smooth_hand(side, fused_hand["points"])
        if smoothed_points is None:
            continue
        fused_hands[side] = {"box": fused_hand["box"], "points": smoothed_points}
    return fused_hands


def print_saved_paths(*paths: Path | None) -> None:
    for path in paths:
        if path is not None:
            print(f"Saved: {path}")
