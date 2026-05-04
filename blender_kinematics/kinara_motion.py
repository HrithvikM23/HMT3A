from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import cast


JointValue = dict[str, float]
JointMap = dict[str, JointValue]

DEFAULT_CONFIDENCE_THRESHOLD = 0.05
MIN_BONE_LENGTH = 0.05


@dataclass(frozen=True, slots=True)
class SkeletonJoint:
    name: str
    parent: str | None


@dataclass(slots=True)
class PersonMotion:
    label: str
    frame_indices: list[int]
    frames: list[JointMap]
    rest_joints: JointMap


@dataclass(slots=True)
class MotionClip:
    source_path: Path
    format_name: str
    fps: float
    frame_count: int
    skeleton: tuple[SkeletonJoint, ...]
    coordinate_system: dict[str, object]
    people: list[PersonMotion]
    metadata: dict[str, object]


DEFAULT_SKELETON: tuple[SkeletonJoint, ...] = (
    SkeletonJoint("HipsRoot", None),
    SkeletonJoint("LeftHip", "HipsRoot"),
    SkeletonJoint("LeftKnee", "LeftHip"),
    SkeletonJoint("LeftAnkle", "LeftKnee"),
    SkeletonJoint("LeftFoot", "LeftAnkle"),
    SkeletonJoint("LeftToeBase", "LeftFoot"),
    SkeletonJoint("RightHip", "HipsRoot"),
    SkeletonJoint("RightKnee", "RightHip"),
    SkeletonJoint("RightAnkle", "RightKnee"),
    SkeletonJoint("RightFoot", "RightAnkle"),
    SkeletonJoint("RightToeBase", "RightFoot"),
    SkeletonJoint("Chest", "HipsRoot"),
    SkeletonJoint("Neck", "Chest"),
    SkeletonJoint("Head", "Neck"),
    SkeletonJoint("LeftShoulder", "Chest"),
    SkeletonJoint("LeftElbow", "LeftShoulder"),
    SkeletonJoint("LeftWrist", "LeftElbow"),
    SkeletonJoint("LeftThumb1", "LeftWrist"),
    SkeletonJoint("LeftThumb2", "LeftThumb1"),
    SkeletonJoint("LeftThumb3", "LeftThumb2"),
    SkeletonJoint("LeftThumb4", "LeftThumb3"),
    SkeletonJoint("LeftIndex1", "LeftWrist"),
    SkeletonJoint("LeftIndex2", "LeftIndex1"),
    SkeletonJoint("LeftIndex3", "LeftIndex2"),
    SkeletonJoint("LeftIndex4", "LeftIndex3"),
    SkeletonJoint("LeftMiddle1", "LeftWrist"),
    SkeletonJoint("LeftMiddle2", "LeftMiddle1"),
    SkeletonJoint("LeftMiddle3", "LeftMiddle2"),
    SkeletonJoint("LeftMiddle4", "LeftMiddle3"),
    SkeletonJoint("LeftRing1", "LeftWrist"),
    SkeletonJoint("LeftRing2", "LeftRing1"),
    SkeletonJoint("LeftRing3", "LeftRing2"),
    SkeletonJoint("LeftRing4", "LeftRing3"),
    SkeletonJoint("LeftPinky1", "LeftWrist"),
    SkeletonJoint("LeftPinky2", "LeftPinky1"),
    SkeletonJoint("LeftPinky3", "LeftPinky2"),
    SkeletonJoint("LeftPinky4", "LeftPinky3"),
    SkeletonJoint("RightShoulder", "Chest"),
    SkeletonJoint("RightElbow", "RightShoulder"),
    SkeletonJoint("RightWrist", "RightElbow"),
    SkeletonJoint("RightThumb1", "RightWrist"),
    SkeletonJoint("RightThumb2", "RightThumb1"),
    SkeletonJoint("RightThumb3", "RightThumb2"),
    SkeletonJoint("RightThumb4", "RightThumb3"),
    SkeletonJoint("RightIndex1", "RightWrist"),
    SkeletonJoint("RightIndex2", "RightIndex1"),
    SkeletonJoint("RightIndex3", "RightIndex2"),
    SkeletonJoint("RightIndex4", "RightIndex3"),
    SkeletonJoint("RightMiddle1", "RightWrist"),
    SkeletonJoint("RightMiddle2", "RightMiddle1"),
    SkeletonJoint("RightMiddle3", "RightMiddle2"),
    SkeletonJoint("RightMiddle4", "RightMiddle3"),
    SkeletonJoint("RightRing1", "RightWrist"),
    SkeletonJoint("RightRing2", "RightRing1"),
    SkeletonJoint("RightRing3", "RightRing2"),
    SkeletonJoint("RightRing4", "RightRing3"),
    SkeletonJoint("RightPinky1", "RightWrist"),
    SkeletonJoint("RightPinky2", "RightPinky1"),
    SkeletonJoint("RightPinky3", "RightPinky2"),
    SkeletonJoint("RightPinky4", "RightPinky3"),
)

DEFAULT_COORDINATE_SYSTEM: dict[str, object] = {
    "space": "kinara_normalized",
    "right_axis": "X",
    "forward_axis": "Y",
    "up_axis": "Z",
    "grounded_axis": "Z",
}


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "person"


def empty_joint_map(skeleton: tuple[SkeletonJoint, ...]) -> JointMap:
    return {
        joint.name: {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "confidence": 0.0,
        }
        for joint in skeleton
    }


def joint_is_valid(joint_value: dict[str, object] | None, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> bool:
    if not isinstance(joint_value, dict):
        return False
    try:
        confidence = float(joint_value.get("confidence", 0.0))
        x_value = float(joint_value.get("x", 0.0))
        y_value = float(joint_value.get("y", 0.0))
        z_value = float(joint_value.get("z", 0.0))
    except (TypeError, ValueError):
        return False
    if confidence < confidence_threshold:
        return False
    return not (math.isclose(x_value, 0.0) and math.isclose(y_value, 0.0) and math.isclose(z_value, 0.0))


def _coerce_joint_value(joint_value: dict[str, object] | None) -> JointValue:
    if not isinstance(joint_value, dict):
        return {"x": 0.0, "y": 0.0, "z": 0.0, "confidence": 0.0}
    return {
        "x": float(joint_value.get("x", 0.0)),
        "y": float(joint_value.get("y", 0.0)),
        "z": float(joint_value.get("z", 0.0)),
        "confidence": max(0.0, float(joint_value.get("confidence", 0.0))),
    }


def _coerce_joint_map(value: object, skeleton: tuple[SkeletonJoint, ...]) -> JointMap:
    if not isinstance(value, dict):
        return empty_joint_map(skeleton)
    joint_map = empty_joint_map(skeleton)
    for joint in skeleton:
        joint_map[joint.name] = _coerce_joint_value(cast(dict[str, object], value.get(joint.name)))
    return joint_map


def _normalize_frame_index(value: object, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return fallback
    return fallback


def _extract_skeleton(metadata: dict[str, object]) -> tuple[SkeletonJoint, ...]:
    raw_skeleton = metadata.get("skeleton")
    if not isinstance(raw_skeleton, list):
        return DEFAULT_SKELETON

    skeleton: list[SkeletonJoint] = []
    for raw_joint in raw_skeleton:
        if not isinstance(raw_joint, dict):
            continue
        name = str(raw_joint.get("name", "")).strip()
        if not name:
            continue
        parent_value = raw_joint.get("parent")
        parent = None if parent_value in (None, "") else str(parent_value)
        skeleton.append(SkeletonJoint(name=name, parent=parent))
    return tuple(skeleton) or DEFAULT_SKELETON


def _default_bone_direction(joint_name: str) -> tuple[float, float, float]:
    if "Head" in joint_name or "Neck" in joint_name or joint_name == "Chest":
        return (0.0, 0.0, 1.0)
    if "Shoulder" in joint_name or "Elbow" in joint_name or "Wrist" in joint_name:
        return (-1.0, 0.0, 0.0) if joint_name.startswith("Left") else (1.0, 0.0, 0.0)
    if "Thumb" in joint_name or "Index" in joint_name or "Middle" in joint_name or "Ring" in joint_name or "Pinky" in joint_name:
        return (-1.0, 0.0, 0.0) if joint_name.startswith("Left") else (1.0, 0.0, 0.0)
    if "Hip" in joint_name:
        return (-0.35, 0.0, -1.0) if joint_name.startswith("Left") else (0.35, 0.0, -1.0)
    if "Knee" in joint_name or "Ankle" in joint_name:
        return (0.0, 0.0, -1.0)
    if "Foot" in joint_name or "Toe" in joint_name:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def _scaled_direction(direction: tuple[float, float, float], length: float = MIN_BONE_LENGTH) -> tuple[float, float, float]:
    dx, dy, dz = direction
    magnitude = math.sqrt((dx * dx) + (dy * dy) + (dz * dz))
    if magnitude <= 1e-8:
        return (0.0, 0.0, length)
    scale = length / magnitude
    return (dx * scale, dy * scale, dz * scale)


def compute_rest_joints(
    frames: list[JointMap],
    skeleton: tuple[SkeletonJoint, ...],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> JointMap:
    if not frames:
        return empty_joint_map(skeleton)

    root_positions: list[tuple[float, float, float]] = []
    root_confidences: list[float] = []
    for frame in frames:
        root_joint = frame.get("HipsRoot")
        if not joint_is_valid(root_joint, confidence_threshold):
            continue
        typed_root = cast(JointValue, root_joint)
        root_positions.append((typed_root["x"], typed_root["y"], typed_root["z"]))
        root_confidences.append(typed_root["confidence"])

    if root_positions:
        root_joint: JointValue = {
            "x": float(median(position[0] for position in root_positions)),
            "y": float(median(position[1] for position in root_positions)),
            "z": float(median(position[2] for position in root_positions)),
            "confidence": float(median(root_confidences)),
        }
    else:
        root_joint = {"x": 0.0, "y": 0.0, "z": 0.0, "confidence": 0.0}

    rest_joints: JointMap = {"HipsRoot": root_joint}

    for joint in skeleton:
        if joint.parent is None:
            continue

        deltas_x: list[float] = []
        deltas_y: list[float] = []
        deltas_z: list[float] = []
        confidences: list[float] = []

        for frame in frames:
            parent_value = frame.get(joint.parent)
            child_value = frame.get(joint.name)
            if not joint_is_valid(parent_value, confidence_threshold) or not joint_is_valid(child_value, confidence_threshold):
                continue
            typed_parent = cast(JointValue, parent_value)
            typed_child = cast(JointValue, child_value)
            deltas_x.append(typed_child["x"] - typed_parent["x"])
            deltas_y.append(typed_child["y"] - typed_parent["y"])
            deltas_z.append(typed_child["z"] - typed_parent["z"])
            confidences.append(min(typed_parent["confidence"], typed_child["confidence"]))

        parent_rest = rest_joints[joint.parent]
        if deltas_x and deltas_y and deltas_z:
            delta = (
                float(median(deltas_x)),
                float(median(deltas_y)),
                float(median(deltas_z)),
            )
            confidence = float(median(confidences))
        else:
            delta = _scaled_direction(_default_bone_direction(joint.name))
            confidence = 0.0

        length = math.sqrt((delta[0] * delta[0]) + (delta[1] * delta[1]) + (delta[2] * delta[2]))
        if length < MIN_BONE_LENGTH:
            delta = _scaled_direction(_default_bone_direction(joint.name))

        rest_joints[joint.name] = {
            "x": parent_rest["x"] + delta[0],
            "y": parent_rest["y"] + delta[1],
            "z": parent_rest["z"] + delta[2],
            "confidence": confidence,
        }

    return {joint.name: rest_joints.get(joint.name, _coerce_joint_value(None)) for joint in skeleton}


def build_held_joint_frames(
    frames: list[JointMap],
    skeleton: tuple[SkeletonJoint, ...],
    rest_joints: JointMap,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[JointMap]:
    held_frames: list[JointMap] = []
    last_valid = {
        joint.name: dict(rest_joints.get(joint.name, _coerce_joint_value(None)))
        for joint in skeleton
    }

    for frame in frames:
        held_frame: JointMap = {}
        for joint in skeleton:
            current = frame.get(joint.name)
            if joint_is_valid(current, confidence_threshold):
                held_value = dict(cast(JointValue, current))
                last_valid[joint.name] = held_value
            else:
                held_value = dict(last_valid[joint.name])
            held_frame[joint.name] = held_value
        held_frames.append(held_frame)

    return held_frames


def load_motion_clip(input_path: str | Path, person_filter: str = "all") -> MotionClip:
    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Kinara motion file must be a JSON object.")

    metadata = cast(dict[str, object], payload.get("metadata", {}))
    skeleton = _extract_skeleton(metadata)
    coordinate_system = cast(dict[str, object], metadata.get("coordinate_system", DEFAULT_COORDINATE_SYSTEM))
    format_name = str(payload.get("format", ""))
    fps = float(payload.get("fps", 0.0))
    raw_frames = cast(list[dict[str, object]], payload.get("frames", []))

    people: list[PersonMotion] = []
    requested_person = person_filter.strip() if person_filter else "all"

    if format_name == "kinara-motion-json-v1":
        joint_frames: list[JointMap] = []
        frame_indices: list[int] = []
        for fallback_index, frame in enumerate(raw_frames):
            joint_frames.append(_coerce_joint_map(frame.get("joints"), skeleton))
            frame_indices.append(_normalize_frame_index(frame.get("frame_index"), fallback_index))

        rest_joints = _coerce_joint_map(metadata.get("rest_joints"), skeleton)
        if not any(joint_is_valid(value, 0.0) for value in rest_joints.values()):
            rest_joints = compute_rest_joints(joint_frames, skeleton)

        people.append(
            PersonMotion(
                label=sanitize_label(path.stem),
                frame_indices=frame_indices,
                frames=joint_frames,
                rest_joints=rest_joints,
            )
        )
    elif format_name == "kinara-multi-person-json-v1":
        tracks: dict[str, list[JointMap]] = {}
        track_indices: dict[str, list[int]] = {}

        for fallback_index, frame in enumerate(raw_frames):
            frame_index = _normalize_frame_index(frame.get("frame_index"), fallback_index)
            keyed_people: dict[str, dict[str, object]] = {}
            raw_people = frame.get("people", [])
            if isinstance(raw_people, list):
                for person_index, raw_person in enumerate(raw_people, start=1):
                    if not isinstance(raw_person, dict):
                        continue
                    label = str(raw_person.get("label") or f"person{raw_person.get('id', person_index)}")
                    key = sanitize_label(label)
                    keyed_people[key] = cast(dict[str, object], raw_person)

            for person_key in set(tracks) | set(keyed_people):
                person_payload = keyed_people.get(person_key)
                joint_map = _coerce_joint_map(None if person_payload is None else person_payload.get("joints"), skeleton)
                tracks.setdefault(person_key, []).append(joint_map)
                track_indices.setdefault(person_key, []).append(frame_index)

        raw_rest_joints = metadata.get("rest_joints")
        typed_rest_joints = cast(dict[str, object], raw_rest_joints) if isinstance(raw_rest_joints, dict) else {}

        for person_key, joint_frames in tracks.items():
            if requested_person.lower() != "all" and sanitize_label(requested_person) != person_key:
                continue

            rest_payload = typed_rest_joints.get(person_key)
            rest_joints = _coerce_joint_map(rest_payload, skeleton)
            if not any(joint_is_valid(value, 0.0) for value in rest_joints.values()):
                rest_joints = compute_rest_joints(joint_frames, skeleton)

            people.append(
                PersonMotion(
                    label=person_key,
                    frame_indices=track_indices[person_key],
                    frames=joint_frames,
                    rest_joints=rest_joints,
                )
            )
    else:
        raise ValueError(
            "Unsupported Kinara motion format. Expected "
            "'kinara-motion-json-v1' or 'kinara-multi-person-json-v1'."
        )

    if not people:
        raise ValueError("No matching animated people were found in the Kinara motion file.")

    return MotionClip(
        source_path=path,
        format_name=format_name,
        fps=fps,
        frame_count=int(payload.get("frame_count", len(raw_frames) if raw_frames else 0)),
        skeleton=skeleton,
        coordinate_system=coordinate_system,
        people=people,
        metadata=metadata,
    )
