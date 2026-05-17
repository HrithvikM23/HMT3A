from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import cast

from utils.body_geometry import derive_foot_points
from utils.skeleton import (
    BODY_NAME_TO_INDEX,
    DEFAULT_EXPORT_COORDINATE_SYSTEM,
    FACE_POINT_INDICES,
    FBX_TIME_UNIT,
    HAND_NAME_TO_INDEX,
    SKELETON,
    JointMap,
    JointValue,
    Point,
)


def _to_world(x: int, y: int, z: float = 0.0) -> tuple[float, float, float]:
    return float(x), float(-y), float(z)


def _to_world_float(x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
    return float(x), float(-y), float(z)


def _average_points(points: list[Point]) -> tuple[float, float, float, float]:
    point_count = len(points)
    sum_x = 0
    sum_y = 0
    sum_conf = 0.0
    for x, y, conf in points:
        sum_x += x
        sum_y += y
        sum_conf += conf
    x, y, z = _to_world(int(round(sum_x / point_count)), int(round(sum_y / point_count)))
    return x, y, z, float(sum_conf / point_count)


def _average_screen_points(points: list[Point]) -> tuple[float, float, float]:
    point_count = max(len(points), 1)
    sum_x = 0.0
    sum_y = 0.0
    sum_conf = 0.0
    for x, y, conf in points:
        sum_x += float(x)
        sum_y += float(y)
        sum_conf += float(conf)
    return sum_x / point_count, sum_y / point_count, sum_conf / point_count


def _make_joint(x: float, y: float, z: float, confidence: float) -> JointValue:
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "confidence": max(0.0, float(confidence)),
    }


def _zero_joint() -> JointValue:
    return {"x": 0.0, "y": 0.0, "z": 0.0, "confidence": 0.0}


def _copy_joint_value(joint_value: JointValue) -> JointValue:
    return {
        "x": float(joint_value["x"]),
        "y": float(joint_value["y"]),
        "z": float(joint_value["z"]),
        "confidence": float(joint_value["confidence"]),
    }


def _lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha


def _derive_head_joints(body_points: list[Point], joint_depths: dict[str, float]) -> tuple[JointValue, JointValue]:
    shoulders = [body_points[5], body_points[6]]
    hips = [body_points[11], body_points[12]]
    shoulder_x, shoulder_y, shoulder_conf = _average_screen_points(shoulders)
    hip_x, hip_y, hip_conf = _average_screen_points(hips)
    face_points = [body_points[index] for index in FACE_POINT_INDICES if body_points[index][2] > 0.0]

    if face_points:
        face_x, face_y, face_conf = _average_screen_points(face_points)
        neck_x = _lerp(shoulder_x, face_x, 0.35)
        neck_y = _lerp(shoulder_y, face_y, 0.35)
        head_x = _lerp(shoulder_x, face_x, 0.85)
        head_y = _lerp(shoulder_y, face_y, 0.85)
        derived_conf = (shoulder_conf + face_conf) * 0.5
    else:
        torso_dx = shoulder_x - hip_x
        torso_dy = shoulder_y - hip_y
        torso_length = math.hypot(torso_dx, torso_dy)
        if torso_length <= 1e-6:
            unit_x, unit_y = 0.0, -1.0
            torso_length = 32.0
        else:
            unit_x = torso_dx / torso_length
            unit_y = torso_dy / torso_length
        neck_x = shoulder_x
        neck_y = shoulder_y
        head_x = shoulder_x + unit_x * max(torso_length * 0.35, 18.0)
        head_y = shoulder_y + unit_y * max(torso_length * 0.35, 18.0)
        derived_conf = (shoulder_conf + hip_conf) * 0.5

    shoulder_depth = float((joint_depths.get("LeftShoulder", 0.0) + joint_depths.get("RightShoulder", 0.0)) * 0.5)
    head_depth = shoulder_depth
    neck_world = _to_world_float(neck_x, neck_y, shoulder_depth)
    head_world = _to_world_float(head_x, head_y, head_depth)
    return (
        _make_joint(neck_world[0], neck_world[1], neck_world[2], derived_conf),
        _make_joint(head_world[0], head_world[1], head_world[2], derived_conf),
    )


def _derive_foot_chain(
    knee_point: Point,
    ankle_point: Point,
    foot_depth: float,
) -> tuple[JointValue, JointValue]:
    foot_point, toe_point = derive_foot_points(knee_point, ankle_point)
    foot_x, foot_y, derived_conf = foot_point
    toe_x, toe_y, _ = toe_point

    foot_world = _to_world_float(foot_x, foot_y, foot_depth)
    toe_world = _to_world_float(toe_x, toe_y, foot_depth)
    return (
        _make_joint(foot_world[0], foot_world[1], foot_world[2], derived_conf),
        _make_joint(toe_world[0], toe_world[1], toe_world[2], derived_conf),
    )


def build_joint_map(
    body_points: list[Point],
    hands_by_side: dict[str, dict[str, object]],
    joint_depths: dict[str, float] | None = None,
) -> JointMap:
    joint_map: JointMap = {joint.name: _zero_joint() for joint in SKELETON}
    joint_depths = joint_depths or {}

    for name, index in BODY_NAME_TO_INDEX.items():
        x, y, conf = body_points[index]
        wx, wy, wz = _to_world(x, y, joint_depths.get(name, 0.0))
        joint_map[name] = _make_joint(wx, wy, wz, conf)

    hips = [body_points[11], body_points[12]]
    shoulders = [body_points[5], body_points[6]]
    root_x, root_y, _, root_conf = _average_points(hips)
    chest_x, chest_y, _, chest_conf = _average_points(shoulders)
    root_z = float((joint_depths.get("LeftHip", 0.0) + joint_depths.get("RightHip", 0.0)) * 0.5)
    chest_z = float((joint_depths.get("LeftShoulder", 0.0) + joint_depths.get("RightShoulder", 0.0)) * 0.5)
    joint_map["HipsRoot"] = _make_joint(root_x, root_y, root_z, root_conf)
    joint_map["Chest"] = _make_joint(chest_x, chest_y, chest_z, chest_conf)
    joint_map["Neck"], joint_map["Head"] = _derive_head_joints(body_points, joint_depths)
    joint_map["LeftFoot"], joint_map["LeftToeBase"] = _derive_foot_chain(
        body_points[13],
        body_points[15],
        joint_depths.get("LeftAnkle", 0.0),
    )
    joint_map["RightFoot"], joint_map["RightToeBase"] = _derive_foot_chain(
        body_points[14],
        body_points[16],
        joint_depths.get("RightAnkle", 0.0),
    )

    for side_label, hand_payload in (("Left", hands_by_side.get("left")), ("Right", hands_by_side.get("right"))):
        if hand_payload is None:
            continue
        hand_points = cast(list[Point], hand_payload["points"])
        for suffix, index in HAND_NAME_TO_INDEX.items():
            x, y, conf = hand_points[index]
            wx, wy, wz = _to_world(x, y, joint_depths.get(f"{side_label}{suffix}", 0.0))
            joint_map[f"{side_label}{suffix}"] = _make_joint(wx, wy, wz, conf)

    return joint_map


def _localize_joint_map(joint_map: JointMap) -> JointMap:
    local_map: JointMap = {}
    for joint in SKELETON:
        current = joint_map[joint.name]
        if joint.parent is None:
            local_map[joint.name] = dict(current)
            continue
        parent = joint_map[joint.parent]
        local_map[joint.name] = {
            "x": current["x"] - parent["x"],
            "y": current["y"] - parent["y"],
            "z": current["z"] - parent["z"],
            "confidence": current["confidence"],
        }
    return local_map


def _frame_joint_map(frame: dict[str, object]) -> JointMap:
    return cast(JointMap, frame["joints"])


def _joint_is_valid(joint_value: JointValue, confidence_threshold: float = 0.05) -> bool:
    if joint_value["confidence"] < confidence_threshold:
        return False
    return not (
        math.isclose(joint_value["x"], 0.0)
        and math.isclose(joint_value["y"], 0.0)
        and math.isclose(joint_value["z"], 0.0)
    )


def _ground_joint_frames_on_axis(frames: list[dict[str, object]], axis: str) -> list[dict[str, object]]:
    min_value: float | None = None
    for frame in frames:
        joints = frame["joints"]
        if not isinstance(joints, dict):
            continue
        typed_joints = cast(JointMap, joints)
        for joint in typed_joints.values():
            if joint["confidence"] <= 0.0:
                continue
            joint_value = joint[axis]
            min_value = joint_value if min_value is None else min(min_value, joint_value)

    if min_value is None:
        return frames

    grounded_frames: list[dict[str, object]] = []
    for frame in frames:
        joints = frame["joints"]
        if not isinstance(joints, dict):
            grounded_frames.append(frame)
            continue

        grounded_joints: JointMap = {}
        typed_joints = cast(JointMap, joints)
        for name, joint in typed_joints.items():
            grounded_joint = {
                "x": joint["x"],
                "y": joint["y"],
                "z": joint["z"],
                "confidence": joint["confidence"],
            }
            grounded_joint[axis] = joint[axis] - min_value
            grounded_joints[name] = grounded_joint

        grounded_frame = dict(frame)
        grounded_frame["joints"] = grounded_joints
        grounded_frames.append(grounded_frame)

    return grounded_frames


def _ground_joint_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    return _ground_joint_frames_on_axis(frames, "y")


def _z_up_joint_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    grounded_frames = _ground_joint_frames(frames)
    z_up_frames: list[dict[str, object]] = []
    for frame in grounded_frames:
        joints = frame["joints"]
        if not isinstance(joints, dict):
            z_up_frames.append(frame)
            continue

        z_up_joints: dict[str, dict[str, float]] = {}
        for name, joint in joints.items():
            if not isinstance(joint, dict):
                continue
            z_up_joints[name] = {
                "x": float(joint["x"]),
                "y": float(joint["z"]),
                "z": float(joint["y"]),
                "confidence": float(joint["confidence"]),
            }

        z_up_frame = dict(frame)
        z_up_frame["joints"] = z_up_joints
        z_up_frames.append(z_up_frame)

    return z_up_frames


def _ground_z_axis_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    return _ground_joint_frames_on_axis(frames, "z")


def _normalize_export_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    return _ground_z_axis_frames(_z_up_joint_frames(frames))


def _build_skeleton_metadata() -> list[dict[str, str | None]]:
    return [{"name": joint.name, "parent": joint.parent} for joint in SKELETON]


def _compute_rest_joints(frames: list[dict[str, object]]) -> JointMap:
    if not frames:
        return _empty_joint_map()

    root_positions: list[tuple[float, float, float]] = []
    root_confidences: list[float] = []
    for frame in frames:
        joint_map = _frame_joint_map(frame)
        root_joint = joint_map.get("HipsRoot", _zero_joint())
        if not _joint_is_valid(root_joint):
            continue
        root_positions.append((root_joint["x"], root_joint["y"], root_joint["z"]))
        root_confidences.append(root_joint["confidence"])

    if root_positions:
        rest_joints: JointMap = {
            "HipsRoot": _make_joint(
                x=statistics.median(position[0] for position in root_positions),
                y=statistics.median(position[1] for position in root_positions),
                z=statistics.median(position[2] for position in root_positions),
                confidence=statistics.median(root_confidences),
            )
        }
    else:
        rest_joints = {"HipsRoot": _zero_joint()}

    for joint in SKELETON:
        if joint.parent is None:
            continue

        delta_xs: list[float] = []
        delta_ys: list[float] = []
        delta_zs: list[float] = []
        confidences: list[float] = []
        for frame in frames:
            joint_map = _frame_joint_map(frame)
            parent_joint = joint_map.get(joint.parent, _zero_joint())
            child_joint = joint_map.get(joint.name, _zero_joint())
            if not _joint_is_valid(parent_joint) or not _joint_is_valid(child_joint):
                continue
            delta_xs.append(child_joint["x"] - parent_joint["x"])
            delta_ys.append(child_joint["y"] - parent_joint["y"])
            delta_zs.append(child_joint["z"] - parent_joint["z"])
            confidences.append(min(parent_joint["confidence"], child_joint["confidence"]))

        if delta_xs and delta_ys and delta_zs:
            delta_x = float(statistics.median(delta_xs))
            delta_y = float(statistics.median(delta_ys))
            delta_z = float(statistics.median(delta_zs))
            delta_length = math.sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z))
            if delta_length <= 1e-6:
                delta_x, delta_y, delta_z = 0.0, 0.0, 0.05
            confidence = float(statistics.median(confidences))
        else:
            delta_x, delta_y, delta_z = 0.0, 0.0, 0.05
            confidence = 0.0

        parent_rest = rest_joints[joint.parent]
        rest_joints[joint.name] = _make_joint(
            parent_rest["x"] + delta_x,
            parent_rest["y"] + delta_y,
            parent_rest["z"] + delta_z,
            confidence,
        )

    return {joint.name: _copy_joint_value(rest_joints.get(joint.name, _zero_joint())) for joint in SKELETON}


def _collect_multi_person_joint_tracks(frames: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    person_frames: dict[str, list[dict[str, object]]] = {}

    for fallback_index, frame in enumerate(frames):
        frame_index = _coerce_frame_index(frame.get("frame_index"))
        present_people = frame.get("people", [])
        keyed_people: dict[str, JointMap] = {}
        if isinstance(present_people, list):
            for person_index, person in enumerate(present_people, start=1):
                if not isinstance(person, dict):
                    continue
                label = str(person.get("label") or f"person{person.get('id', person_index)}")
                person_key = _sanitize_person_label(label)
                joints = person.get("joints")
                if isinstance(joints, dict):
                    keyed_people[person_key] = cast(JointMap, joints)

        if frame_index == 0 and fallback_index != 0:
            frame_index = fallback_index
        for person_key in set(person_frames) | set(keyed_people):
            joint_map = keyed_people.get(person_key, _empty_joint_map())
            person_frames.setdefault(person_key, []).append(
                {
                    "frame_index": frame_index,
                    "joints": joint_map,
                }
            )

    return person_frames


def _normalize_multi_person_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    person_tracks = _collect_multi_person_joint_tracks(frames)
    normalized_tracks = {
        person_key: _normalize_export_frames(person_motion_frames)
        for person_key, person_motion_frames in person_tracks.items()
    }
    normalized_frames: list[dict[str, object]] = []

    for frame_offset, frame in enumerate(frames):
        normalized_people: list[dict[str, object]] = []
        raw_people = frame.get("people", [])
        if isinstance(raw_people, list):
            for person_index, person in enumerate(raw_people, start=1):
                if not isinstance(person, dict):
                    continue
                label = str(person.get("label") or f"person{person.get('id', person_index)}")
                person_key = _sanitize_person_label(label)
                normalized_person = dict(person)
                normalized_person["joints"] = _frame_joint_map(normalized_tracks[person_key][frame_offset])
                normalized_people.append(normalized_person)

        normalized_frame = dict(frame)
        normalized_frame["people"] = normalized_people
        normalized_frames.append(normalized_frame)

    return normalized_frames


def _build_export_metadata(
    metadata: dict[str, object],
    frames: list[dict[str, object]],
    *,
    multi_person: bool,
) -> dict[str, object]:
    enriched_metadata = dict(metadata)
    enriched_metadata["skeleton"] = _build_skeleton_metadata()
    enriched_metadata["coordinate_system"] = dict(DEFAULT_EXPORT_COORDINATE_SYSTEM)
    if multi_person:
        person_tracks = _collect_multi_person_joint_tracks(frames)
        enriched_metadata["rest_joints"] = {
            person_key: _compute_rest_joints(person_frames)
            for person_key, person_frames in person_tracks.items()
        }
    else:
        enriched_metadata["rest_joints"] = _compute_rest_joints(frames)
    return enriched_metadata


def _write_motion_json(
    output_path: Path,
    format_name: str,
    fps: float,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    payload = {
        "format": format_name,
        "fps": fps,
        "frame_count": len(frames),
        "metadata": metadata,
        "frames": frames,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def export_motion_json(
    output_path: Path,
    fps: float,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
    frames_are_normalized: bool = False,
) -> None:
    if not frames_are_normalized:
        frames = _normalize_export_frames(frames)
    _write_motion_json(
        output_path,
        "kinara-motion-json-v1",
        fps,
        frames,
        _build_export_metadata(metadata, frames, multi_person=False),
    )


def export_multi_person_json(
    output_path: Path,
    fps: float,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
    frames_are_normalized: bool = False,
) -> None:
    if not frames_are_normalized:
        frames = _normalize_multi_person_frames(frames)
    _write_motion_json(
        output_path,
        "kinara-multi-person-json-v1",
        fps,
        frames,
        _build_export_metadata(metadata, frames, multi_person=True),
    )


def _sanitize_person_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "person"


def _empty_joint_map() -> JointMap:
    return {joint.name: _zero_joint() for joint in SKELETON}


def _coerce_frame_index(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def export_multi_person_fbx_bundle(
    output_path: Path,
    fps: float,
    frames: list[dict[str, object]],
) -> list[Path]:
    if not frames:
        return []

    person_frames = _collect_multi_person_joint_tracks(frames)

    exported_paths: list[Path] = []
    suffix = output_path.suffix or ".fbx"
    stem = output_path.stem
    for person_key, person_motion_frames in person_frames.items():
        person_output_path = output_path.with_name(f"{stem}_{person_key}{suffix}")
        export_motion_fbx(person_output_path, fps, person_motion_frames)
        exported_paths.append(person_output_path)

    return exported_paths


def _fbx_template_header() -> list[str]:
    return [
        '; FBX 7.4.0 project file',
        'FBXHeaderExtension:  {',
        '  FBXHeaderVersion: 1003',
        '  FBXVersion: 7400',
        '  Creator: "Kinara"',
        '}',
        'GlobalSettings:  {',
        '  Version: 1000',
        '  Properties70:  {',
        '    P: "UpAxis", "int", "Integer", "",2',
        '    P: "UpAxisSign", "int", "Integer", "",1',
        '    P: "FrontAxis", "int", "Integer", "",2',
        '    P: "FrontAxisSign", "int", "Integer", "",1',
        '    P: "CoordAxis", "int", "Integer", "",0',
        '    P: "CoordAxisSign", "int", "Integer", "",1',
        '    P: "UnitScaleFactor", "double", "Number", "",1',
        '  }',
        '}',
    ]


def export_motion_fbx(output_path: Path, fps: float, frames: list[dict[str, object]], frames_are_normalized: bool = False) -> None:
    if not frames:
        return

    if not frames_are_normalized:
        frames = _normalize_export_frames(frames)
    local_frames = [_localize_joint_map(_frame_joint_map(frame)) for frame in frames]
    model_ids: dict[str, int] = {}
    curve_node_ids: dict[str, int] = {}
    curve_ids: dict[tuple[str, str], int] = {}
    animation_stack_id = 100000
    animation_layer_id = 100001
    next_id = 100100

    for joint in SKELETON:
        model_ids[joint.name] = next_id
        next_id += 1
    for joint in SKELETON:
        curve_node_ids[joint.name] = next_id
        next_id += 1
        for axis in ("X", "Y", "Z"):
            curve_ids[(joint.name, axis)] = next_id
            next_id += 1

    key_times = [int(round((frame_index / max(fps, 1.0)) * FBX_TIME_UNIT)) for frame_index in range(len(local_frames))]

    lines = _fbx_template_header()
    lines.append("Objects:  {")
    lines.append(f'  AnimationStack: {animation_stack_id}, "AnimStack::Take 001", "" {{')
    lines.append("    Properties70:  {")
    lines.append('      P: "LocalStart", "KTime", "Time", "",0')
    lines.append(f'      P: "LocalStop", "KTime", "Time", "",{key_times[-1] if key_times else 0}')
    lines.append("    }")
    lines.append("  }")
    lines.append(f'  AnimationLayer: {animation_layer_id}, "AnimLayer::BaseLayer", "" {{')
    lines.append("  }")

    for joint in SKELETON:
        model_id = model_ids[joint.name]
        lines.append(f'  Model: {model_id}, "Model::{joint.name}", "LimbNode" {{')
        lines.append("    Version: 232")
        lines.append("    Properties70:  {")
        lines.append('      P: "Lcl Translation", "Lcl Translation", "", "A",0,0,0')
        lines.append('      P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0')
        lines.append('      P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1')
        lines.append("    }")
        lines.append("    Shading: T")
        lines.append('    Culling: "CullingOff"')
        lines.append("  }")

    for joint in SKELETON:
        curve_node_id = curve_node_ids[joint.name]
        lines.append(f'  AnimationCurveNode: {curve_node_id}, "AnimCurveNode::{joint.name}_T", "" {{')
        lines.append("    Properties70:  {")
        lines.append('      P: "d|X", "Number", "", "A",0')
        lines.append('      P: "d|Y", "Number", "", "A",0')
        lines.append('      P: "d|Z", "Number", "", "A",0')
        lines.append("    }")
        lines.append("  }")

        for axis in ("X", "Y", "Z"):
            curve_id = curve_ids[(joint.name, axis)]
            values = [local_frame[joint.name][axis.lower()] for local_frame in local_frames]
            lines.append(f'  AnimationCurve: {curve_id}, "AnimCurve::{joint.name}_T_{axis}", "" {{')
            lines.append("    Default: 0")
            lines.append("    KeyVer: 4008")
            lines.append(f"    KeyTime: *{len(key_times)} {{")
            lines.append("      a: " + ",".join(str(value) for value in key_times))
            lines.append("    }")
            lines.append(f"    KeyValueFloat: *{len(values)} {{")
            lines.append("      a: " + ",".join(f"{value:.6f}" for value in values))
            lines.append("    }")
            lines.append(f"    KeyAttrFlags: *{len(values)} {{")
            lines.append("      a: " + ",".join("24836" for _ in values))
            lines.append("    }")
            lines.append(f"    KeyAttrDataFloat: *{len(values) * 4} {{")
            lines.append("      a: " + ",".join("0,0,255790911,0" for _ in values))
            lines.append("    }")
            lines.append(f"    KeyAttrRefCount: *{len(values)} {{")
            lines.append("      a: " + ",".join("1" for _ in values))
            lines.append("    }")
            lines.append("  }")

    lines.append("}")
    lines.append("Connections:  {")
    lines.append(f'  C: "OO",{animation_layer_id},{animation_stack_id}')
    for joint in SKELETON:
        parent_id = 0 if joint.parent is None else model_ids[joint.parent]
        lines.append(f'  C: "OO",{model_ids[joint.name]},{parent_id}')
        lines.append(f'  C: "OO",{curve_node_ids[joint.name]},{animation_layer_id}')
        lines.append(f'  C: "OP",{curve_node_ids[joint.name]},{model_ids[joint.name]},"Lcl Translation"')
        for axis in ("X", "Y", "Z"):
            lines.append(f'  C: "OP",{curve_ids[(joint.name, axis)]},{curve_node_ids[joint.name]},"d|{axis}"')
    lines.append("}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
