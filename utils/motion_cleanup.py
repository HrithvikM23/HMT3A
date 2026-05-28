from __future__ import annotations

import math
from typing import cast

from utils.skeleton import JointMap, JointValue


FOOT_LOCK_JOINTS = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")


def cleanup_motion_frames(frames: list[dict[str, object]], config) -> list[dict[str, object]]:
    if not getattr(config, "export_cleanup_enabled", True):
        return frames
    cleaned = _copy_frames(frames)
    _interpolate_missing(cleaned)
    _remove_velocity_spikes(cleaned, float(config.export_cleanup_max_velocity))
    _smooth_frames(cleaned, float(config.export_cleanup_smoothing_alpha))
    if getattr(config, "export_foot_lock_enabled", True):
        _lock_planted_feet(
            cleaned,
            velocity_threshold=float(config.export_foot_lock_velocity),
            max_lift=float(config.export_foot_lock_max_lift),
        )
    return cleaned


def cleanup_multi_person_frames(frames: list[dict[str, object]], config) -> list[dict[str, object]]:
    cleaned_frames = [dict(frame) for frame in frames]
    person_keys = _collect_person_keys(cleaned_frames)
    cleaned_by_key: dict[str, list[dict[str, object]]] = {}

    for person_key in person_keys:
        person_track: list[dict[str, object]] = []
        for frame in cleaned_frames:
            person = _find_person(frame, person_key)
            person_track.append(
                {
                    "frame_index": frame.get("frame_index", 0),
                    "joints": {} if person is None else person.get("joints", {}),
                }
            )
        cleaned_by_key[person_key] = cleanup_motion_frames(person_track, config)

    for frame_index, frame in enumerate(cleaned_frames):
        people = frame.get("people", [])
        if not isinstance(people, list):
            continue
        updated_people = []
        for person in people:
            if not isinstance(person, dict):
                continue
            person_key = _person_key(person)
            updated_person = dict(person)
            updated_person["joints"] = cast(JointMap, cleaned_by_key[person_key][frame_index]["joints"])
            updated_people.append(updated_person)
        frame["people"] = updated_people
    return cleaned_frames


def _copy_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    copied_frames: list[dict[str, object]] = []
    for frame in frames:
        copied_frame = dict(frame)
        joints = frame.get("joints")
        if isinstance(joints, dict):
            copied_frame["joints"] = {
                name: _copy_joint(cast(JointValue, value))
                for name, value in joints.items()
                if isinstance(value, dict)
            }
        copied_frames.append(copied_frame)
    return copied_frames


def _copy_joint(joint: JointValue) -> JointValue:
    return {
        "x": float(joint.get("x", 0.0)),
        "y": float(joint.get("y", 0.0)),
        "z": float(joint.get("z", 0.0)),
        "confidence": float(joint.get("confidence", 0.0)),
    }


def _joint_valid(joint: JointValue | None) -> bool:
    if joint is None or float(joint.get("confidence", 0.0)) <= 0.05:
        return False
    return any(abs(float(joint.get(axis, 0.0))) > 1e-6 for axis in ("x", "y", "z"))


def _distance(a: JointValue, b: JointValue) -> float:
    return math.sqrt(
        (float(a["x"]) - float(b["x"])) ** 2
        + (float(a["y"]) - float(b["y"])) ** 2
        + (float(a["z"]) - float(b["z"])) ** 2
    )


def _blend_joint(a: JointValue, b: JointValue, alpha: float) -> JointValue:
    return {
        "x": float(a["x"]) * (1.0 - alpha) + float(b["x"]) * alpha,
        "y": float(a["y"]) * (1.0 - alpha) + float(b["y"]) * alpha,
        "z": float(a["z"]) * (1.0 - alpha) + float(b["z"]) * alpha,
        "confidence": max(float(a["confidence"]), float(b["confidence"])),
    }


def _interpolate_missing(frames: list[dict[str, object]]) -> None:
    joint_names = _joint_names(frames)
    for joint_name in joint_names:
        valid_indices = [
            index
            for index, frame in enumerate(frames)
            if _joint_valid(_joint(frame, joint_name))
        ]
        if len(valid_indices) < 2:
            continue
        for left_index, right_index in zip(valid_indices, valid_indices[1:]):
            gap = right_index - left_index
            if gap <= 1:
                continue
            left_joint = _joint(frames[left_index], joint_name)
            right_joint = _joint(frames[right_index], joint_name)
            if left_joint is None or right_joint is None:
                continue
            for offset in range(1, gap):
                alpha = offset / gap
                _set_joint(frames[left_index + offset], joint_name, _blend_joint(left_joint, right_joint, alpha))


def _remove_velocity_spikes(frames: list[dict[str, object]], max_velocity: float) -> None:
    if max_velocity <= 0:
        return
    for joint_name in _joint_names(frames):
        for index in range(1, len(frames) - 1):
            previous = _joint(frames[index - 1], joint_name)
            current = _joint(frames[index], joint_name)
            following = _joint(frames[index + 1], joint_name)
            if not (_joint_valid(previous) and _joint_valid(current) and _joint_valid(following)):
                continue
            assert previous is not None and current is not None and following is not None
            if _distance(previous, current) <= max_velocity or _distance(current, following) <= max_velocity:
                continue
            repaired = _blend_joint(previous, following, 0.5)
            repaired["confidence"] = min(float(current["confidence"]), float(repaired["confidence"]))
            _set_joint(frames[index], joint_name, repaired)


def _smooth_frames(frames: list[dict[str, object]], alpha: float) -> None:
    alpha = max(0.0, min(alpha, 1.0))
    if alpha >= 1.0:
        return
    previous_by_joint: dict[str, JointValue] = {}
    for frame in frames:
        joints = frame.get("joints")
        if not isinstance(joints, dict):
            continue
        for joint_name, joint in list(joints.items()):
            typed_joint = cast(JointValue, joint)
            if not _joint_valid(typed_joint):
                continue
            previous = previous_by_joint.get(joint_name)
            if previous is None:
                previous_by_joint[joint_name] = _copy_joint(typed_joint)
                continue
            smoothed = _blend_joint(previous, typed_joint, alpha)
            joints[joint_name] = smoothed
            previous_by_joint[joint_name] = smoothed


def _lock_planted_feet(frames: list[dict[str, object]], velocity_threshold: float, max_lift: float) -> None:
    ground_y = _ground_y(frames)
    if ground_y is None:
        return
    locked: dict[str, JointValue] = {}
    for frame in frames:
        for joint_name in FOOT_LOCK_JOINTS:
            joint = _joint(frame, joint_name)
            if not _joint_valid(joint):
                locked.pop(joint_name, None)
                continue
            assert joint is not None
            previous_locked = locked.get(joint_name)
            near_ground = abs(float(joint["y"]) - ground_y) <= max_lift
            slow = previous_locked is not None and _distance(previous_locked, joint) <= velocity_threshold
            if previous_locked is not None and near_ground and slow:
                _set_joint(frame, joint_name, _copy_joint(previous_locked))
                continue
            if near_ground:
                locked[joint_name] = _copy_joint(joint)
            else:
                locked.pop(joint_name, None)


def _ground_y(frames: list[dict[str, object]]) -> float | None:
    values: list[float] = []
    for frame in frames:
        for joint_name in FOOT_LOCK_JOINTS:
            joint = _joint(frame, joint_name)
            if _joint_valid(joint):
                assert joint is not None
                values.append(float(joint["y"]))
    return min(values) if values else None


def _joint(frame: dict[str, object], joint_name: str) -> JointValue | None:
    joints = frame.get("joints")
    if not isinstance(joints, dict):
        return None
    joint = joints.get(joint_name)
    if not isinstance(joint, dict):
        return None
    return cast(JointValue, joint)


def _set_joint(frame: dict[str, object], joint_name: str, joint: JointValue) -> None:
    joints = frame.setdefault("joints", {})
    if isinstance(joints, dict):
        joints[joint_name] = joint


def _joint_names(frames: list[dict[str, object]]) -> set[str]:
    names: set[str] = set()
    for frame in frames:
        joints = frame.get("joints")
        if isinstance(joints, dict):
            names.update(str(name) for name in joints)
    return names


def _collect_person_keys(frames: list[dict[str, object]]) -> set[str]:
    keys: set[str] = set()
    for frame in frames:
        people = frame.get("people", [])
        if not isinstance(people, list):
            continue
        for person in people:
            if isinstance(person, dict):
                keys.add(_person_key(person))
    return keys


def _find_person(frame: dict[str, object], person_key: str) -> dict[str, object] | None:
    people = frame.get("people", [])
    if not isinstance(people, list):
        return None
    for person in people:
        if isinstance(person, dict) and _person_key(person) == person_key:
            return person
    return None


def _person_key(person: dict[str, object]) -> str:
    return str(person.get("label") or person.get("id") or "person")
