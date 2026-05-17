from __future__ import annotations

from dataclasses import dataclass


FBX_TIME_UNIT = 46186158000


@dataclass(frozen=True, slots=True)
class JointSpec:
    name: str
    parent: str | None


Point = tuple[int, int, float]
JointValue = dict[str, float]
JointMap = dict[str, JointValue]
FACE_POINT_INDICES = (0, 1, 2, 3, 4)


SKELETON: tuple[JointSpec, ...] = (
    JointSpec("HipsRoot", None),
    JointSpec("LeftHip", "HipsRoot"),
    JointSpec("LeftKnee", "LeftHip"),
    JointSpec("LeftAnkle", "LeftKnee"),
    JointSpec("LeftFoot", "LeftAnkle"),
    JointSpec("LeftToeBase", "LeftFoot"),
    JointSpec("RightHip", "HipsRoot"),
    JointSpec("RightKnee", "RightHip"),
    JointSpec("RightAnkle", "RightKnee"),
    JointSpec("RightFoot", "RightAnkle"),
    JointSpec("RightToeBase", "RightFoot"),
    JointSpec("Chest", "HipsRoot"),
    JointSpec("Neck", "Chest"),
    JointSpec("Head", "Neck"),
    JointSpec("LeftShoulder", "Chest"),
    JointSpec("LeftElbow", "LeftShoulder"),
    JointSpec("LeftWrist", "LeftElbow"),
    JointSpec("LeftThumb1", "LeftWrist"),
    JointSpec("LeftThumb2", "LeftThumb1"),
    JointSpec("LeftThumb3", "LeftThumb2"),
    JointSpec("LeftThumb4", "LeftThumb3"),
    JointSpec("LeftIndex1", "LeftWrist"),
    JointSpec("LeftIndex2", "LeftIndex1"),
    JointSpec("LeftIndex3", "LeftIndex2"),
    JointSpec("LeftIndex4", "LeftIndex3"),
    JointSpec("LeftMiddle1", "LeftWrist"),
    JointSpec("LeftMiddle2", "LeftMiddle1"),
    JointSpec("LeftMiddle3", "LeftMiddle2"),
    JointSpec("LeftMiddle4", "LeftMiddle3"),
    JointSpec("LeftRing1", "LeftWrist"),
    JointSpec("LeftRing2", "LeftRing1"),
    JointSpec("LeftRing3", "LeftRing2"),
    JointSpec("LeftRing4", "LeftRing3"),
    JointSpec("LeftPinky1", "LeftWrist"),
    JointSpec("LeftPinky2", "LeftPinky1"),
    JointSpec("LeftPinky3", "LeftPinky2"),
    JointSpec("LeftPinky4", "LeftPinky3"),
    JointSpec("RightShoulder", "Chest"),
    JointSpec("RightElbow", "RightShoulder"),
    JointSpec("RightWrist", "RightElbow"),
    JointSpec("RightThumb1", "RightWrist"),
    JointSpec("RightThumb2", "RightThumb1"),
    JointSpec("RightThumb3", "RightThumb2"),
    JointSpec("RightThumb4", "RightThumb3"),
    JointSpec("RightIndex1", "RightWrist"),
    JointSpec("RightIndex2", "RightIndex1"),
    JointSpec("RightIndex3", "RightIndex2"),
    JointSpec("RightIndex4", "RightIndex3"),
    JointSpec("RightMiddle1", "RightWrist"),
    JointSpec("RightMiddle2", "RightMiddle1"),
    JointSpec("RightMiddle3", "RightMiddle2"),
    JointSpec("RightMiddle4", "RightMiddle3"),
    JointSpec("RightRing1", "RightWrist"),
    JointSpec("RightRing2", "RightRing1"),
    JointSpec("RightRing3", "RightRing2"),
    JointSpec("RightRing4", "RightRing3"),
    JointSpec("RightPinky1", "RightWrist"),
    JointSpec("RightPinky2", "RightPinky1"),
    JointSpec("RightPinky3", "RightPinky2"),
    JointSpec("RightPinky4", "RightPinky3"),
)

BODY_NAME_TO_INDEX = {
    "LeftShoulder": 5,
    "RightShoulder": 6,
    "LeftElbow": 7,
    "RightElbow": 8,
    "LeftWrist": 9,
    "RightWrist": 10,
    "LeftHip": 11,
    "RightHip": 12,
    "LeftKnee": 13,
    "RightKnee": 14,
    "LeftAnkle": 15,
    "RightAnkle": 16,
}

HAND_NAME_TO_INDEX = {
    "Thumb1": 1,
    "Thumb2": 2,
    "Thumb3": 3,
    "Thumb4": 4,
    "Index1": 5,
    "Index2": 6,
    "Index3": 7,
    "Index4": 8,
    "Middle1": 9,
    "Middle2": 10,
    "Middle3": 11,
    "Middle4": 12,
    "Ring1": 13,
    "Ring2": 14,
    "Ring3": 15,
    "Ring4": 16,
    "Pinky1": 17,
    "Pinky2": 18,
    "Pinky3": 19,
    "Pinky4": 20,
}

DEFAULT_EXPORT_COORDINATE_SYSTEM: dict[str, object] = {
    "space": "kinara_normalized",
    "right_axis": "X",
    "forward_axis": "Y",
    "up_axis": "Z",
    "grounded_axis": "Z",
}
