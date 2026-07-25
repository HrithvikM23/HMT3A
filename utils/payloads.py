from __future__ import annotations

from typing import NotRequired, TypedDict

from utils.skeleton import JointMap, Point

Box = tuple[int, int, int, int]


class HandPayload(TypedDict):
    box: Box
    points: list[Point]
    depths: NotRequired[list[float]]
    fallback: NotRequired[bool]


class PersonPayload(TypedDict, total=False):
    id: int
    label: str
    box: Box | None
    score: float | None
    camera_views: list[str]
    body_points: list[Point]
    hands_by_side: dict[str, HandPayload]
    joint_depths: dict[str, float]
    joints: JointMap


class SingleMotionFrame(TypedDict):
    frame_index: int
    joints: JointMap


class MultiPersonMotionFrame(TypedDict):
    frame_index: int
    people: list[PersonPayload]
