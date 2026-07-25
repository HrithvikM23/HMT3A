from __future__ import annotations

import math


def build_hand_box(
    wrist_point: tuple[int, int, float],
    elbow_point: tuple[int, int, float],
    frame_width: int,
    frame_height: int,
    min_box_size: int,
    scale: float,
    forward_shift: float,
) -> tuple[int, int, int, int]:
    wx, wy, wc = wrist_point[0], wrist_point[1], wrist_point[2]
    ex, ey, ec = elbow_point[0], elbow_point[1], elbow_point[2]

    forearm_dx = wx - ex
    forearm_dy = wy - ey
    raw_forearm_len = math.hypot(forearm_dx, forearm_dy)

    if ec > 0.10 and wc > 0.10 and raw_forearm_len >= 5.0 and not (ex == 0 and ey == 0):
        forearm_len = max(int(raw_forearm_len), 1)
        direction_x = forearm_dx / float(forearm_len)
        direction_y = forearm_dy / float(forearm_len)
        actual_shift = forward_shift
    else:
        forearm_len = min_box_size
        direction_x = 0.0
        direction_y = -1.0
        actual_shift = 0.0

    box_size = max(min_box_size, int(forearm_len * scale))
    center_x = int(round(wx + direction_x * box_size * actual_shift))
    center_y = int(round(wy + direction_y * box_size * actual_shift))

    x1 = max(0, center_x - box_size // 2)
    y1 = max(0, center_y - box_size // 2)
    x2 = min(frame_width, center_x + box_size // 2)
    y2 = min(frame_height, center_y + box_size // 2)
    return x1, y1, x2, y2


def expand_box(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    scale: float = 1.25,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    half_width = max(1.0, (x2 - x1) * scale * 0.5)
    half_height = max(1.0, (y2 - y1) * scale * 0.5)
    return (
        max(0, int(round(center_x - half_width))),
        max(0, int(round(center_y - half_height))),
        min(frame_width, int(round(center_x + half_width))),
        min(frame_height, int(round(center_y + half_height))),
    )
