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
    wx, wy, _ = wrist_point
    ex, ey, _ = elbow_point

    forearm_dx = wx - ex
    forearm_dy = wy - ey
    forearm_len = max(int(math.hypot(forearm_dx, forearm_dy)), 1)
    box_size = max(min_box_size, int(forearm_len * scale))
    direction_x = forearm_dx / forearm_len
    direction_y = forearm_dy / forearm_len
    center_x = int(round(wx + direction_x * box_size * forward_shift))
    center_y = int(round(wy + direction_y * box_size * forward_shift))

    x1 = max(0, center_x - box_size // 2)
    y1 = max(0, center_y - box_size // 2)
    x2 = min(frame_width, center_x + box_size // 2)
    y2 = min(frame_height, center_y + box_size // 2)
    return x1, y1, x2, y2
