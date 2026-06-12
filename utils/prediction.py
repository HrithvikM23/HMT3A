from __future__ import annotations

Point = tuple[int, int, float]
Box = tuple[int, int, int, int]


def predict_points(
    current_points: list[Point] | None,
    previous_points: list[Point] | None,
    confidence_decay: float,
) -> list[Point] | None:
    if current_points is None:
        return None
    if previous_points is None or len(previous_points) != len(current_points):
        return decay_points(current_points, confidence_decay)

    predicted: list[Point] = []
    for (x, y, conf), (prev_x, prev_y, _) in zip(current_points, previous_points, strict=True):
        predicted.append(
            (
                int(round(x + (x - prev_x))),
                int(round(y + (y - prev_y))),
                conf * confidence_decay,
            )
        )
    return predicted


def decay_points(points: list[Point], confidence_decay: float) -> list[Point]:
    return [(x, y, conf * confidence_decay) for x, y, conf in points]


def translate_points(points: list[Point], offset_x: float, offset_y: float, confidence_decay: float = 1.0) -> list[Point]:
    return [
        (
            int(round(x + offset_x)),
            int(round(y + offset_y)),
            conf * confidence_decay,
        )
        for x, y, conf in points
    ]


def translate_box(box: Box, offset_x: float, offset_y: float) -> Box:
    x1, y1, x2, y2 = box
    return (
        int(round(x1 + offset_x)),
        int(round(y1 + offset_y)),
        int(round(x2 + offset_x)),
        int(round(y2 + offset_y)),
    )
