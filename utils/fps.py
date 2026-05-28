from __future__ import annotations

import time

import cv2


class FpsMeter:
    def __init__(self, label: str, interval_seconds: float) -> None:
        self.label = label
        self.interval_seconds = interval_seconds
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._last_tick_time = self._start_time
        self._last_frame = 0
        self.current_fps = 0.0
        self.average_fps = 0.0

    def tick(self, frame_index: int) -> None:
        now = time.perf_counter()
        frame_delta = now - self._last_tick_time
        if frame_delta > 0:
            instant_fps = 1.0 / frame_delta
            if self.current_fps <= 0.0:
                self.current_fps = instant_fps
            else:
                self.current_fps = (self.current_fps * 0.85) + (instant_fps * 0.15)
        self._last_tick_time = now

        total_elapsed = now - self._start_time
        self.average_fps = (frame_index + 1) / total_elapsed if total_elapsed > 0 else 0.0

        if self.interval_seconds <= 0:
            return

        elapsed = now - self._last_time
        if elapsed < self.interval_seconds:
            return

        frames = frame_index - self._last_frame + 1
        interval_fps = frames / elapsed if elapsed > 0 else 0.0
        print(f"[fps] mode={self.label} frame={frame_index} current={interval_fps:.1f} average={self.average_fps:.1f}")
        self._last_time = now
        self._last_frame = frame_index + 1


def draw_fps_overlay(frame, fps_meter: FpsMeter, enabled: bool = True) -> None:
    if not enabled:
        return
    text = f"FPS {fps_meter.current_fps:5.1f} | AVG {fps_meter.average_fps:5.1f}"
    origin = (12, 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x - 6, y - text_height - 8),
        (x + text_width + 6, y + baseline + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
