from __future__ import annotations

import time


class FpsMeter:
    def __init__(self, label: str, interval_seconds: float) -> None:
        self.label = label
        self.interval_seconds = interval_seconds
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._last_frame = 0

    def tick(self, frame_index: int) -> None:
        if self.interval_seconds <= 0:
            return
        now = time.perf_counter()
        elapsed = now - self._last_time
        if elapsed < self.interval_seconds:
            return

        frames = frame_index - self._last_frame + 1
        current_fps = frames / elapsed if elapsed > 0 else 0.0
        total_elapsed = now - self._start_time
        average_fps = (frame_index + 1) / total_elapsed if total_elapsed > 0 else 0.0
        print(f"[fps] mode={self.label} frame={frame_index} current={current_fps:.1f} average={average_fps:.1f}")
        self._last_time = now
        self._last_frame = frame_index + 1
