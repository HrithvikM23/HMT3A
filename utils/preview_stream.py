from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

import cv2


class PreviewFrameSink:
    def __init__(self, worker_index: int | str | None = None) -> None:
        raw_path = os.environ.get("KINARA_PREVIEW_FRAME")
        self.path = Path(raw_path) if raw_path else None

        idx = worker_index if worker_index is not None else os.environ.get("KINARA_WORKER_INDEX")
        if self.path is not None and idx is not None and str(idx).strip() != "":
            worker_str = f"_worker_{idx}"
            if worker_str not in self.path.stem:
                self.path = self.path.with_name(f"{self.path.stem}{worker_str}{self.path.suffix}")

        self.interval = max(1, int(os.environ.get("KINARA_PREVIEW_INTERVAL", "2") or "2"))
        self.jpeg_quality = max(40, min(95, int(os.environ.get("KINARA_PREVIEW_QUALITY", "82") or "82")))
        self.keep_frames = max(3, int(os.environ.get("KINARA_PREVIEW_KEEP_FRAMES", "8") or "8"))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
        self._queue = __import__("queue").Queue(maxsize=2)
        self._thread = None
        self._stop_event = __import__("threading").Event()
        
        if self.path is not None:
            self._thread = __import__("threading").Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self.path is not None
        
    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                if item is not None:
                    try:
                        frame, frame_index = item
                        self._process_frame(frame, frame_index)
                    finally:
                        self._queue.task_done()
            except __import__("queue").Empty:
                continue

    def write(self, frame, frame_index: int) -> None:
        if self.path is None or frame_index % self.interval != 0:
            return

        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > 640:
            scale = 640.0 / max_dim
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        try:
            self._queue.put_nowait((frame, frame_index))
        except __import__("queue").Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait((frame, frame_index))
            except (__import__("queue").Empty, __import__("queue").Full):
                pass
                
    def _process_frame(self, frame, frame_index: int) -> None:
        sequence = frame_index // self.interval
        temp_path = self.path.with_name(f"{self.path.stem}_{sequence:08d}.{uuid4().hex}.tmp{self.path.suffix}")
        frame_path = self.path.with_name(f"{self.path.stem}_{sequence:08d}{self.path.suffix}")
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return

        try:
            temp_path.write_bytes(encoded.tobytes())
            for _ in range(6):
                try:
                    temp_path.replace(frame_path)
                    self._cleanup_old_frames()
                    return
                except OSError:
                    time.sleep(0.01)
        except OSError:
            pass
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _cleanup_old_frames(self) -> None:
        if self.path is None:
            return
        frames = sorted(self.path.parent.glob(f"{self.path.stem}_*{self.path.suffix}"))
        remove_count = max(0, len(frames) - self.keep_frames)
        for old_frame in frames[:remove_count]:
            try:
                old_frame.unlink()
            except OSError:
                pass
