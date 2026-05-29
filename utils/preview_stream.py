from __future__ import annotations

import os
from pathlib import Path
import time
from uuid import uuid4

import cv2


class PreviewFrameSink:
    def __init__(self) -> None:
        self.path = Path(os.environ["KINARA_PREVIEW_FRAME"]) if os.environ.get("KINARA_PREVIEW_FRAME") else None
        self.interval = max(1, int(os.environ.get("KINARA_PREVIEW_INTERVAL", "2") or "2"))
        self.jpeg_quality = max(40, min(95, int(os.environ.get("KINARA_PREVIEW_QUALITY", "82") or "82")))
        self.keep_frames = max(3, int(os.environ.get("KINARA_PREVIEW_KEEP_FRAMES", "8") or "8"))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def write(self, frame, frame_index: int) -> None:
        if self.path is None or frame_index % self.interval != 0:
            return

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
            for _ in range(4):
                try:
                    temp_path.replace(frame_path)
                    self._cleanup_old_frames()
                    return
                except OSError:
                    time.sleep(0.02)
        except OSError:
            pass
        finally:
            try:
                temp_path.unlink()
            except OSError:
                pass

    def _cleanup_old_frames(self) -> None:
        if self.path is None:
            return
        frames = sorted(self.path.parent.glob(f"{self.path.stem}_*{self.path.suffix}"))
        for old_frame in frames[:-self.keep_frames]:
            try:
                old_frame.unlink()
            except OSError:
                pass
