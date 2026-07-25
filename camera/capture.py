from __future__ import annotations

from pathlib import Path

import cv2


class VideoInputSource:
    def __init__(self, video_path: int | Path, fallback_fps: float = 30.0):
        source = video_path if isinstance(video_path, int) else str(video_path)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            if isinstance(video_path, int):
                raise RuntimeError(
                    f"Could not open camera index {video_path}. Check that the camera is connected, not in use by another app, "
                    "and that the selected index is correct."
                )
            raise RuntimeError(
                f"Could not open video file: {video_path}. Check that the file exists, the codec is supported by OpenCV, "
                "and the file is not corrupt."
            )

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps <= 0:
            self.fps = fallback_fps

    def read(self):
        return self.cap.read()

    def skip_frames(self, frame_count: int) -> None:
        for _ in range(max(0, frame_count)):
            ok, _ = self.cap.read()
            if not ok:
                break

    def close(self) -> None:
        self.cap.release()


class VideoOutputWriter:
    def __init__(self, output_path: Path, frame_width: int, frame_height: int, fps: float, output_fourcc: str = "mp4v"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.output_fourcc = output_fourcc[:4].ljust(4)
        fourcc = cv2.VideoWriter.fourcc(*output_fourcc[:4].ljust(4))
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        if self.writer is None or not self.writer.isOpened():
            raise RuntimeError(
                f"Could not open video writer: {output_path}. Check that the directory is writable and FourCC "
                f"'{output_fourcc[:4].ljust(4)}' is supported for this output extension."
            )

    def write(self, frame) -> None:
        if self.writer is None:
            raise RuntimeError(
                f"Video writer is not available for {self.output_path}. FourCC '{self.output_fourcc}' may not be usable "
                "in this runtime."
            )
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


class VideoCaptureSession:
    def __init__(self, video_path: int | Path, output_path: Path, fallback_fps: float = 30.0, output_fourcc: str = "mp4v"):
        self.source = VideoInputSource(video_path, fallback_fps=fallback_fps)
        self.frame_width = self.source.frame_width
        self.frame_height = self.source.frame_height
        self.fps = self.source.fps
        try:
            self.writer = VideoOutputWriter(
                output_path,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                fps=self.fps,
                output_fourcc=output_fourcc,
            )
        except Exception:
            self.source.close()
            raise

    def read(self):
        return self.source.read()

    def write(self, frame) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.source.close()
        self.writer.close()
        cv2.destroyAllWindows()
