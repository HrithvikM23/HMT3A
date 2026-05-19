from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / f".vendor_py{sys.version_info.major}{sys.version_info.minor}"
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"

REQUIRED_PROJECT_FILES = (
    Path("main.py"),
    Path("pyproject.toml"),
    Path("backend_selection.py"),
    Path("cli.py"),
    Path("config.py"),
    Path("runtime_profiles.py"),
    Path("kinara") / "__main__.py",
    Path("runtime_config.py"),
    Path("camera") / "capture.py",
    Path("inference") / "rtmpose.py",
    Path("network") / "osc_sender.py",
    Path("pipeline") / "pipeline.py",
    Path("runners") / "common.py",
    Path("runners") / "fused.py",
    Path("runners") / "fused_alignment.py",
    Path("runners") / "multi_person.py",
    Path("runners") / "single.py",
    Path("utils") / "color_profile.py",
    Path("utils") / "body_constraints.py",
    Path("utils") / "body_geometry.py",
    Path("utils") / "bootstrap_cuda.py",
    Path("utils") / "bootstrap_dependencies.py",
    Path("utils") / "bootstrap_packages.py",
    Path("utils") / "bootstrap_paths.py",
    Path("utils") / "bootstrap_state.py",
    Path("utils") / "calibration.py",
    Path("utils") / "exports.py",
    Path("utils") / "fusion.py",
    Path("utils") / "hand_constraints.py",
    Path("utils") / "hand_fallback.py",
    Path("utils") / "model_assets.py",
    Path("utils") / "motion_cleanup.py",
    Path("utils") / "multi_person.py",
    Path("utils") / "normalize.py",
    Path("utils") / "payloads.py",
    Path("utils") / "skeleton.py",
    Path("utils") / "smoothing.py",
    Path("utils") / "triangulation.py",
)

MODULE_TO_PACKAGE = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "onnxruntime": "onnxruntime",
    "mediapipe": "mediapipe==0.10.21",
}


@dataclass(frozen=True, slots=True)
class ModuleStatus:
    module_name: str
    ok: bool
    error: str | None = None


@dataclass(slots=True)
class RuntimeReport:
    nvidia_driver_detected: bool = False
    cuda_bin_dirs: list[Path] = field(default_factory=list)
    cudnn_bin_dirs: list[Path] = field(default_factory=list)
    cuda_include_dirs: list[Path] = field(default_factory=list)
    cudnn_include_dirs: list[Path] = field(default_factory=list)
    path_updates: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TerminalProgress:
    def __init__(self, total_steps: int, width: int = 30) -> None:
        self.total_steps = max(1, total_steps)
        self.width = max(10, width)
        self.current_step = 0
        self._last_render_length = 0

    def _render(self, message: str) -> None:
        ratio = min(1.0, self.current_step / self.total_steps)
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"\r[{bar}] {self.current_step}/{self.total_steps} {int(ratio * 100):3d}% {message}"
        padding = max(0, self._last_render_length - len(line))
        sys.stdout.write(line + (" " * padding))
        sys.stdout.flush()
        self._last_render_length = len(line)

    def note(self, message: str) -> None:
        self._render(message)

    def advance(self, message: str) -> None:
        self.current_step = min(self.total_steps, self.current_step + 1)
        self._render(message)

    def break_line(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._last_render_length = 0

    def finish(self, message: str) -> None:
        self.current_step = self.total_steps
        self._render(message)
        self.break_line()
