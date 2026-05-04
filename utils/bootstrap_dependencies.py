from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import os
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore[no-redef]

try:
    import winreg
except ImportError:  # pragma: no cover
    winreg = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / f".vendor_py{sys.version_info.major}{sys.version_info.minor}"
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"
WINDOWS_DLL_DIRECTORIES: list[object] = []

REQUIRED_PROJECT_FILES = (
    Path("main.py"),
    Path("config.py"),
    Path("camera") / "capture.py",
    Path("inference") / "rtmpose.py",
    Path("network") / "osc_sender.py",
    Path("pipeline") / "pipeline.py",
    Path("utils") / "exports.py",
    Path("utils") / "fusion.py",
    Path("utils") / "hand_constraints.py",
    Path("utils") / "hand_fallback.py",
    Path("utils") / "model_assets.py",
    Path("utils") / "multi_person.py",
    Path("utils") / "normalize.py",
    Path("utils") / "smoothing.py",
)

MODULE_TO_PACKAGE = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "onnxruntime": "onnxruntime",
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


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = os.path.normcase(os.path.normpath(str(path)))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(path)
    return unique_paths


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    normalized = os.path.normcase(os.path.normpath(path_str))
    current_entries = {
        os.path.normcase(os.path.normpath(existing_path))
        for existing_path in sys.path
        if existing_path
    }
    if normalized not in current_entries:
        sys.path.insert(0, path_str)
    site.addsitedir(path_str)


def _prepend_env_path(path: Path) -> bool:
    path_str = str(path)
    current_value = os.environ.get("PATH", "")
    parts = [part for part in current_value.split(os.pathsep) if part]
    normalized_parts = {
        os.path.normcase(os.path.normpath(part))
        for part in parts
    }
    normalized_path = os.path.normcase(os.path.normpath(path_str))
    if normalized_path in normalized_parts:
        return False

    os.environ["PATH"] = os.pathsep.join([path_str, *parts]) if parts else path_str
    return True


def _register_windows_dll_directory(path: Path) -> bool:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return False
    if not path.exists() or not path.is_dir():
        return False

    normalized_candidate = os.path.normcase(os.path.normpath(str(path)))
    for handle in WINDOWS_DLL_DIRECTORIES:
        handle_path = getattr(handle, "_kinara_path", "")
        if handle_path and os.path.normcase(os.path.normpath(handle_path)) == normalized_candidate:
            return False

    handle = os.add_dll_directory(str(path))
    setattr(handle, "_kinara_path", str(path))
    WINDOWS_DLL_DIRECTORIES.append(handle)
    return True


def _prepend_pythonpath(path: Path) -> None:
    path_str = str(path)
    current_value = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current_value.split(os.pathsep) if part]
    normalized_parts = {
        os.path.normcase(os.path.normpath(part))
        for part in parts
    }
    normalized_path = os.path.normcase(os.path.normpath(path_str))
    if normalized_path not in normalized_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([path_str, *parts]) if parts else path_str


def _broadcast_environment_change() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, 0)
    except Exception:
        return


def _persist_user_path(path_updates: list[Path]) -> list[str]:
    if os.name != "nt" or winreg is None or not path_updates:
        return []

    warnings: list[str] = []
    unique_updates = _dedupe_paths(path_updates)
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
            try:
                current_value, current_type = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current_value, current_type = "", winreg.REG_EXPAND_SZ

            parts = [part for part in str(current_value).split(os.pathsep) if part]
            normalized_parts = {
                os.path.normcase(os.path.normpath(part))
                for part in parts
            }
            changed = False

            for update in unique_updates:
                normalized_update = os.path.normcase(os.path.normpath(str(update)))
                if normalized_update in normalized_parts:
                    continue
                parts.insert(0, str(update))
                normalized_parts.add(normalized_update)
                changed = True

            if changed:
                winreg.SetValueEx(key, "Path", 0, current_type, os.pathsep.join(parts))
                _broadcast_environment_change()
    except OSError as exc:
        warnings.append(f"Could not persist CUDA/cuDNN PATH updates: {exc}")

    return warnings


def _distribution_installed(distribution_name: str) -> bool:
    try:
        importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


def _module_status(module_name: str) -> ModuleStatus:
    try:
        importlib.invalidate_caches()
        importlib.import_module(module_name)
        return ModuleStatus(module_name=module_name, ok=True)
    except Exception as exc:
        return ModuleStatus(module_name=module_name, ok=False, error=f"{type(exc).__name__}: {exc}")


def _module_group_status(module_names: tuple[str, ...]) -> list[ModuleStatus]:
    return [_module_status(module_name) for module_name in module_names]


def _find_missing_project_files() -> list[Path]:
    missing_paths: list[Path] = []
    for relative_path in REQUIRED_PROJECT_FILES:
        candidate = PROJECT_ROOT / relative_path
        if not candidate.exists():
            missing_paths.append(relative_path)
    return missing_paths


def _find_nvidia_smi() -> Path | None:
    command_path = shutil.which("nvidia-smi")
    if command_path:
        return Path(command_path)

    common_path = Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    if common_path.exists():
        return common_path
    return None


def _path_has_glob(path: Path, pattern: str) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.glob(pattern))


def _collect_runtime_roots() -> list[Path]:
    roots: list[Path] = []
    env_names = ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDNN_PATH")
    for env_name in env_names:
        raw_value = os.environ.get(env_name)
        if not raw_value:
            continue
        candidate = Path(raw_value)
        if candidate.exists():
            roots.append(candidate)

    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.exists():
        roots.extend(path for path in cuda_root.glob("v*") if path.is_dir())

    cudnn_root = Path(r"C:\Program Files\NVIDIA\CUDNN")
    if cudnn_root.exists():
        roots.extend(path for path in cudnn_root.glob("v*") if path.is_dir())

    local_cudnn_root = Path(os.environ.get("LOCALAPPDATA", "")) / "NVIDIA" / "CUDNN"
    if local_cudnn_root.exists():
        roots.extend(path for path in local_cudnn_root.glob("v*") if path.is_dir())

    roots.extend(_collect_torch_runtime_roots())

    return _dedupe_paths(roots)


def _collect_torch_runtime_roots() -> list[Path]:
    roots: list[Path] = [VENDOR_DIR]

    try:
        roots.extend(Path(site_package) for site_package in site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None
    if user_site:
        roots.append(Path(user_site))

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None and torch_spec.origin:
        roots.append(Path(torch_spec.origin).resolve().parent.parent)

    candidate_dirs: list[Path] = []
    for root in _dedupe_paths([root for root in roots if root.exists()]):
        candidate_dirs.extend(
            [
                root / "torch" / "lib",
                root / "Lib" / "site-packages" / "torch" / "lib",
            ]
        )

    return _dedupe_paths([candidate for candidate in candidate_dirs if candidate.exists() and candidate.is_dir()])


def _bin_candidates(root: Path) -> list[Path]:
    candidates = [root]
    if root.name.lower() != "bin":
        candidates.append(root / "bin")
    candidates.extend(
        [
            root / "lib",
            root / "lib" / "x64",
            root / "bin" / "x64",
            root / "bin" / "12",
            root / "bin" / "11",
            root / "bin" / "10",
        ]
    )
    bin_dir = root / "bin"
    if bin_dir.exists() and bin_dir.is_dir():
        candidates.extend(candidate for candidate in bin_dir.iterdir() if candidate.is_dir())
    return _dedupe_paths([candidate for candidate in candidates if candidate.exists() and candidate.is_dir()])


def _include_candidates(root: Path) -> list[Path]:
    candidates = [root]
    if root.name.lower() != "include":
        candidates.append(root / "include")
    return _dedupe_paths([candidate for candidate in candidates if candidate.exists() and candidate.is_dir()])


def _inspect_runtime() -> RuntimeReport:
    report = RuntimeReport()
    nvidia_smi_path = _find_nvidia_smi()

    if nvidia_smi_path is not None:
        try:
            completed = subprocess.run(
                [str(nvidia_smi_path), "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            report.nvidia_driver_detected = completed.returncode == 0 and bool(completed.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            report.nvidia_driver_detected = False

    for root in _collect_runtime_roots():
        for bin_dir in _bin_candidates(root):
            if _path_has_glob(bin_dir, "cudart*.dll"):
                report.cuda_bin_dirs.append(bin_dir)
            if _path_has_glob(bin_dir, "cudnn*.dll"):
                report.cudnn_bin_dirs.append(bin_dir)

        for include_dir in _include_candidates(root):
            if (include_dir / "cuda_runtime.h").exists() or (include_dir / "cuda.h").exists():
                report.cuda_include_dirs.append(include_dir)
            if (include_dir / "cudnn.h").exists() or any(include_dir.glob("cudnn*.h")):
                report.cudnn_include_dirs.append(include_dir)

    report.cuda_bin_dirs = _dedupe_paths(report.cuda_bin_dirs)
    report.cudnn_bin_dirs = _dedupe_paths(report.cudnn_bin_dirs)
    report.cuda_include_dirs = _dedupe_paths(report.cuda_include_dirs)
    report.cudnn_include_dirs = _dedupe_paths(report.cudnn_include_dirs)

    if report.nvidia_driver_detected and not report.cuda_bin_dirs:
        report.warnings.append("NVIDIA driver detected but CUDA runtime DLLs were not found. Kinara may fall back to CPU.")
    if report.nvidia_driver_detected and not report.cudnn_bin_dirs:
        report.warnings.append("NVIDIA driver detected but cuDNN DLLs were not found. CUDAExecutionProvider may stay unavailable.")
    if report.nvidia_driver_detected and not report.cudnn_include_dirs:
        report.warnings.append("cuDNN headers were not found in common install locations.")

    return report


def _repair_runtime_paths(report: RuntimeReport, persist: bool) -> None:
    candidate_dirs = _dedupe_paths([*report.cudnn_bin_dirs, *report.cuda_bin_dirs])
    for candidate_dir in candidate_dirs:
        if _prepend_env_path(candidate_dir):
            report.path_updates.append(candidate_dir)
        _register_windows_dll_directory(candidate_dir)

    if report.cuda_bin_dirs and "CUDA_PATH" not in os.environ:
        cuda_root = report.cuda_bin_dirs[0].parent if report.cuda_bin_dirs[0].name.lower() == "bin" else report.cuda_bin_dirs[0]
        os.environ["CUDA_PATH"] = str(cuda_root)

    if report.cudnn_bin_dirs and "CUDNN_PATH" not in os.environ:
        cudnn_root = report.cudnn_bin_dirs[0].parent if report.cudnn_bin_dirs[0].name.lower() == "bin" else report.cudnn_bin_dirs[0]
        os.environ["CUDNN_PATH"] = str(cudnn_root)

    if persist:
        report.warnings.extend(_persist_user_path(report.path_updates))


def _ensure_local_environment() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _prepend_sys_path(VENDOR_DIR)
    _prepend_pythonpath(VENDOR_DIR)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))


def _choose_onnxruntime_distribution(report: RuntimeReport) -> str:
    if _distribution_installed("onnxruntime-gpu"):
        return "onnxruntime-gpu"
    if _distribution_installed("onnxruntime"):
        return "onnxruntime"
    if report.nvidia_driver_detected and report.cuda_bin_dirs and report.cudnn_bin_dirs:
        return "onnxruntime-gpu"
    return "onnxruntime"


def _resolve_install_plan(module_statuses: list[ModuleStatus], report: RuntimeReport) -> list[str]:
    missing_modules = {status.module_name for status in module_statuses if not status.ok}
    packages_to_install: list[str] = []

    if {"ultralytics", "torch", "torchvision"} & missing_modules:
        packages_to_install.append("ultralytics")
        missing_modules.difference_update({"ultralytics", "torch", "torchvision", "numpy", "cv2"})

    if "cv2" in missing_modules:
        packages_to_install.append("opencv-python")
        missing_modules.discard("cv2")
        missing_modules.discard("numpy")

    if "numpy" in missing_modules:
        packages_to_install.append("numpy")
        missing_modules.discard("numpy")

    if "onnxruntime" in missing_modules:
        packages_to_install.append(_choose_onnxruntime_distribution(report))
        missing_modules.discard("onnxruntime")

    return packages_to_install


def _ensure_pip() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if completed.returncode == 0:
        return

    subprocess.run(
        [sys.executable, "-m", "ensurepip", "--upgrade"],
        check=True,
        timeout=120,
    )


def _install_packages(packages: list[str]) -> None:
    if not packages:
        return

    _ensure_pip()
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade",
        "--prefer-binary",
        "--progress-bar",
        "on",
        "--target",
        str(VENDOR_DIR),
        *packages,
    ]
    env = os.environ.copy()
    _prepend_pythonpath(VENDOR_DIR)
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    subprocess.run(command, check=True, env=env)
    importlib.invalidate_caches()
    _prepend_sys_path(VENDOR_DIR)


def _probe_runtime(report: RuntimeReport) -> tuple[list[ModuleStatus], list[str]]:
    statuses = _module_group_status(("cv2", "numpy", "torch", "torchvision", "ultralytics", "onnxruntime"))
    warnings = list(report.warnings)

    onnx_status = next((status for status in statuses if status.module_name == "onnxruntime"), None)
    if onnx_status is not None and onnx_status.ok:
        try:
            import onnxruntime as ort

            providers = set(ort.get_available_providers())
            if _distribution_installed("onnxruntime-gpu") and "CUDAExecutionProvider" not in providers:
                discovered_dirs = _dedupe_paths([*report.cudnn_bin_dirs, *report.cuda_bin_dirs])
                joined_dirs = ", ".join(str(path) for path in discovered_dirs[:4])
                detail = f" Checked runtime dirs: {joined_dirs}." if joined_dirs else ""
                warnings.append(
                    "onnxruntime-gpu is installed but CUDAExecutionProvider is unavailable; "
                    f"Kinara will use CPU for hand inference.{detail}"
                )
        except Exception as exc:
            warnings.append(f"Could not inspect ONNX Runtime providers: {type(exc).__name__}: {exc}")

    return statuses, warnings


def _dedupe_warning_messages(warnings: list[str]) -> list[str]:
    unique_warnings: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        normalized = warning.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_warnings.append(normalized)
    return unique_warnings


def ensure_runtime_ready(*, persist_cudnn_path: bool = True, check_only: bool = False) -> None:
    progress = TerminalProgress(total_steps=6)
    progress.note("Starting Kinara runtime bootstrap...")

    progress.advance("Preparing local runtime folders")
    _ensure_local_environment()

    progress.advance("Checking required Kinara files")
    missing_project_files = _find_missing_project_files()
    if missing_project_files:
        progress.break_line()
        missing_list = ", ".join(str(path).replace("\\", "/") for path in missing_project_files)
        raise RuntimeError(f"Kinara is missing required project files: {missing_list}")

    progress.advance("Inspecting CUDA / cuDNN runtime")
    report = _inspect_runtime()
    _repair_runtime_paths(report, persist=persist_cudnn_path)

    progress.advance("Checking Python dependencies")
    initial_statuses, initial_warnings = _probe_runtime(report)
    packages_to_install = _resolve_install_plan(initial_statuses, report)

    if packages_to_install:
        install_message = "Installing missing Python packages"
        if check_only:
            install_message = "Missing packages detected (install skipped: --check-only)"
        progress.advance(install_message)
    else:
        progress.advance("All Python packages already available")

    if packages_to_install and not check_only:
        progress.break_line()
        print(f"Preparing runtime dependencies in {VENDOR_DIR}...")
        print(f"Installing only missing packages: {', '.join(packages_to_install)}")
        _install_packages(packages_to_install)
    elif packages_to_install and check_only:
        progress.break_line()
        print(f"Missing packages: {', '.join(packages_to_install)}")

    progress.advance("Running final verification")
    report = _inspect_runtime()
    _repair_runtime_paths(report, persist=persist_cudnn_path)
    statuses, warnings = _probe_runtime(report)
    warnings = [*initial_warnings, *warnings]

    failed_statuses = [status for status in statuses if not status.ok]
    if failed_statuses:
        progress.break_line()
        lines = []
        for status in failed_statuses:
            package_name = MODULE_TO_PACKAGE.get(status.module_name, status.module_name)
            detail = status.error or "Unknown import failure"
            lines.append(f"- {status.module_name} ({package_name}): {detail}")
        raise RuntimeError("Runtime bootstrap could not satisfy required dependencies:\n" + "\n".join(lines))

    progress.finish("Kinara runtime ready")

    if report.path_updates:
        joined_paths = ", ".join(str(path) for path in report.path_updates)
        print(f"Updated CUDA/cuDNN PATH entries: {joined_paths}")

    for warning in _dedupe_warning_messages(warnings):
        print(f"Runtime warning: {warning}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Kinara's local runtime dependencies.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the runtime without installing missing Python packages.",
    )
    parser.add_argument(
        "--no-persist-cudnn-path",
        action="store_true",
        help="Only patch CUDA/cuDNN PATH values for the current process.",
    )
    return parser


def main() -> int:
    parser = _build_cli_parser()
    args = parser.parse_args()

    try:
        ensure_runtime_ready(
            persist_cudnn_path=not args.no_persist_cudnn_path,
            check_only=args.check_only,
        )
    except RuntimeError as exc:
        print(f"Runtime bootstrap failed: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Dependency installation failed with exit code {exc.returncode}.")
        return exc.returncode or 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
