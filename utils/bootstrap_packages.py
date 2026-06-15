from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore[no-redef]

from utils.bootstrap_paths import dedupe_paths, prepend_pythonpath, prepend_sys_path
from utils.bootstrap_state import MODULE_TO_PACKAGE, VENDOR_DIR, ModuleStatus, RuntimeReport
from utils.logging import safe_print

PRUNED_VENDOR_DISTRIBUTIONS = ("openxlab",)
WINDOWS_CONTROL_C_EXIT = 0xC000013A


def _returncode_text(return_code: int) -> str:
    if return_code == WINDOWS_CONTROL_C_EXIT or return_code == -1073741510:
        return (
            "The dependency installer was interrupted by Windows or cancelled before it finished. "
            "Do not close Kinara or press Stop/Kill while the first run is installing packages. "
            "If antivirus is scanning the app folder, wait and run Check Runtime again."
        )
    return f"Process exited with code {return_code}."


def _valid_python_executable(candidate: str) -> str | None:
    if not candidate:
        return None
    path = Path(candidate.strip().strip('"').strip("'"))
    if path.is_dir():
        path = path / "python.exe"
    if not path.exists() or path.name.lower() != "python.exe":
        return None
    try:
        completed = subprocess.run(
            [str(path), "-c", "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"],
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return str(path) if completed.returncode == 0 else None


def _python_from_launcher() -> str:
    launcher = shutil.which("py")
    if not launcher:
        return ""
    try:
        completed = subprocess.run(
            [launcher, "-3.11", "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _registry_python_candidates() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []

    candidates: list[str] = []
    keys = (
        (winreg.HKEY_CURRENT_USER, r"Software\Python\PythonCore\3.11\InstallPath"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Python\PythonCore\3.11\InstallPath"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Python\PythonCore\3.11\InstallPath"),
    )
    for hive, key_path in keys:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                install_path, _ = winreg.QueryValueEx(key, "")
        except OSError:
            continue
        candidates.append(str(Path(install_path) / "python.exe"))
    return candidates


def _common_python_candidates() -> list[str]:
    candidates: list[str] = []
    for executable in ("python3.11", "python"):
        found = shutil.which(executable)
        if found:
            candidates.append(found)
    candidates.append(_python_from_launcher())

    if sys.platform == "win32":
        roots = [
            os.environ.get("LOCALAPPDATA", ""),
            os.environ.get("ProgramFiles", ""),
            os.environ.get("ProgramFiles(x86)", ""),
        ]
        for root in roots:
            if root:
                candidates.append(str(Path(root) / "Programs" / "Python" / "Python311" / "python.exe"))
                candidates.append(str(Path(root) / "Python311" / "python.exe"))
        candidates.extend(_registry_python_candidates())
    return candidates


def installer_python() -> str:
    candidates = (
        os.environ.get("KINARA_PYTHON", ""),
        getattr(sys, "_base_executable", ""),
        sys.executable,
        *_common_python_candidates(),
    )
    for candidate in candidates:
        resolved = _valid_python_executable(candidate)
        if resolved is not None:
            return resolved

    if os.environ.get("KINARA_ALLOW_BLENDER_PYTHON") == "1":
        resolved = _valid_python_executable(r"C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe")
        if resolved is not None:
            return resolved

    raise RuntimeError(
        "No installable Python runtime was found. Run Kinara from a Python environment or set KINARA_PYTHON "
        "to a python.exe that can run pip. To deliberately use Blender's bundled Python, set "
        "KINARA_ALLOW_BLENDER_PYTHON=1."
    )


def distribution_installed(distribution_name: str) -> bool:
    try:
        importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


def distribution_version(distribution_name: str) -> str | None:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def protobuf_needs_mediapipe_pin() -> bool:
    version = distribution_version("protobuf")
    if version is None:
        return True
    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        return True
    return major >= 5


def prune_vendor_distributions(distribution_names: tuple[str, ...] = PRUNED_VENDOR_DISTRIBUTIONS) -> None:
    if not VENDOR_DIR.exists():
        return
    vendor_root = VENDOR_DIR.resolve()
    normalized_names = {name.lower().replace("-", "_") for name in distribution_names}
    for child in VENDOR_DIR.iterdir():
        normalized_child = child.name.lower().replace("-", "_")
        if not any(
            normalized_child == name
            or normalized_child.startswith(f"{name}-")
            or normalized_child.startswith(f"{name}.")
            for name in normalized_names
        ):
            continue
        try:
            child.resolve().relative_to(vendor_root)
        except ValueError:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass


def prune_opencv_distributions_for_contrib() -> None:
    prune_vendor_distributions((
        "cv2",
        "opencv-contrib-python",
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    ))


def module_status(module_name: str) -> ModuleStatus:
    try:
        importlib.invalidate_caches()
        importlib.import_module(module_name)
        return ModuleStatus(module_name=module_name, ok=True)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        missing_stdlib_modules = [
            stdlib_module
            for stdlib_module in ("timeit", "zoneinfo")
            if f"No module named '{stdlib_module}'" in detail
        ]
        if module_name == "aniposelib" and missing_stdlib_modules:
            module_list = ", ".join(missing_stdlib_modules)
            detail += (
                f" (the packaged Kinara executable is missing Python standard-library module(s): {module_list}; "
                "rebuild/update the app with the current packaging script)"
            )
        return ModuleStatus(module_name=module_name, ok=False, error=detail)


def module_group_status(module_names: tuple[str, ...]) -> list[ModuleStatus]:
    return [module_status(module_name) for module_name in module_names]


def choose_onnxruntime_distribution(report: RuntimeReport) -> str:
    if distribution_installed("onnxruntime-gpu"):
        return "onnxruntime-gpu"
    if report.nvidia_driver_detected and report.cuda_bin_dirs and report.cudnn_bin_dirs:
        return "onnxruntime-gpu"
    if distribution_installed("onnxruntime"):
        return "onnxruntime"
    return "onnxruntime"


def resolve_install_plan(module_statuses: list[ModuleStatus], report: RuntimeReport) -> list[str]:
    missing_modules = {status.module_name for status in module_statuses if not status.ok}
    packages_to_install: list[str] = []
    gpu_runtime_detected = report.nvidia_driver_detected and report.cuda_bin_dirs and report.cudnn_bin_dirs
    mediapipe_requested = "mediapipe" in missing_modules

    if {"ultralytics", "torch", "torchvision"} & missing_modules:
        packages_to_install.append("ultralytics")
        missing_modules.difference_update({"ultralytics", "torch", "torchvision", "numpy", "cv2"})

    calibration_requested = "aniposelib" in missing_modules

    if "cv2" in missing_modules:
        if calibration_requested or mediapipe_requested:
            packages_to_install.append("opencv-contrib-python>=4.9,<4.12")
        else:
            packages_to_install.append("opencv-python")
        missing_modules.discard("cv2")
        missing_modules.discard("numpy")

    if "numpy" in missing_modules:
        packages_to_install.append("numpy")
        missing_modules.discard("numpy")

    onnxruntime_requested = any(status.module_name == "onnxruntime" for status in module_statuses)
    if "onnxruntime" in missing_modules:
        packages_to_install.extend([
            choose_onnxruntime_distribution(report),
            "numpy>=1.26,<2.0",
            "protobuf>=4.25.3,<5",
        ])
        missing_modules.discard("onnxruntime")
    elif onnxruntime_requested and gpu_runtime_detected and not distribution_installed("onnxruntime-gpu"):
        packages_to_install.extend([
            "onnxruntime-gpu",
            "numpy>=1.26,<2.0",
            "protobuf>=4.25.3,<5",
        ])

    if "mediapipe" in missing_modules:
        packages_to_install.extend([
            MODULE_TO_PACKAGE["mediapipe"],
            "numpy>=1.26,<2.0",
            "opencv-contrib-python>=4.9,<4.12",
            "jax==0.7.1",
            "jaxlib==0.7.1",
            "protobuf>=4.25.3,<5",
        ])
        missing_modules.discard("mediapipe")
    elif any(status.module_name == "mediapipe" and status.ok for status in module_statuses) and protobuf_needs_mediapipe_pin():
        packages_to_install.append("protobuf>=4.25.3,<5")

    if "rtmlib" in missing_modules:
        packages_to_install.extend([
            "rtmlib",
            "protobuf>=4.25.3,<5",
        ])
        missing_modules.discard("rtmlib")

    if "aniposelib" in missing_modules:
        packages_to_install.extend([
            "numpy>=1.26,<2.0",
            "opencv-contrib-python>=4.9,<4.12",
            "aniposelib>=0.7,<0.8",
            "protobuf>=4.25.3,<5",
        ])
        missing_modules.discard("aniposelib")
        missing_modules.discard("cv2")
        missing_modules.discard("numpy")

    return list(dict.fromkeys(packages_to_install))


def ensure_pip() -> None:
    python = installer_python()
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [python, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=20,
    )
    if completed.returncode == 0:
        return

    run_logged_subprocess([python, "-m", "ensurepip", "--upgrade"], env=env, timeout=120)


def run_logged_subprocess(command: list[str], *, env: dict[str, str], timeout: int | None = None) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            safe_print(line.rstrip())
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        safe_print(_returncode_text(return_code))
        raise subprocess.CalledProcessError(return_code, command)


def install_packages(packages: list[str]) -> None:
    if not packages:
        return

    ensure_pip()
    prune_vendor_distributions()
    if any(package.startswith(("opencv-contrib-python", "mediapipe")) for package in packages):
        prune_opencv_distributions_for_contrib()
    python = installer_python()
    command = [
        python,
        "-m",
        "pip",
        "install",
        "--no-input",
        "--ignore-installed",
        "--disable-pip-version-check",
        "--upgrade",
        "--prefer-binary",
        "--progress-bar",
        "off",
        "--retries",
        "3",
        "--timeout",
        "60",
        "--target",
        str(VENDOR_DIR),
        *packages,
    ]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    prepend_pythonpath(VENDOR_DIR)
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    run_logged_subprocess(command, env=env)
    importlib.invalidate_caches()
    prepend_sys_path(VENDOR_DIR)


def probe_runtime(report: RuntimeReport, module_names: tuple[str, ...] | None = None) -> tuple[list[ModuleStatus], list[str]]:
    statuses = module_group_status(module_names or tuple(MODULE_TO_PACKAGE))
    warnings = list(report.warnings)

    onnx_status = next((status for status in statuses if status.module_name == "onnxruntime"), None)
    if onnx_status is not None and onnx_status.ok:
        try:
            import onnxruntime as ort

            providers = set(ort.get_available_providers())
            if distribution_installed("onnxruntime-gpu") and "CUDAExecutionProvider" not in providers:
                discovered_dirs = dedupe_paths([*report.cudnn_bin_dirs, *report.cuda_bin_dirs])
                joined_dirs = ", ".join(str(path) for path in discovered_dirs[:4])
                detail = f" Checked runtime dirs: {joined_dirs}." if joined_dirs else ""
                warnings.append(
                    "onnxruntime-gpu is installed but CUDAExecutionProvider is unavailable; "
                    f"Kinara will use CPU for hand inference.{detail}"
                )
        except Exception as exc:
            warnings.append(f"Could not inspect ONNX Runtime providers: {type(exc).__name__}: {exc}")

    torch_status = next((status for status in statuses if status.module_name == "torch"), None)
    if report.nvidia_driver_detected and torch_status is not None and torch_status.ok:
        try:
            import torch

            if not torch.cuda.is_available():
                warnings.append(
                    "PyTorch is installed but CUDA is unavailable; YOLO body inference will run on CPU. "
                    "Install a CUDA-enabled PyTorch build to accelerate body tracking."
                )
        except Exception as exc:
            warnings.append(f"Could not inspect PyTorch CUDA status: {type(exc).__name__}: {exc}")

    return statuses, warnings


def dedupe_warning_messages(warnings: list[str]) -> list[str]:
    unique_warnings: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        normalized = warning.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_warnings.append(normalized)
    return unique_warnings
