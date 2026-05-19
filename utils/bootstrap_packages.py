from __future__ import annotations

import importlib
import os
import subprocess
import sys

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore[no-redef]

from utils.bootstrap_paths import dedupe_paths, prepend_pythonpath, prepend_sys_path
from utils.bootstrap_state import MODULE_TO_PACKAGE, ModuleStatus, RuntimeReport, VENDOR_DIR


def distribution_installed(distribution_name: str) -> bool:
    try:
        importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


def module_status(module_name: str) -> ModuleStatus:
    try:
        importlib.invalidate_caches()
        importlib.import_module(module_name)
        return ModuleStatus(module_name=module_name, ok=True)
    except Exception as exc:
        return ModuleStatus(module_name=module_name, ok=False, error=f"{type(exc).__name__}: {exc}")


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
        packages_to_install.append(choose_onnxruntime_distribution(report))
        missing_modules.discard("onnxruntime")
    elif gpu_runtime_detected and not distribution_installed("onnxruntime-gpu"):
        packages_to_install.append("onnxruntime-gpu")

    if "mediapipe" in missing_modules:
        packages_to_install.append(MODULE_TO_PACKAGE["mediapipe"])
        missing_modules.discard("mediapipe")

    return packages_to_install


def ensure_pip() -> None:
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


def install_packages(packages: list[str]) -> None:
    if not packages:
        return

    ensure_pip()
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
    prepend_pythonpath(VENDOR_DIR)
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    subprocess.run(command, check=True, env=env)
    importlib.invalidate_caches()
    prepend_sys_path(VENDOR_DIR)


def probe_runtime(report: RuntimeReport) -> tuple[list[ModuleStatus], list[str]]:
    statuses = module_group_status(tuple(MODULE_TO_PACKAGE))
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
