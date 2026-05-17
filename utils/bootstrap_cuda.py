from __future__ import annotations

import importlib.util
import os
import shutil
import site
import subprocess
from pathlib import Path

from utils.bootstrap_paths import (
    dedupe_paths,
    path_has_glob,
    path_is_dir,
    persist_user_path,
    prepend_env_path,
    register_windows_dll_directory,
    safe_iter_dirs,
)
from utils.bootstrap_state import RuntimeReport, VENDOR_DIR


def find_nvidia_smi() -> Path | None:
    command_path = shutil.which("nvidia-smi")
    if command_path:
        return Path(command_path)

    common_path = Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    if common_path.exists():
        return common_path
    return None


def collect_runtime_roots() -> list[Path]:
    roots: list[Path] = []
    env_names = ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDNN_PATH")
    for env_name in env_names:
        raw_value = os.environ.get(env_name)
        if not raw_value:
            continue
        candidate = Path(raw_value)
        if path_is_dir(candidate):
            roots.append(candidate)

    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if path_is_dir(cuda_root):
        roots.extend(path for path in cuda_root.glob("v*") if path_is_dir(path))

    cudnn_root = Path(r"C:\Program Files\NVIDIA\CUDNN")
    if path_is_dir(cudnn_root):
        roots.extend(path for path in cudnn_root.glob("v*") if path_is_dir(path))

    local_cudnn_root = Path(os.environ.get("LOCALAPPDATA", "")) / "NVIDIA" / "CUDNN"
    if path_is_dir(local_cudnn_root):
        roots.extend(path for path in local_cudnn_root.glob("v*") if path_is_dir(path))

    roots.extend(collect_torch_runtime_roots())
    return dedupe_paths(roots)


def collect_torch_runtime_roots() -> list[Path]:
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
    for root in dedupe_paths([root for root in roots if path_is_dir(root)]):
        candidate_dirs.extend(
            [
                root / "torch" / "lib",
                root / "Lib" / "site-packages" / "torch" / "lib",
            ]
        )

    return dedupe_paths([candidate for candidate in candidate_dirs if path_is_dir(candidate)])


def bin_candidates(root: Path) -> list[Path]:
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
    for child_dir in safe_iter_dirs(bin_dir):
        candidates.append(child_dir)
        candidates.extend(safe_iter_dirs(child_dir))
    return dedupe_paths([candidate for candidate in candidates if path_is_dir(candidate)])


def include_candidates(root: Path) -> list[Path]:
    candidates = [root]
    if root.name.lower() != "include":
        candidates.append(root / "include")
    include_dir = root / "include"
    candidates.extend(safe_iter_dirs(include_dir))
    return dedupe_paths([candidate for candidate in candidates if path_is_dir(candidate)])


def inspect_runtime() -> RuntimeReport:
    report = RuntimeReport()
    nvidia_smi_path = find_nvidia_smi()

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

    for root in collect_runtime_roots():
        for bin_dir in bin_candidates(root):
            if path_has_glob(bin_dir, "cudart*.dll"):
                report.cuda_bin_dirs.append(bin_dir)
            if path_has_glob(bin_dir, "cudnn*.dll"):
                report.cudnn_bin_dirs.append(bin_dir)

        for include_dir in include_candidates(root):
            if (include_dir / "cuda_runtime.h").exists() or (include_dir / "cuda.h").exists():
                report.cuda_include_dirs.append(include_dir)
            if (include_dir / "cudnn.h").exists() or any(include_dir.glob("cudnn*.h")):
                report.cudnn_include_dirs.append(include_dir)

    report.cuda_bin_dirs = dedupe_paths(report.cuda_bin_dirs)
    report.cudnn_bin_dirs = dedupe_paths(report.cudnn_bin_dirs)
    report.cuda_include_dirs = dedupe_paths(report.cuda_include_dirs)
    report.cudnn_include_dirs = dedupe_paths(report.cudnn_include_dirs)

    if report.nvidia_driver_detected and not report.cuda_bin_dirs:
        report.warnings.append("NVIDIA driver detected but CUDA runtime DLLs were not found. Kinara may fall back to CPU.")
    if report.nvidia_driver_detected and not report.cudnn_bin_dirs:
        report.warnings.append("NVIDIA driver detected but cuDNN DLLs were not found. CUDAExecutionProvider may stay unavailable.")
    if report.nvidia_driver_detected and not report.cudnn_include_dirs:
        report.warnings.append("cuDNN headers were not found in common install locations.")

    return report


def repair_runtime_paths(report: RuntimeReport, persist: bool) -> None:
    candidate_dirs = dedupe_paths([*report.cudnn_bin_dirs, *report.cuda_bin_dirs])
    for candidate_dir in candidate_dirs:
        if prepend_env_path(candidate_dir):
            report.path_updates.append(candidate_dir)
        register_windows_dll_directory(candidate_dir)

    if report.cuda_bin_dirs and "CUDA_PATH" not in os.environ:
        cuda_root = report.cuda_bin_dirs[0].parent if report.cuda_bin_dirs[0].name.lower() == "bin" else report.cuda_bin_dirs[0]
        os.environ["CUDA_PATH"] = str(cuda_root)

    if report.cudnn_bin_dirs and "CUDNN_PATH" not in os.environ:
        cudnn_root = report.cudnn_bin_dirs[0].parent if report.cudnn_bin_dirs[0].name.lower() == "bin" else report.cudnn_bin_dirs[0]
        os.environ["CUDNN_PATH"] = str(cudnn_root)

    if persist:
        report.warnings.extend(persist_user_path(report.path_updates))
