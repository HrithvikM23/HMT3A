from __future__ import annotations

import math
import os

from core.config import PipelineConfig


def is_gpu_backend(config: PipelineConfig) -> bool:
    """Detect whether the current config targets a GPU-backed inference run.

    Returns True when any of these conditions hold:
    - An ONNX execution provider containing 'cuda', 'tensorrt', or 'dml' is requested.
    - ``yolo_device`` or ``rtmpose_device`` is set to a CUDA-like value
      (``"cuda"``, ``"cuda:0"``, ``"0"``, ``0``, etc.) or to ``"mps"``.
    - A GPU-oriented body backend (yolo / rtmpose / rtmpose-wholebody) is
      selected and no explicit ``"cpu"`` device override is present.
    """
    # Check ONNX provider names for GPU providers
    gpu_provider_keywords = {"cuda", "tensorrt", "dml", "rocm"}
    for provider in config.provider_names:
        provider_lower = str(provider).lower()
        if any(keyword in provider_lower for keyword in gpu_provider_keywords):
            return True

    # Check explicit device settings
    for device_value in (config.yolo_device, config.rtmpose_device):
        if device_value is None:
            continue
        # Integer device index (e.g. 0, 1) always means GPU
        if isinstance(device_value, int):
            return True
        device_str = str(device_value).strip().lower()
        # "cuda", "cuda:0", "cuda:1", "mps", etc.
        if device_str in ("mps",) or device_str.startswith("cuda"):
            return True
        # Bare numeric string like "0", "1" — PyTorch GPU device index
        if device_str.isdigit():
            return True

    # GPU-oriented backends without explicit "cpu" override
    if config.body_backend in {"yolo", "rtmpose", "rtmpose-wholebody"}:
        device_text = " ".join(
            str(v) if v is not None else ""
            for v in (config.yolo_device, config.rtmpose_device)
        ).lower()
        if "cpu" not in device_text:
            return True

    return False


def auto_parallel_workers(config: PipelineConfig, total_frames: int = 0, fps: float = 0.0) -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1

    pct = max(10.0, min(100.0, float(getattr(config, "max_cpu_percent", 60.0))))
    max_workers = max(1, math.floor(cpu_count * (pct / 100.0)))

    if total_frames <= 0 or fps <= 0:
        return max(1, min(2, max_workers))

    duration_seconds = total_frames / fps
    if duration_seconds < 3.0:
        return 1

    return max(1, min(2, max_workers))


def resolve_parallel_workers(config: PipelineConfig, *, total_frames: int = 0, fps: float = 0.0) -> int:
    mode = getattr(config, "execution_mode", "auto")
    cpu_count = os.cpu_count() or 1
    if mode == "serial":
        return 1
    if config.parallel_workers > 0:
        return min(config.parallel_workers, cpu_count)
    return auto_parallel_workers(config, total_frames, fps)


def resolve_parallel_chunk_seconds(config: PipelineConfig, *, total_frames: int = 0, fps: float = 0.0) -> float:
    if config.parallel_chunk_seconds > 0:
        return max(0.25, config.parallel_chunk_seconds)

    duration_seconds = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0
    workers = resolve_parallel_workers(config, total_frames=total_frames, fps=fps)
    if workers > 1 and duration_seconds > 0:
        # Dynamically size chunks so all requested worker processes are assigned a chunk
        target_sec = duration_seconds / workers
        return max(0.25, round(target_sec, 2))

    if duration_seconds >= 120:
        return 10.0
    return 5.0


def resolve_parallel_overlap_seconds(config: PipelineConfig) -> float:
    if config.parallel_overlap_seconds == 0:
        return 0.5
    return max(0.0, config.parallel_overlap_seconds)
