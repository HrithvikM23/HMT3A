from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from core.backend_selection import needs_mediapipe, needs_onnx_hand, needs_rtmpose_body, needs_yolo_body
from core.config import PipelineConfig
from utils.privacy import public_path


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except Exception:
        return []
    return [str(provider) for provider in ort.get_available_providers()]


def _torch_cuda_available() -> bool | None:
    try:
        import torch
    except Exception:
        return None
    return bool(torch.cuda.is_available())


def build_runtime_report(config: PipelineConfig) -> dict[str, object]:
    required_modules = ["cv2", "numpy"]
    if needs_mediapipe(config.body_backend, config.hand_backend, config.enable_backend_fallbacks):
        required_modules.append("mediapipe")
    if needs_yolo_body(config.body_backend, config.enable_backend_fallbacks):
        required_modules.extend(["torch", "torchvision", "ultralytics"])
    if needs_rtmpose_body(config.body_backend):
        required_modules.extend(["rtmlib", "onnxruntime"])
    if needs_onnx_hand(config.hand_backend, config.enable_backend_fallbacks):
        required_modules.append("onnxruntime")

    module_status = {
        module_name: _module_available(module_name)
        for module_name in dict.fromkeys(required_modules)
    }
    uses_onnxruntime = "onnxruntime" in module_status
    project_root = config.project_root
    return {
        "python": sys.version.split()[0],
        "python_executable": public_path(sys.executable, project_root=project_root),
        "project_root": public_path(project_root, project_root=project_root),
        "profile": config.profile,
        "body_backend": config.body_backend,
        "hand_backend": config.hand_backend,
        "backend_fallbacks": config.enable_backend_fallbacks,
        "required_modules": module_status,
        "onnx_requested_providers": list(config.provider_names) if uses_onnxruntime else [],
        "onnx_available_providers": _onnx_providers() if module_status.get("onnxruntime") else [],
        "torch_cuda_available": _torch_cuda_available() if module_status.get("torch") else None,
        "models": {
            "body_variant": config.body_model_variant,
            "body_path": public_path(config.body_model_path, project_root=project_root),
            "hand_variant": config.hand_model_variant,
            "hand_path": public_path(config.hand_model_path, project_root=project_root),
            "mediapipe_pose_model": config.mediapipe_pose_model,
            "mediapipe_pose_model_path": (
                public_path(config.mediapipe_pose_model_path, project_root=project_root)
            ),
            "rtmpose_mode": config.rtmpose_mode,
            "rtmpose_backend": config.rtmpose_backend,
            "rtmpose_device": config.rtmpose_device,
        },
        "outputs": {
            "rendered": public_path(config.rendered_output_path, project_root=project_root),
            "json": public_path(config.json_output_path, project_root=project_root),
            "fbx": public_path(config.fbx_output_path, project_root=project_root),
            "metadata": public_path(config.metadata_output_path, project_root=project_root),
        },
    }


def runtime_report_lines(report: dict[str, object]) -> list[str]:
    modules = report.get("required_modules", {})
    if not isinstance(modules, dict):
        modules = {}
    available = ", ".join(name for name, ok in modules.items() if ok) or "none"
    missing = ", ".join(name for name, ok in modules.items() if not ok) or "none"

    lines = [
        f"Runtime profile: {report['profile']}",
        f"Backends: body={report['body_backend']} hand={report['hand_backend']} fallbacks={report['backend_fallbacks']}",
        f"Python: {report['python']} ({Path(str(report['python_executable'])).name})",
        f"Modules available: {available}",
        f"Modules missing: {missing}",
    ]

    onnx_providers = report.get("onnx_available_providers") or []
    requested_providers = report.get("onnx_requested_providers") or []
    if requested_providers or onnx_providers:
        lines.append(f"ONNX providers requested: {', '.join(requested_providers) or 'none'}")
        lines.append(f"ONNX providers available: {', '.join(onnx_providers) or 'none'}")

    cuda = report.get("torch_cuda_available")
    if cuda is not None:
        lines.append(f"PyTorch CUDA available: {cuda}")

    return lines
