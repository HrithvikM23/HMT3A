from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import PipelineConfig
from core.runtime_report import build_runtime_report


def _git_revision(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return None
    revision = completed.stdout.strip()
    return revision or None


def _path_value(path: Path | str | None) -> str | None:
    return None if path is None else str(path)


def build_run_metadata(
    config: PipelineConfig,
    *,
    mode: str,
    fps: float,
    frame_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format": "kinara-run-metadata-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(config.project_root),
        "mode": mode,
        "fps": fps,
        "frame_count": frame_count,
        "source": str(config.video_path),
        "profile": config.profile,
        "runtime": build_runtime_report(config),
        "configuration": {
            "max_people": config.max_people,
            "body_backend": config.body_backend,
            "hand_backend": config.hand_backend,
            "backend_fallbacks": config.enable_backend_fallbacks,
            "single_camera_depth_mode": config.single_camera_depth_mode,
            "triangulate_3d": config.enable_3d_triangulation,
            "osc_enabled": config.osc_enabled,
            "body_detect_interval": config.body_detect_interval,
            "hand_detect_interval": config.hand_detect_interval,
            "processing_width": config.processing_width,
            "body_conf_threshold": config.body_conf_threshold,
            "hand_kp_threshold": config.hand_kp_threshold,
            "identity_hints": {key: list(value) for key, value in config.identity_hints.items()},
            "camera_calibration_path": _path_value(config.camera_calibration_path),
            "calibration_3d_path": _path_value(config.calibration_3d_path),
        },
        "outputs": {
            "rendered": str(config.rendered_output_path),
            "json": str(config.json_output_path),
            "fbx": str(config.fbx_output_path),
            "metadata": str(config.metadata_output_path),
        },
    }
    if extra:
        metadata["extra"] = extra
    return metadata


def write_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
