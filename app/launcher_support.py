from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

APP_TITLE = "Kinara"
APP_USER_MODEL_ID = "Kinara.Kinara.Launcher"
COLOR_PRESETS = (
    "black",
    "orange",
    "blue",
    "gray",
    "silver",
    "red",
    "green",
    "yellow",
    "purple",
    "pink",
    "brown",
    "white",
)
MANAGED_DESTS = {
    "source",
    "output",
    "output_dir",
    "max_people",
    "identity_hints",
    "no_preview",
}


def default_text(value: object) -> str:
    if value in (None, argparse.SUPPRESS):
        return ""
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def tile_text(source: str) -> str:
    if source == "No source selected":
        return "Add files or choose camera input"
    if source == "UDP_DEVELOPMENT_MODE":
        return "Waiting for UDP stream - development mode"
    if source.isdigit():
        return f"Local camera {source}"
    return Path(source).name


def quote(value: str) -> str:
    return f'"{value}"' if " " in value else value


def installer_python_path() -> str:
    candidates = [
        os.environ.get("KINARA_PYTHON", ""),
        sys.executable if not getattr(sys, "frozen", False) else "",
    ]
    if os.environ.get("KINARA_ALLOW_BLENDER_PYTHON") == "1":
        candidates.append(r"C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe")
    for candidate in candidates:
        path = Path(candidate) if candidate else None
        if path is not None and path.is_dir():
            path = path / "python.exe"
        if path is not None and path.exists() and path.name.lower() == "python.exe":
            return str(path)
    return ""


def app_icon_path(project_root: Path) -> Path | None:
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            roots.append(Path(bundle_root))
    else:
        roots.append(project_root)

    for root in roots:
        for relative in (Path("assets") / "kinara.ico", Path("assets") / "kinara-mark.png", Path("assets") / "kinara.png"):
            candidate = root / relative
            if candidate.exists():
                return candidate
    return None
