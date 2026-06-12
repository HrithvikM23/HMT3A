from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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
    "skip_runtime_check",
    "calibrate_cameras",
    "calibration_output",
    "triangulate_3d",
    "calibration_3d",
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
    if source.isdigit():
        return f"Local camera {source}"
    return Path(source).name


def quote(value: str) -> str:
    return f'"{value}"' if " " in value else value


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


def installer_python_path() -> str:
    candidates = [
        os.environ.get("KINARA_PYTHON", ""),
        sys.executable if not getattr(sys, "frozen", False) else "",
    ]
    candidates.extend(_common_python_candidates())
    if os.environ.get("KINARA_ALLOW_BLENDER_PYTHON") == "1":
        candidates.append(r"C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe")
    for candidate in candidates:
        resolved = _valid_python_executable(candidate)
        if resolved is not None:
            return resolved
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
