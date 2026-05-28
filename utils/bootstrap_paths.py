from __future__ import annotations

import ctypes
import os
import site
import sys
from pathlib import Path

from utils.bootstrap_state import PROJECT_ROOT, REQUIRED_PROJECT_FILES, ULTRALYTICS_CONFIG_DIR, VENDOR_DIR

try:
    import winreg
except ImportError:  # pragma: no cover
    winreg = None  # type: ignore[assignment]


WINDOWS_DLL_DIRECTORIES: list[object] = []


def dedupe_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = os.path.normcase(os.path.normpath(str(path)))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(path)
    return unique_paths


def path_is_dir(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir()
    except OSError:
        return False


def safe_iter_dirs(path: Path) -> list[Path]:
    if not path_is_dir(path):
        return []
    try:
        return [candidate for candidate in path.iterdir() if path_is_dir(candidate)]
    except OSError:
        return []


def path_has_glob(path: Path, pattern: str) -> bool:
    if not path_is_dir(path):
        return False
    try:
        return any(path.glob(pattern))
    except OSError:
        return False


def prepend_sys_path(path: Path) -> None:
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


def prepend_env_path(path: Path) -> bool:
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


def register_windows_dll_directory(path: Path) -> bool:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return False
    if not path_is_dir(path):
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


def prepend_pythonpath(path: Path) -> None:
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


def broadcast_environment_change() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, 0)
    except Exception:
        return


def persist_user_path(path_updates: list[Path]) -> list[str]:
    if os.name != "nt" or winreg is None or not path_updates:
        return []

    warnings: list[str] = []
    unique_updates = dedupe_paths(path_updates)
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
                broadcast_environment_change()
    except OSError as exc:
        warnings.append(f"Could not persist CUDA/cuDNN PATH updates: {exc}")

    return warnings


def find_missing_project_files() -> list[Path]:
    missing_paths: list[Path] = []
    for relative_path in REQUIRED_PROJECT_FILES:
        candidate = PROJECT_ROOT / relative_path
        if not candidate.exists():
            missing_paths.append(relative_path)
    return missing_paths


def ensure_local_environment() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    prepend_sys_path(VENDOR_DIR)
    prepend_pythonpath(VENDOR_DIR)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
