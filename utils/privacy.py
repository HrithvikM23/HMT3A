from __future__ import annotations

import os
from pathlib import Path


REDACTED_PATH = "<REDACTED_PATH>"
HOME_PLACEHOLDER = "<HOME>"
PROJECT_PLACEHOLDER = "<PROJECT_ROOT>"


def _normalized(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def public_path(value: Path | str | int | None, *, project_root: Path | None = None) -> str | int | None:
    if value is None or isinstance(value, int):
        return value

    text = str(value)
    if not text:
        return text

    path = Path(text)
    if project_root is not None:
        try:
            relative = _normalized(path).relative_to(_normalized(project_root))
        except (OSError, ValueError):
            pass
        else:
            return str(Path(PROJECT_PLACEHOLDER) / relative).replace("\\", "/")

    home = Path.home()
    try:
        relative_home = _normalized(path).relative_to(_normalized(home))
    except (OSError, ValueError):
        relative_home = None
    if relative_home is not None:
        parts = relative_home.parts
        if parts and parts[0].lower() in {"downloads", "desktop", "documents", "onedrive"}:
            return str(Path(HOME_PLACEHOLDER) / parts[0] / "<FILE>").replace("\\", "/")
        return str(Path(HOME_PLACEHOLDER) / relative_home).replace("\\", "/")

    drive, _ = os.path.splitdrive(text)
    if drive:
        return REDACTED_PATH

    return text.replace("\\", "/")


def redact_value(value: object, *, project_root: Path | None = None) -> object:
    if isinstance(value, Path | str | int) or value is None:
        return public_path(value, project_root=project_root)
    if isinstance(value, dict):
        return {key: redact_value(item, project_root=project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item, project_root=project_root) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, project_root=project_root) for item in value]
    return value
