from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

RUN_LOG_PATH: Path | None = None


def _default_log_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / ".kinara_logs"
    return Path.cwd() / ".kinara_logs"


def default_run_log_path(prefix: str = "kinara", root: str | Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = Path(root) if root else _default_log_root()
    return log_root / f"{prefix}_{timestamp}.txt"


def configure_run_log(path: str | Path | None = None, *, prefix: str = "kinara") -> Path:
    global RUN_LOG_PATH
    selected_path = Path(path) if path else default_run_log_path(prefix)
    if selected_path.suffix.lower() != ".txt":
        selected_path = selected_path / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH = selected_path
    _write_log_line(f"\n=== Kinara run started {datetime.now().isoformat(timespec='seconds')} ===\n")
    return selected_path


def current_run_log_path() -> Path | None:
    return RUN_LOG_PATH


def _write_log_line(text: str) -> None:
    if RUN_LOG_PATH is None:
        return
    try:
        with RUN_LOG_PATH.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text)
    except OSError:
        return


class SafeTextIO:
    def __init__(self, wrapped: TextIO) -> None:
        self._wrapped = wrapped

    def write(self, text: str) -> int:
        try:
            written = self._wrapped.write(text)
        except OSError:
            written = 0
        if text:
            _write_log_line(text)
        return written

    def flush(self) -> None:
        try:
            self._wrapped.flush()
        except OSError:
            return

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)


def install_safe_stdio() -> None:
    if not isinstance(sys.stdout, SafeTextIO):
        sys.stdout = SafeTextIO(sys.stdout)  # type: ignore[assignment]
    if not isinstance(sys.stderr, SafeTextIO):
        sys.stderr = SafeTextIO(sys.stderr)  # type: ignore[assignment]


def safe_print(message: str = "", *, end: str = "\n", flush: bool = True) -> None:
    try:
        sys.stdout.write(f"{message}{end}")
        if flush:
            sys.stdout.flush()
    except OSError:
        return


def _emit(level: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    safe_print(f"[{timestamp}] {level}: {message}")


def log_info(message: str) -> None:
    _emit("INFO", message)


def log_warning(message: str) -> None:
    _emit("WARNING", message)


def log_error(message: str) -> None:
    _emit("ERROR", message)
