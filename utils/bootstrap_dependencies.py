from __future__ import annotations

import argparse
import subprocess
import sys
from types import SimpleNamespace

from core.backend_selection import needs_mediapipe, needs_onnx_hand, needs_yolo_body, resolve_backend_selection
from utils.bootstrap_cuda import inspect_runtime, repair_runtime_paths
from utils.bootstrap_packages import (
    dedupe_warning_messages,
    install_packages,
    probe_runtime,
    resolve_install_plan,
)
from utils.bootstrap_paths import ensure_local_environment, find_missing_project_files
from utils.bootstrap_state import MODULE_TO_PACKAGE, VENDOR_DIR, TerminalProgress


def _option_value(argv: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, token in enumerate(argv):
        if token.startswith(prefix):
            return token.split("=", 1)[1]
        if token == option and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _selected_runtime_modules(argv: list[str] | None = None) -> tuple[str, ...]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = SimpleNamespace(
        landmark_backend=_option_value(tokens, "--landmark-backend") or "yolo",
        body_backend=_option_value(tokens, "--body-backend"),
        hand_backend=_option_value(tokens, "--hand-backend"),
        backend_fallbacks="--backend-fallbacks" in tokens,
    )
    body_backend, hand_backend, enable_fallbacks = resolve_backend_selection(args)

    modules = ["cv2", "numpy"]
    if needs_yolo_body(body_backend, enable_fallbacks):
        modules.extend(["torch", "torchvision", "ultralytics"])
    if needs_onnx_hand(hand_backend, enable_fallbacks):
        modules.append("onnxruntime")
    if needs_mediapipe(body_backend, hand_backend, enable_fallbacks):
        modules.append("mediapipe")

    return tuple(dict.fromkeys(module for module in modules if module in MODULE_TO_PACKAGE))


def ensure_runtime_ready(*, persist_cudnn_path: bool = True, check_only: bool = False) -> None:
    frozen_app = getattr(sys, "frozen", False)
    runtime_modules = _selected_runtime_modules()
    progress = TerminalProgress(total_steps=6)
    progress.note("Starting Kinara runtime bootstrap...")

    progress.advance("Preparing local runtime folders")
    ensure_local_environment()

    progress.advance("Checking required Kinara files")
    if frozen_app:
        missing_project_files = []
    else:
        missing_project_files = find_missing_project_files()
        if missing_project_files:
            progress.break_line()
            missing_list = ", ".join(str(path).replace("\\", "/") for path in missing_project_files)
            raise RuntimeError(f"Kinara is missing required project files: {missing_list}")

    progress.advance("Inspecting CUDA / cuDNN runtime")
    report = inspect_runtime()
    repair_runtime_paths(report, persist=persist_cudnn_path)

    progress.advance("Checking Python dependencies")
    initial_statuses, initial_warnings = probe_runtime(report, runtime_modules)
    packages_to_install = resolve_install_plan(initial_statuses, report)

    if packages_to_install:
        install_message = "Installing missing Python packages"
        if check_only:
            install_message = "Missing packages detected (install skipped)"
        progress.advance(install_message)
    else:
        progress.advance("All Python packages already available")

    if packages_to_install and not check_only:
        progress.break_line()
        print(f"Preparing runtime dependencies in {VENDOR_DIR}...")
        print(f"Installing only missing packages: {', '.join(packages_to_install)}")
        install_packages(packages_to_install)
    elif packages_to_install and check_only:
        progress.break_line()
        print(f"Missing packages (--check-only): {', '.join(packages_to_install)}")

    progress.advance("Running final verification")
    report = inspect_runtime()
    repair_runtime_paths(report, persist=persist_cudnn_path)
    statuses, warnings = probe_runtime(report, runtime_modules)
    warnings = [*initial_warnings, *warnings]

    failed_statuses = [status for status in statuses if not status.ok]
    if failed_statuses:
        progress.break_line()
        lines = []
        for status in failed_statuses:
            package_name = MODULE_TO_PACKAGE.get(status.module_name, status.module_name)
            detail = status.error or "Unknown import failure"
            lines.append(f"- {status.module_name} ({package_name}): {detail}")
        raise RuntimeError("Runtime bootstrap could not satisfy required dependencies:\n" + "\n".join(lines))

    progress.finish("Kinara runtime ready")

    if report.path_updates:
        joined_paths = ", ".join(str(path) for path in report.path_updates)
        print(f"Updated CUDA/cuDNN PATH entries: {joined_paths}")

    for warning in dedupe_warning_messages(warnings):
        print(f"Runtime warning: {warning}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Kinara's local runtime dependencies.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the runtime without installing missing Python packages.",
    )
    parser.add_argument(
        "--no-persist-cudnn-path",
        action="store_true",
        help="Only patch CUDA/cuDNN PATH values for the current process.",
    )
    return parser


def main() -> int:
    parser = _build_cli_parser()
    args = parser.parse_args()

    try:
        ensure_runtime_ready(
            persist_cudnn_path=not args.no_persist_cudnn_path,
            check_only=args.check_only,
        )
    except RuntimeError as exc:
        print(f"Runtime bootstrap failed: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Dependency installation failed with exit code {exc.returncode}.")
        return exc.returncode or 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
