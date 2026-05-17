from __future__ import annotations

import argparse
import subprocess

from utils.bootstrap_cuda import inspect_runtime, repair_runtime_paths
from utils.bootstrap_packages import (
    dedupe_warning_messages,
    install_packages,
    probe_runtime,
    resolve_install_plan,
)
from utils.bootstrap_paths import ensure_local_environment, find_missing_project_files
from utils.bootstrap_state import MODULE_TO_PACKAGE, VENDOR_DIR, TerminalProgress


def ensure_runtime_ready(*, persist_cudnn_path: bool = True, check_only: bool = False) -> None:
    progress = TerminalProgress(total_steps=6)
    progress.note("Starting Kinara runtime bootstrap...")

    progress.advance("Preparing local runtime folders")
    ensure_local_environment()

    progress.advance("Checking required Kinara files")
    missing_project_files = find_missing_project_files()
    if missing_project_files:
        progress.break_line()
        missing_list = ", ".join(str(path).replace("\\", "/") for path in missing_project_files)
        raise RuntimeError(f"Kinara is missing required project files: {missing_list}")

    progress.advance("Inspecting CUDA / cuDNN runtime")
    report = inspect_runtime()
    repair_runtime_paths(report, persist=persist_cudnn_path)

    progress.advance("Checking Python dependencies")
    initial_statuses, initial_warnings = probe_runtime(report)
    packages_to_install = resolve_install_plan(initial_statuses, report)

    if packages_to_install:
        install_message = "Installing missing Python packages"
        if check_only:
            install_message = "Missing packages detected (install skipped: --check-only)"
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
        print(f"Missing packages: {', '.join(packages_to_install)}")

    progress.advance("Running final verification")
    report = inspect_runtime()
    repair_runtime_paths(report, persist=persist_cudnn_path)
    statuses, warnings = probe_runtime(report)
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
