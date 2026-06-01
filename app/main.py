from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cli import build_parser, explicit_option_dests, resolve_sources
from core.config_file import config_preparser, load_config_defaults
from core.runtime_config import build_config_for_assignment, build_fused_config, prepare_runtime_config
from core.runtime_profiles import apply_runtime_profile
from utils.bootstrap_dependencies import ensure_runtime_ready
from utils.logging import configure_run_log, install_safe_stdio, log_error, log_info, safe_print

install_safe_stdio()


def main() -> None:
    log_path = configure_run_log(os.environ.get("KINARA_LOG_FILE"), prefix="kinara_runner")
    safe_print(f"Log file: {log_path}")
    parser = build_parser()
    pre_args, _ = config_preparser().parse_known_args(sys.argv[1:])
    config_dests: set[str] = set()
    if pre_args.config is not None:
        try:
            config_dests = load_config_defaults(parser, pre_args.config)
        except ValueError as exc:
            log_error(str(exc))
            return

    explicit_dests = config_dests | explicit_option_dests(parser, sys.argv[1:])
    args = parser.parse_args()
    apply_runtime_profile(args, explicit_dests)

    try:
        ensure_runtime_ready(check_only=bool(args.dry_run or args.runtime_check))
    except Exception as exc:
        log_error(f"Runtime check failed: {exc}")
        return

    from runners.fused import run_fused_assignments
    from runners.single import run_assignment
    from utils.calibration import calibrate_cameras, calibration_available

    if args.runtime_check:
        dry_assignment = type("Assignment", (), {"source": 0, "label": "CAM_0"})()
        config = build_config_for_assignment(args, dry_assignment, False)
        prepare_runtime_config(config, prepare_assets=False)
        return

    assignments = resolve_sources(args)
    if not assignments:
        return

    if args.dry_run:
        if len(assignments) > 1:
            config = build_fused_config(args, assignments)
        else:
            config = build_config_for_assignment(args, assignments[0], False)
        if prepare_runtime_config(config, prepare_assets=False):
            log_info("Dry run passed. No video was opened and no models were run.")
        return

    if args.calibrate_cameras:
        if not calibration_available():
            safe_print("Error: calibrated camera support is not installed in this runtime.")
            return
        output_path = args.calibration_output or (args.output_dir or assignments[0].source.parent if not isinstance(assignments[0].source, int) else None)
        if output_path is None:
            safe_print("Error: --calibration-output is required when calibrating from live camera indices.")
            return
        output_path = output_path if output_path.suffix else output_path / "camera_calibration.toml"
        try:
            saved_path = calibrate_cameras(
                assignments,
                output_path,
                squares_x=args.charuco_squares_x,
                squares_y=args.charuco_squares_y,
                square_size=args.charuco_square_size,
                marker_scale=args.charuco_marker_scale,
                marker_bits=args.charuco_marker_bits,
                dict_size=args.charuco_dict_size,
                legacy_pattern=args.charuco_legacy_pattern,
                detection_strictness=args.charuco_detection_strictness,
            )
        except Exception as exc:
            safe_print(f"Error: camera calibration failed: {exc}")
            return
        safe_print(f"Saved: {saved_path}")
        report_path = saved_path.with_suffix(".quality.json")
        if report_path.exists():
            safe_print(f"Saved: {report_path}")
        return

    if len(assignments) > 1:
        safe_print("Running synchronized multi-camera fusion...")
        run_fused_assignments(assignments, args)
        return

    for assignment in assignments:
        safe_print(f"Running pipeline for {assignment.label}...")
        config = build_config_for_assignment(args, assignment, False)
        run_assignment(config)


if __name__ == "__main__":
    main()
