from __future__ import annotations

import sys

from cli import build_parser, explicit_option_dests, resolve_sources
from runtime_profiles import apply_runtime_profile
from utils.bootstrap_dependencies import ensure_runtime_ready


def main() -> None:
    parser = build_parser()
    explicit_dests = explicit_option_dests(parser, sys.argv[1:])
    args = parser.parse_args()
    apply_runtime_profile(args, explicit_dests)

    ensure_runtime_ready()

    from utils.calibration import calibrate_cameras, calibration_available
    from runners.fused import run_fused_assignments
    from runners.single import run_assignment
    from runtime_config import build_config_for_assignment

    assignments = resolve_sources(args)
    if not assignments:
        return

    if args.calibrate_cameras:
        if not calibration_available():
            print("Error: calibrated camera support is not installed in this runtime.")
            return
        output_path = args.calibration_output or (args.output_dir or assignments[0].source.parent if not isinstance(assignments[0].source, int) else None)
        if output_path is None:
            print("Error: --calibration-output is required when calibrating from live camera indices.")
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
            )
        except Exception as exc:
            print(f"Error: camera calibration failed: {exc}")
            return
        print(f"Saved: {saved_path}")
        report_path = saved_path.with_suffix(".quality.json")
        if report_path.exists():
            print(f"Saved: {report_path}")
        return

    if len(assignments) > 1:
        print("Running synchronized multi-camera fusion...")
        run_fused_assignments(assignments, args)
        return

    for assignment in assignments:
        print(f"Running pipeline for {assignment.label}...")
        config = build_config_for_assignment(args, assignment, False)
        run_assignment(config)


if __name__ == "__main__":
    main()
