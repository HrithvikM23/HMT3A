from __future__ import annotations

import os
import subprocess
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
from utils.bootstrap_paths import ensure_local_environment
from utils.logging import configure_run_log, install_safe_stdio, log_error, log_info, safe_print

install_safe_stdio()
ensure_local_environment()


def _checked_config(builder, *args, **kwargs):
    try:
        return builder(*args, **kwargs)
    except ValueError as exc:
        log_error(str(exc))
        raise SystemExit(1) from exc
    except OSError as exc:
        log_error(f"Could not prepare output paths: {exc}")
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        log_error(str(exc))
        raise SystemExit(1) from exc


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
            raise SystemExit(1) from exc

    explicit_dests = config_dests | explicit_option_dests(parser, sys.argv[1:])
    args = parser.parse_args()
    apply_runtime_profile(args, explicit_dests)

    if not args.skip_runtime_check:
        try:
            ensure_runtime_ready(check_only=bool(args.dry_run))
        except subprocess.CalledProcessError as exc:
            log_error(f"Runtime dependency installation failed with exit code {exc.returncode}. See the run log above.")
            raise SystemExit(1) from exc
        except Exception as exc:
            log_error(f"Runtime check failed: {exc}")
            raise SystemExit(1) from exc

    from runners.fused import run_fused_assignments
    from runners.single import run_assignment
    from utils.calibration import calibrate_cameras, calibration_available

    if args.runtime_check:
        dry_assignment = type("Assignment", (), {"source": 0, "label": "CAM_0"})()
        config = _checked_config(build_config_for_assignment, args, dry_assignment, False)
        if not prepare_runtime_config(config, prepare_assets=True):
            raise SystemExit(1)
        return

    assignments = resolve_sources(args)
    if not assignments:
        raise SystemExit(1)

    if args.dry_run:
        if len(assignments) > 1:
            config = _checked_config(build_fused_config, args, assignments)
        else:
            config = _checked_config(build_config_for_assignment, args, assignments[0], False)
        if prepare_runtime_config(config, prepare_assets=False):
            log_info("Dry run passed. No video was opened and no models were run.")
            return
        raise SystemExit(1)

    if args.calibrate_cameras:
        if not calibration_available():
            safe_print("Error: calibrated camera support is not installed in this runtime.")
            raise SystemExit(1)
        if args.calibration_output is not None:
            output_path = args.calibration_output
        elif args.output_dir is not None:
            output_path = args.output_dir
        elif not isinstance(assignments[0].source, int):
            output_path = assignments[0].source.parent
        else:
            output_path = None
        if output_path is None:
            safe_print("Error: --calibration-output is required when calibrating from live camera indices.")
            raise SystemExit(1)
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
                retry_scale=args.charuco_retry_scale,
                minimum_markers=args.charuco_min_markers,
                retry_sharpen=args.charuco_retry_sharpen,
            )
        except Exception as exc:
            safe_print(f"Error: camera calibration failed: {exc}")
            raise SystemExit(1) from exc
        safe_print(f"Saved: {saved_path}")
        report_path = saved_path.with_suffix(".quality.json")
        if report_path.exists():
            safe_print(f"Saved: {report_path}")
        if args.triangulate_3d:
            args.calibrate_cameras = False
            args.calibration_3d = saved_path
            safe_print("Calibration complete. Starting triangulation with the saved calibration file...")
            if len(assignments) < 2:
                safe_print("Error: triangulation needs at least two sources.")
                raise SystemExit(1)
            if not run_fused_assignments(assignments, args):
                raise SystemExit(1)
        return

    if len(assignments) > 1:
        safe_print("Running synchronized multi-camera fusion...")
        if not run_fused_assignments(assignments, args):
            raise SystemExit(1)
        return

    for assignment in assignments:
        safe_print(f"Running pipeline for {assignment.label}...")
        config = _checked_config(build_config_for_assignment, args, assignment, False)
        if not run_assignment(config):
            raise SystemExit(1)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
