from __future__ import annotations

import base64
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

IS_FROZEN = getattr(sys, "frozen", False)
EXE_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parent.parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", str(EXE_DIR))) if IS_FROZEN else Path(__file__).resolve().parent.parent

PROJECT_ROOT = RESOURCE_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import webview
from app.launcher_support import APP_TITLE, installer_python_path, quote
from utils.logging import default_run_log_path, install_safe_stdio

install_safe_stdio()

CHARUCO_A3_PRESET = {
    "calibrate_cameras": True,
    "charuco_squares_x": "11",
    "charuco_squares_y": "8",
    "charuco_square_size": "36",
    "charuco_marker_scale": "0.6667",
    "charuco_marker_bits": "4",
    "charuco_dict_size": "50",
    "charuco_detection_strictness": "balanced",
    "charuco_retry_scale": "",
    "charuco_min_markers": "",
    "charuco_retry_sharpen": False,
}

CHARUCO_RESCUE_PRESET = {
    **CHARUCO_A3_PRESET,
    "charuco_detection_strictness": "lenient",
    "charuco_retry_scale": "3.5",
    "charuco_min_markers": "6",
    "charuco_retry_sharpen": True,
}

CHARUCO_PAPER_PRESETS = {
    "A3": {"charuco_squares_x": "11", "charuco_squares_y": "8", "charuco_square_size": "36"},
    "A4": {"charuco_squares_x": "9", "charuco_squares_y": "6", "charuco_square_size": "28"},
    "Letter": {"charuco_squares_x": "9", "charuco_squares_y": "6", "charuco_square_size": "26"},
    "Legal": {"charuco_squares_x": "11", "charuco_squares_y": "7", "charuco_square_size": "30"},
}

MODEL_WEIGHT_VALUES = {
    "Lite": {"hand_model_variant": "low", "hand_input_size": "512"},
    "Full": {"hand_model_variant": "high", "hand_input_size": "640"},
    "Heavy": {"hand_model_variant": "max", "hand_input_size": "768"},
}

PRESETS = {
    "MediaPipe": {
        "profile": "fastest",
        "landmark_backend": "mediapipe",
        "body_backend": "mediapipe",
        "hand_backend": "mediapipe",
        "processing_width": "640",
    },
    "RTMPose": {
        "landmark_backend": "rtmpose",
        "body_backend": "rtmpose",
        "hand_backend": "onnx",
        "rtmpose_device": "cuda",
        "hand_model_variant": "max",
    },
    "ONNX": {
        "landmark_backend": "yolo",
        "body_backend": "yolo",
        "hand_backend": "onnx",
        "hand_model_variant": "max",
    },
    "RTMPose WholeBody": {
        "landmark_backend": "rtmpose-wholebody",
        "rtmpose_device": "cuda",
        "rtmpose_tracking": True,
    },
}


class KinaraWebAPI:
    def __init__(self) -> None:
        self._window: webview.Window | None = None
        self._process: subprocess.Popen[str] | None = None
        self.sources: list[str] = []
        self.destination: str = str(EXE_DIR / "outputs")
        self.people_count: int = 1
        self.person_colors: dict[str, str] = {}
        self.advanced_values: dict[str, Any] = {}
        self.calibrate_mode: bool = False
        self.triangulate_after_calibration: bool = False
        self.enable_triangulation: bool = False
        self.triangulation_path: str = ""
        self._preview_thread: threading.Thread | None = None
        self._preview_running: bool = False
        self._preview_error_logged: bool = False
        self._lock = threading.Lock()

    def set_window(self, window: Any) -> None:
        self._window = window
        if window is not None and hasattr(window, "events") and hasattr(window.events, "closed"):
            try:
                window.events.closed += self._on_window_closed
            except Exception:
                pass

    def _on_window_closed(self) -> None:
        self._window = None
        with self._lock:
            self._preview_running = False
        self.stop_run()

    def safe_evaluate_js(self, script: str) -> None:
        if self._window is None:
            return
        try:
            self._window.evaluate_js(script)
        except (Exception, BaseException):
            self._window = None

    def log(self, text: str) -> None:
        import json
        self.safe_evaluate_js(f"window.onKinaraLog({json.dumps(text)});")

    def set_status(self, status: str, status_type: str = "idle") -> None:
        import json
        self.safe_evaluate_js(f"window.onKinaraStatus({json.dumps(status)}, {json.dumps(status_type)});")

    def get_initial_command(self) -> str:
        return self._build_command_string()

    def set_sources(self, sources: list[str]) -> str:
        self.sources = sources
        return self._build_command_string()

    def set_destination(self, path: str) -> str:
        self.destination = path
        return self._build_command_string()

    def set_people_count(self, count: int) -> str:
        self.people_count = max(1, min(12, int(count)))
        return self._build_command_string()

    def set_person_color(self, person_key: str, color: str) -> str:
        self.person_colors[person_key] = color
        return self._build_command_string()

    def set_checkbox(self, key: str, value: bool) -> str:
        if key == "calibrate_cameras":
            self.calibrate_mode = bool(value)
        elif key == "triangulate_after_calibration":
            self.triangulate_after_calibration = bool(value)
        elif key == "triangulate_3d":
            self.enable_triangulation = bool(value)
        elif key == "charuco_rescue_mode":
            preset = CHARUCO_RESCUE_PRESET if value else CHARUCO_A3_PRESET
            self.advanced_values.update(preset)
        return self._build_command_string()

    def set_triangulation_path(self, path: str) -> str:
        self.triangulation_path = path
        return self._build_command_string()

    def set_paper_size(self, size: str) -> str:
        values = CHARUCO_PAPER_PRESETS.get(size)
        if values:
            self.advanced_values.update(values)
        return self._build_command_string()

    def apply_preset(self, preset_title: str, option: str | None, weight: str | None) -> str:
        self.advanced_values.clear()
        base_values = PRESETS.get(preset_title, {})
        self.advanced_values.update(base_values)

        if preset_title == "MediaPipe" and option:
            model_map = {"Lite": "pose_landmark_lite.tflite", "Full": "pose_landmark_full.tflite", "Heavy": "pose_landmark_heavy.tflite"}
            self.advanced_values["model"] = model_map.get(option, "pose_landmark_full.tflite")
        elif preset_title == "RTMPose" and option:
            mode_map = {"Lite": "lightweight", "Full": "balanced", "Heavy": "performance"}
            self.advanced_values["rtmpose_mode"] = mode_map.get(option, "balanced")
        elif preset_title == "ONNX" and option:
            yolo_map = {
                "Nano": ("yolo11n-pose.pt", "640"),
                "Small": ("yolo11s-pose.pt", "640"),
                "Medium": ("yolo11m-pose.pt", "768"),
                "Large": ("yolo11l-pose.pt", "832"),
                "X-Large": ("yolo11x-pose.pt", "960"),
            }
            if option in yolo_map:
                model, size = yolo_map[option]
                self.advanced_values["model"] = model
                self.advanced_values["body_input_size"] = size
        elif preset_title == "RTMPose WholeBody" and option:
            self.advanced_values["rtmpose_mode"] = option.lower()

        if weight and weight in MODEL_WEIGHT_VALUES:
            self.advanced_values.update(MODEL_WEIGHT_VALUES[weight])

        return self._build_command_string()

    def apply_charuco_preset(self, preset_type: str) -> str:
        preset = CHARUCO_RESCUE_PRESET if preset_type == "Rescue" else CHARUCO_A3_PRESET
        self.advanced_values.update(preset)
        return self._build_command_string()

    def reset_defaults(self) -> str:
        self.sources.clear()
        self.destination = str(EXE_DIR / "outputs")
        self.people_count = 1
        self.person_colors.clear()
        self.advanced_values.clear()
        self.calibrate_mode = False
        self.triangulate_after_calibration = False
        self.enable_triangulation = False
        self.triangulation_path = ""
        return self._build_command_string()

    def set_parallel_workers(self, count: int) -> str:
        self.advanced_values["parallel_workers"] = str(count)
        return self._build_command_string()

    def get_benchmark_telemetry(self, workers: int = 4, sample_seconds: float = 60.0) -> dict[str, Any]:
        workers_count = max(1, int(workers))
        sample_sec = max(1.0, float(sample_seconds))
        # Estimated baseline FPS — not measured from actual hardware profiling
        serial_fps = 42.5
        parallel_fps = serial_fps * (1.0 + (workers_count - 1) * 0.82)
        total_frames = sample_sec * 30.0
        serial_time = total_frames / serial_fps
        parallel_time = total_frames / parallel_fps
        speedup = serial_time / max(0.001, parallel_time)
        return {
            "workers": workers_count,
            "sample_seconds": sample_sec,
            "total_frames": int(total_frames),
            "serial_time_sec": round(serial_time, 2),
            "parallel_time_sec": round(parallel_time, 2),
            "time_saved_sec": round(serial_time - parallel_time, 2),
            "serial_fps": round(serial_fps, 1),
            "parallel_fps": round(parallel_fps, 1),
            "speedup_metric": f"{speedup:.2f}x",
            "speedup_factor": round(speedup, 2),
        }

    def browse_files(self) -> list[str]:
        if self._window is None:
            return []
        try:
            result = self._window.create_file_dialog(
                cast(int, webview.OPEN_DIALOG),
                allow_multiple=True,
                file_types=("Video files (*.mp4;*.avi;*.mov;*.mkv)", "All files (*.*)"),
            )
            return list(result) if result else []
        except (Exception, BaseException):
            self._window = None
            return []

    def browse_destination(self) -> str:
        if self._window is None:
            return ""
        try:
            result = self._window.create_file_dialog(cast(int, webview.FOLDER_DIALOG))
            return result[0] if result else ""
        except (Exception, BaseException):
            self._window = None
            return ""

    def browse_triangulation(self) -> str:
        if self._window is None:
            return ""
        try:
            result = self._window.create_file_dialog(
                cast(int, webview.OPEN_DIALOG),
                file_types=("Calibration files (*.toml;*.json)", "All files (*.*)"),
            )
            return result[0] if result else ""
        except (Exception, BaseException):
            self._window = None
            return ""

    def start_run(self) -> dict[str, Any]:
        if self._process is not None and self._process.poll() is None:
            return {"log": "Kinara is already running."}
        if not self.sources:
            return {"log": "Please select at least one input source."}

        command = self._build_command_list()
        self._start_process(command, status_text="Running", enable_preview=True)
        return {"log": f"Started run: {' '.join(quote(p) for p in command)}"}

    def check_runtime(self) -> dict[str, Any]:
        if self._process is not None and self._process.poll() is None:
            return {"log": "Kinara is already running."}

        command = self._build_runner_command([
            *self._advanced_args(),
            *self._workflow_args(for_runtime_check=True),
            "--max-people", str(self.people_count),
            "--runtime-check",
            "--no-preview",
        ])
        self._start_process(command, status_text="Checking", enable_preview=False)
        return {"log": "Checking Kinara runtime dependencies..."}

    def stop_run(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self.set_status("Stopping...", "idle")
            try:
                self._process.terminate()
            except (ProcessLookupError, OSError):
                pass

    def kill_run(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self.set_status("Killed", "error")
            try:
                self._process.kill()
            except (ProcessLookupError, OSError):
                pass

    def _start_process(self, command: list[str], status_text: str, enable_preview: bool) -> None:
        log_path = default_run_log_path("kinara_run", root=PROJECT_ROOT / ".kinara_logs")
        self.log(f"> {' '.join(quote(part) for part in command)}")
        self.log(f"Log file: {log_path}")
        self.set_status(status_text, "running" if enable_preview else "checking")

        env = os.environ.copy()
        env["KINARA_PYTHON"] = installer_python_path()
        env["KINARA_LOG_FILE"] = str(log_path)
        runtime_dir = PROJECT_ROOT / ".kinara_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        # Clear stale preview files from previous runs
        for old_file in runtime_dir.glob("preview*.*"):
            try:
                old_file.unlink()
            except OSError:
                pass

        self.safe_evaluate_js("window.resetPreviewStage();")

        preview_frame_path = runtime_dir / "preview.jpg"
        env["KINARA_PREVIEW_FRAME"] = str(preview_frame_path)

        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            threading.Thread(target=self._stream_stdout, daemon=True).start()
            if enable_preview:
                with self._lock:
                    self._preview_running = True
                self._preview_thread = threading.Thread(target=self._watch_preview_frames, args=(preview_frame_path,), daemon=True)
                self._preview_thread.start()
        except Exception as err:
            self.log(f"Failed to launch process: {err}")
            self.set_status("Error", "error")

    def _stream_stdout(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        for line in iter(self._process.stdout.readline, ""):
            if line:
                self.log(line.rstrip())
        self._process.stdout.close()
        rc = self._process.wait()
        with self._lock:
            self._preview_running = False
        self.set_status("Finished" if rc == 0 else f"Exited {rc}", "idle" if rc == 0 else "error")

    def _watch_preview_frames(self, frame_path: Path) -> None:
        self._last_preview_file: Path | None = None
        with self._lock:
            running = self._preview_running
        while running:
            time.sleep(0.04)
            try:
                frame_paths = sorted(frame_path.parent.glob(f"{frame_path.stem}*{frame_path.suffix}"))
                if not frame_paths:
                    latest = frame_path if frame_path.exists() else None
                else:
                    latest = frame_paths[-1]

                if latest is not None and latest != self._last_preview_file:
                    data = None
                    for _ in range(3):
                        try:
                            data = latest.read_bytes()
                            if data:
                                break
                        except OSError:
                            time.sleep(0.01)

                    if data:
                        b64 = base64.b64encode(data).decode("utf-8")
                        self._last_preview_file = latest

                        # Determine camera ID from filename
                        cam_id = "CAM_0"
                        fname_upper = latest.name.upper()
                        if "CAM_" in fname_upper:
                            parts = fname_upper.split("CAM_")
                            if len(parts) > 1 and parts[1][0].isdigit():
                                cam_id = f"CAM_{parts[1][0]}"
                        elif "WORKER_" in fname_upper:
                            parts = fname_upper.split("WORKER_")
                            if len(parts) > 1 and parts[1][0].isdigit():
                                cam_id = f"CAM_{parts[1][0]}"

                        self.safe_evaluate_js(f"window.onKinaraPreviewFrame('{b64}', '{cam_id}');")
            except Exception:
                pass

            with self._lock:
                running = self._preview_running

    def _build_command_string(self) -> str:
        return " ".join(quote(p) for p in self._build_command_list())

    def _build_runner_command(self, args: list[str]) -> list[str]:
        if IS_FROZEN:
            return [sys.executable, "--kinara-runner", *args]
        return [sys.executable, str(PROJECT_ROOT / "app" / "kinara_launcher.py"), "--kinara-runner", *args]

    def _build_command_list(self) -> list[str]:
        args: list[str] = []
        for source in self.sources:
            args.extend(["--source", source])
        if self.destination:
            args.extend(["--output-dir", self.destination])
        args.extend(["--max-people", str(self.people_count)])
        if self.people_count >= 2:
            for idx in range(1, self.people_count + 1):
                color = self.person_colors.get(f"person{idx}")
                if color:
                    args.extend(["--identity", f"person{idx}={color}"])
        args.extend(self._advanced_args())
        args.extend(self._workflow_args())
        args.append("--skip-runtime-check")
        args.append("--no-preview")
        return self._build_runner_command(args)

    def _workflow_args(self, for_runtime_check: bool = False) -> list[str]:
        args: list[str] = []
        if self.calibrate_mode:
            args.append("--calibrate-cameras")
            if not for_runtime_check and self.triangulate_after_calibration:
                args.append("--triangulate-3d")
        if self.enable_triangulation and self.triangulation_path:
            args.extend(["--triangulate-3d", "--calibration-3d", self.triangulation_path])
        return args

    def _advanced_args(self) -> list[str]:
        args: list[str] = []
        for k, v in self.advanced_values.items():
            if v is True:
                args.append(f"--{k.replace('_', '-')}")
            elif v and v is not False:
                args.extend([f"--{k.replace('_', '-')}", str(v)])
        return args


def main() -> None:
    api = KinaraWebAPI()
    ui_html = PROJECT_ROOT / "app" / "ui" / "index.html"
    window = webview.create_window(
        title=f"{APP_TITLE} Motion Capture Host",
        url=str(ui_html),
        width=1380,
        height=840,
        min_size=(1120, 700),
        resizable=True,
        maximized=True,
        js_api=api,
    )
    api.set_window(window)
    webview.start(debug=False)


if __name__ == "__main__":
    main()
