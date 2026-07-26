from __future__ import annotations

import base64
import math
import os
import subprocess
import queue
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
        self._user_stopped: bool = False
        self._lock = threading.Lock()
        
        self._js_queue = queue.Queue(maxsize=200)
        self._preview_queue = queue.Queue(maxsize=5)
        self._js_thread: threading.Thread | None = None

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
            
        if script.startswith("window.onKinaraPreviewFrame"):
            try:
                self._preview_queue.put_nowait(script)
            except queue.Full:
                try:
                    self._preview_queue.get_nowait()
                    self._preview_queue.put_nowait(script)
                except (queue.Empty, queue.Full):
                    pass
        else:
            try:
                self._js_queue.put_nowait(script)
            except queue.Full:
                try:
                    self._js_queue.get_nowait()
                    self._js_queue.put_nowait(script)
                except (queue.Empty, queue.Full):
                    pass

    def _execute_js(self, script: str) -> None:
        if self._window is None:
            return
        try:
            self._window.evaluate_js(script)
        except (Exception, BaseException):
            self._window = None
        except:
            self._window = None

    def _js_consumer(self) -> None:
        log_batch = []
        last_log_push = time.time()
        
        def flush_logs():
            if log_batch:
                import json
                joined = "\n".join(log_batch)
                self._execute_js(f"window.onKinaraLog({json.dumps(joined)});")
                log_batch.clear()

        while True:
            with self._lock:
                running = self._preview_running
                
            try:
                script = self._preview_queue.get_nowait()
                self._execute_js(script)
                continue
            except queue.Empty:
                pass
                
            try:
                script = self._js_queue.get(timeout=0.05)
                if script.startswith("window.onKinaraLog"):
                    try:
                        import json
                        log_text = json.loads(script[19:-2])
                        log_batch.append(log_text)
                    except:
                        self._execute_js(script)
                else:
                    flush_logs()
                    self._execute_js(script)
            except queue.Empty:
                if not running and self._js_queue.empty() and self._preview_queue.empty():
                    break
                    
            if log_batch and time.time() - last_log_push >= 0.1:
                flush_logs()
                last_log_push = time.time()
                
        flush_logs()

    def log(self, text: str) -> None:
        import json
        self.safe_evaluate_js(f"window.onKinaraLog({json.dumps(text)});")

    def set_status(self, status: str, status_type: str = "idle") -> None:
        import json
        self.safe_evaluate_js(f"window.onKinaraStatus({json.dumps(status)}, {json.dumps(status_type)});")

    def get_initial_command(self) -> str:
        return self._build_command_string()

    def _file_dialog_open(self) -> Any:
        if hasattr(webview, "FileDialog") and hasattr(webview.FileDialog, "OPEN"):
            return webview.FileDialog.OPEN
        return getattr(webview, "OPEN_DIALOG", 10)

    def _file_dialog_folder(self) -> Any:
        if hasattr(webview, "FileDialog") and hasattr(webview.FileDialog, "FOLDER"):
            return webview.FileDialog.FOLDER
        return getattr(webview, "FOLDER_DIALOG", 20)

    def browse_files(self) -> list[str]:
        if self._window is None:
            return []
        try:
            result = self._window.create_file_dialog(
                self._file_dialog_open(),
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
            result = self._window.create_file_dialog(self._file_dialog_folder())
            return result[0] if result else ""
        except (Exception, BaseException):
            self._window = None
            return ""

    def browse_triangulation(self) -> str:
        if self._window is None:
            return ""
        try:
            result = self._window.create_file_dialog(
                self._file_dialog_open(),
                file_types=("Calibration files (*.toml;*.json)", "All files (*.*)"),
            )
            return result[0] if result else ""
        except (Exception, BaseException):
            self._window = None
            return ""

    def add_camera(self, camera_index_or_path: str) -> str:
        if camera_index_or_path and camera_index_or_path not in self.sources:
            self.sources.append(camera_index_or_path)
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

    def set_advanced_option(self, key: str, value: Any) -> str:
        """Set a single advanced CLI option by key."""
        if value == '' or value is None:
            self.advanced_values.pop(key, None)
        else:
            self.advanced_values[key] = value
        return self._build_command_string()

    def set_profile(self, profile: str) -> str:
        """Set the performance profile."""
        self.advanced_values['profile'] = profile
        return self._build_command_string()

    def set_workflow_option(self, key: str, value: Any) -> str:
        """Set a workflow-level option (triangulation, calibration, etc.)."""
        if key == 'enable_triangulation':
            self.enable_triangulation = bool(value)
        elif key == 'calibrate_mode':
            self.calibrate_mode = bool(value)
        else:
            self.advanced_values[key] = value
        return self._build_command_string()

    def apply_preset_weight(self, weight: str) -> str:
        """Apply a hand model weight preset."""
        values = MODEL_WEIGHT_VALUES.get(weight)
        if values:
            self.advanced_values.update(values)
        return self._build_command_string()

    def set_execution_mode(self, mode: str) -> str:
        """Set the execution mode (auto/serial/parallel) and pipeline-parallel flag."""
        if mode == 'pipeline-parallel':
            self.advanced_values['pipeline_parallel'] = True
            self.advanced_values.pop('execution_mode', None)
        else:
            self.advanced_values.pop('pipeline_parallel', None)
            if mode == 'auto' or mode == '':
                self.advanced_values.pop('execution_mode', None)
            else:
                self.advanced_values['execution_mode'] = mode
        return self._build_command_string()

    def set_parallel_workers(self, count: int) -> str:
        """Set parallel worker count (0 = auto 60% CPU cap)."""
        val = int(count)
        if val <= 0:
            self.advanced_values.pop("parallel_workers", None)
        else:
            cpu_cap = os.cpu_count() or 1
            self.advanced_values["parallel_workers"] = str(min(val, cpu_cap))
        return self._build_command_string()

    def set_max_cpu_percent(self, pct: float) -> str:
        """Set max CPU percent allocation (10.0 to 100.0)."""
        val = max(10.0, min(100.0, float(pct)))
        if val == 60.0:
            self.advanced_values.pop("max_cpu_percent", None)
        else:
            self.advanced_values["max_cpu_percent"] = f"{val:.1f}"
        return self._build_command_string()

    def get_system_info(self) -> dict[str, Any]:
        """Return system hardware specs for dynamic UI constraints."""
        cpu_count = os.cpu_count() or 1
        return {
            "cpu_count": cpu_count,
            "cpu_60_cap": max(1, math.floor(cpu_count * 0.60)),
        }

    def start_run(self, custom_command: str | None = None) -> dict[str, Any]:
        if self._process is not None and self._process.poll() is None:
            return {"log": "Kinara is already running."}
        if not self.sources and not (custom_command and "--source" in custom_command):
            return {"log": "Please select at least one input source."}

        if custom_command and custom_command.strip():
            import shlex
            command = shlex.split(custom_command)
            if "parallel_workers" in self.advanced_values and "--parallel-workers" not in command:
                command.extend(["--parallel-workers", str(self.advanced_values["parallel_workers"])])
            if "max_cpu_percent" in self.advanced_values and "--max-cpu-percent" not in command:
                command.extend(["--max-cpu-percent", str(self.advanced_values["max_cpu_percent"])])
        else:
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
        self._user_stopped = True
        with self._lock:
            self._preview_running = False
        if self._process is not None and self._process.poll() is None:
            self.set_status("Stopping...", "idle")
            self._terminate_process_tree(force=True)
            self.log("[Launcher] Stopped execution and worker processes.")
        self.safe_evaluate_js("window.resetPreviewStage();")

    def kill_run(self) -> None:
        self._user_stopped = True
        with self._lock:
            self._preview_running = False
        if self._process is not None and self._process.poll() is None:
            self.set_status("Killed", "error")
            self._terminate_process_tree(force=True)
            self.log("[Launcher] Terminated process tree.")
        self.safe_evaluate_js("window.resetPreviewStage();")

    def _terminate_process_tree(self, force: bool = False) -> None:
        if self._process is None:
            return
        pid = self._process.pid
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                if force:
                    self._process.kill()
                else:
                    self._process.terminate()
        except Exception:
            pass
        finally:
            self._process = None

    def _start_process(self, command: list[str], status_text: str, enable_preview: bool) -> None:
        self._user_stopped = False
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
            with self._lock:
                self._preview_running = True
                
            if self._js_thread is None or not self._js_thread.is_alive():
                self._js_thread = threading.Thread(target=self._js_consumer, daemon=True)
                self._js_thread.start()
                
            threading.Thread(target=self._stream_stdout, daemon=True).start()
            if enable_preview:
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
            user_stopped = self._user_stopped
        if not user_stopped:
            self.set_status("Finished" if rc == 0 else f"Exited {rc}", "idle" if rc == 0 else "error")

    def _watch_preview_frames(self, frame_path: Path) -> None:
        self._last_preview_file: Path | None = None
        _PREVIEW_MIN_INTERVAL = 0.1
        last_frame_time = 0.0
        
        with self._lock:
            running = self._preview_running
        while running:
            time.sleep(0.04)
            current_time = time.time()
            if current_time - last_frame_time < _PREVIEW_MIN_INTERVAL:
                with self._lock:
                    running = self._preview_running
                continue
                
            try:
                all_files = [p for p in frame_path.parent.glob(f"{frame_path.stem}*{frame_path.suffix}") if p.exists()]
                if not all_files:
                    latest = frame_path if frame_path.exists() else None
                else:
                    all_files.sort(key=lambda p: (p.stat().st_mtime, p.name))
                    latest = all_files[-1]

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
                        last_frame_time = time.time()

                        # Determine camera ID and worker ID from filename
                        cam_id = "CAM_0"
                        worker_id = "WORKER_0"
                        fname_upper = latest.name.upper()
                        if "WORKER_" in fname_upper:
                            parts = fname_upper.split("WORKER_")
                            if len(parts) > 1 and parts[1][0].isdigit():
                                worker_id = f"WORKER_{parts[1][0]}"

                        if len(self.sources) > 1:
                            if "CAM_" in fname_upper:
                                parts = fname_upper.split("CAM_")
                                if len(parts) > 1 and parts[1][0].isdigit():
                                    cam_id = f"CAM_{parts[1][0]}"
                            elif "WORKER_" in fname_upper:
                                parts = fname_upper.split("WORKER_")
                                if len(parts) > 1 and parts[1][0].isdigit():
                                    cam_id = f"CAM_{parts[1][0]}"

                        self.safe_evaluate_js(f"window.onKinaraPreviewFrame('{b64}', '{cam_id}', '{worker_id}');")
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
