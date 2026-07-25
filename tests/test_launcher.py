import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kinara_web_launcher import KinaraWebAPI, PRESETS, CHARUCO_RESCUE_PRESET
import app.kinara_launcher as kinara_launcher


def test_launcher_module_main_entrypoint(monkeypatch):
    called = []

    def fake_run_web_launcher():
        called.append("web_launcher")

    monkeypatch.setattr("app.kinara_web_launcher.main", fake_run_web_launcher)
    monkeypatch.setattr(sys, "argv", ["kinara_launcher.py"])
    kinara_launcher.main()
    assert called == ["web_launcher"]


def test_launcher_module_runner_entrypoint(monkeypatch):
    called = []

    def fake_run_pipeline():
        called.append("pipeline")

    monkeypatch.setattr("app.main.main", fake_run_pipeline)
    monkeypatch.setattr(sys, "argv", ["kinara_launcher.py", "--kinara-runner", "--max-people", "1"])
    kinara_launcher.main()
    assert called == ["pipeline"]


def test_preset_list_contains_expected_presets():
    assert "MediaPipe" in PRESETS
    assert "RTMPose" in PRESETS
    assert "ONNX" in PRESETS
    assert "RTMPose WholeBody" in PRESETS


def test_preset_state_mixing_fix():
    api = KinaraWebAPI()
    # Apply RTMPose Heavy preset first
    cmd_rtm = api.apply_preset("RTMPose", "Heavy", "Heavy")
    assert "--landmark-backend rtmpose" in cmd_rtm
    assert "--rtmpose-mode performance" in cmd_rtm

    # Now apply MediaPipe Full preset
    cmd_mp = api.apply_preset("MediaPipe", "Full", "Full")
    assert "--landmark-backend mediapipe" in cmd_mp
    assert "--model pose-landmark-full.tflite" in cmd_mp or "--model pose_landmark_full.tflite" in cmd_mp
    assert "--rtmpose-mode" not in cmd_mp


def test_people_count_and_identity_constraints():
    api = KinaraWebAPI()
    # Single person (1)
    api.set_people_count(1)
    cmd_1 = api._build_command_string()
    assert "--max-people 1" in cmd_1
    assert "--identity" not in cmd_1

    # Multi person (2)
    api.set_people_count(2)
    api.set_person_color("person1", "red")
    api.set_person_color("person2", "blue")
    cmd_2 = api._build_command_string()
    assert "--max-people 2" in cmd_2
    assert "--identity person1=red" in cmd_2
    assert "--identity person2=blue" in cmd_2


def test_people_count_clamping():
    api = KinaraWebAPI()
    api.set_people_count(0)
    assert api.people_count == 1
    api.set_people_count(15)
    assert api.people_count == 12


def test_rescue_preset_lenient_strictness():
    assert CHARUCO_RESCUE_PRESET.get("charuco_detection_strictness") == "lenient"


def test_check_runtime_button(monkeypatch):
    api = KinaraWebAPI()
    started_commands = []

    def fake_start_process(command, status_text, enable_preview):
        started_commands.append((command, status_text, enable_preview))

    monkeypatch.setattr(api, "_start_process", fake_start_process)
    res = api.check_runtime()
    assert "Checking Kinara runtime dependencies..." in res.get("log", "")
    assert len(started_commands) == 1
    command, status_text, enable_preview = started_commands[0]
    assert "--runtime-check" in command
    assert "--max-people" in command
    assert status_text == "Checking"
    assert enable_preview is False
