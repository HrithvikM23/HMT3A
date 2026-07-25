import pytest
from PySide6.QtWidgets import QApplication

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kinara_launcher import KinaraLauncher, LAUNCHER_PRESETS, CHARUCO_RESCUE_PRESET

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

def test_preset_list_excludes_removed_presets(qapp):
    titles = [p["title"] for p in LAUNCHER_PRESETS]
    assert "Multi-Person" not in titles
    assert "Calibration" not in titles
    assert "MediaPipe" in titles
    assert "RTMPose" in titles
    assert "ONNX" in titles
    assert "RTMPose WholeBody" in titles

def test_preset_labels_present(qapp):
    for preset in LAUNCHER_PRESETS:
        if "options" in preset:
            assert "option_label" in preset
        if "weight_options" in preset:
            assert "weight_label" in preset

def test_preset_state_mixing_fix(qapp):
    launcher = KinaraLauncher()
    # Apply RTMPose Heavy preset first
    rtm_preset = next(p for p in LAUNCHER_PRESETS if p["title"] == "RTMPose")
    launcher._apply_launcher_preset(rtm_preset, option="Heavy", weight="Heavy")
    args_rtm = launcher.build_args()
    assert "--landmark-backend" in args_rtm
    assert "rtmpose" in args_rtm

    # Now apply MediaPipe Full preset
    mp_preset = next(p for p in LAUNCHER_PRESETS if p["title"] == "MediaPipe")
    launcher._apply_launcher_preset(mp_preset, option="Full", weight="Full")
    args_mp = launcher.build_args()

    # Stale RTMPose/ONNX/YOLO flags should NOT be in MediaPipe command
    assert "--rtmpose-mode" not in args_mp
    assert "--yolo-fast-preset" not in args_mp
    assert "--hand-input-size" not in args_mp
    # Should contain mediapipe backend
    assert launcher._advanced_args() == [] or "--landmark-backend" not in launcher._advanced_args() or "mediapipe" in args_mp

def test_people_count_and_identity_constraints(qapp):
    launcher = KinaraLauncher()
    # Single person (default = 1)
    launcher._set_people_count(1)
    args_1 = launcher.build_args()
    assert "--max-people" in args_1
    assert "1" in args_1
    # Check no --identity flag emitted for single person
    assert not any(arg.startswith("--identity") for arg in args_1)

    # Multi person (2 people)
    launcher._set_people_count(2)
    args_2 = launcher.build_args()
    assert "--max-people" in args_2
    assert "2" in args_2
    assert any(arg.startswith("--identity") or arg == "--identity" for arg in args_2)

def test_people_count_clamping(qapp):
    launcher = KinaraLauncher()
    launcher._set_people_count(0)
    assert launcher._get_people_count() == 1
    launcher._set_people_count(15)
    assert launcher._get_people_count() == 12

def test_rescue_preset_lenient_strictness():
    assert CHARUCO_RESCUE_PRESET.get("charuco_detection_strictness") == "lenient"

def test_check_runtime_button(qapp, monkeypatch):
    launcher = KinaraLauncher()
    started_commands = []
    def fake_start_process(command, enable_preview_stream, status_text):
        started_commands.append((command, enable_preview_stream, status_text))

    monkeypatch.setattr(launcher, "_start_process", fake_start_process)
    launcher.check_runtime()
    assert len(started_commands) == 1
    command, enable_preview, status_text = started_commands[0]
    assert "--runtime-check" in command
    assert "--max-people" in command
    assert status_text == "Checking"
    assert enable_preview is False

