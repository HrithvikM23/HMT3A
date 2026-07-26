import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kinara_web_launcher import KinaraWebAPI

def test_web_api_initial_command():
    api = KinaraWebAPI()
    cmd = api.get_initial_command()
    assert "--max-people" in cmd
    assert "1" in cmd

def test_web_api_apply_mediapipe_preset():
    api = KinaraWebAPI()
    cmd = api.apply_preset("MediaPipe", "Heavy", "Heavy")
    assert "--landmark-backend mediapipe" in cmd
    assert "--model pose_landmark_heavy.tflite" in cmd
    assert "--hand-model-variant max" in cmd

def test_web_api_apply_rtmpose_preset():
    api = KinaraWebAPI()
    cmd = api.apply_preset("RTMPose", "Heavy", "Heavy")
    assert "--landmark-backend rtmpose" in cmd
    assert "--rtmpose-mode performance" in cmd

def test_web_api_people_count_and_identity_constraints():
    api = KinaraWebAPI()
    # Single person (1)
    api.set_people_count(1)
    cmd_1 = api._build_command_string()
    assert "--max-people 1" in cmd_1
    assert "--identity" not in cmd_1

    # Multi-person (2)
    api.set_people_count(2)
    api.set_person_color("person1", "black,orange")
    api.set_person_color("person2", "white,blue")
    cmd_2 = api._build_command_string()
    assert "--max-people 2" in cmd_2
    assert "--identity person1=black,orange" in cmd_2
    assert "--identity person2=white,blue" in cmd_2

def test_web_api_reset_defaults():
    api = KinaraWebAPI()
    api.apply_preset("ONNX", "X-Large", "Heavy")
    api.set_people_count(4)
    api.reset_defaults()
    cmd = api._build_command_string()
    assert "--max-people 1" in cmd
    assert "--identity" not in cmd
    assert "--landmark-backend" not in cmd

def test_web_api_rescue_mode_toggle():
    api = KinaraWebAPI()
    cmd_on = api.set_checkbox("charuco_rescue_mode", True)
    assert "--charuco-detection-strictness lenient" in cmd_on
    assert "--charuco-retry-scale 3.5" in cmd_on

    cmd_off = api.set_checkbox("charuco_rescue_mode", False)
    assert "--charuco-detection-strictness balanced" in cmd_off


def test_web_api_safe_evaluate_js_handles_disposed_object():
    class DummyDisposedWindow:
        def evaluate_js(self, script):
            raise Exception("Cannot access a disposed object. Object name: 'WebView2'.")

    api = KinaraWebAPI()
    dummy_win = DummyDisposedWindow()
    api.set_window(dummy_win)
    assert api._window is dummy_win

    # safe_evaluate_js now enqueues; _execute_js is where the disposed window
    # exception is caught and _window is set to None.
    api._execute_js("window.onKinaraLog('test');")
    assert api._window is None


def test_web_api_window_closed_event():
    class DummyEvent:
        def __init__(self):
            self.handlers = []

        def __iadd__(self, handler):
            self.handlers.append(handler)
            return self

    class DummyEvents:
        def __init__(self):
            self.closed = DummyEvent()

    class DummyWindow:
        def __init__(self):
            self.events = DummyEvents()

    api = KinaraWebAPI()
    win = DummyWindow()
    api.set_window(win)
    assert len(win.events.closed.handlers) == 1

    # Trigger close callback directly
    win.events.closed.handlers[0]()
    assert api._window is None
    assert api._preview_running is False


