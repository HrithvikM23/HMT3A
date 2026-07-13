from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import default_run_log_path, install_safe_stdio

install_safe_stdio()

from PySide6.QtCore import QEasingCurve, QEvent, QProcess, QProcessEnvironment, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabBar,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.launcher_support import (
    APP_TITLE,
    APP_USER_MODEL_ID,
    COLOR_PRESETS,
    MANAGED_DESTS,
    app_icon_path,
    default_text,
    installer_python_path,
    quote,
    tile_text,
)
from core.cli import build_parser

ADVANCED_CATEGORY_ORDER = (
    "Runtime",
    "Model",
    "RTMPose",
    "Legacy YOLO",
    "Hands",
    "Tracking",
    "Calibration",
    "Output",
    "Visuals",
    "Smoothing",
    "Cleanup",
)

ADVANCED_CATEGORY_BY_DEST = {
    "config": "Runtime",
    "dry_run": "Runtime",
    "runtime_check": "Runtime",
    "benchmark_frames": "Runtime",
    "profile": "Model",
    "landmark_backend": "Model",
    "body_backend": "Model",
    "hand_backend": "Model",
    "backend_fallbacks": "Model",
    "mediapipe_delegate": "Model",
    "model": "Model",
    "body_input_size": "Model",
    "processing_width": "Model",
    "rtmpose_mode": "RTMPose",
    "rtmpose_backend": "RTMPose",
    "rtmpose_device": "RTMPose",
    "rtmpose_det_frequency": "RTMPose",
    "rtmpose_tracking": "RTMPose",
    "yolo_fast_preset": "Legacy YOLO",
    "body_iou_threshold": "Legacy YOLO",
    "yolo_tracker": "Legacy YOLO",
    "yolo_device": "Legacy YOLO",
    "yolo_half": "Legacy YOLO",
    "yolo_fuse": "Legacy YOLO",
    "yolo_warmup": "Legacy YOLO",
    "yolo_person_class_filter": "Legacy YOLO",
    "hand_model_variant": "Hands",
    "hand_model": "Hands",
    "hand_input_name": "Hands",
    "hand_input_size": "Hands",
    "hand_det_threshold": "Hands",
    "hand_kp_threshold": "Hands",
    "hand_box_min_size": "Hands",
    "hand_box_scale": "Hands",
    "hand_detect_interval": "Hands",
    "hand_crop_retries": "Hands",
    "person_box_scale": "Tracking",
    "person_track_hold_frames": "Tracking",
    "person_match_threshold": "Tracking",
    "person_cross_wrist_ratio": "Tracking",
    "body_conf_threshold": "Tracking",
    "body_detect_interval": "Tracking",
    "body_hold_frames": "Tracking",
    "hand_hold_frames": "Tracking",
    "hold_confidence_decay": "Tracking",
    "camera_calibration": "Calibration",
    "calibration_3d": "Calibration",
    "triangulate_3d": "Calibration",
    "triangulation_min_cameras": "Calibration",
    "triangulation_use_outlier_rejection": "Calibration",
    "triangulation_max_cameras_to_drop": "Calibration",
    "triangulation_reprojection_error": "Calibration",
    "triangulation_max_error": "Calibration",
    "triangulation_smoothing_alpha": "Calibration",
    "sync_offsets": "Calibration",
    "fused_depth_scale": "Calibration",
    "single_camera_depth": "Calibration",
    "calibrate_cameras": "Calibration",
    "calibration_output": "Calibration",
    "charuco_squares_x": "Calibration",
    "charuco_squares_y": "Calibration",
    "charuco_square_size": "Calibration",
    "charuco_marker_scale": "Calibration",
    "charuco_marker_bits": "Calibration",
    "charuco_dict_size": "Calibration",
    "charuco_legacy_pattern": "Calibration",
    "charuco_detection_strictness": "Calibration",
    "charuco_retry_scale": "Calibration",
    "charuco_min_markers": "Calibration",
    "charuco_retry_sharpen": "Calibration",
    "providers": "Output",
    "osc_host": "Output",
    "osc_port": "Output",
    "osc_enabled": "Output",
    "preview_title": "Output",
    "fallback_fps": "Output",
    "output_fourcc": "Output",
    "fps_log_interval": "Output",
    "fps_overlay_enabled": "Output",
    "auto_performance": "Output",
    "body_line_color": "Visuals",
    "body_point_color": "Visuals",
    "hand_box_color": "Visuals",
    "hand_line_color": "Visuals",
    "hand_point_color": "Visuals",
    "body_line_thickness": "Visuals",
    "body_point_radius": "Visuals",
    "hand_box_thickness": "Visuals",
    "hand_line_thickness": "Visuals",
    "hand_point_radius": "Visuals",
    "body_smoothing_alpha": "Smoothing",
    "hand_smoothing_alpha": "Smoothing",
    "body_constraints": "Smoothing",
    "body_length_smoothing_alpha": "Smoothing",
    "body_length_correction": "Smoothing",
    "export_cleanup": "Cleanup",
    "export_cleanup_smoothing_alpha": "Cleanup",
    "export_cleanup_max_velocity": "Cleanup",
    "foot_lock": "Cleanup",
    "foot_lock_velocity": "Cleanup",
    "foot_lock_max_lift": "Cleanup",
}

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

LAUNCHER_PRESETS = (
    {
        "title": "Demo Stable",
        "description": "Reliable defaults for a live demo on normal footage.",
        "section": "Runtime",
        "people": 1,
        "values": {
            "profile": "fastest",
            "landmark_backend": "mediapipe",
            "processing_width": "640",
            "body_detect_interval": "1",
            "hand_detect_interval": "2",
            "hand_crop_retries": "1",
        },
    },
    {
        "title": "Quality Export",
        "description": "Sharper processing and smoother output for recorded clips.",
        "section": "Model",
        "people": 1,
        "values": {
            "profile": "quality",
            "processing_width": "960",
            "body_smoothing_alpha": "0.55",
            "hand_smoothing_alpha": "0.55",
            "export_cleanup": True,
        },
    },
    {
        "title": "Multi-Person",
        "description": "Tracks two people with longer identity hold settings.",
        "section": "Tracking",
        "people": 2,
        "values": {
            "max_people": "2",
            "person_track_hold_frames": "45",
            "person_match_threshold": "0.35",
            "body_hold_frames": "8",
            "hand_hold_frames": "10",
        },
    },
    {
        "title": "RTMPose GPU",
        "description": "Use the RTMPose path when CUDA/runtime packages are ready.",
        "section": "RTMPose",
        "people": 1,
        "values": {
            "landmark_backend": "rtmpose",
            "rtmpose_device": "cuda",
            "rtmpose_mode": "balanced",
            "rtmpose_tracking": True,
        },
    },
    {
        "title": "MediaPipe GPU Try",
        "description": "Try MediaPipe Tasks GPU delegate and fall back to CPU if unsupported.",
        "section": "Model",
        "people": 1,
        "values": {
            "landmark_backend": "mediapipe",
            "mediapipe_delegate": "gpu",
        },
    },
    {
        "title": "ChArUco A3",
        "description": "Recommended A3 board calibration setup.",
        "section": "Calibration",
        "people": 1,
        "values": CHARUCO_A3_PRESET,
    },
    {
        "title": "ChArUco Rescue",
        "description": "More forgiving board detection for distant or compressed calibration videos.",
        "section": "Calibration",
        "people": 1,
        "values": CHARUCO_RESCUE_PRESET,
    },
)


def advanced_category(dest: str) -> str:
    return ADVANCED_CATEGORY_BY_DEST.get(dest, "Runtime")


def main() -> None:
    if "--kinara-runner" in sys.argv:
        runner_args = [arg for arg in sys.argv[1:] if arg != "--kinara-runner"]
        sys.argv = [sys.argv[0], *runner_args]
        from app.main import main as run_pipeline

        run_pipeline()
        return

    set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setApplicationDisplayName(APP_TITLE)
    app.setOrganizationName("Kinara")
    app.setFont(QFont("Segoe UI", 10))
    icon_path = app_icon_path(PROJECT_ROOT)
    if icon_path is not None:
        app.setWindowIcon(load_app_icon(icon_path))
    window = KinaraLauncher()
    window.show()
    sys.exit(app.exec())


class KinaraLauncher(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            app.setFont(QFont("Segoe UI", 10))
        self.setWindowTitle(APP_TITLE)
        icon_path = app_icon_path(PROJECT_ROOT)
        if icon_path is not None:
            self.setWindowIcon(load_app_icon(icon_path))
        self.resize(1380, 840)
        self.setMinimumSize(1120, 700)

        self.sources: list[str] = []
        self.process: QProcess | None = None
        self.person_color_controls: list[QComboBox] = []
        self.advanced_controls: dict[str, tuple[argparse.Action, QWidget]] = {}
        self.advanced_rows: dict[str, QFrame] = {}
        self.advanced_section_buttons: dict[str, QToolButton] = {}
        self.advanced_section_bodies: dict[str, QWidget] = {}
        self.advanced_section_frames: dict[str, QFrame] = {}
        self.advanced_row_texts: dict[str, str] = {}
        self.advanced_search: QLineEdit | None = None
        self.python_path: QLineEdit | None = None
        self.people_count: QSpinBox | None = None
        self.people_box: QVBoxLayout | None = None
        self.calibration_mode: QCheckBox | None = None
        self.calibration_output_path: QLineEdit | None = None
        self.triangulation_enabled: QCheckBox | None = None
        self.triangulation_calibration_path: QLineEdit | None = None
        self.preview_frame_path = self._runtime_dir() / "preview.jpg"
        self._last_preview_mtime = 0.0
        self._preview_pixmap: QPixmap | None = None
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(80)
        self.preview_timer.timeout.connect(self._load_preview_frame)
        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.setInterval(3000)
        self.stop_timer.timeout.connect(self.kill_run)
        self._animations: list[QPropertyAnimation] = []
        self.theme_button: QToolButton | None = None
        self.current_theme = "dark"

        self.setStyleSheet(app_style("dark"))
        self._build_ui()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._refresh_people()
        self._refresh_sources()
        self._refresh_preview()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(0)

        workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        workspace_splitter.setObjectName("workspaceSplitter")
        workspace_splitter.setChildrenCollapsible(False)
        layout.addWidget(workspace_splitter)

        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setMinimumWidth(620)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 12, 0)
        left.setSpacing(12)
        workspace_splitter.addWidget(left_panel)

        preview_header = QHBoxLayout()
        preview_titles = QVBoxLayout()
        preview_titles.setSpacing(2)
        preview_title = QLabel("Video inputs")
        preview_title.setObjectName("pageTitle")
        preview_subtitle = QLabel("Single source fills the stage; multiple sources tile automatically.")
        preview_subtitle.setObjectName("muted")
        preview_titles.addWidget(preview_title)
        preview_titles.addWidget(preview_subtitle)
        preview_header.addLayout(preview_titles, 1)
        self.source_summary = QLabel("0 sources")
        self.source_summary.setObjectName("summaryPill")
        preview_header.addWidget(self.source_summary)
        left.addLayout(preview_header)

        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.setObjectName("contentSplitter")
        content_splitter.setChildrenCollapsible(False)
        left.addWidget(content_splitter, 1)

        self.preview = QFrame()
        self.preview.setObjectName("preview")
        self.preview.setMinimumHeight(330)
        self.preview_layout = QGridLayout(self.preview)
        self.preview_layout.setContentsMargins(18, 18, 18, 18)
        self.preview_layout.setSpacing(16)
        content_splitter.addWidget(self.preview)
        self.live_preview_label = QLabel("Processed preview will appear here when a run starts")
        self.live_preview_label.setObjectName("livePreview")
        self.live_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_preview_label.setScaledContents(False)
        self.live_preview_label.hide()

        command_bar = QFrame()
        command_bar.setObjectName("commandBar")
        controls = QHBoxLayout(command_bar)
        controls.setContentsMargins(10, 10, 10, 10)
        controls.setSpacing(10)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_run)
        self.check_button = QPushButton("Check Runtime")
        self.check_button.setObjectName("secondaryButton")
        self.check_button.clicked.connect(self.check_runtime)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self.stop_run)
        self.kill_button = QPushButton("Kill")
        self.kill_button.setObjectName("killButton")
        self.kill_button.clicked.connect(self.kill_run)
        self.status = QLabel("Idle")
        self.status.setObjectName("status")
        controls.addWidget(self.start_button)
        controls.addWidget(self.check_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.kill_button)
        controls.addWidget(self.status)
        controls.addStretch(1)
        left.addWidget(command_bar)

        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(110)
        self.log.setMaximumHeight(320)
        content_splitter.addWidget(self.log)
        content_splitter.setStretchFactor(0, 5)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([560, 165])

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(560)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(12)
        workspace_splitter.addWidget(sidebar)
        workspace_splitter.setStretchFactor(0, 4)
        workspace_splitter.setStretchFactor(1, 1)
        workspace_splitter.setSizes([940, 400])

        brand = QFrame()
        brand.setObjectName("brandHeader")
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(14, 14, 14, 14)
        brand_layout.setSpacing(8)

        title_row = QHBoxLayout()
        title = QLabel("Kinara")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title_row.addWidget(title, 1)
        brand_layout.addLayout(title_row)

        subtitle = QLabel("Motion capture launcher")
        subtitle.setObjectName("brandSubtitle")
        brand_layout.addWidget(subtitle)

        command_row = QHBoxLayout()
        self.theme_button = QToolButton()
        self.theme_button.setObjectName("toolbarIconButton")
        self.theme_button.setText("☀")
        self.theme_button.setToolTip("Switch to Light")
        self.theme_button.clicked.connect(self._toggle_theme)
        reset_button = QToolButton()
        reset_button.setObjectName("toolbarIconButton")
        reset_button.setText("↻")
        reset_button.setToolTip("Reset defaults")
        reset_button.clicked.connect(self.reset_defaults)
        command_row.addWidget(self.theme_button)
        command_row.addWidget(reset_button)
        brand_layout.addLayout(command_row)
        side_layout.addWidget(brand)

        advanced_tab = self._advanced_tab()
        tabs = QTabWidget()
        tabs.addTab(self._camera_tab(), "Capture")
        tabs.addTab(self._file_tab(), "Files")
        tabs.addTab(self._presets_tab(), "Presets")
        tabs.addTab(self._calibration_tab(), "Calibration")
        tabs.addTab(self._triangulation_tab(), "Triangulation")
        tabs.addTab(advanced_tab, "Tune")
        side_layout.addWidget(tabs, 1)

    def _camera_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tabPage")
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.addWidget(section_label("Camera input"))

        open_camera = QPushButton("Open Camera")
        open_camera.setObjectName("wideButton")
        open_camera.clicked.connect(self.use_camera)
        layout.addWidget(open_camera)

        hint = QLabel("Use local camera input for quick tests, or add recorded files from the Files tab.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return tab

    def _file_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tabPage")
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        layout.addWidget(section_label("Sources"))
        source_buttons = QHBoxLayout()
        add_files = QPushButton("Add files")
        add_files.setObjectName("wideButton")
        add_files.clicked.connect(self.add_files)
        remove = QPushButton("Remove selected")
        remove.setObjectName("secondaryButton")
        remove.clicked.connect(self.remove_selected_sources)
        clear = QPushButton("Clear")
        clear.setObjectName("secondaryButton")
        clear.clicked.connect(self.clear_sources)
        source_buttons.addWidget(add_files)
        source_buttons.addWidget(remove)
        source_buttons.addWidget(clear)
        layout.addLayout(source_buttons)

        self.source_list = QListWidget()
        self.source_list.setMinimumHeight(116)
        self.source_list.setAlternatingRowColors(True)
        self.source_list.itemDoubleClicked.connect(lambda _item: self.remove_selected_sources())
        layout.addWidget(self.source_list)

        layout.addWidget(section_label("Destination"))
        dest_row = QHBoxLayout()
        self.destination = QLineEdit(str(Path.cwd() / "outputs"))
        browse_dest = QPushButton("Browse")
        browse_dest.setObjectName("secondaryButton")
        browse_dest.clicked.connect(self.choose_destination)
        dest_row.addWidget(self.destination, 1)
        dest_row.addWidget(browse_dest)
        layout.addLayout(dest_row)

        layout.addStretch(1)
        return tab

    def _advanced_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tabPage")
        outer = QVBoxLayout(tab)
        hint = QLabel("Settings changed here apply to this launcher session.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self.advanced_search = QLineEdit()
        self.advanced_search.setPlaceholderText("Search advanced settings")
        self.advanced_search.textChanged.connect(self._filter_advanced_controls)
        outer.addWidget(self.advanced_search)

        scroll = QScrollArea()
        scroll.setObjectName("tabScroll")
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(8)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        category_layouts: dict[str, QVBoxLayout] = {}
        for category_name in ADVANCED_CATEGORY_ORDER:
            section, section_layout = self._advanced_section(category_name)
            category_layouts[category_name] = section_layout
            content_layout.addWidget(section)

        category_layouts["Runtime"].addWidget(self._people_runtime_control())
        category_layouts["Runtime"].addWidget(self._python_runtime_control())
        category_layouts["Calibration"].addWidget(self._calibration_quick_panel())

        parser = build_parser()
        for action in parser._actions:
            if not action.option_strings or action.dest in MANAGED_DESTS or action.dest == "help":
                continue
            category = advanced_category(action.dest)
            self._add_advanced_control(action, category_layouts[category])
        for layout in category_layouts.values():
            layout.addStretch(1)
        content_layout.addStretch(1)
        return tab

    def _presets_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tabPage")
        outer = QVBoxLayout(tab)
        outer.setSpacing(6)

        hint = QLabel("Quick setups")
        hint.setObjectName("hint")
        outer.addWidget(hint)

        scroll = QScrollArea()
        scroll.setObjectName("tabScroll")
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)
        for preset in LAUNCHER_PRESETS:
            layout.addWidget(self._preset_card(preset))
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        return tab

    def _preset_card(self, preset: dict[str, object]) -> QFrame:
        card = QFrame()
        card.setObjectName("presetCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(8)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        title = QLabel(str(preset["title"]))
        title.setObjectName("advancedLabel")
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pill = QLabel(str(preset.get("section", "Preset")))
        pill.setObjectName("presetPill")

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title_row.addWidget(title, 1)
        title_row.addWidget(pill)
        text_layout.addLayout(title_row)

        description = QLabel(str(preset["description"]))
        description.setObjectName("hint")
        description.setWordWrap(False)
        description.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        description.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        text_layout.addWidget(description)
        layout.addLayout(text_layout, 1)

        apply_button = QPushButton("Apply")
        apply_button.setObjectName("primaryButton")
        apply_button.setFixedWidth(68)
        apply_button.clicked.connect(lambda _checked=False, selected=preset: self._apply_launcher_preset(selected))
        layout.addWidget(apply_button)
        return card

    def _calibration_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tabPage")
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        hero = QFrame()
        hero.setObjectName("workflowPanel")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(12, 12, 12, 12)
        hero_layout.setSpacing(10)

        title = QLabel("Camera Calibration")
        title.setObjectName("advancedLabel")
        hero_layout.addWidget(title)

        hint = QLabel("Select synchronized ChArUco videos in Files, enable calibration mode, then Start. The run will save a camera calibration file and quality report.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        hero_layout.addWidget(hint)

        self.calibration_mode = QCheckBox("Calibration mode")
        self.calibration_mode.setChecked(False)
        hero_layout.addWidget(self.calibration_mode)

        output_row = QHBoxLayout()
        self.calibration_output_path = QLineEdit("")
        self.calibration_output_path.setPlaceholderText(r"Output .toml path or folder")
        browse = QPushButton("Browse")
        browse.setObjectName("secondaryButton")
        browse.clicked.connect(self.choose_calibration_output)
        output_row.addWidget(self.calibration_output_path, 1)
        output_row.addWidget(browse)
        hero_layout.addLayout(output_row)

        preset_row = QHBoxLayout()
        a3 = QPushButton("A3 Board")
        a3.setObjectName("primaryButton")
        a3.clicked.connect(lambda: self._apply_charuco_preset(CHARUCO_A3_PRESET))
        rescue = QPushButton("Rescue")
        rescue.setObjectName("secondaryButton")
        rescue.clicked.connect(lambda: self._apply_charuco_preset(CHARUCO_RESCUE_PRESET))
        show = QPushButton("Tune ChArUco")
        show.setObjectName("secondaryButton")
        show.clicked.connect(lambda: self._open_advanced_section("Calibration"))
        preset_row.addWidget(a3)
        preset_row.addWidget(rescue)
        preset_row.addWidget(show)
        hero_layout.addLayout(preset_row)
        layout.addWidget(hero)

        layout.addWidget(section_label("Checklist"))
        for text in (
            "Use two or more recorded sources from the Files tab.",
            "Keep the whole ChArUco board sharp, large, and visible in every camera.",
            "Use Rescue only for distant, compressed, or blurry footage.",
        ):
            label = QLabel(text)
            label.setObjectName("hint")
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addStretch(1)
        return tab

    def _triangulation_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tabPage")
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        panel = QFrame()
        panel.setObjectName("workflowPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(10)

        title = QLabel("3D Triangulation")
        title.setObjectName("advancedLabel")
        panel_layout.addWidget(title)

        hint = QLabel("Enable triangulation after calibration. Pick the saved camera calibration file, usually camera_calibration.toml.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        panel_layout.addWidget(hint)

        self.triangulation_enabled = QCheckBox("Enable 3D triangulation")
        self.triangulation_enabled.setChecked(False)
        panel_layout.addWidget(self.triangulation_enabled)

        file_row = QHBoxLayout()
        self.triangulation_calibration_path = QLineEdit("")
        self.triangulation_calibration_path.setPlaceholderText(r"camera_calibration.toml / .json")
        browse = QPushButton("Browse")
        browse.setObjectName("secondaryButton")
        browse.clicked.connect(self.choose_triangulation_calibration)
        file_row.addWidget(self.triangulation_calibration_path, 1)
        file_row.addWidget(browse)
        panel_layout.addLayout(file_row)

        tune = QPushButton("Tune Triangulation")
        tune.setObjectName("secondaryButton")
        tune.clicked.connect(lambda: self._open_advanced_section("Calibration"))
        panel_layout.addWidget(tune)
        layout.addWidget(panel)

        layout.addWidget(section_label("Flow"))
        for text in (
            "1. Calibrate first and save the camera calibration file.",
            "2. Select the same camera sources or matching recorded views.",
            "3. Browse to the calibration file here, then Start.",
        ):
            label = QLabel(text)
            label.setObjectName("hint")
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addStretch(1)
        return tab

    def _advanced_section(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        section = QFrame()
        section.setObjectName("accordionSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QToolButton()
        header.setObjectName("accordionHeader")
        header.setText(title)
        header.setCheckable(True)
        header.setChecked(title == "Runtime")
        header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header.setArrowType(Qt.ArrowType.DownArrow if header.isChecked() else Qt.ArrowType.RightArrow)
        layout.addWidget(header)

        body = QWidget()
        body.setObjectName("accordionBody")
        body.setVisible(header.isChecked())
        if not header.isChecked():
            body.setMaximumHeight(0)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(10)
        layout.addWidget(body)

        header.toggled.connect(lambda checked, section_title=title: self._toggle_advanced_section(section_title, checked))
        self.advanced_section_buttons[title] = header
        self.advanced_section_bodies[title] = body
        self.advanced_section_frames[title] = section
        return section, body_layout

    def _toggle_advanced_section(self, title: str, checked: bool) -> None:
        if self.advanced_search is not None and self.advanced_search.text().strip():
            return
        button = self.advanced_section_buttons[title]
        body = self.advanced_section_bodies[title]
        button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._animate_section_body(body, checked)
        if not checked:
            return
        for other_title, other_button in self.advanced_section_buttons.items():
            if other_title == title or not other_button.isChecked():
                continue
            other_button.blockSignals(True)
            other_button.setChecked(False)
            other_button.setArrowType(Qt.ArrowType.RightArrow)
            other_button.blockSignals(False)
            self._animate_section_body(self.advanced_section_bodies[other_title], False)

    def _python_runtime_control(self) -> QFrame:
        row = QFrame()
        row.setObjectName("advancedRow")
        layout = QVBoxLayout(row)
        layout.setContentsMargins(10, 8, 10, 8)
        label = QLabel("Python 3.11 runtime")
        label.setObjectName("advancedLabel")
        layout.addWidget(label)

        picker = QHBoxLayout()
        self.python_path = QLineEdit(installer_python_path())
        self.python_path.setPlaceholderText(r"C:\Users\...\Python311 or ...\Python311\python.exe")
        browse = QToolButton()
        browse.setObjectName("iconButton")
        browse.setToolTip("Browse for python.exe")
        browse.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        browse.clicked.connect(self.choose_python_runtime)
        picker.addWidget(self.python_path, 1)
        picker.addWidget(browse)
        layout.addLayout(picker)

        help_label = QLabel("Used for installing/checking runtime packages. A folder ending in Python311 is accepted.")
        help_label.setObjectName("hint")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.advanced_row_texts["python_runtime"] = "python 3.11 runtime installer package dependencies"
        return row

    def _people_runtime_control(self) -> QFrame:
        row = QFrame()
        row.setObjectName("advancedRow")
        layout = QVBoxLayout(row)
        layout.setContentsMargins(10, 8, 10, 8)

        label = QLabel("People to track")
        label.setObjectName("advancedLabel")
        layout.addWidget(label)

        amount_row = QHBoxLayout()
        amount_row.addWidget(QLabel("Amount"))
        self.people_count = QSpinBox()
        self.people_count.setRange(1, 12)
        self.people_count.setValue(1)
        self.people_count.valueChanged.connect(self._refresh_people)
        amount_row.addWidget(self.people_count)
        amount_row.addStretch(1)
        layout.addLayout(amount_row)

        self.people_box = QVBoxLayout()
        layout.addLayout(self.people_box)

        hint = QLabel("Color hints help keep identities stable when multiple people are visible.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.advanced_rows["runtime_people"] = row
        self.advanced_row_texts["runtime_people"] = "runtime people max people amount identity color hints multi person"
        return row

    def _calibration_quick_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("calibrationPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("ChArUco demo setup")
        title.setObjectName("advancedLabel")
        layout.addWidget(title)

        text = QLabel("Use the A3 preset for the recommended board, or rescue mode for compressed/distant videos.")
        text.setObjectName("hint")
        text.setWordWrap(True)
        layout.addWidget(text)

        buttons = QHBoxLayout()
        a3 = QPushButton("A3 Board")
        a3.setObjectName("primaryButton")
        a3.clicked.connect(lambda: self._apply_charuco_preset(CHARUCO_A3_PRESET))
        rescue = QPushButton("Rescue")
        rescue.setObjectName("secondaryButton")
        rescue.clicked.connect(lambda: self._apply_charuco_preset(CHARUCO_RESCUE_PRESET))
        open_section = QPushButton("Show Options")
        open_section.setObjectName("secondaryButton")
        open_section.clicked.connect(lambda: self._open_advanced_section("Calibration"))
        buttons.addWidget(a3)
        buttons.addWidget(rescue)
        buttons.addWidget(open_section)
        layout.addLayout(buttons)
        self.advanced_row_texts["charuco_quick_panel"] = "charuco demo setup a3 board rescue calibration preset"
        return panel

    def _add_advanced_control(self, action: argparse.Action, parent_layout: QVBoxLayout) -> None:
        row = QFrame()
        row.setObjectName("advancedRow")
        layout = QVBoxLayout(row)
        layout.setContentsMargins(10, 8, 10, 8)
        label = QLabel(action.option_strings[-1])
        label.setObjectName("advancedLabel")
        layout.addWidget(label)

        if isinstance(action, argparse._StoreTrueAction):
            control = QCheckBox("enabled")
            control.setChecked(bool(action.default))
        elif action.choices:
            control = QComboBox()
            values = [str(choice) for choice in action.choices]
            control.addItems(values)
            default = "" if action.default in (None, argparse.SUPPRESS) else str(action.default)
            if default in values:
                control.setCurrentText(default)
        else:
            control = QLineEdit(default_text(action.default))
        layout.addWidget(control)
        if action.dest == "landmark_backend" and isinstance(control, QComboBox):
            control.currentTextChanged.connect(self._sync_backend_controls)

        if action.help:
            help_label = QLabel(action.help)
            help_label.setObjectName("hint")
            help_label.setWordWrap(True)
            layout.addWidget(help_label)

        parent_layout.addWidget(row)
        self.advanced_controls[action.dest] = (action, control)
        self.advanced_rows[action.dest] = row
        self.advanced_row_texts[action.dest] = " ".join(
            part
            for part in (
                action.dest.replace("_", " "),
                " ".join(action.option_strings),
                str(action.help or ""),
            )
            if part
        ).lower()
        self._sync_backend_controls()

    def _apply_charuco_preset(self, values: dict[str, object]) -> None:
        for dest, value in values.items():
            self._set_advanced_control_value(dest, value)
        self._open_advanced_section("Calibration")
        self.status.setText("ChArUco preset applied")

    def _apply_launcher_preset(self, preset: dict[str, object]) -> None:
        people = preset.get("people")
        if isinstance(people, int) and self.people_count is not None:
            self.people_count.setValue(people)

        values = preset.get("values", {})
        if isinstance(values, dict):
            for dest, value in values.items():
                self._set_advanced_control_value(str(dest), value)

        section = preset.get("section")
        if isinstance(section, str):
            self._open_advanced_section(section)
        self.status.setText(f"{preset['title']} preset applied")

    def reset_defaults(self) -> None:
        self._set_theme("dark")
        if self.people_count is not None:
            self.people_count.setValue(1)
        if self.calibration_mode is not None:
            self.calibration_mode.setChecked(False)
        if self.calibration_output_path is not None:
            self.calibration_output_path.clear()
        if self.triangulation_enabled is not None:
            self.triangulation_enabled.setChecked(False)
        if self.triangulation_calibration_path is not None:
            self.triangulation_calibration_path.clear()
        for action, control in self.advanced_controls.values():
            if isinstance(control, QCheckBox):
                control.setChecked(bool(action.default))
            elif isinstance(control, QComboBox):
                default = default_text(action.default)
                if default in [control.itemText(index) for index in range(control.count())]:
                    control.setCurrentText(default)
                elif control.count():
                    control.setCurrentIndex(0)
            elif isinstance(control, QLineEdit):
                control.setText(default_text(action.default))
        if self.advanced_search is not None:
            self.advanced_search.clear()
        self._open_advanced_section("Runtime")
        self._sync_backend_controls()
        self.status.setText("Defaults restored")

    def _set_advanced_control_value(self, dest: str, value: object) -> None:
        if dest == "max_people" and self.people_count is not None:
            self.people_count.setValue(int(value))
            return
        if dest == "calibrate_cameras" and self.calibration_mode is not None:
            self.calibration_mode.setChecked(bool(value))
            return
        if dest == "calibration_output" and self.calibration_output_path is not None:
            self.calibration_output_path.setText(str(value))
            return
        if dest == "triangulate_3d" and self.triangulation_enabled is not None:
            self.triangulation_enabled.setChecked(bool(value))
            return
        if dest == "calibration_3d" and self.triangulation_calibration_path is not None:
            self.triangulation_calibration_path.setText(str(value))
            return
        control = self.advanced_controls.get(dest, (None, None))[1]
        if isinstance(control, QCheckBox):
            control.setChecked(bool(value))
        elif isinstance(control, QComboBox):
            control.setCurrentText(str(value))
        elif isinstance(control, QLineEdit):
            control.setText(str(value))

    def _set_theme(self, theme: str) -> None:
        self.current_theme = "light" if theme == "light" else "dark"
        self.setStyleSheet(app_style(theme))
        if self.theme_button is not None:
            if self.current_theme == "light":
                self.theme_button.setText("☾")
                self.theme_button.setToolTip("Switch to Dark")
            else:
                self.theme_button.setText("☀")
                self.theme_button.setToolTip("Switch to Light")

    def _toggle_theme(self) -> None:
        self._set_theme("light" if self.current_theme == "dark" else "dark")

    def _open_advanced_section(self, title: str) -> None:
        if self.advanced_search is not None:
            self.advanced_search.clear()
        button = self.advanced_section_buttons.get(title)
        if button is not None:
            if button.isChecked():
                self._toggle_advanced_section(title, True)
            else:
                button.setChecked(True)

    def _filter_advanced_controls(self, query: str) -> None:
        normalized_query = query.strip().lower()
        if not normalized_query:
            for row in self.advanced_rows.values():
                row.setVisible(True)
            for section_title, frame in self.advanced_section_frames.items():
                frame.setVisible(True)
                button = self.advanced_section_buttons[section_title]
                body = self.advanced_section_bodies[section_title]
                body.setVisible(button.isChecked())
                body.setMaximumHeight(16777215 if button.isChecked() else 0)
                button.setArrowType(Qt.ArrowType.DownArrow if button.isChecked() else Qt.ArrowType.RightArrow)
            return

        matched_sections: set[str] = set()
        for dest, row in self.advanced_rows.items():
            matched = normalized_query in self.advanced_row_texts.get(dest, "")
            row.setVisible(matched)
            if matched:
                matched_sections.add(advanced_category(dest))

        runtime_row = self.python_path.parentWidget() if self.python_path is not None else None
        if isinstance(runtime_row, QFrame):
            runtime_matched = normalized_query in self.advanced_row_texts.get("python_runtime", "")
            runtime_row.setVisible(runtime_matched)
            if runtime_matched:
                matched_sections.add("Runtime")

        for section_title, frame in self.advanced_section_frames.items():
            matched = section_title in matched_sections or normalized_query in section_title.lower()
            frame.setVisible(matched)
            self.advanced_section_bodies[section_title].setVisible(matched)
            self.advanced_section_bodies[section_title].setMaximumHeight(16777215 if matched else 0)
            self.advanced_section_buttons[section_title].setArrowType(Qt.ArrowType.DownArrow if matched else Qt.ArrowType.RightArrow)

    def use_camera(self) -> None:
        self.sources = ["0"]
        self._refresh_sources()
        self._refresh_preview()

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video source file(s)",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if not paths:
            return
        self.sources.extend(paths)
        self._refresh_sources()
        self._refresh_preview()

    def clear_sources(self) -> None:
        self.sources.clear()
        self._refresh_sources()
        self._refresh_preview()

    def remove_selected_sources(self) -> None:
        selected_rows = sorted((index.row() for index in self.source_list.selectedIndexes()), reverse=True)
        if not selected_rows:
            return
        for row in selected_rows:
            if 0 <= row < len(self.sources):
                self.sources.pop(row)
        self._refresh_sources()
        self._refresh_preview()

    def choose_destination(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output destination", self.destination.text())
        if path:
            self.destination.setText(path)

    def choose_python_runtime(self) -> None:
        start_dir = ""
        if self.python_path is not None and self.python_path.text().strip():
            current = Path(self.python_path.text().strip())
            start_dir = str(current if current.is_dir() else current.parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Python 3.11 executable",
            start_dir,
            "Python executable (python.exe);;All files (*.*)",
        )
        if path and self.python_path is not None:
            self.python_path.setText(path)

    def choose_calibration_output(self) -> None:
        start_dir = ""
        if self.calibration_output_path is not None and self.calibration_output_path.text().strip():
            current = Path(self.calibration_output_path.text().strip())
            start_dir = str(current if current.is_dir() else current.parent)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save camera calibration",
            str(Path(start_dir) / "camera_calibration.toml") if start_dir else "camera_calibration.toml",
            "Calibration files (*.toml *.json);;TOML files (*.toml);;JSON files (*.json);;All files (*.*)",
        )
        if path and self.calibration_output_path is not None:
            self.calibration_output_path.setText(path)

    def choose_triangulation_calibration(self) -> None:
        start_dir = ""
        if self.triangulation_calibration_path is not None and self.triangulation_calibration_path.text().strip():
            current = Path(self.triangulation_calibration_path.text().strip())
            start_dir = str(current if current.is_dir() else current.parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select camera calibration",
            start_dir,
            "Calibration files (*.toml *.json);;TOML files (*.toml);;JSON files (*.json);;All files (*.*)",
        )
        if path and self.triangulation_calibration_path is not None:
            self.triangulation_calibration_path.setText(path)

    def _refresh_sources(self) -> None:
        self.source_list.clear()
        for index, source in enumerate(self.sources):
            self.source_list.addItem(f"CAM_{index}: {source}")
        count = len(self.sources)
        self.source_summary.setText(f"{count} source" if count == 1 else f"{count} sources")
        self._pulse_widget(self.source_summary, low=0.82, duration=160)
        self._pulse_widget(self.source_list, low=0.9, duration=180)

    def _refresh_people(self) -> None:
        if self.people_box is None or self.people_count is None:
            return
        clear_layout(self.people_box)
        self.person_color_controls.clear()
        for index in range(self.people_count.value()):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"person{index + 1}"))
            combo = QComboBox()
            combo.addItems(COLOR_PRESETS)
            combo.setCurrentText(COLOR_PRESETS[min(index, len(COLOR_PRESETS) - 1)])
            row.addWidget(combo, 1)
            self.people_box.addLayout(row)
            self.person_color_controls.append(combo)
            self._pulse_widget(combo, low=0.88, duration=160)

    def _refresh_preview(self) -> None:
        clear_layout(self.preview_layout, preserve={self.live_preview_label})
        self.live_preview_label.hide()
        sources = self.sources or ["No source selected"]
        count = len(sources)
        rows, cols = (1, 1) if count == 1 else ((1, 2) if count == 2 else (2, 2))
        for index, source in enumerate(sources[:4]):
            tile = self._source_tile(index, source, has_sources=bool(self.sources))
            self.preview_layout.addWidget(tile, index // cols, index % cols)
            self._fade_in(tile, duration=180 + index * 35)
        for row in range(rows):
            self.preview_layout.setRowStretch(row, 1)
        for col in range(cols):
            self.preview_layout.setColumnStretch(col, 1)

    def _show_live_preview(self) -> None:
        clear_layout(self.preview_layout, preserve={self.live_preview_label})
        self.live_preview_label.setText("Waiting for processed frames...")
        self.live_preview_label.show()
        self.preview_layout.addWidget(self.live_preview_label, 0, 0)
        self._fade_in(self.live_preview_label, duration=180)
        self.preview_layout.setRowStretch(0, 1)
        self.preview_layout.setColumnStretch(0, 1)

    def _source_tile(self, index: int, source: str, *, has_sources: bool) -> QFrame:
        tile = QFrame()
        tile.setObjectName("tile")
        tile.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(tile)
        label = QLabel(f"CAM_{index}" if has_sources else "Preview")
        label.setObjectName("tileTitle")
        layout.addWidget(label)
        layout.addStretch(1)
        body = QLabel(tile_text(source))
        body.setObjectName("tileBody")
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setWordWrap(True)
        layout.addWidget(body, 2)
        layout.addStretch(1)
        return tile

    def _fade_in(self, widget: QWidget, *, duration: int = 180) -> None:
        self._animate_opacity(widget, start=0.0, end=1.0, duration=duration)

    def _animate_opacity(self, widget: QWidget, *, start: float, end: float, duration: int) -> None:
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", widget)
        animation.setDuration(duration)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        def finish() -> None:
            if animation in self._animations:
                self._animations.remove(animation)
            widget.setGraphicsEffect(None)

        animation.finished.connect(finish)
        self._animations.append(animation)
        animation.start()

    def _animate_section_body(self, body: QWidget, expanded: bool) -> None:
        if expanded:
            body.setVisible(True)
            body.setMaximumHeight(0)
            target_height = max(body.sizeHint().height(), 1)
            start_height = 0
        else:
            start_height = max(body.height(), body.sizeHint().height(), 1)
            target_height = 0

        animation = QPropertyAnimation(body, b"maximumHeight", body)
        animation.setDuration(190)
        animation.setStartValue(start_height)
        animation.setEndValue(target_height)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        def finish() -> None:
            if animation in self._animations:
                self._animations.remove(animation)
            if expanded:
                body.setMaximumHeight(16777215)
            else:
                body.setVisible(False)
                body.setMaximumHeight(0)

        animation.finished.connect(finish)
        self._animations.append(animation)
        animation.start()

    def _pulse_widget(self, widget: QWidget, *, low: float = 0.78, duration: int = 120) -> None:
        if widget.graphicsEffect() is not None:
            return
        self._animate_opacity(widget, start=low, end=1.0, duration=duration)

    def eventFilter(self, watched, event) -> bool:
        if isinstance(watched, (QAbstractButton, QComboBox, QSpinBox, QTabBar, QListWidget)):
            if event.type() == QEvent.Type.MouseButtonPress:
                self._pulse_widget(watched, low=0.72, duration=130)
        elif isinstance(watched, QLineEdit) and event.type() == QEvent.Type.FocusIn:
            self._pulse_widget(watched, low=0.86, duration=160)
        return super().eventFilter(watched, event)

    def build_args(self) -> list[str]:
        args: list[str] = []
        for source in self.sources:
            args.extend(["--source", source])
        if self.destination.text().strip():
            args.extend(["--output-dir", self.destination.text().strip()])
        people_count = self.people_count.value() if self.people_count is not None else 1
        args.extend(["--max-people", str(people_count)])
        for index, combo in enumerate(self.person_color_controls, start=1):
            color = combo.currentText().strip()
            if color:
                args.extend(["--identity", f"person{index}={color}"])
        args.extend(self._advanced_args())
        args.extend(self._workflow_args())
        args.append("--skip-runtime-check")
        args.append("--no-preview")
        return args

    def _workflow_args(self) -> list[str]:
        args: list[str] = []
        if self.calibration_mode is not None and self.calibration_mode.isChecked():
            args.append("--calibrate-cameras")
            if self.calibration_output_path is not None and self.calibration_output_path.text().strip():
                args.extend(["--calibration-output", self.calibration_output_path.text().strip()])
        if self.triangulation_enabled is not None and self.triangulation_enabled.isChecked():
            args.append("--triangulate-3d")
            if self.triangulation_calibration_path is not None and self.triangulation_calibration_path.text().strip():
                args.extend(["--calibration-3d", self.triangulation_calibration_path.text().strip()])
        return args

    def _advanced_args(self) -> list[str]:
        args: list[str] = []
        wholebody_selected = self._wholebody_selected()
        for action, control in self.advanced_controls.values():
            if wholebody_selected and action.dest in {"body_backend", "hand_backend"}:
                continue
            option = action.option_strings[-1]
            if isinstance(action, argparse._StoreTrueAction):
                checked = isinstance(control, QCheckBox) and control.isChecked()
                if checked != bool(action.default):
                    args.append(option)
                continue
            value = control.currentText().strip() if isinstance(control, QComboBox) else control.text().strip()
            default = default_text(action.default)
            if not value or value == default:
                continue
            if action.nargs in ("+", "*") or action.dest in {"providers", "sync_offsets"}:
                for part in value.split(","):
                    if part.strip():
                        args.extend([option, part.strip()])
            else:
                args.extend([option, value])
        return args

    def _wholebody_selected(self) -> bool:
        control = self.advanced_controls.get("landmark_backend", (None, None))[1]
        return isinstance(control, QComboBox) and control.currentText() == "rtmpose-wholebody"

    def _sync_backend_controls(self) -> None:
        wholebody_selected = self._wholebody_selected()
        for dest in ("body_backend", "hand_backend"):
            control = self.advanced_controls.get(dest, (None, None))[1]
            row = self.advanced_rows.get(dest)
            if control is not None:
                control.setEnabled(not wholebody_selected)
            if row is not None:
                row.setEnabled(not wholebody_selected)

    def start_run(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            show_message(self, QMessageBox.Icon.Information, "Kinara is already running.")
            return
        if not self.sources:
            show_message(self, QMessageBox.Icon.Warning, "Choose at least one file or camera source.")
            return
        command = self._runner_command(self.build_args())
        self._start_process(command, enable_preview_stream=True, status_text="Running")

    def check_runtime(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            show_message(self, QMessageBox.Icon.Information, "Kinara is already running.")
            return
        people_count = self.people_count.value() if self.people_count is not None else 1
        command = self._runner_command([
            *self._advanced_args(),
            *self._workflow_args(),
            "--max-people",
            str(people_count),
            "--runtime-check",
            "--no-preview",
        ])
        self._start_process(command, enable_preview_stream=False, status_text="Checking")

    def _start_process(self, command: list[str], *, enable_preview_stream: bool, status_text: str) -> None:
        log_path = default_run_log_path("kinara_run", root=Path(self._app_dir()) / ".kinara_logs")
        self.log.clear()
        self.log.append("> " + " ".join(quote(part) for part in command))
        self.log.append(f"Log file: {log_path}")
        self.status.setText(status_text)
        if enable_preview_stream:
            self._prepare_preview_stream()
            self._show_live_preview()
            self.preview_timer.start()
        else:
            self.preview_timer.stop()

        self.process = QProcess(self)
        self.process.setWorkingDirectory(self._app_dir())
        self.process.setProgram(command[0])
        self.process.setArguments(command[1:])
        environment = QProcessEnvironment.systemEnvironment()
        installer = self._selected_python_runtime()
        if installer:
            environment.insert("KINARA_PYTHON", installer)
        source_dir = self._source_dir()
        app_dir = Path(self._app_dir())
        runtime_vendor_dir = app_dir / f".vendor_py{sys.version_info.major}{sys.version_info.minor}"
        python_paths = [str(runtime_vendor_dir), str(source_dir)]
        existing_pythonpath = environment.value("PYTHONPATH")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        environment.insert("PYTHONPATH", os.pathsep.join(python_paths))
        environment.insert("KINARA_RUNTIME_ROOT", str(app_dir))
        environment.insert("KINARA_LOG_FILE", str(log_path))
        environment.insert("KINARA_PREVIEW_FRAME", str(self.preview_frame_path))
        environment.insert("KINARA_PREVIEW_INTERVAL", "2")
        environment.insert("KINARA_PREVIEW_QUALITY", "82")
        environment.insert("KINARA_PREVIEW_KEEP_FRAMES", "8")
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._append_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self.process.start()

    def _selected_python_runtime(self) -> str:
        if self.python_path is None:
            return installer_python_path()
        value = self.python_path.text().strip().strip('"').strip("'")
        if not value:
            return ""
        path = Path(value)
        if path.is_dir():
            path = path / "python.exe"
        return str(path)

    def stop_run(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.status.setText("Stopping")
            self.process.terminate()
            self.stop_timer.start()

    def kill_run(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.status.setText("Killing")
            self.process.kill()

    def _append_process_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.append(text.rstrip())

    def _process_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self.stop_timer.stop()
        self.status.setText("Finished" if code == 0 else f"Exited {code}")
        self.preview_timer.stop()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self.stop_timer.stop()
        self.status.setText("Error")
        self.log.append(f"Process error: {error.name}")
        self.preview_timer.stop()

    def _runner_command(self, args: list[str]) -> list[str]:
        if getattr(sys, "frozen", False):
            python = self._selected_python_runtime()
            runner = self._source_dir() / "app" / "main.py"
            if python and runner.exists():
                return [python, str(runner), *args]
            return [sys.executable, "--kinara-runner", *args]
        return [sys.executable, str(Path(__file__).resolve()), "--kinara-runner", *args]

    def _app_dir(self) -> str:
        if getattr(sys, "frozen", False):
            return str(Path(sys.executable).resolve().parent)
        return str(PROJECT_ROOT)

    def _source_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            bundle_root = getattr(sys, "_MEIPASS", None)
            if bundle_root:
                return Path(bundle_root)
        return Path(self._app_dir())

    def _runtime_dir(self) -> Path:
        path = Path(self._app_dir()) / ".kinara_runtime"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _prepare_preview_stream(self) -> None:
        self.preview_frame_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_preview_mtime = 0.0
        self._preview_pixmap = None
        for path in self.preview_frame_path.parent.glob(f"{self.preview_frame_path.stem}_*{self.preview_frame_path.suffix}"):
            try:
                path.unlink()
            except OSError:
                pass

    def _load_preview_frame(self) -> None:
        frame_paths = sorted(self.preview_frame_path.parent.glob(f"{self.preview_frame_path.stem}_*{self.preview_frame_path.suffix}"))
        if not frame_paths:
            return
        frame_path = frame_paths[-1]
        stat = frame_path.stat()
        if stat.st_mtime <= self._last_preview_mtime:
            return

        try:
            data = frame_path.read_bytes()
        except OSError:
            return
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        if pixmap.isNull():
            return
        self._last_preview_mtime = stat.st_mtime
        self._preview_pixmap = pixmap
        self._render_preview_pixmap()

    def _render_preview_pixmap(self) -> None:
        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            return
        size = self.live_preview_label.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        scaled = self._preview_pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.live_preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_preview_pixmap()


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("section")
    return label


def clear_layout(layout, preserve=None) -> None:
    preserved_widgets = set() if preserve is None else set(preserve)
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            if widget in preserved_widgets:
                widget.setParent(layout.parentWidget())
            else:
                widget.deleteLater()
        child_layout = item.layout()
        if child_layout is not None:
            clear_layout(child_layout, preserve=preserved_widgets)


def load_app_icon(path: Path) -> QIcon:
    icon = QIcon(str(path))
    if path.suffix.lower() == ".ico":
        icon.addFile(str(path), mode=QIcon.Mode.Normal, state=QIcon.State.Off)
    return icon


def set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def show_message(parent: QWidget, icon: QMessageBox.Icon, text: str) -> None:
    message = QMessageBox(parent)
    message.setWindowTitle(APP_TITLE)
    message.setText(text)
    message.setIcon(icon)
    icon_path = app_icon_path(PROJECT_ROOT)
    if icon_path is not None:
        message.setWindowIcon(load_app_icon(icon_path))
    message.exec()


THEME_COLORS = {
    "dark": {
        "text": "#eef3ff",
        "muted": "#a9b6d3",
        "title": "#ffffff",
        "root": "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #08172f, stop: 0.42 #1d2b67, stop: 1 #4a2377)",
        "panel": "rgba(10, 18, 35, 242)",
        "panel_border": "rgba(132, 166, 255, 118)",
        "surface": "#101a31",
        "surface_alt": "#15223d",
        "surface_soft": "#1b2b4e",
        "border": "#38507b",
        "button": "#1d2f55",
        "button_hover": "#294477",
        "input": "#0b1428",
        "selected": "#334f87",
        "primary": "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #4dbdff, stop: 1 #9d77ff)",
        "primary_text": "#ffffff",
        "status_bg": "#2d2541",
        "status_border": "#755cb2",
        "status_text": "#ddccff",
        "danger_bg": "#4b1d2b",
        "danger_text": "#ffd7e0",
        "kill_bg": "#6b2032",
        "calibration": "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #122f54, stop: 1 #30205f)",
        "brand": "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #123865, stop: 1 #5a33a0)",
        "brand_border": "#7193ff",
        "tab_bg": "#0b1428",
    },
    "light": {
        "text": "#172033",
        "muted": "#65728a",
        "title": "#22305f",
        "root": "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #dff4ff, stop: 0.52 #eee7ff, stop: 1 #f8efff)",
        "panel": "rgba(255, 255, 255, 238)",
        "panel_border": "rgba(122, 159, 230, 86)",
        "surface": "#fbfdff",
        "surface_alt": "#f3f7ff",
        "surface_soft": "#edf2ff",
        "border": "#c8d7f5",
        "button": "#edf2ff",
        "button_hover": "#dce7ff",
        "input": "#ffffff",
        "selected": "#dce7ff",
        "primary": "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #70c8ff, stop: 1 #9c7bff)",
        "primary_text": "#ffffff",
        "status_bg": "#fff7d9",
        "status_border": "#efd990",
        "status_text": "#7a5b00",
        "danger_bg": "#ffe5ea",
        "danger_text": "#8b1d35",
        "kill_bg": "#ffccd7",
        "calibration": "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #eef9ff, stop: 1 #f4efff)",
        "brand": "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 0.5 #eef8ff, stop: 1 #f3edff)",
        "brand_border": "#bfd2ff",
        "tab_bg": "#fbfdff",
    },
}


def app_style(theme: str) -> str:
    colors = THEME_COLORS.get(theme, THEME_COLORS["dark"])
    return f"""
* {{
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 10pt;
    color: {colors["text"]};
}}
QMainWindow, QWidget#appRoot {{
    background: {colors["root"]};
}}
QFrame#preview {{
    background: {colors["panel"]};
    border: 1px solid {colors["panel_border"]};
    border-radius: 8px;
}}
QFrame#sidebar {{
    background: {colors["panel"]};
    border: 1px solid {colors["panel_border"]};
    border-radius: 8px;
}}
QFrame#brandHeader {{
    background: {colors["brand"]};
    border: 1px solid {colors["brand_border"]};
    border-radius: 8px;
}}
QFrame#commandBar {{
    background: {colors["panel"]};
    border: 1px solid {colors["panel_border"]};
    border-radius: 8px;
}}
QSplitter#workspaceSplitter, QSplitter#contentSplitter {{
    background: transparent;
}}
QSplitter#workspaceSplitter::handle, QSplitter#contentSplitter::handle {{
    background: {colors["panel_border"]};
    border-radius: 3px;
}}
QSplitter#workspaceSplitter::handle:horizontal {{
    width: 8px;
    margin-top: 6px;
    margin-bottom: 6px;
}}
QSplitter#contentSplitter::handle:vertical {{
    height: 8px;
    margin-left: 8px;
    margin-right: 8px;
}}
QSplitter#workspaceSplitter::handle:hover, QSplitter#contentSplitter::handle:hover {{
    background: #9c7bff;
}}
QFrame#tile {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
}}
QLabel#title, QLabel#pageTitle, QLabel#section, QLabel#tileTitle {{
    color: {colors["title"]};
}}
QLabel#pageTitle {{
    font-size: 18pt;
    font-weight: 800;
}}
QLabel#muted, QLabel#hint, QLabel#tileBody {{
    color: {colors["muted"]};
}}
QLabel#brandSubtitle {{
    color: {colors["muted"]};
    font-weight: 600;
}}
QLabel#brandPill, QLabel#presetPill {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    color: {colors["title"]};
    font-size: 8pt;
    font-weight: 800;
    padding: 3px 6px;
}}
QLabel#section {{
    font-weight: 700;
    font-size: 11pt;
    margin-top: 8px;
}}
QLabel#tileTitle {{
    font-weight: 700;
    font-size: 14pt;
}}
QLabel#tileBody {{
    font-size: 11pt;
}}
QLabel#livePreview {{
    background: {colors["input"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    color: {colors["muted"]};
    font-size: 12pt;
    padding: 12px;
}}
QLabel#status {{
    background: {colors["status_bg"]};
    border: 1px solid {colors["status_border"]};
    border-radius: 8px;
    color: {colors["status_text"]};
    font-weight: 700;
    padding: 8px 12px;
}}
QLabel#summaryPill {{
    background: {colors["surface_soft"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    color: {colors["title"]};
    font-weight: 700;
    padding: 8px 12px;
}}
QPushButton {{
    background: {colors["button"]};
    color: {colors["text"]};
    border: 0;
    border-radius: 8px;
    padding: 9px 13px;
    min-height: 18px;
}}
QPushButton:hover {{
    background: {colors["button_hover"]};
}}
QPushButton:pressed {{
    padding-top: 10px;
    padding-bottom: 8px;
}}
QPushButton#primaryButton {{
    background: {colors["primary"]};
    color: {colors["primary_text"]};
    font-weight: 800;
    min-width: 86px;
}}
QPushButton#dangerButton {{
    background: {colors["danger_bg"]};
    color: {colors["danger_text"]};
}}
QPushButton#killButton {{
    background: {colors["kill_bg"]};
    color: {colors["danger_text"]};
    font-weight: 800;
}}
QPushButton#secondaryButton {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
}}
QPushButton#wideButton {{
    text-align: left;
}}
QLineEdit, QSpinBox, QComboBox, QListWidget, QTextEdit {{
    background: {colors["input"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    padding: 8px;
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QTextEdit:focus {{
    border: 1px solid #9c7bff;
}}
QListWidget::item {{
    border-radius: 6px;
    padding: 6px;
}}
QListWidget::item:alternate {{
    background: {colors["surface_alt"]};
}}
QListWidget::item:selected {{
    background: {colors["selected"]};
    color: {colors["title"]};
}}
QTextEdit#log {{
    background: {colors["input"]};
    font-family: Consolas;
    font-size: 9pt;
}}
QTabWidget::pane {{
    border: 0;
}}
QTabBar::tab {{
    background: {colors["surface_soft"]};
    color: {colors["muted"]};
    padding: 9px 14px;
    border-radius: 8px;
    margin-right: 5px;
    font-weight: 700;
}}
QTabBar::tab:selected {{
    background: {colors["surface"]};
    color: {colors["title"]};
    border: 1px solid {colors["border"]};
}}
QTabBar::tab:hover {{
    background: {colors["button_hover"]};
    color: {colors["title"]};
}}
QFrame#accordionSection {{
    background: transparent;
    border: 0;
}}
QWidget#accordionBody {{
    background: transparent;
}}
QWidget#tabPage, QWidget#scrollContent, QScrollArea#tabScroll QWidget#qt_scrollarea_viewport {{
    background: {colors["tab_bg"]};
}}
QToolButton#accordionHeader {{
    background: {colors["surface_soft"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    padding: 9px 10px;
    margin-top: 4px;
    font-weight: 700;
    text-align: left;
}}
QToolButton#accordionHeader:checked {{
    background: {colors["surface"]};
    color: {colors["title"]};
}}
QToolButton#accordionHeader:hover {{
    background: {colors["button_hover"]};
}}
QFrame#advancedRow {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
}}
QFrame#calibrationPanel, QFrame#presetCard, QFrame#workflowPanel {{
    background: {colors["calibration"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
}}
QFrame#presetCard {{
    min-height: 48px;
}}
QLabel#advancedLabel {{
    color: {colors["title"]};
    font-weight: 600;
}}
QCheckBox {{
    spacing: 8px;
}}
QScrollArea {{
    border: 0;
    background: transparent;
}}
QScrollArea#tabScroll {{
    background: {colors["tab_bg"]};
}}
QScrollArea#tabScroll > QWidget {{
    background: {colors["tab_bg"]};
}}
QToolButton#iconButton {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    padding: 8px;
    min-width: 34px;
    min-height: 32px;
}}
QToolButton#iconButton:hover {{
    background: {colors["button_hover"]};
}}
QToolButton#toolbarIconButton {{
    background: {colors["input"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    color: {colors["title"]};
    font-size: 14pt;
    font-weight: 800;
    min-width: 38px;
    max-width: 38px;
    min-height: 34px;
    max-height: 34px;
}}
QToolButton#toolbarIconButton:hover {{
    background: {colors["button_hover"]};
}}
"""


if __name__ == "__main__":
    main()
