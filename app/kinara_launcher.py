from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import default_run_log_path, install_safe_stdio

install_safe_stdio()

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
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
    icon_path = app_icon_path(PROJECT_ROOT)
    if icon_path is not None:
        app.setWindowIcon(load_app_icon(icon_path))
    window = KinaraLauncher()
    window.show()
    sys.exit(app.exec())


class KinaraLauncher(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
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

        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._refresh_people()
        self._refresh_sources()
        self._refresh_preview()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(12)
        layout.addLayout(left, 3)

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

        self.preview = QFrame()
        self.preview.setObjectName("preview")
        self.preview_layout = QGridLayout(self.preview)
        self.preview_layout.setContentsMargins(18, 18, 18, 18)
        self.preview_layout.setSpacing(16)
        left.addWidget(self.preview, 1)
        self.live_preview_label = QLabel("Processed preview will appear here when a run starts")
        self.live_preview_label.setObjectName("livePreview")
        self.live_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_preview_label.setScaledContents(False)
        self.live_preview_label.hide()

        controls = QHBoxLayout()
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
        left.addLayout(controls)

        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(145)
        self.log.setMaximumHeight(190)
        left.addWidget(self.log)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(380)
        sidebar.setMaximumWidth(430)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(12)
        layout.addWidget(sidebar, 1)

        title = QLabel("Kinara")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        side_layout.addWidget(title)
        subtitle = QLabel("Desktop launcher")
        subtitle.setObjectName("muted")
        side_layout.addWidget(subtitle)

        tabs = QTabWidget()
        tabs.addTab(self._camera_tab(), "Camera")
        tabs.addTab(self._file_tab(), "File")
        tabs.addTab(self._advanced_tab(), "Advanced")
        side_layout.addWidget(tabs, 1)

    def _camera_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.addWidget(section_label("Camera input"))

        open_camera = QPushButton("Open Camera")
        open_camera.setObjectName("wideButton")
        open_camera.clicked.connect(self.use_camera)
        layout.addWidget(open_camera)

        udp = QPushButton("Wait for UDP")
        udp.setObjectName("wideButton")
        udp.clicked.connect(self.use_udp)
        layout.addWidget(udp)

        hint = QLabel(
            "UDP/LAN/WLAN/USB phone camera support is in development mode. "
            "This reserves the app workflow while the Android sender is built."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return tab

    def _file_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        layout.addWidget(section_label("Sources"))
        source_buttons = QHBoxLayout()
        add_files = QPushButton("Add files")
        add_files.setObjectName("wideButton")
        add_files.clicked.connect(self.add_files)
        clear = QPushButton("Clear")
        clear.setObjectName("secondaryButton")
        clear.clicked.connect(self.clear_sources)
        source_buttons.addWidget(add_files)
        source_buttons.addWidget(clear)
        layout.addLayout(source_buttons)

        self.source_list = QListWidget()
        self.source_list.setMinimumHeight(116)
        self.source_list.setAlternatingRowColors(True)
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

        layout.addWidget(section_label("People"))
        people_row = QHBoxLayout()
        people_row.addWidget(QLabel("Amount"))
        self.people_count = QSpinBox()
        self.people_count.setRange(1, 12)
        self.people_count.setValue(1)
        self.people_count.valueChanged.connect(self._refresh_people)
        people_row.addWidget(self.people_count)
        people_row.addStretch(1)
        layout.addLayout(people_row)

        self.people_box = QVBoxLayout()
        layout.addLayout(self.people_box)
        layout.addStretch(1)
        return tab

    def _advanced_tab(self) -> QWidget:
        tab = QWidget()
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
        scroll.setWidgetResizable(True)
        content = QWidget()
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

        category_layouts["Runtime"].addWidget(self._python_runtime_control())

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
        body.setVisible(checked)
        if not checked:
            return
        for other_title, other_button in self.advanced_section_buttons.items():
            if other_title == title or not other_button.isChecked():
                continue
            other_button.blockSignals(True)
            other_button.setChecked(False)
            other_button.setArrowType(Qt.ArrowType.RightArrow)
            other_button.blockSignals(False)
            self.advanced_section_bodies[other_title].setVisible(False)

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
        browse = QPushButton("Browse")
        browse.setObjectName("secondaryButton")
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

    def _filter_advanced_controls(self, query: str) -> None:
        normalized_query = query.strip().lower()
        if not normalized_query:
            for row in self.advanced_rows.values():
                row.setVisible(True)
            for section_title, frame in self.advanced_section_frames.items():
                frame.setVisible(True)
                button = self.advanced_section_buttons[section_title]
                self.advanced_section_bodies[section_title].setVisible(button.isChecked())
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
            self.advanced_section_buttons[section_title].setArrowType(Qt.ArrowType.DownArrow if matched else Qt.ArrowType.RightArrow)

    def use_camera(self) -> None:
        self.sources = ["0"]
        self._refresh_sources()
        self._refresh_preview()

    def use_udp(self) -> None:
        self.sources = ["UDP_DEVELOPMENT_MODE"]
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

    def _refresh_sources(self) -> None:
        self.source_list.clear()
        for index, source in enumerate(self.sources):
            self.source_list.addItem(f"CAM_{index}: {source}")
        count = len(self.sources)
        self.source_summary.setText(f"{count} source" if count == 1 else f"{count} sources")

    def _refresh_people(self) -> None:
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

    def _refresh_preview(self) -> None:
        clear_layout(self.preview_layout, preserve={self.live_preview_label})
        self.live_preview_label.hide()
        sources = self.sources or ["No source selected"]
        count = len(sources)
        rows, cols = (1, 1) if count == 1 else ((1, 2) if count == 2 else (2, 2))
        for index, source in enumerate(sources[:4]):
            tile = self._source_tile(index, source, has_sources=bool(self.sources))
            self.preview_layout.addWidget(tile, index // cols, index % cols)
        for row in range(rows):
            self.preview_layout.setRowStretch(row, 1)
        for col in range(cols):
            self.preview_layout.setColumnStretch(col, 1)

    def _show_live_preview(self) -> None:
        clear_layout(self.preview_layout, preserve={self.live_preview_label})
        self.live_preview_label.setText("Waiting for processed frames...")
        self.live_preview_label.show()
        self.preview_layout.addWidget(self.live_preview_label, 0, 0)
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

    def build_args(self) -> list[str]:
        args: list[str] = []
        for source in self.sources:
            if source != "UDP_DEVELOPMENT_MODE":
                args.extend(["--source", source])
        if self.destination.text().strip():
            args.extend(["--output-dir", self.destination.text().strip()])
        args.extend(["--max-people", str(self.people_count.value())])
        for index, combo in enumerate(self.person_color_controls, start=1):
            color = combo.currentText().strip()
            if color:
                args.extend(["--identity", f"person{index}={color}"])
        args.extend(self._advanced_args())
        args.append("--no-preview")
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
        if "UDP_DEVELOPMENT_MODE" in self.sources:
            show_message(self, QMessageBox.Icon.Information, "UDP camera input is reserved for development mode.")
            return

        command = self._runner_command(self.build_args())
        self._start_process(command, enable_preview_stream=True, status_text="Running")

    def check_runtime(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            show_message(self, QMessageBox.Icon.Information, "Kinara is already running.")
            return
        command = self._runner_command([*self._advanced_args(), "--runtime-check", "--no-preview"])
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
        value = self.python_path.text().strip()
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
            return [sys.executable, "--kinara-runner", *args]
        return [sys.executable, str(Path(__file__).resolve()), "--kinara-runner", *args]

    def _app_dir(self) -> str:
        if getattr(sys, "frozen", False):
            return str(Path(sys.executable).resolve().parent)
        return str(PROJECT_ROOT)

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


APP_STYLE = """
QMainWindow, QWidget {
    background: #0f1115;
    color: #eef2f7;
    font-family: Segoe UI;
    font-size: 10pt;
}
QFrame#preview {
    background: #151922;
    border: 1px solid #2a3140;
    border-radius: 22px;
}
QFrame#sidebar {
    background: #171b22;
    border: 1px solid #242b36;
    border-radius: 18px;
}
QFrame#tile {
    background: #0b0d12;
    border: 1px solid #303848;
    border-radius: 16px;
}
QLabel#title {
    color: #ffffff;
}
QLabel#pageTitle {
    color: #ffffff;
    font-size: 18pt;
    font-weight: 800;
}
QLabel#muted, QLabel#hint {
    color: #9aa5b5;
}
QLabel#section {
    color: #ffffff;
    font-weight: 700;
    font-size: 11pt;
    margin-top: 8px;
}
QLabel#tileTitle {
    color: #ffffff;
    font-weight: 700;
    font-size: 14pt;
}
QLabel#tileBody {
    color: #8d98aa;
    font-size: 11pt;
}
QLabel#livePreview {
    background: #090b0f;
    border: 1px solid #303848;
    border-radius: 16px;
    color: #8d98aa;
    font-size: 12pt;
    padding: 12px;
}
QLabel#status {
    background: #1d2430;
    border: 1px solid #303848;
    border-radius: 10px;
    color: #efc46d;
    font-weight: 700;
    padding: 8px 12px;
}
QLabel#summaryPill {
    background: #1d2430;
    border: 1px solid #303848;
    border-radius: 12px;
    color: #dbe4f0;
    font-weight: 700;
    padding: 8px 12px;
}
QPushButton {
    background: #252d3b;
    color: #eef2f7;
    border: 0;
    border-radius: 10px;
    padding: 9px 13px;
    min-height: 18px;
}
QPushButton:hover {
    background: #303a4d;
}
QPushButton#primaryButton {
    background: #49c37b;
    color: #07110c;
    font-weight: 800;
}
QPushButton#dangerButton {
    background: #5d2530;
}
QPushButton#killButton {
    background: #842633;
    font-weight: 800;
}
QPushButton#secondaryButton {
    background: #1b212c;
    border: 1px solid #303848;
}
QPushButton#wideButton {
    text-align: left;
}
QLineEdit, QSpinBox, QComboBox, QListWidget, QTextEdit {
    background: #0f1115;
    color: #eef2f7;
    border: 1px solid #303848;
    border-radius: 10px;
    padding: 8px;
}
QListWidget::item {
    border-radius: 7px;
    padding: 6px;
}
QListWidget::item:alternate {
    background: #131821;
}
QListWidget::item:selected {
    background: #273244;
}
QTextEdit#log {
    background: #090b0f;
    font-family: Consolas;
    font-size: 9pt;
}
QTabWidget::pane {
    border: 0;
}
QTabBar::tab {
    background: #202733;
    color: #d7deeb;
    padding: 8px 13px;
    border-radius: 10px;
    margin-right: 5px;
}
QTabBar::tab:selected {
    background: #344055;
    color: #ffffff;
}
QFrame#accordionSection {
    background: transparent;
    border: 0;
}
QWidget#accordionBody {
    background: transparent;
}
QToolButton#accordionHeader {
    background: #202733;
    color: #d7deeb;
    border: 1px solid #303848;
    border-radius: 10px;
    padding: 9px 10px;
    margin-top: 4px;
    font-weight: 700;
    text-align: left;
}
QToolButton#accordionHeader:checked {
    background: #344055;
    color: #ffffff;
}
QToolButton#accordionHeader:hover {
    background: #303a4d;
}
QFrame#advancedRow {
    background: #11161f;
    border: 1px solid #303848;
    border-radius: 12px;
}
QLabel#advancedLabel {
    color: #d7deeb;
    font-weight: 600;
}
"""


if __name__ == "__main__":
    main()
