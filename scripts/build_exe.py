from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DIST_ROOT = ARTIFACTS_ROOT / "windows"
BUILD_ROOT = ARTIFACTS_ROOT / "pyinstaller" / "build"
SPEC_ROOT = ARTIFACTS_ROOT / "pyinstaller" / "spec"
BUNDLE_MODELS = Path(tempfile.gettempdir()) / "kinara_bundle_models"
BUILT_LAUNCHER = DIST_ROOT / "Kinara.exe"


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
        raise RuntimeError(output or f"Command failed with exit code {completed.returncode}: {' '.join(command)}")
    return completed.stdout.strip()


def _valid_python(candidate: str | Path | None) -> Path | None:
    if not candidate:
        return None
    path = Path(str(candidate).strip().strip('"').strip("'"))
    if path.is_dir():
        path = path / "python.exe"
    if not path.exists() or path.name.lower() != "python.exe":
        return None
    completed = subprocess.run(
        [str(path), "-c", "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"],
        capture_output=True,
        check=False,
    )
    return path if completed.returncode == 0 else None


def resolve_python() -> Path:
    candidates: list[str | Path | None] = [
        os.environ.get("KINARA_BUILD_PYTHON"),
        os.environ.get("KINARA_PYTHON"),
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        sys.executable,
    ]
    path_python = shutil.which("python")
    if path_python:
        candidates.append(path_python)
    py_launcher = shutil.which("py")
    if py_launcher:
        completed = subprocess.run(
            [py_launcher, "-3.11", "-c", "import sys; print(sys.executable)"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            candidates.append(completed.stdout.strip())

    for candidate in candidates:
        resolved = _valid_python(candidate)
        if resolved is not None:
            return resolved
    raise RuntimeError("Python 3.11 was not found. Set KINARA_BUILD_PYTHON or KINARA_PYTHON to python.exe.")


def collect_model_bundle(pyinstaller_args: list[str]) -> None:
    models_root = PROJECT_ROOT / "models"
    if not models_root.exists():
        return
    model_files = []
    for file_path in models_root.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(models_root)
        relative_text = str(relative).replace("/", "\\")
        if relative_text.startswith("body\\pose_landmark_") and relative_text.endswith(".tflite"):
            continue
        if relative_text.startswith("hand\\mediapipe\\"):
            continue
        model_files.append(file_path)

    for file_path in model_files:
        destination = BUNDLE_MODELS / file_path.relative_to(models_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)

    if model_files:
        pyinstaller_args.extend(["--add-data", f"{BUNDLE_MODELS};models"])


def build() -> None:
    python = resolve_python()
    os.environ.pop("PYTHONNOUSERSITE", None)
    print(f"Using Python: {python.name}")
    pyinstaller_version = _run([str(python), "-m", "PyInstaller", "--version"])
    print(f"Using PyInstaller: {pyinstaller_version}")

    if BUNDLE_MODELS.exists():
        shutil.rmtree(BUNDLE_MODELS)
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    SPEC_ROOT.mkdir(parents=True, exist_ok=True)

    args = [
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name", "Kinara",
        "--distpath", str(DIST_ROOT),
        "--workpath", str(BUILD_ROOT),
        "--specpath", str(SPEC_ROOT),
        "--paths", str(PROJECT_ROOT),
        "--exclude-module", "cv2",
        "--exclude-module", "mediapipe",
        "--exclude-module", "ultralytics",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "onnxruntime",
        "--exclude-module", "jax",
        "--exclude-module", "jaxlib",
        "--exclude-module", "polars",
        "--exclude-module", "pandas",
        "--exclude-module", "scipy",
        "--exclude-module", "matplotlib",
        "--hidden-import", "webview",
        "--hidden-import", "plistlib",
        "--hidden-import", "timeit",
        "--hidden-import", "zoneinfo",
        "--hidden-import", "aniposelib",
        "--hidden-import", "aniposelib.boards",
        "--hidden-import", "aniposelib.cameras",
    ]

    try:
        if os.environ.get("KINARA_BUNDLE_HEAVY_MODELS") == "1":
            print("Bundling pre-downloaded model weights into executable...")
            collect_model_bundle(args)
        else:
            print("Building ultra-lightweight executable (models will be downloaded on-demand into .kinara_runtime)...")
        ultralytics_config = PROJECT_ROOT / ".ultralytics"
        if ultralytics_config.exists():
            args.extend(["--add-data", f"{ultralytics_config};.ultralytics"])
        assets_root = PROJECT_ROOT / "assets"
        if assets_root.exists():
            args.extend(["--add-data", f"{assets_root};assets"])
        ui_root = PROJECT_ROOT / "app" / "ui"
        if ui_root.exists():
            args.extend(["--add-data", f"{ui_root};app/ui"])
        icon_path = PROJECT_ROOT / "assets" / "kinara.ico"
        if icon_path.exists():
            args.extend(["--icon", str(icon_path)])

        args.append(str(PROJECT_ROOT / "app" / "kinara_launcher.py"))
        subprocess.run([str(python), "-m", "PyInstaller", *args], cwd=PROJECT_ROOT, check=True)
    finally:
        if BUNDLE_MODELS.exists():
            shutil.rmtree(BUNDLE_MODELS)

    print(f"Built launcher: {BUILT_LAUNCHER}")


def main() -> int:
    try:
        build()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
