from __future__ import annotations

import sys
from pathlib import Path

IS_FROZEN = getattr(sys, "frozen", False)
EXE_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parent.parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", str(EXE_DIR))) if IS_FROZEN else Path(__file__).resolve().parent.parent

PROJECT_ROOT = RESOURCE_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import install_safe_stdio

install_safe_stdio()


def main() -> None:
    if "--kinara-runner" in sys.argv:
        runner_args = [arg for arg in sys.argv[1:] if arg != "--kinara-runner"]
        sys.argv = [sys.argv[0], *runner_args]
        from app.main import main as run_pipeline

        run_pipeline()
        return

    from app.kinara_web_launcher import main as run_web_launcher

    run_web_launcher()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
