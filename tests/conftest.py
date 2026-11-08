# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Shared pytest setup. Points Ultralytics at a repository-local config directory, sets a throwaway
inference key for child processes, provides fixtures (hang guard, clip and model paths, process cleanup,
unique IPC addresses) and skips test modules whose optional packages are not installed."""
import faulthandler
import importlib.util
import os
from pathlib import Path
import secrets
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(REPO_ROOT / ".ultralytics")  # must be set before Ultralytics loads
os.environ.setdefault("TRAFFIC_INFERENCE_KEY", secrets.token_hex(32))  # child processes inherit it

import psutil
import pytest

HANG_TIMEOUT_S = 480
HANG_LOG = Path(__file__).resolve().parents[1] / "hang.log"


@pytest.fixture(autouse=True)
def hang_guard():
    """A leaked lease must end the run with a traceback, not hang it."""
    with HANG_LOG.open("w") as log:  # pytest captures stderr, so the traceback goes to a file
        faulthandler.dump_traceback_later(HANG_TIMEOUT_S, exit=True, file=log)
        yield
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def clip_path() -> Path:
    """Path of the sample video clip shipped in the repository's data folder."""
    return REPO_ROOT / "data" / "pexels-36304379.mp4"


@pytest.fixture
def models_dir() -> Path:
    """Folder holding the exported ONNX models; fails the test with a hint if they have not been exported."""
    folder = Path(__file__).resolve().parents[1] / "models"
    if not (folder / "b1.onnx").exists():
        pytest.fail("Run tools/export_models.py first")
    return folder


@pytest.fixture
def processes():
    """Processes a test starts. A failed test must not leave a server holding the GPU."""
    started = []
    yield started
    for process in started:
        if process.poll() is None:
            for child in psutil.Process(process.pid).children(recursive=True):
                child.kill()
            process.kill()
            process.wait(30)


def _address(name: str) -> str:
    """Build an IPC address unique to this process and name: a named pipe on Windows, a socket path elsewhere."""
    if sys.platform == "win32":
        return rf"\\.\pipe\test-{os.getpid()}-{name}"
    return os.path.join(tempfile.gettempdir(), f"t{os.getpid()}-{abs(hash(name)) % 10**8}.sock")  # under 108 bytes


@pytest.fixture
def make_address():
    """Factory that turns a short name into a unique IPC address, for tests that need several."""
    return _address


@pytest.fixture
def pipe_address(request) -> str:
    """Unique per test run: a second Listener on a name still in use fails."""
    return _address(request.node.name)


def _missing(package: str) -> bool:
    """Tell whether a package, possibly dotted, cannot be found for import."""
    try:
        return importlib.util.find_spec(package) is None
    except ModuleNotFoundError:  # the parent package of a dotted name is missing
        return True


# Test modules and the packages they import at module level, directly or through pipeline.overlay (cv2),
# pipeline.display and pipeline.gl_util (glfw, OpenGL, cuda.bindings). The images leave these out.
_DISPLAY = ("cv2", "glfw", "OpenGL", "cuda.bindings")
_NEEDS = {
    "test_bytetrack.py": ("ultralytics",),
    "test_detector.py": ("cv2",),
    "test_display.py": _DISPLAY,
    "test_display_wait.py": _DISPLAY,
    "test_overlay.py": ("cv2",),
    "test_status_text.py": _DISPLAY,
}
collect_ignore = [module for module, packages in _NEEDS.items() if any(_missing(name) for name in packages)]
