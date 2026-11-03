# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Shared pytest setup. Points Ultralytics at a repository-local config directory, arms a hang watchdog around
every test and provides the paths of the sample clip and the ONNX model."""
import faulthandler
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(REPO_ROOT / ".ultralytics")  # must be set before Ultralytics loads

import pytest

HANG_TIMEOUT_S = 300
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
def model_path() -> Path:
    """Path of the exported YOLO ONNX model in the repository root; the GPU tests need it to exist."""
    return REPO_ROOT / "yolo26n.onnx"
