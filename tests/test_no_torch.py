# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Import-weight tests: the instance and server modules must not pull in heavy packages when imported.

Each check imports the modules in a fresh interpreter and inspects sys.modules afterwards.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
DISPLAY_MODULES = {"pipeline.overlay", "pipeline.gl_util", "pipeline.display"}


def _missing(package: str) -> bool:
    """Return True when the package cannot be found, including when its parent package is absent."""
    try:
        return importlib.util.find_spec(package) is None
    except ModuleNotFoundError:  # the parent package of a dotted name is missing
        return True


MODULES = ["pipeline.bundle", "pipeline.config", "pipeline.protocol", "pipeline.metrics", "pipeline.stage",
           "pipeline.source", "pipeline.overlay", "pipeline.gl_util", "pipeline.bytetrack", "pipeline.gpu",
           "pipeline.decoder", "pipeline.remote", "pipeline.tracking", "pipeline.app", "pipeline.display",
           "pipeline.fleet", "pipeline.statslog", "pipeline.memory"]


def test_instance_modules_do_not_load_heavy_packages():
    """Importing every instance module loads none of torch, onnxruntime, ultralytics or rich.

    The display modules are left out when their own dependencies (cv2, glfw, OpenGL, cuda.bindings) are not installed.
    """
    imported = [name for name in MODULES
                if not (name in DISPLAY_MODULES and any(_missing(p) for p in ("cv2", "glfw", "OpenGL", "cuda.bindings")))]
    code = ("import sys\n" + "".join(f"import {name}\n" for name in imported)
            + "heavy = [n for n in ('torch', 'onnxruntime', 'ultralytics', 'rich') if n in sys.modules]\n"
            + "sys.exit(f'loaded {heavy}' if heavy else 0)\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=V3, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


SERVER_MODULES = ["server.config", "server.service", "server.queueing", "server.letterbox", "pipeline.statslog"]


def test_server_modules_do_not_load_rich():
    """Importing the server's lightweight modules does not load rich, which only the monitor needs."""
    code = ("import sys\n" + "".join(f"import {name}\n" for name in SERVER_MODULES)
            + "sys.exit('loaded rich' if 'rich' in sys.modules else 0)\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=V3, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
