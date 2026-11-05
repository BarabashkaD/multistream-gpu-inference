# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU tests of the frame import path: a second process shares decoded frames through CUDA IPC.

These need a CUDA device and are marked `gpu`, so they are skipped on machines without one.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from server.imports import FrameImports

V3 = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.gpu


def test_a_second_process_reads_identical_frames(clip_path):
    """A child process decodes and exports frames, and this process imports them with identical pixels.

    Each frame's SHA-256 from the child must match the imported tensor, and at least one frame must reuse a
    cached mapping (no handle sent). Whatever happens, the child is killed so it cannot keep GPU memory.
    """
    count = 24  # the decoder cycles through 5 to 8 pointers, so this many frames must reuse a cached mapping
    imports = FrameImports(1920, 1080, 0)
    frame = None
    reused = 0
    child = subprocess.Popen([sys.executable, str(V3 / "tests" / "ipc_child.py"), str(clip_path), str(count)], cwd=V3,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "PYTHONPATH": str(V3)})
    try:
        torch.zeros(1, device="cuda:0")
        for _ in range(count):
            pointer, handle, digest = child.stdout.readline().split()
            reused += handle == "-"
            frame = imports.view(0, int(pointer), None if handle == "-" else bytes.fromhex(handle))
            assert frame.shape == (1620, 1920)
            assert hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest() == digest
            child.stdin.write("\n")
            child.stdin.flush()
    finally:
        del frame  # close_stream raises if any view of a mapping is still alive
        try:
            imports.close_stream(0)
            child.stdin.close()
            child.wait(30)
        finally:
            if child.poll() is None:  # a failure must not leave GPU memory held
                child.kill()
                child.wait(30)
    assert child.returncode == 0
    assert reused > 0


def test_an_unknown_pointer_without_a_handle_is_an_error():
    """Asking for a pointer that was never mapped, with no handle to map it from, raises KeyError."""
    imports = FrameImports(1920, 1080, 0)
    with pytest.raises(KeyError):
        imports.view(0, 0x1234, None)


def test_closing_an_unknown_stream_is_harmless():
    """Closing a stream that was never opened does nothing and does not raise."""
    FrameImports(1920, 1080, 0).close_stream(9)
