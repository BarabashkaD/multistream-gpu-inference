# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of server.imports.FrameImports: mapping, caching and closing of imported frame memory.

They run on the CPU: plain memory stands in for mapped frames, so no CUDA device is needed.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import server.imports as imports_module
from server.imports import FrameImports, _Opened

_KDL_CPU = 1
_FRAME_BYTES = 6 * 4  # NV12 of a 4 by 4 frame
_EXPORTED = 0x1000  # the exporter's pointer, which is not the address this process maps it at


class Fake:
    """A cached mapping stand-in that records only the pointer it is closed by."""

    def __init__(self, pointer):
        """Store the pointer, with a placeholder tensor and the released flag already set."""
        self.pointer = pointer
        self.tensor = object()
        self.released = True


@pytest.fixture
def mapped(monkeypatch):
    """Plain memory stands in for mapped frames, and the DLPack struct says CPU, so CUDA never loads."""
    backing = (ctypes.c_uint8 * (2 * _FRAME_BYTES))()
    base = ctypes.addressof(backing)
    closed = []
    monkeypatch.setattr(imports_module, "_KDL_CUDA", _KDL_CPU)
    monkeypatch.setattr(imports_module, "open_handle", lambda handle: base + handle[0] * _FRAME_BYTES)
    monkeypatch.setattr(imports_module, "close_handle", closed.append)
    return SimpleNamespace(first=base, second=base + _FRAME_BYTES, closed=closed, backing=backing)


@pytest.fixture
def deleter_calls(monkeypatch):
    """Wrap the DLPack deleter so a test can count how many times it fires; returns the list of calls."""
    calls = []
    original = imports_module._released

    @imports_module._DELETER
    def counting(managed):
        """Record the call, then run the real deleter."""
        calls.append(1)
        original(managed)

    monkeypatch.setattr(imports_module, "_released", counting)
    return calls


def key_of(opened) -> int:
    """Return the address of the opened mapping's DLPack struct, which keys the table of live views."""
    return ctypes.addressof(opened.managed)


def test_close_stream_tries_every_mapping_and_raises_the_first_error(monkeypatch):
    """When several mappings fail to close, close_stream still closes all of them and raises the first error."""
    closed = []

    def close_handle(pointer):
        """Record the close and fail for pointers 1 and 2."""
        closed.append(pointer)
        if pointer in (1, 2):
            raise RuntimeError(f"close {pointer} failed")

    monkeypatch.setattr(imports_module, "close_handle", close_handle)
    imports = FrameImports(4, 4, 0)
    imports._streams = {7: {1: Fake(1), 2: Fake(2), 3: Fake(3)}}
    with pytest.raises(RuntimeError, match="close 1 failed"):
        imports.close_stream(7)
    assert closed == [1, 2, 3]
    assert 7 not in imports._streams


def test_the_deleter_fires_once_when_the_last_view_dies(mapped, deleter_calls):
    """The DLPack deleter runs exactly once, only after the last tensor derived from the mapping is gone."""
    opened = _Opened(mapped.first, 4, 4, 0)
    key = key_of(opened)
    view = opened.tensor
    derived = view[:2]
    del opened.tensor
    del view
    assert deleter_calls == []
    assert not opened.released
    assert key in imports_module._live
    del derived
    assert deleter_calls == [1]
    assert opened.released
    assert key not in imports_module._live


def test_close_stream_closes_the_mapping_when_no_other_view_exists(mapped):
    """With no view still alive, close_stream closes the mapping and drops its struct from the live table."""
    imports = FrameImports(4, 4, 0)
    imports.view(0, _EXPORTED, b"\x00")
    opened = imports._streams[0][_EXPORTED]
    imports.close_stream(0)
    assert mapped.closed == [mapped.first]
    assert opened.released
    assert key_of(opened) not in imports_module._live


def test_close_stream_keeps_the_mapping_under_a_surviving_view(mapped):
    """A surviving view makes close_stream raise and leave the mapping open, and it is never unmapped later."""
    imports = FrameImports(4, 4, 0)
    frame = imports.view(0, _EXPORTED, b"\x00")
    key = key_of(imports._streams[0][_EXPORTED])
    with pytest.raises(RuntimeError, match="a view of stream 0 is still alive"):
        imports.close_stream(0)
    assert mapped.closed == []
    assert key in imports_module._live
    spray = [ctypes.create_string_buffer(b"\x41" * 63, 64) for _ in range(200000)]  # would reuse a freed struct
    assert int(frame.sum()) == 0
    del frame  # torch reads the deleter from the struct here
    assert key not in imports_module._live
    assert len(spray) == 200000
    assert mapped.closed == []  # a leak is safe, and nothing may unmap it later


def test_a_surviving_view_does_not_stop_the_other_mappings_closing(mapped):
    """One mapping with a live view does not prevent the stream's other mappings from being closed."""
    imports = FrameImports(4, 4, 0)
    frame = imports.view(0, _EXPORTED, b"\x00")
    imports.view(0, _EXPORTED + 1, b"\x01")
    with pytest.raises(RuntimeError, match="still alive"):
        imports.close_stream(0)
    assert mapped.closed == [mapped.second]
    del frame


def test_a_cached_mapping_gives_the_same_view(mapped):
    """Asking again for a known pointer without a handle returns the very same tensor, and one close frees it."""
    imports = FrameImports(4, 4, 0)
    assert imports.view(0, _EXPORTED, b"\x00") is imports.view(0, _EXPORTED, None)
    imports.close_stream(0)
    assert mapped.closed == [mapped.first]


def test_a_construction_failure_closes_the_handle(mapped, monkeypatch):
    """If building the tensor fails, the freshly opened mapping is closed and nothing is cached or left live."""
    monkeypatch.setattr(imports_module, "_KDL_CUDA", 99)  # torch knows no such device
    live = dict(imports_module._live)
    imports = FrameImports(4, 4, 0)
    with pytest.raises(BufferError):
        imports.view(0, _EXPORTED, b"\x00")
    assert mapped.closed == [mapped.first]
    assert imports._streams[0] == {}
    assert imports_module._live == live


_EXIT_WITH_OPEN_STREAMS = """
import ctypes
import server.imports as si
si._KDL_CUDA = 1
backing = (ctypes.c_uint8 * 24)()
si.open_handle = lambda handle: ctypes.addressof(backing)
si.close_handle = lambda pointer: None
imports = si.FrameImports(4, 4, 0)
"""

_OPEN_STREAMS = """
def serve():  # as run_server.main when it stops with streams still open: close_stream never runs
    imports = si.FrameImports(4, 4, 0)
    imports.view(0, 0x1000, b"x")
    imports.view(1, 0x1000, b"x")
    return imports.view(2, 0x1000, b"x")

frame = serve()
"""

_SURVIVOR = """
frame = imports.view(3, 0x1000, b"x")
imports.close_stream(3)  # uncaught: the traceback keeps frame alive until after the module globals are gone
"""


@pytest.mark.parametrize(("ending", "status", "error"), [(_OPEN_STREAMS, 0, ""),
                                                        (_SURVIVOR, 1, "a view of stream 3 is still alive")],
                         ids=["open-streams", "surviving-view"])
def test_shutdown_never_frees_a_struct_under_a_view_or_runs_a_broken_deleter(tmp_path, ending, status, error):
    """A child interpreter exiting with open streams or a surviving view neither crashes nor runs a stale deleter.

    The child runs with memory poisoning on and must exit with the expected status and message, with no
    ignored exception and no fatal error on stderr.
    """
    v3 = Path(__file__).resolve().parents[1]
    script = tmp_path / "shutdown.py"  # a file, not -c: only then does the traceback outlive the module globals
    script.write_text(_EXIT_WITH_OPEN_STREAMS + ending)
    finished = subprocess.run([sys.executable, "-X", "faulthandler", str(script)], cwd=v3,
                              capture_output=True, text=True, timeout=120,
                              env={**os.environ, "PYTHONPATH": str(v3), "PYTHONMALLOC": "debug"})  # poisons freed memory
    assert finished.returncode == status, finished.stderr
    assert error in finished.stderr
    assert "Exception ignored" not in finished.stderr
    assert "Fatal Python error" not in finished.stderr


def test_an_interrupted_construction_closes_the_handle(mapped, monkeypatch):
    """A KeyboardInterrupt while building the tensor still closes the mapping and leaves no live struct."""
    def interrupted(capsule):
        """Raise KeyboardInterrupt in place of the DLPack import."""
        raise KeyboardInterrupt

    monkeypatch.setattr(torch.utils.dlpack, "from_dlpack", interrupted)
    live = dict(imports_module._live)
    with pytest.raises(KeyboardInterrupt):
        FrameImports(4, 4, 0).view(0, _EXPORTED, b"\x00")
    assert mapped.closed == [mapped.first]
    assert imports_module._live == live
