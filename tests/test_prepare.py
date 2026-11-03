# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for the prepare stage with a fake converter and no GPU: ordering, cleanup, late-frame dropping, and that the
app imports without loading the OpenGL libraries."""
import subprocess
import sys
import threading
import time
from pathlib import Path

from pipeline.bundle import Bundle, BundlePool, Part
from pipeline.config import PipelineConfig
from pipeline.metrics import Metrics
from pipeline.prepare import PrepareStage
from pipeline.stage import END


class FakeConverter:
    """Stand-in converter that records open, convert and close calls and can make convert fail."""

    def __init__(self, fail: bool = False):
        """Takes whether convert should raise."""
        self.calls: list = []
        self.fail = fail

    def open(self) -> None:
        """Records the call."""
        self.calls.append("open")

    def convert(self, bundle) -> None:
        """Records the call with the frame id, then raises if the converter was made to fail."""
        self.calls.append(("convert", bundle.frame_id))
        if self.fail:
            raise RuntimeError("convert failed")

    def close(self) -> None:
        """Records the call."""
        self.calls.append("close")


def _pool(count: int, released: list):
    """Builds a pool of `count` bundles without display; released buffer pointers are appended to `released`."""
    return BundlePool([Bundle(i) for i in range(count)], released.append, display_enabled=False)


def _publish(pool, stop, frame_id, due_at=None):
    """Acquires a bundle and publishes a frame into it, with an optional source due time."""
    bundle = pool.acquire(stop)
    pool.publish_frame(bundle, 1000 + frame_id, object(), frame_id, 0, frame_id, due_at=due_at)


def _run(converter, pool, config, metrics, stop, failures):
    """Runs a prepare stage to the end of its input and fails if the thread does not finish in 10 seconds."""
    stage = PrepareStage(converter, pool, config, metrics, stop, failures)
    stage.start()
    stage.join(10)
    assert not stage.is_alive()


def test_frames_reach_detect_queue_in_order_with_input_marked_and_end_forwarded():
    """Proves frames reach the detect queue in order with the input part marked, the end marker follows them, and the
    converter is opened once before and closed once after the conversions."""
    stop, failures, released = threading.Event(), [], []
    pool = _pool(2, released)
    for frame_id in range(2):
        _publish(pool, stop, frame_id)
    pool.to_infer.put(END)
    converter = FakeConverter()

    _run(converter, pool, PipelineConfig(display=False), Metrics(), stop, failures)

    assert not failures, failures
    first, second, end = pool.to_detect.get_nowait(), pool.to_detect.get_nowait(), pool.to_detect.get_nowait()
    assert (first.frame_id, second.frame_id) == (0, 1)
    assert first.parts & Part.INPUT and second.parts & Part.INPUT
    assert end is END
    assert converter.calls == ["open", ("convert", 0), ("convert", 1), "close"]


def test_close_is_called_once_even_if_convert_raises():
    """Proves the converter is closed exactly once when a conversion fails, and the failure is recorded under the
    prepare stage and stops the run."""
    stop, failures, released = threading.Event(), [], []
    pool = _pool(1, released)
    _publish(pool, stop, 0)
    converter = FakeConverter(fail=True)

    _run(converter, pool, PipelineConfig(display=False), Metrics(), stop, failures)

    assert converter.calls.count("open") == 1
    assert converter.calls.count("close") == 1
    assert [name for name, _ in failures] == ["prepare"]
    assert stop.is_set()


def test_frame_behind_its_source_time_is_dropped_and_released():
    """Proves with the drop policy on, a frame later than the allowed lag is counted as dropped, its buffer is
    released and it is not converted."""
    stop, failures, released = threading.Event(), [], []
    pool = _pool(1, released)
    _publish(pool, stop, 0, due_at=time.monotonic() - 10.0)
    pool.to_infer.put(END)
    metrics = Metrics()
    converter = FakeConverter()
    config = PipelineConfig(display=False, drop_policy="oldest", max_lag_s=2.0)

    _run(converter, pool, config, metrics, stop, failures)

    assert not failures, failures
    assert metrics.snapshot()["counts"]["dropped"] == 1
    assert released == [1000]
    assert ("convert", 0) not in converter.calls
    assert pool.to_detect.get_nowait() is END


def test_late_frame_is_kept_when_drop_policy_is_none():
    """Proves with no drop policy a late frame is still converted and passed on, and nothing is counted as dropped."""
    stop, failures, released = threading.Event(), [], []
    pool = _pool(1, released)
    _publish(pool, stop, 0, due_at=time.monotonic() - 10.0)
    pool.to_infer.put(END)
    metrics = Metrics()
    config = PipelineConfig(display=False, drop_policy="none", max_lag_s=2.0)

    _run(FakeConverter(), pool, config, metrics, stop, failures)

    assert not failures, failures
    assert metrics.snapshot()["counts"].get("dropped", 0) == 0
    assert released == []
    assert pool.to_detect.get_nowait().frame_id == 0


def test_importing_the_app_does_not_import_glfw():
    """Proves importing the app module in a fresh interpreter loads neither glfw nor OpenGL, so the CUDA prepare
    path works without them."""
    v2 = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import pipeline.app; sys.exit(int('glfw' in sys.modules or 'OpenGL' in sys.modules))"],
        cwd=v2, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_pipeline_module_uses_glbegin():
    """Proves no pipeline source file uses the immediate-mode begin and end calls, which suspend OpenGL error
    checking."""
    # PyOpenGL suspends error checking for every thread between glBegin and glEnd (the flag is process-wide),
    # which can hide real GL errors raised on the prepare thread.
    folder = Path(__file__).resolve().parent.parent / "pipeline"
    offenders = [path.name for path in folder.glob("*.py")
                 if "glBegin(" in path.read_text(encoding="utf-8") or "glEnd(" in path.read_text(encoding="utf-8")]
    assert not offenders, offenders
