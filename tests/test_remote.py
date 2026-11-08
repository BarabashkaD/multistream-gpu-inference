# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of pipeline.remote: the stages that submit frames to the server and collect its results.

They run on the CPU with a multiprocessing pipe standing in for the server connection.
"""
from multiprocessing import Pipe
import threading
import time

import numpy as np
import pytest

from pipeline.bundle import Bundle, BundlePool, Part
from pipeline.config import InstanceConfig
from pipeline.metrics import Metrics
import pipeline.remote as remote
from pipeline.protocol import Frame, Reject, Result
from pipeline.remote import CollectStage, InFlight, SubmitStage
from pipeline.stage import END


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    """Replace GPU selection and handle export with fakes so no CUDA is needed; the handle is b"handle-<pointer>"."""
    monkeypatch.setattr(remote, "use_gpu", lambda gpu_id: None)
    monkeypatch.setattr(remote, "export_handle", lambda pointer: b"handle-%d" % pointer)


def rig(**overrides):
    """Build a test rig: a four-bundle pool, a config with the overrides, metrics, a stop event and two lists.

    Returns (pool, config, metrics, stop, failures, released), where `released` collects the pointers the pool
    gives back.
    """
    released = []
    pool = BundlePool([Bundle(i) for i in range(4)], released.append, display_enabled=False)
    config = InstanceConfig(address="unused", **overrides)
    return pool, config, Metrics(), threading.Event(), [], released


def publish(pool, stop, frame_id, pointer=0x10, due_at=None):
    """Take a bundle from the pool, publish a frame into it and return the bundle."""
    bundle = pool.acquire(stop)
    pool.publish_frame(bundle, pointer, frame_id, 0, frame_id, due_at=due_at)
    return bundle


def test_submit_sends_a_handle_only_once():
    """For frames sharing one buffer, only the first Frame message carries the memory handle."""
    pool, config, metrics, stop, failures, _ = rig()
    near, far = Pipe()
    in_flight = InFlight()
    publish(pool, stop, 0)
    publish(pool, stop, 1)
    pool.to_infer.put(END)
    stage = SubmitStage(far, pool, in_flight, config, metrics, stop, failures)
    stage.start()
    stage.join(2.0)
    assert not failures
    assert near.recv() == Frame(0, 0x10, b"handle-16")
    assert near.recv() == Frame(1, 0x10, None)
    assert len(in_flight) == 2
    assert in_flight.closed


def test_submit_drops_a_late_frame_at_real_time():
    """At real-time pace with the oldest-drop policy, a frame already past its due time is dropped, not sent."""
    pool, config, metrics, stop, failures, released = rig(pace="realtime", drop_policy="oldest", max_lag_s=0.5)
    near, far = Pipe()
    in_flight = InFlight()
    publish(pool, stop, 0, due_at=time.monotonic() - 10.0)
    pool.to_infer.put(END)
    stage = SubmitStage(far, pool, in_flight, config, metrics, stop, failures)
    stage.start()
    stage.join(2.0)
    assert not near.poll(0.05)
    assert metrics.snapshot()["counts"]["dropped"] == 1
    assert released == [0x10]


@pytest.mark.parametrize("overrides", [{"pace": "max"}, {"drop_policy": "none"}])
def test_no_drop_at_max_pace_or_with_the_policy_off(overrides):
    """A late frame is still sent when the pace is "max" or the drop policy is "none"."""
    pool, config, metrics, stop, failures, _ = rig(**overrides)
    near, far = Pipe()
    publish(pool, stop, 0, due_at=time.monotonic() - 10.0)
    pool.to_infer.put(END)
    stage = SubmitStage(far, pool, InFlight(), config, metrics, stop, failures)
    stage.start()
    stage.join(2.0)
    assert near.poll(1.0)


def test_collect_hands_a_result_to_the_tracker_and_finishes():
    """A result for a frame in flight is decoded into the bundle's rows and passed on, then END follows on close."""
    pool, config, metrics, stop, failures, _ = rig()
    near, far = Pipe()
    in_flight = InFlight()
    bundle = publish(pool, stop, 3)
    pool.to_infer.get()
    in_flight.add(bundle, time.monotonic())
    rows = np.array([[1, 2, 3, 4, 0.9, 2]], np.float32)
    near.send(Result(3, rows.tobytes()))
    stage = CollectStage(far, pool, in_flight, config, metrics, stop, failures)
    stage.start()
    delivered = pool.to_track.get(timeout=2.0)
    assert delivered is bundle
    assert np.array_equal(bundle.rows, rows)
    assert bundle.parts & Part.DETECTIONS
    in_flight.close()
    assert pool.to_track.get(timeout=2.0) is END
    stage.join(2.0)
    assert not failures


def test_collect_ignores_a_result_nobody_is_waiting_for():
    """A result whose frame id is not in flight is skipped without a failure, and the stage still ends cleanly."""
    pool, config, metrics, stop, failures, _ = rig()
    near, far = Pipe()
    near.send(Result(99, b""))
    in_flight = InFlight()
    in_flight.close()
    stage = CollectStage(far, pool, in_flight, config, metrics, stop, failures)
    stage.start()
    assert pool.to_track.get(timeout=2.0) is END
    stage.join(2.0)
    assert not failures


def test_a_silent_server_is_a_failure():
    """When a frame in flight gets no answer within the server timeout, the stage records a TimeoutError and stops."""
    pool, config, metrics, stop, failures, _ = rig(server_timeout_s=0.05)
    near, far = Pipe()
    in_flight = InFlight()
    in_flight.add(publish(pool, stop, 0), time.monotonic())
    stage = CollectStage(far, pool, in_flight, config, metrics, stop, failures)
    stage.start()
    stage.join(2.0)
    assert failures and isinstance(failures[0][1], TimeoutError)
    assert stop.is_set()


def test_a_busy_server_does_not_hide_a_stuck_frame():
    """Results for other frames arriving steadily do not postpone the timeout of a frame that never gets one."""
    pool, config, metrics, stop, failures, _ = rig(server_timeout_s=0.1)
    near, far = Pipe()
    in_flight = InFlight()
    in_flight.add(publish(pool, stop, 0), time.monotonic() - 1.0)
    stage = CollectStage(far, pool, in_flight, config, metrics, stop, failures)
    stage.start()
    failed_while_busy = False
    deadline = time.monotonic() + 1.0
    number = 100
    while time.monotonic() < deadline and not failed_while_busy:
        near.send(Result(number, b""))
        number += 1
        time.sleep(0.01)
        failed_while_busy = bool(failures)
    stage.join(2.0)
    assert failed_while_busy, "the timeout only fired after results stopped arriving"
    assert isinstance(failures[0][1], TimeoutError)


def test_a_reject_stops_the_stream_with_its_reason():
    """A Reject message ends the stage with a failure whose text contains the server's reason."""
    pool, config, metrics, stop, failures, _ = rig()
    near, far = Pipe()
    near.send(Reject("cuIpcOpenMemHandle failed"))
    stage = CollectStage(far, pool, InFlight(), config, metrics, stop, failures)
    stage.start()
    stage.join(2.0)
    assert "cuIpcOpenMemHandle failed" in str(failures[0][1])


def test_a_broken_pipe_is_a_failure():
    """When the server end of the pipe is closed, the stage records a failure and sets stop."""
    pool, config, metrics, stop, failures, _ = rig()
    near, far = Pipe()
    near.close()
    stage = CollectStage(far, pool, InFlight(), config, metrics, stop, failures)
    stage.start()
    stage.join(2.0)
    assert failures
    assert stop.is_set()


def test_collect_stops_on_the_stop_event():
    """An idle collect stage exits promptly once the stop event is set."""
    pool, config, metrics, stop, failures, _ = rig()
    near, far = Pipe()
    stage = CollectStage(far, pool, InFlight(), config, metrics, stop, failures)
    stage.start()
    stop.set()
    stage.join(2.0)
    assert not stage.is_alive()
