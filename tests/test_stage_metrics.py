# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for the stage helpers (queue put and get that honour the stop event, failure handling) and the metrics
counters and rates."""
import queue
import threading

import pytest

from pipeline.metrics import Metrics
from pipeline.stage import END, Stage, get_until, put_until


def test_put_until_gives_up_when_stopped():
    """Proves put_until returns False instead of blocking when the queue is full and the stop event is set."""
    target = queue.Queue(maxsize=1)
    target.put(1)
    stop = threading.Event()
    stop.set()
    assert put_until(target, 2, stop) is False


def test_put_until_delivers():
    """Proves put_until puts the item on a queue with room and returns True."""
    target = queue.Queue(maxsize=1)
    assert put_until(target, 2, threading.Event()) is True
    assert target.get_nowait() == 2


def test_get_until_returns_none_when_stopped():
    """Proves get_until returns None instead of blocking when the queue is empty and the stop event is set."""
    stop = threading.Event()
    stop.set()
    assert get_until(queue.Queue(), stop) is None


def test_get_until_returns_end_sentinel():
    """Proves get_until passes the end marker through to the caller."""
    source = queue.Queue()
    source.put(END)
    assert get_until(source, threading.Event()) is END


def test_stage_failure_is_recorded_and_stops_everything():
    """Proves an exception in a stage's work is recorded with the stage name and sets the stop event."""
    class Broken(Stage):
        """Stage whose work always fails."""

        def work(self):
            """Raises, to simulate a stage failure."""
            raise ValueError("boom")

    stop = threading.Event()
    failures = []
    stage = Broken("broken", stop, failures)
    stage.start()
    stage.join(2)
    assert stop.is_set()
    assert failures[0][0] == "broken"
    assert isinstance(failures[0][1], ValueError)


def test_metrics_snapshot():
    """Proves a snapshot reports the elapsed time, the counts, the average rate and the median time per stage."""
    now = [10.0]
    metrics = Metrics(clock=lambda: now[0])
    metrics.count("decoded", 50)
    metrics.record_ms("infer", 2.0)
    metrics.record_ms("infer", 4.0)
    now[0] = 20.0
    snapshot = metrics.snapshot()
    assert snapshot["elapsed_s"] == pytest.approx(10.0)
    assert snapshot["counts"]["decoded"] == 50
    assert snapshot["fps"]["decoded"] == pytest.approx(5.0)
    assert snapshot["median_ms"]["infer"] == pytest.approx(3.0)


def test_current_fps_reports_a_steady_rate():
    """Proves the current rate is about 25 for a steady 25 events per second, over the default and a shorter
    window."""
    now = [0.0]
    metrics = Metrics(clock=lambda: now[0])
    for _ in range(50):  # 25 events per second for two seconds
        now[0] += 0.04
        metrics.count("tracked")
    assert metrics.current_fps()["tracked"] == pytest.approx(25.0, abs=1.0)
    assert metrics.current_fps(window_s=0.5)["tracked"] == pytest.approx(25.0, abs=2.0)


def test_current_fps_ages_out_old_events():
    """Proves events older than the window no longer count towards the current rate."""
    now = [0.0]
    metrics = Metrics(clock=lambda: now[0])
    for _ in range(10):  # a fast burst, then a slow trickle
        now[0] += 0.01
        metrics.count("tracked")
    now[0] = 5.0
    metrics.count("tracked")
    assert metrics.current_fps()["tracked"] == pytest.approx(1.0)


def test_current_fps_of_a_stopped_counter_is_zero():
    """Proves a counter that stopped receiving events has a current rate of zero while its total is kept."""
    now = [0.0]
    metrics = Metrics(clock=lambda: now[0])
    metrics.count("tracked", 3)
    now[0] = 10.0
    assert metrics.current_fps()["tracked"] == 0.0
    assert metrics.snapshot()["counts"]["tracked"] == 3


def test_steady_fps_excludes_startup_and_shutdown():
    """Proves the steady rate is measured from the first to the last event, leaving out start-up and shutdown time
    and any counter with a single event, while the plain rate uses the whole run."""
    now = [0.0]
    metrics = Metrics(clock=lambda: now[0])
    now[0] = 10.0  # start-up delay before the first event
    metrics.count("tracked")
    now[0] = 12.0
    metrics.count("tracked", 1)
    now[0] = 14.0
    metrics.count("tracked", 1)
    metrics.count("once")  # a single event has no rate
    now[0] = 100.0  # shutdown delay after the last event
    snapshot = metrics.snapshot()
    assert snapshot["steady_fps"]["tracked"] == pytest.approx(2 / 4.0)
    assert "once" not in snapshot["steady_fps"]
    assert snapshot["fps"]["tracked"] == pytest.approx(3 / 100.0)
