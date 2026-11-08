# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of pipeline.stage (queue helpers and the Stage thread base) and pipeline.metrics (counters and timings)."""
import queue
import threading

import pytest

from pipeline.metrics import Metrics, percentile, period_ms
from pipeline.stage import END, Stage, get_until, put_until


def test_put_until_gives_up_when_stopped():
    """put_until on a full queue returns False once the stop event is set, instead of blocking."""
    target = queue.Queue(maxsize=1)
    target.put(1)
    stop = threading.Event()
    stop.set()
    assert put_until(target, 2, stop) is False


def test_put_until_delivers():
    """put_until on a queue with room puts the item and returns True."""
    target = queue.Queue(maxsize=1)
    assert put_until(target, 2, threading.Event()) is True
    assert target.get_nowait() == 2


def test_get_until_returns_none_when_stopped():
    """get_until on an empty queue returns None once the stop event is set."""
    stop = threading.Event()
    stop.set()
    assert get_until(queue.Queue(), stop) is None


def test_get_until_returns_end_sentinel():
    """get_until passes the END sentinel through unchanged."""
    source = queue.Queue()
    source.put(END)
    assert get_until(source, threading.Event()) is END


def test_stage_failure_is_recorded_and_stops_everything():
    """An exception in a stage's work is recorded with the stage name and sets the shared stop event."""
    class Broken(Stage):
        """A stage whose work always fails."""

        def work(self):
            """Raise ValueError at once."""
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
    """A snapshot reports elapsed time, counts, whole-run rates and median timings."""
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
    """current_fps measures about 25 events per second for a steady 25 Hz counter, over the default or a short window."""
    now = [0.0]
    metrics = Metrics(clock=lambda: now[0])
    for _ in range(50):  # 25 events per second for two seconds
        now[0] += 0.04
        metrics.count("tracked")
    assert metrics.current_fps()["tracked"] == pytest.approx(25.0, abs=1.0)
    assert metrics.current_fps(window_s=0.5)["tracked"] == pytest.approx(25.0, abs=2.0)


def test_current_fps_ages_out_old_events():
    """Events older than the window no longer count, so a burst followed by one late event gives about 1 per second."""
    now = [0.0]
    metrics = Metrics(clock=lambda: now[0])
    for _ in range(10):  # a fast burst, then a slow trickle
        now[0] += 0.01
        metrics.count("tracked")
    now[0] = 5.0
    metrics.count("tracked")
    assert metrics.current_fps()["tracked"] == pytest.approx(1.0)


def test_current_fps_of_a_stopped_counter_is_zero():
    """A counter that stopped long ago has a current rate of zero, though its total count is kept."""
    now = [0.0]
    metrics = Metrics(clock=lambda: now[0])
    metrics.count("tracked", 3)
    now[0] = 10.0
    assert metrics.current_fps()["tracked"] == 0.0
    assert metrics.snapshot()["counts"]["tracked"] == 3


def test_steady_fps_excludes_startup_and_shutdown():
    """steady_fps spans only the first to last event, ignores a counter with one event, and differs from fps."""
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


def test_snapshot_reports_a_p95():
    """For samples 1 to 100 ms the snapshot's p95 is 95 and its median is 50.5."""
    metrics = Metrics()
    for value in range(1, 101):
        metrics.record_ms("latency", float(value))
    snapshot = metrics.snapshot()
    assert snapshot["p95_ms"]["latency"] == 95.0
    assert snapshot["median_ms"]["latency"] == 50.5


def test_take_period_returns_each_sample_once():
    """take_period hands out the samples recorded since the last call and then forgets them."""
    metrics = Metrics()
    metrics.record_ms("decode", 1.0)
    metrics.record_ms("decode", 3.0)
    assert metrics.take_period() == {"decode": [1.0, 3.0]}
    assert metrics.take_period() == {}
    metrics.record_ms("decode", 5.0)
    metrics.record_ms("track", 2.0)
    assert metrics.take_period() == {"decode": [5.0], "track": [2.0]}


def test_take_period_keeps_at_most_the_sample_window():
    """After 2000 samples take_period returns only the newest 1024."""
    metrics = Metrics()
    for value in range(2000):
        metrics.record_ms("decode", float(value))
    taken = metrics.take_period()["decode"]
    assert len(taken) == 1024
    assert taken[-1] == 1999.0


def test_take_period_does_not_change_the_snapshot():
    """Draining the period samples leaves the whole-run median in the snapshot intact."""
    metrics = Metrics()
    for value in (1.0, 2.0, 3.0):
        metrics.record_ms("decode", value)
    metrics.take_period()
    assert metrics.snapshot()["median_ms"]["decode"] == 2.0


@pytest.mark.parametrize("values,expected", [([7.0], 7.0), ([1.0, 2.0], 2.0), ([float(v) for v in range(1, 21)], 19.0),
                                             ([float(v) for v in range(1, 101)], 95.0)])
def test_percentile_is_nearest_rank(values, expected):
    """percentile(values, 0.95) returns the nearest-rank value for lists of 1, 2, 20 and 100 samples."""
    assert percentile(values, 0.95) == expected


def test_period_ms_rounds_and_skips_empty_names():
    """period_ms gives p50 and p95 per name and leaves out names that have no samples."""
    assert period_ms({"decode": [1.0, 2.0, 4.0], "track": []}) == {"decode": {"p50": 2.0, "p95": 4.0}}
