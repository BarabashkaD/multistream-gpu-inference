# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for how long the display stage waits for the next frame to choose, with a fake pool and without a GPU."""
import threading
import time

from pipeline.config import PipelineConfig
from pipeline.display import DisplayStage


class _Pool:
    """Stand-in pool that hands out the given choices in order, then None, and counts the calls."""

    def __init__(self, results):
        """Takes the choices to return, one per call."""
        self.results = list(results)
        self.calls = 0

    def take_for_display(self):
        """Returns the next choice, or None when they are used up."""
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return None


def _stage(pool, stop=None):
    """Builds a display stage for a 25 fps display over the fake pool; `stop` defaults to an unset event."""
    return DisplayStage(PipelineConfig(display_fps=25.0), pool, (64, 48), {}, None,
                        stop or threading.Event(), threading.Event())


def test_next_choice_returns_a_frame_that_finishes_during_the_wait():
    """Proves the stage keeps polling the pool and returns a frame that becomes ready before the wait ends."""
    choice = object()
    pool = _Pool([None, None, choice])
    assert _stage(pool)._next_choice() is choice
    assert pool.calls == 3


def test_next_choice_gives_up_after_half_a_tick():
    """Proves with nothing to show the stage returns None after about half a display tick (20 ms at 25 fps)."""
    started = time.monotonic()
    assert _stage(_Pool([]))._next_choice() is None
    elapsed = time.monotonic() - started
    assert 0.015 <= elapsed < 0.1


def test_next_choice_does_not_wait_when_stopping():
    """Proves that once the stop event is set the stage asks the pool once and returns without waiting."""
    stop = threading.Event()
    stop.set()
    pool = _Pool([])
    started = time.monotonic()
    assert _stage(pool, stop)._next_choice() is None
    assert time.monotonic() - started < 0.015
    assert pool.calls == 1
