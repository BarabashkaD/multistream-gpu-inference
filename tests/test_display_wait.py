# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for how long the display stage waits for the next frame choice before giving up."""
import threading
import time

from pipeline.config import InstanceConfig
from pipeline.display import DisplayStage


class _Pool:
    """Stand-in pool that hands out prepared display choices and counts how often it was asked."""

    def __init__(self, results):
        """Queue the choices to return, in order; None means nothing is ready yet."""
        self.results = list(results)
        self.calls = 0

    def take_for_display(self):
        """Return the next prepared choice, or None when none are left."""
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return None


def _stage(pool, stop=None):
    """Build a 25 fps display stage around the given pool and optional stop event."""
    return DisplayStage(InstanceConfig(address="unused", display_fps=25.0), pool, (64, 48), {}, None,
                        stop or threading.Event(), threading.Event())


def test_next_choice_returns_a_frame_that_finishes_during_the_wait():
    """Proves the stage keeps polling and returns a choice that becomes ready within the wait."""
    choice = object()
    pool = _Pool([None, None, choice])
    assert _stage(pool)._next_choice() is choice
    assert pool.calls == 3


def test_next_choice_gives_up_after_half_a_tick():
    """Proves the stage returns None after waiting about half a display tick when nothing becomes ready."""
    started = time.monotonic()
    assert _stage(_Pool([]))._next_choice() is None
    elapsed = time.monotonic() - started
    assert 0.015 <= elapsed < 0.1


def test_next_choice_does_not_wait_when_stopping():
    """Proves that once stop is set the stage asks the pool once and returns without waiting."""
    stop = threading.Event()
    stop.set()
    pool = _Pool([])
    started = time.monotonic()
    assert _stage(pool, stop)._next_choice() is None
    assert time.monotonic() - started < 0.015
    assert pool.calls == 1
