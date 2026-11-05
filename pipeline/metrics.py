# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Thread-safe event counters and timing samples, with rate and percentile summaries."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import math
import statistics
import threading
import time
from typing import Callable

_TIMING_SAMPLES = 1024
_RECENT_EVENTS = 4096


class Metrics:
    """Counters and millisecond timings shared by the stage threads; every method is safe to call from any thread.

    Timing samples and recent events are bounded, so memory stays constant however long the run is.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        """Start the elapsed-time clock now. The clock is injectable so tests can control time."""
        self._lock = threading.Lock()
        self._counts: Counter = Counter()
        self._first: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._timings: dict[str, deque] = defaultdict(lambda: deque(maxlen=_TIMING_SAMPLES))
        self._recorded: Counter = Counter()  # samples ever recorded, per timing name
        self._taken: dict[str, int] = {}  # _recorded at the previous take_period()
        self._recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=_RECENT_EVENTS))  # (time, amount)
        self._clock = clock
        self._start = clock()

    def count(self, name: str, amount: int = 1) -> None:
        """Add `amount` to the named counter and remember when, for the rate calculations."""
        now = self._clock()
        with self._lock:
            self._counts[name] += amount
            self._recent[name].append((now, amount))
            self._first.setdefault(name, now)
            self._last[name] = now

    def record_ms(self, name: str, value: float) -> None:
        """Record one timing sample in milliseconds; only the latest samples are kept."""
        with self._lock:
            self._timings[name].append(value)
            self._recorded[name] += 1

    def take_period(self) -> dict[str, list[float]]:
        """Per timing name, the samples recorded since the previous call: at most the last _TIMING_SAMPLES.
        Called by one reporter thread only."""
        with self._lock:
            result = {}
            for name, values in self._timings.items():
                new = self._recorded[name] - self._taken.get(name, 0)
                self._taken[name] = self._recorded[name]
                if new > 0:
                    result[name] = list(values)[-min(new, len(values)):]
            return result

    def current_fps(self, window_s: float = 1.0) -> dict[str, float]:
        """Events per second over the last window_s seconds, 0.0 for a counter with no recent events."""
        with self._lock:
            cutoff = self._clock() - window_s
            return {name: sum(amount for when, amount in events if when > cutoff) / window_s
                    for name, events in self._recent.items()}

    def snapshot(self) -> dict:
        """Summarise the run so far: elapsed seconds, counts, average and steady rates, and median and 95th
        percentile timings per name."""
        with self._lock:
            elapsed = max(self._clock() - self._start, 1e-6)
            counts = dict(self._counts)
            steady = {name: (count - 1) / (self._last[name] - self._first[name])
                      for name, count in counts.items()
                      if count >= 2 and self._last[name] > self._first[name]}
            timings = {name: list(values) for name, values in self._timings.items()}
        return {
            "elapsed_s": elapsed,
            "counts": counts,
            "fps": {name: value / elapsed for name, value in counts.items()},
            "steady_fps": steady,  # first to last event, so start-up and shutdown are excluded
            "median_ms": {name: statistics.median(values) for name, values in timings.items() if values},
            "p95_ms": {name: percentile(values, 0.95) for name, values in timings.items() if values},
        }


def percentile(values: list[float], fraction: float) -> float:
    """Nearest rank: the smallest value with at least `fraction` of the values at or below it."""
    ordered = sorted(values)
    return ordered[max(math.ceil(len(ordered) * fraction) - 1, 0)]


def period_ms(samples: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Reduce each name's samples to rounded p50 and p95 values, skipping names with no samples."""
    return {name: {"p50": round(statistics.median(values), 2), "p95": round(percentile(values, 0.95), 2)}
            for name, values in samples.items() if values}
