# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Statistics lines: one JSON object per line on standard output and, optionally, in a stats file.

A periodic line has "kind": "period"; the final summary has "kind": "final". The monitor reads these files.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import psutil

from pipeline.metrics import Metrics, period_ms

_CREATED = psutil.Process().create_time()  # wall-clock start of this process


def elapsed_s() -> float:
    """Seconds since this process started, to one decimal; used as the "time" field of every line."""
    return round(time.time() - _CREATED, 1)


def rounded(values: dict) -> dict:
    """A copy of a name-to-number dict with every value rounded to two decimals."""
    return {key: round(value, 2) for key, value in values.items()}


class StatsLog:
    """Thread-safe: the reporter thread and the main thread both write.

    Every line goes to standard output and, when a path is given, to that file, flushed at once so the monitor
    sees it. Only one final line is ever written."""

    def __init__(self, path: Path | None = None):
        """Opens (and truncates) the stats file when path is given; raises OSError if it cannot be opened."""
        self._lock = threading.Lock()
        self._finished = False
        self._file = open(path, "w", encoding="utf-8") if path is not None else None  # truncated: this run only

    def write(self, line: dict) -> None:
        """Dropped silently once the final line has been written."""
        self._emit(line, final=False)

    def write_final(self, line: dict) -> None:
        """Write the last line this log will ever carry; later write() calls are dropped."""
        self._emit(line, final=True)

    def _emit(self, line: dict, final: bool) -> None:
        """Writes one JSON line to standard output and the file under the lock, unless the log is already final."""
        text = json.dumps(line)
        with self._lock:
            if self._finished:
                return
            self._finished = final
            print(text, flush=True)
            if self._file is not None:
                self._file.write(text + "\n")
                self._file.flush()

    def close(self) -> None:
        """Closes the stats file; safe to call twice. Lines still reach standard output but not the file."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def instance_line(metrics: Metrics, pool_stats: dict, stream: int, period_s: float, memory_mb: float) -> dict:
    """One periodic statistics line for a client instance.

    Takes the metrics' cumulative snapshot and the timings since the previous period; the latter are consumed, so
    call it once per period. Fields are JSON-ready numbers and dicts."""
    snapshot = metrics.snapshot()
    return {"kind": "period", "role": "instance", "stream": stream, "time": elapsed_s(), "period_s": period_s,
            "current": rounded(metrics.current_fps(window_s=period_s)), "fps": rounded(snapshot["fps"]),
            "counts": snapshot["counts"], "period_ms": period_ms(metrics.take_period()), "pool": pool_stats,
            "memory_mb": round(memory_mb)}


def server_line(metrics: Metrics, batches: dict, queued: int, period_s: float, memory_mb: float,
                available: dict) -> dict:
    """One periodic statistics line for the server: queue depth, batch counts and per-period timings.

    Like instance_line it consumes the timings since the previous period, so call it once per period."""
    return {"kind": "period", "role": "server", "time": elapsed_s(), "period_s": period_s,
            "current": rounded(metrics.current_fps(window_s=period_s)), "queued": queued, "batches": batches,
            "available": available, "period_ms": period_ms(metrics.take_period()), "memory_mb": round(memory_mb)}
