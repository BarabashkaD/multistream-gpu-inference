# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The tracker stage: turns each frame's detections into tracks and hands the finished bundle back to the pool."""
from __future__ import annotations

import threading
import time

import numpy as np

from pipeline.bundle import BundlePool
from pipeline.bytetrack import ByteTrack
from pipeline.metrics import Metrics
from pipeline.stage import END, Stage, get_until


class TrackerCore:
    """A ByteTrack tracker that drops detections below a confidence and keeps tracks inside the frame."""

    def __init__(self, conf: float, source_size: tuple[int, int]):
        """Keep detections with at least `conf` confidence; source_size is the frame width and height."""
        self.conf = conf
        self.width, self.height = source_size
        self.tracker = ByteTrack()

    def update(self, rows: np.ndarray) -> np.ndarray:
        """Track one frame's detection rows and return the confirmed tracks with boxes clipped to the frame."""
        tracks = self.tracker.update(rows[rows[:, 4] >= self.conf])
        tracks = np.array(tracks, np.float32)
        tracks[:, [0, 2]] = tracks[:, [0, 2]].clip(0, self.width)
        tracks[:, [1, 3]] = tracks[:, [1, 3]].clip(0, self.height)
        return tracks

    def reset(self) -> None:
        """Forget all tracks, as when the clip starts over."""
        self.tracker.reset()


class TrackerStage(Stage):
    """Tracks bundles in the order they arrive from the server, resetting the tracker when the clip loops."""

    def __init__(self, core, pool: BundlePool, finished: threading.Event, metrics: Metrics,
                 stop: threading.Event, failures: list, delay_s: float = 0.0):
        """Wire the stage to its tracker core and pool; `finished` is set when the stream has been fully tracked."""
        super().__init__("tracker", stop, failures)
        self.core = core
        self.pool = pool
        self.finished = finished
        self.metrics = metrics
        self.delay_s = delay_s  # simulates a slow consumer, for the benchmark

    def work(self) -> None:
        """Track each queued bundle, store its tracks in the pool and release the stage's lease, until the end marker.

        Records tracking time and end-to-end latency, and sets `finished` at the end marker.
        """
        current_loop = None
        while True:
            bundle = get_until(self.pool.to_track, self.stop_event)
            if bundle is None:
                return
            if bundle is END:
                self.finished.set()
                return
            if current_loop is not None and bundle.loop != current_loop:
                self.core.reset()
            current_loop = bundle.loop
            started = time.perf_counter()
            tracks = self.core.update(bundle.rows)
            self.metrics.record_ms("track", (time.perf_counter() - started) * 1e3)
            self.pool.set_tracks(bundle, tracks)
            self.metrics.count("tracked")
            # At max pace frames run ahead of due_at, and behind schedule they start after it.
            self.metrics.record_ms("latency", (time.monotonic() - min(bundle.due_at, bundle.published_at)) * 1e3)
            self.pool.release(bundle)
            if self.delay_s:
                self.stop_event.wait(self.delay_s)
