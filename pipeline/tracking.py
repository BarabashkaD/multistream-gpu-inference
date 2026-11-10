# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The tracker stage: turns each frame's detections into tracks and hands the finished bundle back to the pool."""
from __future__ import annotations

import threading
import time

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from pipeline.bundle import BundlePool
from pipeline.metrics import Metrics
from pipeline.nv12 import Letterbox
from pipeline.stage import END, Stage, get_until

_EMPTY = np.zeros((0, 8), np.float32)


class TrackerCore:
    """A ByteTrack tracker that drops detections below a confidence and keeps tracks inside the frame."""

    def __init__(self, letterbox: Letterbox, source_size: tuple[int, int], tracker_yaml: str, conf: float):
        """Keep detections with at least `conf` confidence; source_size is the frame width and height.

        The tracker settings come from tracker_yaml, resolved by Ultralytics. The letterbox maps detector boxes back
        to source pixels.
        """
        self.letterbox = letterbox
        self.width, self.height = source_size
        self.conf = conf
        self.tracker = BYTETracker(args=IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml))))

    def update(self, rows: np.ndarray) -> np.ndarray:
        """Track one frame's detection rows and return the confirmed tracks with boxes clipped to the frame.

        Rows are in detector coordinates; the result is in source pixels and is a shared empty array when no track
        is confirmed, so callers must not modify it.
        """
        kept = rows[rows[:, 4] >= self.conf].copy()
        kept[:, :4] = self.letterbox.to_source(kept[:, :4])
        boxes = Boxes(kept, (self.height, self.width))
        tracks = self.tracker.update(boxes, None)
        if len(tracks) == 0:
            return _EMPTY
        tracks = np.asarray(tracks, np.float32)
        tracks[:, [0, 2]] = tracks[:, [0, 2]].clip(0, self.width)
        tracks[:, [1, 3]] = tracks[:, [1, 3]].clip(0, self.height)
        return tracks

    def reset(self) -> None:
        """Forget all tracks, as when the clip starts over."""
        self.tracker.reset()


class TrackerStage(Stage):
    """Tracks bundles in the order they arrive from inference, resetting the tracker when the clip loops."""

    def __init__(self, core, pool: BundlePool, finished: threading.Event, metrics: Metrics,
                 stop: threading.Event, failures: list):
        """Wire the stage to its tracker core and pool; `finished` is set when the stream has been fully tracked."""
        super().__init__("tracker", stop, failures)
        self.core = core
        self.pool = pool
        self.finished = finished
        self.metrics = metrics

    def work(self) -> None:
        """Track each queued bundle, store its tracks in the pool and release the stage's lease, until the end marker.

        Resets the core when the loop number changes, records tracking time, and sets `finished` at the end marker.
        """
        current_loop = None
        while True:
            bundle = get_until(self.pool.to_track, self.stop_event)
            if bundle is None:
                return
            if bundle is END:
                self.finished.set()
                return
            try:
                if current_loop is not None and bundle.loop != current_loop:
                    self.core.reset()
                current_loop = bundle.loop
                started = time.perf_counter()
                tracks = self.core.update(bundle.detections[0].cpu().numpy())
                self.metrics.record_ms("track", (time.perf_counter() - started) * 1e3)
                self.pool.set_tracks(bundle, tracks)
                self.metrics.record_ms("published_to_tracked", (time.monotonic() - bundle.published_at) * 1e3)
                self.metrics.record_ms("due_to_tracked", (time.monotonic() - bundle.due_at) * 1e3)
                self.metrics.count("tracked")
            except BaseException:
                self.pool.abandon(bundle)
                raise
            else:
                self.pool.release(bundle)
