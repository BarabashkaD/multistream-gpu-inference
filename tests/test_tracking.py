# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of pipeline.tracking: the tracker core's filtering and clipping, and the stage's reset on a new loop."""
import threading

import numpy as np

from pipeline.bundle import Bundle, BundlePool, Part
from pipeline.metrics import Metrics
from pipeline.stage import END
from pipeline.tracking import TrackerCore, TrackerStage


def test_core_filters_by_confidence_and_clips():
    """Detections below the confidence limit never reach the tracker, and output boxes are clipped to the frame."""
    core = TrackerCore(0.5, (100, 100))
    core.tracker.update = lambda rows: (setattr(core, "seen", rows.copy()),
                                        np.array([[-5, -5, 200, 200, 1, 0.9, 0, 0]], np.float32))[1]
    tracks = core.update(np.array([[1, 2, 3, 4, 0.9, 0], [1, 2, 3, 4, 0.2, 0]], np.float32))
    assert len(core.seen) == 1
    assert tracks[0, :4].tolist() == [0, 0, 100, 100]


def test_no_tracks_gives_an_empty_array():
    """With no detections, update returns an empty array that still has eight columns."""
    assert TrackerCore(0.1, (100, 100)).update(np.zeros((0, 6), np.float32)).shape == (0, 8)


def test_a_new_loop_resets_the_tracker():
    """When the clip loops, the stage resets the tracker once, finishes at END and returns every bundle to the pool."""
    pool = BundlePool([Bundle(i) for i in range(2)], lambda pointer: None, display_enabled=False)
    stop, finished = threading.Event(), threading.Event()
    resets = []

    class Core:
        """A tracker core that returns no tracks and records each reset."""

        def update(self, rows):
            """Return an empty track array."""
            return np.zeros((0, 8), np.float32)

        def reset(self):
            """Record that the tracker was reset."""
            resets.append(True)

    for frame_id, loop in ((0, 0), (1, 1)):
        bundle = pool.acquire(stop)
        pool.publish_frame(bundle, 0x10 + frame_id, frame_id, loop, frame_id)
        pool.to_infer.get()
        bundle.rows = np.zeros((0, 6), np.float32)
        pool.mark(bundle, Part.DETECTIONS)
        pool.to_track.put(bundle)
    pool.to_track.put(END)
    stage = TrackerStage(Core(), pool, finished, Metrics(), stop, [])
    stage.start()
    stage.join(2.0)
    assert finished.is_set()
    assert resets == [True]
    assert pool.stats()["free"] == 2
