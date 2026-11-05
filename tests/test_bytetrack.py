# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests that the local ByteTrack matches the Ultralytics reference tracker, track for track, on synthetic
motion and on detections recorded from the sample clip."""
from pathlib import Path

import numpy as np
import pytest
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from pipeline.bytetrack import ByteTrack

WIDTH, HEIGHT = 1920, 1080


def reference():
    """Build an Ultralytics ByteTrack tracker with its default settings and track ids restarted from 1."""
    BaseTrack.reset_id()
    return BYTETracker(args=IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))


def reference_update(tracker, rows):
    """Feed one frame of detection rows to the reference tracker and return its tracks as an N x 8 array."""
    tracks = tracker.update(Boxes(rows.copy(), (HEIGHT, WIDTH)), None)
    return np.asarray(tracks, np.float32).reshape(-1, 8)


def assert_same(ours, theirs, label):
    """Assert two track arrays hold the same ids, boxes (within 0.01 pixel) and other fields, in any row order."""
    assert len(ours) == len(theirs), label
    order_a, order_b = np.argsort(ours[:, 4]), np.argsort(theirs[:, 4])
    a, b = ours[order_a], theirs[order_b]
    assert np.array_equal(a[:, 4], b[:, 4]), label
    assert np.abs(a[:, :4] - b[:, :4]).max(initial=0.0) < 0.01, label
    assert np.abs(a[:, 5:] - b[:, 5:]).max(initial=0.0) < 1e-4, label


def synthetic(seed, frames=300, objects=12):
    """Moving boxes with noise, missed detections, weak scores and objects entering and leaving."""
    random = np.random.default_rng(seed)
    start = random.uniform([0, 0], [WIDTH - 200, HEIGHT - 200], (objects, 2))
    velocity = random.uniform(-6, 6, (objects, 2))
    size = random.uniform(30, 180, (objects, 2))
    score = random.uniform(0.15, 0.95, objects)
    born = random.integers(0, frames // 2, objects)
    died = born + random.integers(40, frames, objects)
    for frame in range(frames):
        rows = []
        for index in range(objects):
            if not born[index] <= frame < died[index] or random.random() < 0.1:
                continue
            corner = start[index] + velocity[index] * frame + random.normal(0, 1.0, 2)
            confidence = float(np.clip(score[index] + random.normal(0, 0.05), 0.1, 0.99))
            rows.append([*corner, *(corner + size[index]), confidence, index % 3])
        yield np.array(rows, np.float32).reshape(-1, 6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_ultralytics_on_synthetic_motion(seed):
    """Proves both trackers give the same tracks on every frame of seeded synthetic motion."""
    ours, theirs = ByteTrack(), reference()
    for number, rows in enumerate(synthetic(seed)):
        assert_same(ours.update(rows), reference_update(theirs, rows), f"seed {seed} frame {number}")


def test_matches_ultralytics_on_clip_detections():
    """Proves both trackers give the same tracks on every frame of detections recorded from the sample clip."""
    data = np.load(Path(__file__).parent / "data" / "clip_rows.npz")
    edges = np.cumsum(data["counts"])[:-1]
    ours, theirs = ByteTrack(), reference()
    for number, rows in enumerate(np.split(data["rows"], edges)):
        assert_same(ours.update(rows), reference_update(theirs, rows), f"frame {number}")


def test_matches_ultralytics_when_a_removed_track_returns():
    """A track removed on one frame can still be found on the next, and is gone for good if it is lost again."""
    box = np.array([[100, 100, 300, 300, 0.9, 0]], np.float32)
    nothing = box[:0]
    ours, theirs = ByteTrack(), reference()
    ids = []
    for number, rows in enumerate([box] * 5 + [nothing] * 31 + [box] * 3 + [nothing] * 2 + [box] * 3):
        output = ours.update(rows)
        assert_same(output, reference_update(theirs, rows), f"frame {number}")
        ids.append(output[:, 4].tolist())
    assert ids[36] == [1]
    assert ids[-1] == [2]


def test_empty_frames_are_fine():
    """Proves a frame without detections gives an empty N x 8 result instead of an error."""
    tracker = ByteTrack()
    assert tracker.update(np.zeros((0, 6), np.float32)).shape == (0, 8)


def test_reset_restarts_ids():
    """Proves reset() starts track numbering over, so the same detection gets the same id again."""
    tracker = ByteTrack()
    rows = np.array([[10, 10, 100, 100, 0.9, 0]], np.float32)
    first = tracker.update(rows)
    tracker.reset()
    assert tracker.update(rows)[0, 4] == first[0, 4]


def test_instances_do_not_share_ids():
    """Proves each tracker counts its own ids, so one stream's tracks do not shift another's."""
    rows = np.array([[10, 10, 100, 100, 0.9, 0]], np.float32)
    assert ByteTrack().update(rows)[0, 4] == ByteTrack().update(rows)[0, 4]
