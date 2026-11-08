# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
# Portions derived from Ultralytics (https://github.com/ultralytics/ultralytics), AGPL-3.0.
"""ByteTrack in NumPy and lap.

Importing Ultralytics costs an instance 885 MB, because it loads PyTorch, so its tracker cannot run here. This is a
port of the Ultralytics tracker to NumPy, held to Ultralytics' results frame by frame in tests/test_bytetrack.py.
Parity is why the float widths below look fussy: a threshold or a tie can resolve differently in another width.
"""
from __future__ import annotations

from collections import deque
from itertools import count
from typing import Iterable, NamedTuple

import lap
import numpy as np

_TRACKED, _LOST, _REMOVED = 1, 2, 3
_LOW_SCORE_MATCH = 0.5
_UNCONFIRMED_MATCH = 0.7
_DUPLICATE_COST = 0.15
_REMOVED_MEMORY = 1000
_IOU_EPS = 1e-7


class _KalmanXYAH:
    """Constant-velocity filter over centre x, centre y, aspect ratio, height and their velocities."""

    def __init__(self):
        """Build the constant-velocity motion matrix and the noise weights, which scale with the box height."""
        self._motion = np.eye(8)
        self._motion[range(4), range(4, 8)] = 1.0
        self._position_weight = 1.0 / 20
        self._velocity_weight = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the starting mean (measurement with zero velocity) and covariance for a new track."""
        measurement = np.asarray(measurement, np.float64)
        height = measurement[3]
        position, velocity = 2 * self._position_weight * height, 10 * self._velocity_weight * height
        spread = [position, position, 1e-2, position, velocity, velocity, 1e-5, velocity]
        return np.r_[measurement, np.zeros(4)], np.diag(np.square(spread))

    def predict_many(self, means: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance every track one frame at once, returning the predicted means and covariances."""
        height = means[:, 3]
        position, velocity = self._position_weight * height, self._velocity_weight * height
        aspect, aspect_rate = np.full_like(height, 1e-2), np.full_like(height, 1e-5)
        spread = (position, position, aspect, position, velocity, velocity, aspect_rate, velocity)
        noise = np.zeros((len(means), 8, 8))
        noise[:, range(8), range(8)] = np.square(np.stack(spread, axis=1))
        left = np.dot(self._motion, covariances).transpose((1, 0, 2))
        return np.dot(means, self._motion.T), np.dot(left, self._motion.T) + noise

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Correct one track's predicted state with a matched measurement and return the new mean and covariance."""
        size = self._position_weight * mean[3]
        projected = covariance[:4, :4] + np.diag(np.square([size, size, 1e-1, size]))
        gain = np.linalg.solve(projected, covariance[:, :4].T).T
        new_mean = mean + np.dot(measurement - mean[:4], gain.T)
        return new_mean, covariance - np.linalg.multi_dot((gain, projected, gain.T))


class _Detections(NamedTuple):
    """The detections of one frame that passed a score band, as parallel columns."""

    boxes: np.ndarray  # x1, y1, x2, y2
    measurements: np.ndarray  # centre x, centre y, aspect ratio, height
    scores: np.ndarray
    classes: np.ndarray
    indices: np.ndarray  # row number in the frame's input

    def take(self, chosen: list[int]) -> _Detections:
        """Return the detections at the chosen positions, keeping every column aligned."""
        return _Detections(*(column[chosen] for column in self))


def _detections(rows: np.ndarray, keep: np.ndarray) -> _Detections:
    """Build the boxes, centre/aspect/height measurements and columns for the rows selected by the boolean mask."""
    picked = rows[keep]
    centre = (picked[:, :2] + picked[:, 2:4]) / 2
    size = picked[:, 2:4] - picked[:, :2]
    # Ultralytics holds the row in float64 for this one subtraction, so its corner rounds differently from float32.
    corner = (centre.astype(np.float64) - size.astype(np.float64) / 2).astype(np.float32)
    measurements = np.concatenate((corner + size / 2, size), axis=1)
    measurements[:, 2] /= measurements[:, 3]
    boxes = np.concatenate((corner, size + corner), axis=1)
    return _Detections(boxes, measurements, picked[:, 4], picked[:, 5], np.flatnonzero(keep))


class _Track:
    """One tracked object: its Kalman state, lifecycle state and the detection it last matched."""

    __slots__ = ("track_id", "mean", "covariance", "state", "confirmed", "frame", "start", "score", "cls", "index")

    def __init__(self, track_id: int, mean: np.ndarray, covariance: np.ndarray, confirmed: bool, frame: int,
                 detections: _Detections, row: int):
        """Start a tracked track at `frame` from the given detection row."""
        self.track_id, self.mean, self.covariance = track_id, mean, covariance
        self.state, self.confirmed = _TRACKED, confirmed
        self.frame = self.start = frame
        self._take_row(detections, row)

    def absorb(self, kalman: _KalmanXYAH, detections: _Detections, row: int, frame: int) -> None:
        """Update the state with a matched detection and mark the track tracked and confirmed at `frame`."""
        self.mean, self.covariance = kalman.update(self.mean, self.covariance, detections.measurements[row])
        self.state, self.confirmed, self.frame = _TRACKED, True, frame
        self._take_row(detections, row)

    def _take_row(self, detections: _Detections, row: int) -> None:
        """Copy the detection's score, class and input row number onto the track."""
        self.score, self.cls, self.index = detections.scores[row], detections.classes[row], detections.indices[row]


def _track_boxes(tracks: list[_Track]) -> np.ndarray:
    """x1, y1, x2, y2 from each track's state, in float64."""
    boxes = np.array([track.mean[:4] for track in tracks], np.float64).reshape(-1, 4)
    boxes[:, 2] *= boxes[:, 3]
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _iou_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - IoU in float32, which is the width Ultralytics compares with its thresholds."""
    a_x1, a_y1, a_x2, a_y2 = a.astype(np.float32).T
    b_x1, b_y1, b_x2, b_y2 = b.astype(np.float32).T
    inter = (np.minimum(a_x2[:, None], b_x2) - np.maximum(a_x1[:, None], b_x1)).clip(0) * (
        np.minimum(a_y2[:, None], b_y2) - np.maximum(a_y1[:, None], b_y1)).clip(0)
    area_a, area_b = (a_x2 - a_x1) * (a_y2 - a_y1), (b_x2 - b_x1) * (b_y2 - b_y1)
    return 1 - inter / (area_b + area_a[:, None] - inter + _IOU_EPS)


def _fuse(cost: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Weight an IoU cost matrix by each detection's score, so low-confidence detections match less readily."""
    if cost.size == 0:
        return cost
    return 1 - (1 - cost) * scores


def _assign(cost: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return matched (row, column) pairs, unmatched rows and unmatched columns, all ascending."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    _, columns, rows = lap.lapjv(cost, extend_cost=True, cost_limit=threshold)
    matched = np.flatnonzero(columns >= 0)
    pairs = list(zip(matched.tolist(), columns[matched].tolist()))
    return pairs, np.flatnonzero(columns < 0).tolist(), np.flatnonzero(rows < 0).tolist()


def _join(first: list[_Track], second: list[_Track]) -> list[_Track]:
    """Concatenate two track lists, skipping tracks of the second whose id is already in the first."""
    seen = {track.track_id for track in first}
    return first + [track for track in second if track.track_id not in seen]


def _without(tracks: list[_Track], ids: Iterable[int]) -> list[_Track]:
    """Return the tracks whose id is not in ids."""
    ids = set(ids)
    return [track for track in tracks if track.track_id not in ids]


def _drop_duplicates(tracked: list[_Track], lost: list[_Track]) -> tuple[list[_Track], list[_Track]]:
    """Of a tracked and a lost track on the same box, keep the older one. A tie drops the tracked one."""
    drop_tracked, drop_lost = set(), set()
    for a, b in zip(*np.where(_iou_cost(_track_boxes(tracked), _track_boxes(lost)) < _DUPLICATE_COST)):
        if tracked[a].frame - tracked[a].start > lost[b].frame - lost[b].start:
            drop_lost.add(b)
        else:
            drop_tracked.add(a)
    return ([track for index, track in enumerate(tracked) if index not in drop_tracked],
            [track for index, track in enumerate(lost) if index not in drop_lost])


class ByteTrack:
    """update() takes rows of x1, y1, x2, y2, confidence, class and returns x1, y1, x2, y2, track_id, score, class,
    input_index for each confirmed track."""

    def __init__(self, high: float = 0.25, low: float = 0.1, new_track: float = 0.25, match: float = 0.8,
                 buffer: int = 30, fuse_score: bool = True):
        """Set the thresholds: high and low score bands, the minimum score to start a track, the match cost limit,
        how many frames a lost track is kept, and whether scores are fused into the matching cost."""
        self.high, self.low, self.new_track, self.match = high, low, new_track, match
        self.buffer, self.fuse_score = buffer, fuse_score
        self._kalman = _KalmanXYAH()
        self.reset()

    def reset(self) -> None:
        """Forget every track and restart frame counting and track ids."""
        self._tracked: list[_Track] = []
        self._lost: list[_Track] = []
        self._removed_ids: deque[int] = deque(maxlen=_REMOVED_MEMORY)
        self._frame = 0
        self._ids = count(1)

    def update(self, rows: np.ndarray) -> np.ndarray:
        """Advance the tracker one frame and return its confirmed tracks.

        Matches strong detections to tracked and lost tracks, then weak ones to the tracked leftovers, then
        unconfirmed tracks, starts new tracks from unmatched strong detections and retires tracks lost too long.
        """
        # float32 throughout, because scores > 0.1 can flip in float64.
        rows = np.asarray(rows, np.float32).reshape(-1, 6)
        self._frame += 1
        activated, refound, lost, removed = [], [], [], []
        strong, weak = self._split(rows)

        unconfirmed = [track for track in self._tracked if not track.confirmed]
        pool = _join([track for track in self._tracked if track.confirmed], self._lost)
        self._predict(pool)  # unconfirmed tracks stay where they were created, as they do in Ultralytics

        pairs, left_pool, left_strong = _assign(self._cost(pool, strong), self.match)
        self._apply(pairs, pool, strong, activated, refound)

        # The pool also holds lost tracks and any removed last frame. They can match above, but only tracked ones go on.
        remaining = [pool[row] for row in left_pool if pool[row].state == _TRACKED]
        left_remaining = range(len(remaining))
        if remaining and len(weak.scores):
            cost = _iou_cost(_track_boxes(remaining), weak.boxes)
            pairs, left_remaining, _ = _assign(cost, _LOW_SCORE_MATCH)
            self._apply(pairs, remaining, weak, activated, refound)
        for row in left_remaining:
            remaining[row].state = _LOST
            lost.append(remaining[row])

        strong = strong.take(left_strong)
        left_strong = range(len(strong.scores))
        if unconfirmed:
            pairs, left_unconfirmed, left_strong = _assign(self._cost(unconfirmed, strong), _UNCONFIRMED_MATCH)
            self._apply(pairs, unconfirmed, strong, activated, refound)
            for row in left_unconfirmed:
                unconfirmed[row].state = _REMOVED
                removed.append(unconfirmed[row])

        for row in left_strong:
            if strong.scores[row] >= self.new_track:
                activated.append(self._start(strong, row))
        for track in self._lost:
            if self._frame - track.frame > self.buffer:
                track.state = _REMOVED
                removed.append(track)

        self._merge(activated, refound, lost, removed)
        return self._output()

    def _split(self, rows: np.ndarray) -> tuple[_Detections, _Detections]:
        """Split the frame's rows into strong and weak detections by score, dropping boxes with no area."""
        scores = rows[:, 4]
        valid = (rows[:, 2] - rows[:, 0] > 0) & (rows[:, 3] - rows[:, 1] > 0)  # zero height would divide by zero
        strong = valid & (scores >= self.high)
        weak = valid & (scores > self.low) & (scores < self.high)
        return _detections(rows, strong), _detections(rows, weak)

    def _predict(self, tracks: list[_Track]) -> None:
        """Predict each track's next state in place; tracks that are not currently tracked lose their height velocity."""
        if not tracks:
            return
        means = np.array([track.mean for track in tracks])
        covariances = np.array([track.covariance for track in tracks])
        for row, track in enumerate(tracks):
            if track.state != _TRACKED:
                means[row, 7] = 0
        means, covariances = self._kalman.predict_many(means, covariances)
        for track, mean, covariance in zip(tracks, means, covariances):
            track.mean, track.covariance = mean, covariance

    def _cost(self, tracks: list[_Track], detections: _Detections) -> np.ndarray:
        """The matching cost between tracks and detections: IoU distance, fused with scores when enabled."""
        cost = _iou_cost(_track_boxes(tracks), detections.boxes)
        return _fuse(cost, detections.scores) if self.fuse_score else cost

    def _apply(self, pairs: list[tuple[int, int]], tracks: list[_Track], detections: _Detections,
               activated: list[_Track], refound: list[_Track]) -> None:
        """Absorb each matched detection into its track, filing the track as activated or, if it was lost, refound."""
        for row, column in pairs:
            track = tracks[row]
            (activated if track.state == _TRACKED else refound).append(track)
            track.absorb(self._kalman, detections, column, self._frame)

    def _start(self, detections: _Detections, row: int) -> _Track:
        """Create a track with the next id from a detection; only tracks born on the first frame start confirmed."""
        mean, covariance = self._kalman.initiate(detections.measurements[row])
        return _Track(next(self._ids), mean, covariance, self._frame == 1, self._frame, detections, row)

    def _merge(self, activated: list[_Track], refound: list[_Track], lost: list[_Track],
               removed: list[_Track]) -> None:
        """Fold this frame's activated, refound, lost and removed tracks into the tracked and lost lists."""
        tracked = [track for track in self._tracked if track.state == _TRACKED]
        tracked = _join(_join(tracked, activated), refound)
        still_lost = _without(self._lost, (track.track_id for track in tracked)) + lost
        # Only earlier frames' removals apply here, so a lost track removed this frame stays lost for one more frame.
        still_lost = _without(still_lost, self._removed_ids)
        self._tracked, self._lost = _drop_duplicates(tracked, still_lost)
        self._removed_ids.extend(track.track_id for track in removed)

    def _output(self) -> np.ndarray:
        """Return the confirmed tracked tracks as rows of box, id, score, class and input row (empty if none)."""
        tracks = [track for track in self._tracked if track.confirmed]
        output = np.zeros((len(tracks), 8), np.float32)
        if tracks:
            output[:, :4] = _track_boxes(tracks)
            output[:, 4:] = [(track.track_id, track.score, track.cls, track.index) for track in tracks]
        return output
