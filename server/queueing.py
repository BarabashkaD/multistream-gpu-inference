# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""A per-stream frame queue that hands the batch thread frames fairly across streams."""
from __future__ import annotations

from collections import deque
import threading

from pipeline.protocol import Frame


class FairQueue:
    """One FIFO per stream. Batches are filled round-robin, from a start stream that rotates."""

    def __init__(self):
        """Creates an empty queue with no streams."""
        self._streams: dict[int, deque] = {}
        self._condition = threading.Condition()
        self._waiting = 0
        self._turn = 0
        self.last_available = 0  # frames waiting when the last non-empty take chose its picks

    def __len__(self) -> int:
        """The number of frames waiting across all streams."""
        with self._condition:
            return self._waiting

    def add(self, stream: int, frame: Frame) -> None:
        """Appends a frame to its stream's FIFO, creating the FIFO on first use, and wakes a waiting take."""
        with self._condition:
            self._streams.setdefault(stream, deque()).append(frame)
            self._waiting += 1
            self._condition.notify_all()

    def remove_stream(self, stream: int) -> None:
        """Drops a stream and every frame still queued for it; an unknown stream is ignored."""
        with self._condition:
            self._waiting -= len(self._streams.pop(stream, ()))

    def _streams_with_frames(self) -> int:
        """Counts the streams that have at least one frame queued; the caller holds the condition."""
        return sum(1 for frames in self._streams.values() if frames)

    def take(self, sizes: tuple[int, ...], gather_s: float, timeout_s: float, streams: int) -> list[tuple[int, Frame]]:
        """Wait up to timeout_s for one frame, then up to gather_s until a full batch is waiting or each of `streams`
        streams has a frame queued. Takes every waiting frame, at most the largest size. May return an empty list."""
        with self._condition:
            if not self._condition.wait_for(lambda: self._waiting > 0, timeout_s):
                return []
            largest = max(sizes)
            self._condition.wait_for(
                lambda: self._waiting >= largest or self._streams_with_frames() >= streams, gather_s)
            if self._waiting == 0:  # a removed stream can empty the queue while it gathers
                return []
            self.last_available = self._waiting
            size = min(self._waiting, largest)
            order = sorted(self._streams)
            start = self._turn % len(order)
            self._turn += 1
            order = order[start:] + order[:start]
            picks: list[tuple[int, Frame]] = []
            while len(picks) < size:
                for stream in order:
                    if self._streams[stream] and len(picks) < size:
                        picks.append((stream, self._streams[stream].popleft()))
            self._waiting -= size
            return picks
