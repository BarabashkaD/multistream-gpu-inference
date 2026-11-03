# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Frame bundles and the pool that lends them out, tracks their progress and reclaims them."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntFlag
import queue
import threading
import time
from typing import Any, Callable


class Part(IntFlag):
    """The pieces of work a bundle needs before it is complete: decoded frame, prepared input, detections and tracks."""

    NONE = 0
    FRAME = 1
    INPUT = 2
    DETECTIONS = 4
    TRACKS = 8


COMPLETE = Part.FRAME | Part.INPUT | Part.DETECTIONS | Part.TRACKS


@dataclass
class Bundle:
    """One frame in flight: its decoded memory pointer, tensors, results and progress, reused across frames by the pool.

    The generation increases each time the pool hands the bundle out. leases counts the holders that still need it;
    the pool reclaims it only at zero. The tensors are allocated once and kept for the whole run.
    """

    index: int
    input_tensor: Any = None
    detections: Any = None
    generation: int = 0
    frame_id: int = -1
    loop: int = 0
    pts: int = 0
    published_at: float = 0.0
    due_at: float = 0.0
    nv12_pointer: int | None = None
    nv12: Any = None
    tracks: Any = None
    parts: Part = Part.NONE
    leases: int = 0
    displayed: bool = False
    abandoned: bool = False


@dataclass
class DisplayChoice:
    """The bundle picked for display, whether it is complete, and the tracks to draw with the id of their frame."""

    bundle: Bundle
    complete: bool
    tracks: Any
    tracks_frame_id: int


class BundlePool:
    """Fixed bundle set. Leases protect bundles in use; reclaim runs on every update.

    release_frame runs under the pool lock on whichever stage thread triggered reclaim,
    so it must not block and must not touch the decoder. The application passes a queue's put,
    which the decoder stage drains.
    """

    def __init__(self, bundles: list[Bundle], release_frame: Callable[[int], None], *,
                 display_enabled: bool, max_age_s: float = 2.0, display_grace_s: float = 0.2,
                 clock: Callable[[], float] = time.monotonic):
        """Take ownership of the bundles. Bundles older than max_age_s are reclaimed even if unfinished.

        With display enabled a complete bundle also waits to be shown or passed over; display_grace_s is how long
        the display holds back for a complete frame before it falls back to a newer incomplete one. The clock is
        injectable so tests can control time.
        """
        self._free = deque(bundles)
        self._active: dict[int, Bundle] = {}
        self._condition = threading.Condition()
        self._release_frame = release_frame
        self._display_enabled = display_enabled
        self._max_age_s = max_age_s
        self._display_grace_s = display_grace_s
        self._clock = clock
        self._size = len(bundles)
        self._last_shown = -1
        self._latest_tracks: Any = None
        self._latest_tracks_frame = -1
        self._last_complete_at = float("-inf")
        self._expired = 0
        self._abandoned = 0
        self._superseded = 0
        self.to_infer: queue.Queue = queue.Queue()
        self.to_detect: queue.Queue = queue.Queue()
        self.to_track: queue.Queue = queue.Queue()

    def acquire(self, stop: threading.Event) -> Bundle | None:
        """Take a free bundle with one lease, reclaiming finished or expired ones while waiting.

        Returns None if stop is set before a bundle becomes free.
        """
        with self._condition:
            while not self._free:
                if stop.is_set():
                    return None
                self._reclaim()
                if not self._free:
                    self._condition.wait(0.05)
            bundle = self._free.popleft()
            bundle.generation += 1
            bundle.parts = Part.NONE
            bundle.leases = 1
            bundle.displayed = False
            bundle.abandoned = False
            return bundle

    def publish_frame(self, bundle: Bundle, pointer: int, nv12: Any, frame_id: int, loop: int, pts: int,
                      due_at: float | None = None) -> None:
        """Record the decoded frame in the bundle, make it visible to display and queue it for preparation.

        due_at is when the frame should be shown and defaults to now. The caller's lease passes to the queue.
        """
        with self._condition:
            bundle.nv12_pointer = pointer
            bundle.nv12 = nv12
            bundle.frame_id = frame_id
            bundle.loop = loop
            bundle.pts = pts
            bundle.published_at = self._clock()
            bundle.due_at = bundle.published_at if due_at is None else due_at
            bundle.parts = Part.FRAME
            self._active[frame_id] = bundle
        self.to_infer.put(bundle)  # the decoder's lease moves to the queue

    def mark(self, bundle: Bundle, part: Part) -> None:
        """Record that a part of the bundle's work is done and reclaim anything that has become finished."""
        with self._condition:
            bundle.parts |= part
            self._reclaim()

    def set_tracks(self, bundle: Bundle, tracks: Any) -> None:
        """Attach tracks to the bundle, mark it tracked, and remember them as the latest tracks if the frame is newest."""
        with self._condition:
            bundle.tracks = tracks
            bundle.parts |= Part.TRACKS
            self._last_complete_at = self._clock()
            if bundle.frame_id > self._latest_tracks_frame:
                self._latest_tracks = tracks
                self._latest_tracks_frame = bundle.frame_id
            self._reclaim()

    def release(self, bundle: Bundle) -> None:
        """Give up one lease on the bundle; it is reclaimed once no lease remains and its work is finished."""
        with self._condition:
            bundle.leases -= 1
            self._reclaim()

    def abandon(self, bundle: Bundle) -> None:
        """Give up one lease and mark the bundle abandoned, so it is reclaimed without its remaining work."""
        with self._condition:
            bundle.abandoned = True
            bundle.leases -= 1
            self._abandoned += 1
            self._reclaim()

    def take_for_display(self) -> DisplayChoice | None:
        """Pick the newest complete bundle not yet shown, or the newest incomplete one, and lease it for display.

        Returns None when nothing newer than the last shown frame exists, or when only incomplete frames exist and a
        complete one was finished within the grace period. Older unshown frames are counted as superseded. An
        incomplete frame is shown with the latest known tracks instead of its own. The caller must release the lease.
        """
        with self._condition:
            newer = [b for b in self._active.values() if b.frame_id > self._last_shown and not b.abandoned]
            if not newer:
                return None
            complete = [b for b in newer if b.parts == COMPLETE]
            if not complete and self._clock() - self._last_complete_at < self._display_grace_s:
                return None  # a matching frame is due soon, so do not jump past the inference queue
            bundle = max(complete or newer, key=lambda b: b.frame_id)
            is_complete = bundle.parts == COMPLETE
            self._superseded += sum(b.frame_id < bundle.frame_id for b in newer)
            self._last_shown = bundle.frame_id
            bundle.displayed = True
            bundle.leases += 1
            tracks = bundle.tracks if is_complete else self._latest_tracks
            tracks_frame = bundle.frame_id if is_complete else self._latest_tracks_frame
            self._reclaim()
            return DisplayChoice(bundle, is_complete, tracks, tracks_frame)

    def finish(self) -> None:
        """Stop waiting for the display, so complete bundles are reclaimed without being shown."""
        with self._condition:
            self._display_enabled = False
            self._reclaim()

    def stats(self) -> dict:
        """Return a snapshot of the pool: size, free, active and leased counts and the expired, abandoned and
        superseded totals."""
        with self._condition:
            return {
                "size": self._size,
                "free": len(self._free),
                "active": len(self._active),
                "expired": self._expired,
                "abandoned": self._abandoned,
                "superseded": self._superseded,
                "leased": sum(b.leases > 0 for b in self._active.values()),
            }

    def _finished(self, bundle: Bundle) -> bool:
        """Whether nothing more is needed from the bundle: abandoned, or complete and no longer awaiting display."""
        if bundle.abandoned:
            return True
        if bundle.parts != COMPLETE:
            return False
        return not self._display_enabled or bundle.displayed or bundle.frame_id < self._last_shown

    def _reclaim(self) -> None:
        """Return every unleased bundle that is finished or older than the maximum age to the free list.

        Expired unfinished bundles are counted. The frame's memory pointer is handed to release_frame, and waiting
        acquirers are woken. The caller must hold the lock.
        """
        now = self._clock()
        for frame_id, bundle in list(self._active.items()):
            if bundle.leases > 0:
                continue
            expired = now - bundle.published_at > self._max_age_s
            if not self._finished(bundle) and not expired:
                continue
            if expired and not self._finished(bundle):
                self._expired += 1
            del self._active[frame_id]
            pointer = bundle.nv12_pointer
            bundle.nv12 = None
            bundle.nv12_pointer = None
            bundle.tracks = None
            self._free.append(bundle)
            if pointer is not None:
                self._release_frame(pointer)
            self._condition.notify_all()
