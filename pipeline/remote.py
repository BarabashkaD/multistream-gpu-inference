# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The instance's side of the inference server: frames out, results back."""
from __future__ import annotations

import threading
import time

from pipeline.bundle import BundlePool, Part
from pipeline.config import InstanceConfig
from pipeline.gpu import export_handle, use_gpu
from pipeline.metrics import Metrics
from pipeline.protocol import Frame, Reject, Result, rows_from_bytes
from pipeline.stage import END, Stage, get_until, put_until

_POLL_S = 0.05


class InFlight:
    """Bundles the server holds. Each keeps its lease here until its result arrives."""

    def __init__(self):
        """Start empty and open; closed becomes True once no more frames will be submitted."""
        self._lock = threading.Lock()
        self._bundles: dict[int, tuple] = {}
        self.closed = False

    def __len__(self) -> int:
        """The number of bundles still waiting for a result."""
        with self._lock:
            return len(self._bundles)

    def add(self, bundle, now: float) -> None:
        """Record that the bundle was sent at time `now`, keyed by its frame id."""
        with self._lock:
            self._bundles[bundle.frame_id] = (bundle, now)

    def pop(self, frame_id: int):
        """Remove and return (bundle, sent time) for a frame id, or (None, 0.0) if it is not in flight."""
        with self._lock:
            return self._bundles.pop(frame_id, (None, 0.0))

    def oldest_age(self, now: float) -> float:
        """Seconds since the longest-waiting bundle was sent, or 0.0 when none is in flight."""
        with self._lock:
            return max((now - sent for _, sent in self._bundles.values()), default=0.0)

    def close(self) -> None:
        """Mark that no more frames will be submitted."""
        self.closed = True


class SubmitStage(Stage):
    """Sends decoded frames to the server as pointers, with an IPC handle the first time each pointer is used.

    Frames already later than the allowed lag are dropped here rather than sent, under the realtime oldest policy.
    """

    def __init__(self, conn, pool: BundlePool, in_flight: InFlight, config: InstanceConfig, metrics: Metrics,
                 stop: threading.Event, failures: list):
        """Wire the stage to the server connection, the bundle pool and the in-flight record."""
        super().__init__("submit", stop, failures)
        self.conn = conn
        self.pool = pool
        self.in_flight = in_flight
        self.config = config
        self.metrics = metrics
        self.exported: set[int] = set()

    def work(self) -> None:
        """Send each queued bundle to the server until the end marker, then close the in-flight record.

        The bundle is recorded as in flight before the send, so a fast reply always finds it.
        """
        use_gpu(self.config.gpu_id)
        while True:
            bundle = get_until(self.pool.to_infer, self.stop_event)
            if bundle is None:
                return
            if bundle is END:
                self.in_flight.close()
                return
            self.metrics.record_ms("source_lag", (time.monotonic() - bundle.due_at) * 1e3)
            if self._too_old(bundle):
                self.pool.abandon(bundle)
                self.metrics.count("dropped")
                continue
            pointer = bundle.nv12_pointer
            handle = None
            if pointer not in self.exported:
                handle = export_handle(pointer)
                self.exported.add(pointer)
            self.in_flight.add(bundle, time.monotonic())  # before the send, so a fast reply finds it
            self.conn.send(Frame(bundle.frame_id, pointer, handle))
            self.metrics.count("submitted")

    def _too_old(self, bundle) -> bool:
        """Whether the bundle is later than the maximum lag; only the realtime oldest-drop policy ever says yes."""
        if self.config.drop_policy != "oldest" or self.config.pace != "realtime":
            return False
        return time.monotonic() - bundle.due_at > self.config.max_lag_s


class CollectStage(Stage):
    """Receives results from the server and passes each bundle on to tracking."""

    def __init__(self, conn, pool: BundlePool, in_flight: InFlight, config: InstanceConfig, metrics: Metrics,
                 stop: threading.Event, failures: list):
        """Wire the stage to the server connection, the bundle pool and the in-flight record."""
        super().__init__("collect", stop, failures)
        self.conn = conn
        self.pool = pool
        self.in_flight = in_flight
        self.config = config
        self.metrics = metrics

    def work(self) -> None:
        """Deliver results as they arrive; once submitting is over and nothing is in flight, signal the tracker.

        Raises TimeoutError if the oldest in-flight frame gets no result within the server timeout.
        """
        while not self.stop_event.is_set():
            if self.conn.poll(_POLL_S):
                self._deliver(self.conn.recv())
            elif self.in_flight.closed and not len(self.in_flight):
                put_until(self.pool.to_track, END, self.stop_event)
                return
            if self.in_flight.oldest_age(time.monotonic()) > self.config.server_timeout_s:
                raise TimeoutError(f"no result from the server for {self.config.server_timeout_s} s")

    def _deliver(self, message) -> None:
        """Attach a result's detections to its bundle and queue it for tracking; ignore results for unknown frames.

        Raises RuntimeError if the server rejected the stream or sent something other than a result.
        """
        if isinstance(message, Reject):
            raise RuntimeError(f"the server dropped this stream: {message.reason}")
        if not isinstance(message, Result):
            raise RuntimeError(f"unexpected message from the server: {type(message).__name__}")
        bundle, sent_at = self.in_flight.pop(message.frame_id)
        if bundle is None:
            return
        self.metrics.record_ms("remote", (time.monotonic() - sent_at) * 1e3)
        bundle.rows = rows_from_bytes(message.rows)
        self.pool.mark(bundle, Part.DETECTIONS)
        self.metrics.count("inferred")
        self.metrics.count("rows", len(bundle.rows))
        self.pool.to_track.put(bundle)  # the lease moves with the bundle
