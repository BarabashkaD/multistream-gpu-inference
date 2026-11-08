# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The inference server loop: admits pipeline instances, batches their frames, runs the detector and replies."""
from __future__ import annotations

from collections import Counter
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener, wait
import sys
import threading
import time

import psutil

from pipeline.memory import private_mb
from pipeline.metrics import Metrics
from pipeline.protocol import Bye, Closed, Frame, Hello, Reject, Result, Start, auth_key
from pipeline.statslog import StatsLog, server_line
from server.config import ServerConfig
from server.queueing import FairQueue

_HELLO_TIMEOUT_S = 5.0
_TAKE_TIMEOUT_S = 0.05
_RECEIVE_WAIT_S = 0.2
_RECEIVE_IDLE_S = 0.05


class InferenceServer:
    """Only the thread calling step() touches the detector, the preparer and the imports.

    Other threads admit instances, receive their messages and report statistics; they share the stream tables
    through one lock. The server is done once every expected instance has joined and left."""

    def __init__(self, config: ServerConfig, detector, make_preparer, imports, metrics: Metrics,
                 stop: threading.Event):
        """Stores the collaborators and creates empty stream tables.

        make_preparer is called with the source frame size when the first batch is formed; `stop` ends every
        thread of the server when set."""
        self.config = config
        self.detector = detector
        self.make_preparer = make_preparer
        self.imports = imports
        self.metrics = metrics
        self.stop = stop
        self.queue = FairQueue()
        self.conns: dict = {}
        self.joined = 0
        self.source_size: tuple[int, int] | None = None
        self.preparer = None
        self.batches: Counter = Counter()
        self.available: Counter = Counter()  # frames waiting per batch, by bucket label
        self.stream_stats: list[dict] = []
        self.dead: list[int] = []
        self.errors: list[dict] = []
        self.names = detector.names  # copied here so the accept thread never touches the detector
        self._retired: dict[int, tuple[Bye | None, str | None]] = {}
        self._gone: set[int] = set()  # streams that have left; only ever grows
        self._spent: list = []  # kept open until exit, because receive() may be inside wait() on them
        self._lock = threading.Lock()

    def admit(self, conn) -> None:
        """Handles a new connection from the accept thread: reads its Hello and registers or rejects the stream.

        A connection that dies while joining is closed and forgotten."""
        try:
            self._admit(conn)
        except (EOFError, OSError):  # the instance died while joining
            conn.close()

    def _admit(self, conn) -> None:
        """Waits for a Hello, then registers the stream or sends a Reject.

        A stream that was seen before, or a frame size that differs from the first stream's, is rejected. When the
        last expected instance has joined, every member receives Start with the class names. A connection that
        sends no Hello in time, or something else, is closed without a reply."""
        if not conn.poll(_HELLO_TIMEOUT_S):
            conn.close()
            return
        hello = conn.recv()
        if not isinstance(hello, Hello):
            conn.close()
            return
        size = (hello.width, hello.height)
        with self._lock:
            reason = None
            if hello.stream in self.conns or hello.stream in self._gone:
                reason = f"stream {hello.stream} has already joined"
            elif self.source_size not in (None, size):
                reason = f"frame size {size} differs from {self.source_size}"
            if reason is None:
                self.source_size = size
                self.conns[hello.stream] = conn
                self.joined += 1
            everyone = list(self.conns.values()) if self.joined == self.config.clients else []
        if reason is not None:
            conn.send(Reject(reason))
            conn.close()
            return
        for member in everyone:
            try:
                member.send(Start(self.names))
            except OSError:
                pass  # the receive thread will see this stream's pipe close

    def handle(self, stream: int, message) -> None:
        """Acts on one message of a stream: queues a Frame, or retires the stream on Bye. Others are ignored.

        A frame of a stream that has left is dropped."""
        if isinstance(message, Frame):
            with self._lock:
                if stream not in self._gone:
                    self.queue.add(stream, message)
        elif isinstance(message, Bye):
            self.retire(stream, message)

    def retire(self, stream: int, bye: Bye | None, reason: str | None = None) -> None:
        """No more frames from this stream. The batch thread closes its handles between batches."""
        with self._lock:
            if stream in self._gone:
                return
            self._gone.add(stream)
            self._retired[stream] = (bye, reason)
            self.queue.remove_stream(stream)

    def receive(self) -> None:
        """Reads messages from every live stream until stop is set; a stream whose pipe breaks is retired.

        Runs on its own thread. It idles briefly while no stream is connected."""
        while not self.stop.is_set():
            with self._lock:
                owners = {conn: stream for stream, conn in self.conns.items() if stream not in self._gone}
            if not owners:
                self.stop.wait(_RECEIVE_IDLE_S)
                continue
            for conn in wait(list(owners), timeout=_RECEIVE_WAIT_S):
                stream = owners[conn]
                try:
                    self.handle(stream, conn.recv())
                except (EOFError, OSError):
                    self.retire(stream, None)

    def _close_retired(self) -> None:
        """Finishes the streams retired since the last call, on the batch thread.

        Closes their imported handles, records them as dead or with their statistics, and tells the instance with
        Reject (when there is a reason) and Closed. A handle that fails to close is recorded as an error and Closed
        is withheld, so the instance keeps its decoder memory."""
        with self._lock:
            retired, self._retired = self._retired, {}
        for stream, (bye, reason) in retired.items():
            kept_mapping = False
            try:
                self.imports.close_stream(stream)
            except RuntimeError as error:  # a dead exporter must not stop the other streams
                print(f"stream {stream}: closing its handles failed: {error}", file=sys.stderr, flush=True)
                self.errors.append({"stream": stream, "error": str(error)})
                reason = f"closing its handles failed: {error}"
                kept_mapping = True
            with self._lock:
                conn = self.conns.pop(stream)
            self._spent.append(conn)
            if bye is None:
                self.dead.append(stream)
            else:
                self.stream_stats.append(bye.stats)
            try:
                if reason is not None:
                    conn.send(Reject(reason))
                if not kept_mapping:  # Closed tells the instance it may free its decoder memory
                    conn.send(Closed())  # best effort: an instance that died cannot read it
            except OSError:
                pass

    def _send(self, stream: int, message) -> None:
        """Sends a message to a stream if it is still connected; a broken pipe retires the stream."""
        conn = self.conns.get(stream)
        if conn is None:
            return
        try:
            conn.send(message)
        except OSError:
            self.retire(stream, None)

    def _active_streams(self) -> int:
        """Counts the connected streams that have not left."""
        with self._lock:
            return sum(1 for stream in self.conns if stream not in self._gone)

    def step(self) -> None:
        """Forms and serves one batch; returns early when no frame arrives in time or none can be opened.

        Closes retired streams, takes frames from the queue, imports their buffers, converts them into the
        smallest configured batch size that fits (padding with copies of a real frame), runs the detector and sends
        each stream its detections above the confidence threshold. A stream whose frame cannot be opened is retired
        with the reason. Records timings and counts in the metrics."""
        self._close_retired()
        started = time.perf_counter()
        picks = self.queue.take(self.config.batch_sizes, self.config.gather_ms / 1e3, _TAKE_TIMEOUT_S,
                                self._active_streams())
        if not picks:
            return
        available = self.queue.last_available  # right after take: only this thread consumes the queue
        began = time.perf_counter()
        self.metrics.record_ms("batch_wait", (began - started) * 1e3)
        if self.preparer is None:
            self.preparer = self.make_preparer(*self.source_size)
        views = self._open(picks)
        good = [view for view in views if view is not None]
        if not good:
            return
        self._count_available(available)
        size = min(item for item in self.config.batch_sizes if item >= len(picks))
        frames = [good[0] if view is None else view for view in views]
        frames += [good[0]] * (size - len(frames))  # copies of a real frame; their rows are never sent
        self.preparer.convert(frames, self.detector.input(size))
        prepared = time.perf_counter()
        rows = self.detector.run(size)
        inferred = time.perf_counter()
        for (stream, frame), view, row in zip(picks, views, rows):
            if view is None:
                continue
            kept = row[row[:, 4] >= self.config.min_conf].copy()
            kept[:, :4] = self.preparer.to_source(kept[:, :4])
            self._send(stream, Result(frame.frame_id, kept.tobytes()))
        self.metrics.record_ms("prepare", (prepared - began) * 1e3 / len(good))
        self.metrics.record_ms("infer", (inferred - prepared) * 1e3 / len(good))
        self.metrics.record_ms("reply", (time.perf_counter() - inferred) * 1e3 / len(good))
        self.metrics.count("inferred", len(good))
        if size > len(picks):
            self.metrics.count("padded", size - len(picks))
        with self._lock:
            self.batches[size] += 1

    def batch_counts(self) -> dict[str, int]:
        """Batches run so far, by batch size (as text), in ascending size order."""
        with self._lock:
            return {str(size): count for size, count in sorted(self.batches.items())}

    def _count_available(self, waiting: int) -> None:
        """Counts one batch in the bucket of frames that were waiting, capping at the largest batch size."""
        largest = max(self.config.batch_sizes)
        label = f"{largest}+" if waiting >= largest else str(waiting)
        with self._lock:
            self.available[label] += 1

    def available_counts(self) -> dict[str, int]:
        """Batches by the number of frames waiting when each was formed; the largest size's bucket ends in '+'."""
        with self._lock:
            return {label: count for label, count in sorted(
                self.available.items(), key=lambda item: (item[0].endswith("+"), int(item[0].rstrip("+"))))}

    def _open(self, picks: list) -> list:
        """Returns a tensor for each picked frame, or None for a frame that cannot be used.

        None marks a stream that left after the take, or whose handle could not be opened; the latter is retired
        with the error as its reason."""
        views = []
        for stream, frame in picks:
            with self._lock:
                gone = stream in self._gone
            if gone:  # it left after the take; its handles are closed before the next batch
                views.append(None)
                continue
            try:
                views.append(self.imports.view(stream, frame.pointer, frame.handle))
            except (KeyError, RuntimeError, BufferError) as error:  # its handle cannot be opened
                print(f"stream {stream}: {error}", file=sys.stderr, flush=True)
                self.errors.append({"stream": stream, "error": str(error)})
                self.retire(stream, None, str(error))
                views.append(None)
        return views

    def done(self) -> bool:
        """True once every expected instance has joined and every connection has been closed."""
        with self._lock:
            return self.joined == self.config.clients and not self.conns

    def summary(self) -> dict:
        """The final statistics: the metrics snapshot plus batch and availability counts, per-stream statistics,
        dead streams, errors and the private memory of the process in MB."""
        snapshot = self.metrics.snapshot()
        snapshot["role"] = "server"
        snapshot["batches"] = self.batch_counts()
        snapshot["available"] = self.available_counts()
        snapshot["streams"] = self.stream_stats
        snapshot["dead"] = self.dead
        snapshot["errors"] = self.errors
        snapshot["memory_mb"] = round(private_mb(psutil.Process()))
        return snapshot


def report(server: InferenceServer, log: StatsLog) -> None:
    """Writes a statistics line to the log every report interval until the server's stop is set."""
    process = psutil.Process()
    interval_s = server.config.report_interval_s
    while not server.stop.wait(interval_s):
        log.write(server_line(server.metrics, server.batch_counts(), len(server.queue), interval_s,
                              private_mb(process), server.available_counts()))


def accept(listener: Listener, server: InferenceServer) -> None:
    """Accepts connections and passes them to server.admit until every expected instance has joined or stop is set.

    A peer that fails authentication, or dies while connecting, is skipped."""
    while server.joined < server.config.clients and not server.stop.is_set():  # a rejected instance uses no place
        try:
            conn = listener.accept()
        except (AuthenticationError, EOFError, OSError):
            if server.stop.is_set():
                return
            server.stop.wait(0.05)  # a persistently failing listener must not spin
            continue  # a peer without the key, or one that died while connecting
        server.admit(conn)


def run(server: InferenceServer, log: StatsLog | None = None) -> dict:
    """Serves until every instance has left.

    Listens on the configured address, starts the accept, receive and report threads, prints "ready" and runs
    server.step on the calling thread. Returns server.summary(). Raises TimeoutError when not every instance joins
    within the start timeout, and re-raises an error that killed the receive thread. Ctrl+C ends the run normally."""
    log = log or StatsLog()
    errors: list[BaseException] = []

    def receive() -> None:
        """Runs server.receive on its thread; an error is kept for run to raise, and stops the server."""
        try:
            server.receive()
        except BaseException as error:  # a dead receive thread must end the run, not hang it
            errors.append(error)
            server.stop.set()

    with Listener(server.config.address, authkey=auth_key()) as listener:
        threading.Thread(target=accept, args=(listener, server), name="accept", daemon=True).start()
        threading.Thread(target=receive, name="receive", daemon=True).start()
        reporter = threading.Thread(target=report, args=(server, log), name="report", daemon=True)
        reporter.start()
        print("ready", flush=True)
        started = time.monotonic()
        try:
            while not server.done() and not server.stop.is_set():
                if server.joined < server.config.clients and time.monotonic() - started > server.config.start_timeout_s:
                    raise TimeoutError(f"only {server.joined} of {server.config.clients} instances joined")
                server.step()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop.set()
            reporter.join(5.0)
    if errors:
        raise errors[0]
    return server.summary()
