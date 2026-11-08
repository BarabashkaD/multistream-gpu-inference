# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Runs one pipeline instance: wires the stages together, joins the server, reports and shuts down in order."""
from __future__ import annotations

import queue
import sys
import threading

import psutil

from pipeline.bundle import Bundle, BundlePool
from pipeline.config import InstanceConfig
from pipeline.decoder import DecoderStage
from pipeline.gpu import use_gpu
from pipeline.memory import private_mb
from pipeline.metrics import Metrics
from pipeline.protocol import Bye, Closed, Hello, Reject, Start, connect
from pipeline.remote import CollectStage, InFlight, SubmitStage
from pipeline.source import MemorySource, ReaderStage
from pipeline.statslog import StatsLog, instance_line
from pipeline.tracking import TrackerCore, TrackerStage

_JOIN_TIMEOUT_S = 5.0
_FAREWELL_TIMEOUT_S = 10.0
# Keeps stages referenced so their decoder is never freed while the server may map its memory.
# At process exit the operating system releases everything in the right order for the server.
_stuck_stages: list = []


def _join_stages(stages, timeout_s: float) -> None:
    """Join every stage, each for up to timeout_s; raise RuntimeError naming the stages that are still alive."""
    for stage in stages:
        stage.join(timeout_s)
    stuck = [stage for stage in stages if stage.is_alive()]
    if stuck:
        raise RuntimeError("stages did not stop within the timeout: " + ", ".join(stage.name for stage in stuck))


def _report(metrics: Metrics, pool: BundlePool, config: InstanceConfig, log: StatsLog, done: threading.Event) -> None:
    """Write one statistics line every report interval until `done` is set; runs on the reporter thread."""
    process = psutil.Process()
    while not done.wait(config.report_interval_s):
        log.write(instance_line(metrics, pool.stats(), config.stream, config.report_interval_s, private_mb(process)))


def _join_server(conn, config: InstanceConfig, width: int, height: int) -> dict:
    """Announce this stream and its frame size to the server and return the class names from its Start reply.

    Raises TimeoutError if no reply arrives within the connect timeout and RuntimeError if the server rejects the
    stream or answers with anything but Start.
    """
    conn.send(Hello(config.stream, width, height))
    if not conn.poll(config.connect_timeout_s):
        raise TimeoutError("the server did not start this stream")
    reply = conn.recv()
    if isinstance(reply, Reject):
        raise RuntimeError(f"the server rejected this stream: {reply.reason}")
    if not isinstance(reply, Start):
        raise RuntimeError(f"unexpected reply from the server: {type(reply).__name__}")
    return reply.names


def _leave_server(conn, stats: dict) -> bool:
    """Send the final statistics and wait for the server to confirm it has released this stream's frames.

    Returns True once the server holds no mapping of this process's decoder memory, False if it stays silent for
    the farewell timeout.
    """
    try:
        conn.send(Bye(stats))
        while conn.poll(_FAREWELL_TIMEOUT_S):
            if isinstance(conn.recv(), Closed):
                return True
    except (EOFError, OSError):
        return True  # the server never closes a live pipe, so it has gone, and its mappings with it
    return False


def run_instance(config: InstanceConfig, log: StatsLog | None = None) -> dict:
    """Run one stream to the end of its input, or until interrupted, and return its final statistics snapshot.

    Starts the reader, decoder, submit, collect and tracker stages plus a reporter thread, then waits on the display
    window or on the tracker finishing. Shutdown order matters: the pool is finished before the stages are joined,
    and the server must confirm it released the stream's frames before the decoder may be freed. Raises RuntimeError
    when a stage failed or did not stop in time, and the server errors of _join_server; connect failures propagate.
    """
    log = log or StatsLog()
    use_gpu(config.gpu_id)
    stop, finished = threading.Event(), threading.Event()
    failures: list = []
    source = MemorySource(config.input)
    source_size = (source.width, source.height)
    conn = connect(config.address, config.connect_timeout_s)
    names = _join_server(conn, config, *source_size)
    returned: queue.SimpleQueue = queue.SimpleQueue()
    pool = BundlePool([Bundle(index) for index in range(config.pool_size)], returned.put,
                      display_enabled=config.display, max_age_s=config.max_age_s,
                      display_grace_s=config.display_grace_s)
    packets: queue.Queue = queue.Queue(maxsize=config.packet_queue)
    metrics = Metrics()
    in_flight = InFlight()
    stages = [
        ReaderStage(source, packets, config, metrics, stop, failures),
        DecoderStage(source, packets, pool, returned, config, metrics, stop, failures),
        SubmitStage(conn, pool, in_flight, config, metrics, stop, failures),
        CollectStage(conn, pool, in_flight, config, metrics, stop, failures),
        TrackerStage(TrackerCore(config.conf, source_size), pool, finished, metrics, stop, failures,
                     delay_s=config.track_delay_ms / 1e3),
    ]
    _stuck_stages.extend(stages)  # pinned until the server confirms it holds no mapping of the decoder memory
    reporting_done = threading.Event()
    reporter = threading.Thread(target=_report, args=(metrics, pool, config, log, reporting_done),
                                name="reporter", daemon=True)
    try:
        try:
            for stage in stages:
                stage.start()
            reporter.start()
            try:
                _wait(config, pool, source_size, names, metrics, stop, finished)
            except KeyboardInterrupt:
                stop.set()
            finally:
                reporting_done.set()
                if not finished.is_set():
                    stop.set()
                pool.finish()  # before the joins, so the decoder thread can unlock every frame
                try:
                    _join_stages(stages, _JOIN_TIMEOUT_S)
                except RuntimeError:
                    if not failures:
                        raise
        finally:
            _release_display(config)
    finally:
        reporting_done.set()
        stop.set()
        if reporter.is_alive():
            reporter.join(_JOIN_TIMEOUT_S)  # no periodic line may follow the final one
        snapshot = metrics.snapshot()
        snapshot["stream"] = config.stream
        snapshot["pool"] = pool.stats()
        snapshot["role"] = "instance"
        snapshot["memory_mb"] = round(private_mb(psutil.Process()))
        if _leave_server(conn, {key: snapshot[key] for key in ("stream", "counts", "steady_fps", "median_ms",
                                                              "p95_ms")}):
            for stage in stages:
                _stuck_stages.remove(stage)
        else:
            print(f"the server did not confirm within {_FAREWELL_TIMEOUT_S:.0f} s that it released this stream's "
                  "frames, so the decoder is kept until exit", file=sys.stderr, flush=True)
    if failures:
        name, error = failures[0]
        raise RuntimeError(f"stage {name} failed: {error}") from error
    return snapshot


def _release_display(config: InstanceConfig) -> None:
    """Terminate GLFW when a display was used; glfw is imported here so headless runs never load it."""
    if config.display:
        import glfw

        glfw.terminate()


def _wait(config, pool, source_size, names, metrics, stop, finished) -> None:
    """Block until the stream finishes or stop is set, running the display window on this thread when enabled."""
    if config.display:
        from pipeline.display import DisplayStage

        DisplayStage(config, pool, source_size, names, metrics, stop, finished).run()
        return
    while not finished.wait(0.1):
        if stop.is_set():
            return
