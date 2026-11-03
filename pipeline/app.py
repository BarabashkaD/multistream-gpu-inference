# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Runs the bundle pipeline: wires the stages together, reports while it runs and shuts down in order."""
from __future__ import annotations

import json
import queue
import threading

import torch

from pipeline.bundle import Bundle, BundlePool
from pipeline.config import PipelineConfig
from pipeline.decoder import DecoderStage
from pipeline.detector import Detector, InferenceStage
from pipeline.metrics import Metrics
from pipeline.nv12 import Letterbox
from pipeline.prepare import CudaConverter, PrepareStage
from pipeline.source import MemorySource, ReaderStage
from pipeline.tracking import TrackerCore, TrackerStage

_JOIN_TIMEOUT_S = 5.0
_stuck_stages: list = []  # keeps stages that outlived their join referenced, so their decoder is never freed


def _join_stages(stages, timeout_s: float) -> None:
    """Join every stage, each for up to timeout_s; raise RuntimeError naming the stages that are still alive."""
    for stage in stages:
        stage.join(timeout_s)
    stuck = [stage for stage in stages if stage.is_alive()]
    if stuck:
        _stuck_stages.extend(stuck)
        raise RuntimeError("stages did not stop within the timeout: " + ", ".join(stage.name for stage in stuck))


def _build_bundles(config: PipelineConfig, detector: Detector) -> list[Bundle]:
    """Create the fixed set of bundles, each with a float32 input tensor and a detection tensor on the GPU."""
    device = f"cuda:{config.gpu_id}"
    size = config.detector_size
    return [Bundle(index,
                   torch.empty((1, 3, size, size), dtype=torch.float32, device=device),
                   torch.empty(detector.output_shape, dtype=torch.float32, device=device))
            for index in range(config.pool_size)]


def _report(metrics: Metrics, pool: BundlePool, interval_s: float, done: threading.Event) -> None:
    """Print one JSON line of current and average rates and pool counts every interval until `done` is set.

    Runs on the reporter thread.
    """
    while not done.wait(interval_s):
        snapshot = metrics.snapshot()
        current = metrics.current_fps()
        print(json.dumps({"current": {k: round(v, 1) for k, v in current.items()},
                          "fps": {k: round(v, 1) for k, v in snapshot["fps"].items()},
                          "pool": pool.stats()}), flush=True)


def run_pipeline(config: PipelineConfig) -> dict:
    """Run the pipeline to the end of its input, or until interrupted, and return the final statistics snapshot.

    Starts the reader, decoder, prepare, inference and tracker stages plus a reporter thread, then waits on the
    display window or on the tracker finishing. The pool is finished before the stages are joined, so the decoder
    thread can unlock every frame. Raises RuntimeError when a stage failed or did not stop in time; graphics
    teardown errors are reported the same way and never replace a stage failure. The snapshot includes the pool's
    statistics under "pool".
    """
    stop, finished = threading.Event(), threading.Event()
    failures: list = []
    source = MemorySource(config.input)
    source_size = (source.width, source.height)
    letterbox = Letterbox(source.width, source.height, config.detector_size)
    detector = Detector(config.model, config.gpu_id)
    returned: queue.SimpleQueue = queue.SimpleQueue()
    pool = BundlePool(_build_bundles(config, detector), returned.put,
                      display_enabled=config.display, max_age_s=config.max_age_s,
                      display_grace_s=config.display_grace_s)
    packets: queue.Queue = queue.Queue(maxsize=config.packet_queue)
    core = TrackerCore(letterbox, source_size, config.tracker, config.conf)
    metrics = Metrics()  # created after model loading; "fps" still includes pipeline start-up and shutdown, see "steady_fps"
    converter = _make_converter(config, letterbox, source_size)
    stages = [
        ReaderStage(source, packets, config, metrics, stop, failures),
        DecoderStage(source, packets, pool, returned, config, metrics, stop, failures),
        PrepareStage(converter, pool, config, metrics, stop, failures),
        InferenceStage(detector, pool, metrics, stop, failures),
        TrackerStage(core, pool, finished, metrics, stop, failures),
    ]
    reporting_done = threading.Event()
    reporter = threading.Thread(target=_report, args=(metrics, pool, config.report_interval_s, reporting_done),
                                name="reporter", daemon=True)
    try:
        converter.setup_main()  # GLFW windows belong to the main thread; this runs before any stage starts
        for stage in stages:
            stage.start()
        reporter.start()
        try:
            _wait(config, pool, source_size, detector, metrics, stop, finished)
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
        if not any(stage.is_alive() for stage in stages):
            try:
                _release_graphics(config, converter)  # a stage that is still alive may own the GL context
            except Exception as error:  # must not replace a stage failure or an exception already propagating
                failures.append(("graphics", error))
    if failures:
        name, error = failures[0]
        raise RuntimeError(f"stage {name} failed: {error}") from error
    snapshot = metrics.snapshot()
    snapshot["pool"] = pool.stats()
    return snapshot


def _make_converter(config: PipelineConfig, letterbox: Letterbox, source_size: tuple[int, int]):
    """Return the frame converter chosen by --prepare: an OpenGL one for "gl", otherwise the CUDA one."""
    if config.prepare == "gl":
        # Imported here: --prepare cuda does not load GLFW or OpenGL for preparation, and a --no-display run loads neither.
        from pipeline.gl_prepare import GlConverter

        return GlConverter(letterbox, source_size, config.gpu_id)
    return CudaConverter(letterbox, source_size)


def _release_graphics(config: PipelineConfig, converter) -> None:
    """Tear down the converter's main-thread resources, then terminate GLFW if a display or OpenGL preparation used it.

    glfw is imported here so headless CUDA runs never load it. GLFW is terminated even if the teardown fails.
    """
    try:
        converter.teardown_main()
    finally:
        if config.display or config.prepare == "gl":
            import glfw

            glfw.terminate()


def _wait(config, pool, source_size, detector, metrics, stop, finished) -> None:
    """Block until the stream finishes or stop is set, running the display window on this thread when enabled."""
    if config.display:
        from pipeline.display import DisplayStage

        DisplayStage(config, pool, source_size, detector.names, metrics, stop, finished).run()
        return
    while not finished.wait(0.1):
        if stop.is_set():
            return
