# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The prepare stage: converts each decoded NV12 frame into the detector's input tensor, or drops it when too old.

The conversion is done by a converter object with a fixed set of hooks; CudaConverter here uses PyTorch CUDA
kernels, and pipeline.gl_prepare provides an OpenGL one.
"""
from __future__ import annotations

import threading
import time

import torch

from pipeline.bundle import BundlePool, Part
from pipeline.config import PipelineConfig
from pipeline.metrics import Metrics
from pipeline.nv12 import Letterbox, nv12_to_input
from pipeline.stage import END, Stage, get_until, put_until


class CudaConverter:
    """NV12 to detector input with PyTorch CUDA kernels. open, convert and close run on the prepare thread."""

    def __init__(self, letterbox: Letterbox, source_size: tuple[int, int]):
        """Keep the letterbox geometry and the source frame size; nothing is allocated."""
        self.letterbox = letterbox
        self.width, self.height = source_size

    def setup_main(self) -> None:
        """Main-thread setup hook; this converter needs none."""
        pass

    def open(self) -> None:
        """Prepare-thread setup hook; this converter needs none."""
        pass

    def convert(self, bundle) -> None:
        """Fill the bundle's input tensor from its NV12 frame and wait for the GPU to finish."""
        nv12_to_input(bundle.nv12, self.letterbox, self.width, self.height, bundle.input_tensor)
        torch.cuda.current_stream(bundle.input_tensor.device).synchronize()

    def close(self) -> None:
        """Prepare-thread cleanup hook; this converter holds nothing to release."""
        pass

    def teardown_main(self) -> None:
        """Main-thread cleanup hook; this converter needs none."""
        pass


class PrepareStage(Stage):
    """Converts frames from the decoder into detector input and passes them on to inference."""

    def __init__(self, converter, pool: BundlePool, config: PipelineConfig, metrics: Metrics,
                 stop: threading.Event, failures: list):
        """Wire the stage to its converter, the bundle pool and the drop settings in the config."""
        super().__init__("prepare", stop, failures)
        self.converter = converter
        self.pool = pool
        self.config = config
        self.metrics = metrics

    def work(self) -> None:
        """Open the converter on this thread, prepare frames until the end marker or a stop, and always close it."""
        self.converter.open()
        try:
            self._prepare_frames()
        finally:
            self.converter.close()

    def _prepare_frames(self) -> None:
        """Convert each queued bundle and mark its input done, abandoning those that are too old.

        Passes the end marker on to inference. The bundle's lease moves to the next queue with it, or is given up
        when the bundle is abandoned.
        """
        while True:
            bundle = get_until(self.pool.to_infer, self.stop_event)
            if bundle is None:
                return
            if bundle is END:
                put_until(self.pool.to_detect, END, self.stop_event)
                return
            self.metrics.record_ms("source_lag", (time.monotonic() - bundle.due_at) * 1e3)
            if self._too_old(bundle):
                self.pool.abandon(bundle)
                self.metrics.count("dropped")
                continue
            started = time.perf_counter()
            self.converter.convert(bundle)
            self.metrics.record_ms("prepare", (time.perf_counter() - started) * 1e3)
            self.pool.mark(bundle, Part.INPUT)
            self.pool.to_detect.put(bundle)  # the lease moves with the bundle

    def _too_old(self, bundle) -> bool:
        """Whether the oldest-first drop policy is on and the frame is further behind its source time than allowed."""
        if self.config.drop_policy != "oldest":
            return False
        return time.monotonic() - bundle.due_at > self.config.max_lag_s
