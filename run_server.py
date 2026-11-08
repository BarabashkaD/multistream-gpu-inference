#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Inference server. See docs/design/external-inference.md."""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # must precede NumPy: each thread otherwise commits memory

import threading

import torch

from pipeline.metrics import Metrics
from pipeline.protocol import remove_stale_socket
from pipeline.statslog import StatsLog
from server.config import parse_server_config
from server.detector import BatchDetector
from server.imports import FrameImports
from server.letterbox import Letterbox
from server.prepare import BatchPreparer
from server.service import InferenceServer, run


def main() -> None:
    """Loads the detector, serves pipeline instances until they have all left and writes the final statistics.

    Reads the options from the command line, needs a CUDA GPU and the exported models, and removes a stale socket
    at the address first."""
    config = parse_server_config()
    remove_stale_socket(config.address)
    log = StatsLog(config.stats_file)
    device = torch.device(f"cuda:{config.gpu_id}")
    torch.zeros(1, device=device)  # the batch thread is this thread, so its CUDA context is current here
    detector = BatchDetector(config.models, config.batch_sizes, config.gpu_id)
    imports = FrameImports(0, 0, config.gpu_id)  # the frame size arrives with the first instance

    def make_preparer(width: int, height: int) -> BatchPreparer:
        """Builds the preparer for the source size of the first instance and tells the imports that size."""
        imports.width, imports.height = width, height
        return BatchPreparer(Letterbox(width, height, *detector.input_size), device)

    server = InferenceServer(config, detector, make_preparer, imports, Metrics(), threading.Event())
    try:
        log.write_final({"kind": "final", **run(server, log)})
    finally:
        log.close()


if __name__ == "__main__":
    main()
