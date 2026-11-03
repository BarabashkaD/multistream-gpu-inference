# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU tests for the detector: model metadata, per-slot output binding and the inference stage that fills bundles."""
import threading

import pytest
import torch

from pipeline.bundle import Bundle, BundlePool, Part
from pipeline.detector import Detector, InferenceStage
from pipeline.metrics import Metrics
from pipeline.stage import END

pytestmark = pytest.mark.gpu


def test_detector_reads_model_metadata(model_path):
    """Proves the detector reports the model's output shape and class names."""
    detector = Detector(model_path, 0)
    assert detector.output_shape == (1, 300, 6)
    assert detector.names[2] == "car"


def test_output_can_be_rebound_per_slot(model_path):
    """Proves running the same input into two different output tensors fills both with the same non-zero result."""
    detector = Detector(model_path, 0)
    frame = torch.rand((1, 3, 640, 640), device="cuda:0")
    first = torch.zeros((1, 300, 6), device="cuda:0")
    second = torch.zeros((1, 300, 6), device="cuda:0")
    detector.run(frame, first)
    detector.run(frame, second)
    assert torch.equal(first, second)
    assert float(second.abs().sum()) > 0


def test_stage_fills_detections_and_forwards_in_order(model_path):
    """Proves the inference stage marks the detections part on each bundle, passes the bundles to the tracker in
    frame order and forwards the end marker after them."""
    stop, failures = threading.Event(), []
    detector = Detector(model_path, 0)
    bundles = [Bundle(i, torch.empty((1, 3, 640, 640), device="cuda:0"),
                      torch.zeros((1, 300, 6), device="cuda:0")) for i in range(2)]
    pool = BundlePool(bundles, lambda pointer: None, display_enabled=False)
    nv12 = torch.full((1620, 1920), 128, dtype=torch.uint8, device="cuda:0")
    for frame_id in range(2):
        bundle = pool.acquire(stop)
        pool.publish_frame(bundle, 1000 + frame_id, nv12, frame_id, 0, frame_id)
        pool.to_infer.get_nowait()  # the prepare stage is not under test; hand the bundle on directly
        bundle.input_tensor.fill_(0.5)
        pool.mark(bundle, Part.INPUT)
        pool.to_detect.put(bundle)
    pool.to_detect.put(END)

    stage = InferenceStage(detector, pool, Metrics(), stop, failures)
    stage.start()
    stage.join(30)
    assert not failures, failures

    first, second, end = pool.to_track.get_nowait(), pool.to_track.get_nowait(), pool.to_track.get_nowait()
    assert (first.frame_id, second.frame_id) == (0, 1)
    assert first.parts & Part.INPUT and first.parts & Part.DETECTIONS
    assert end is END

