# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU tests for the batch detector: model sizes and shapes, batched against single results, and the error for
missing models."""
import cv2
import numpy as np
import pytest
import torch

from server.detector import BatchDetector

pytestmark = pytest.mark.gpu


def clip_inputs(clip_path, indexes) -> torch.Tensor:
    """Real frames as a 640x384 RGB batch, resized on the CPU and padded with 114."""
    capture = cv2.VideoCapture(str(clip_path))
    frames = []
    for index in indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        assert ok
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        frames.append(cv2.copyMakeBorder(small, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114)))
    capture.release()
    batch = np.stack(frames).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return torch.from_numpy(batch).to("cuda:0")


def test_sizes_and_shapes(models_dir):
    """Proves the detector reports the expected input size and class names and returns correctly shaped batches."""
    detector = BatchDetector(models_dir, (1, 4), 0)
    assert detector.input_size == (640, 384)
    assert detector.names[2] == "car"
    assert detector.input(4).shape == (4, 3, 384, 640)
    assert detector.run(4).shape == (4, 300, 6)


def test_batched_rows_match_single_rows(models_dir, clip_path):
    """Proves a batch of 4 real frames gives the same kept detections as running each frame alone."""
    detector = BatchDetector(models_dir, (1, 4), 0)
    frames = clip_inputs(clip_path, (0, 60, 120, 180))
    detector.input(4).copy_(frames)
    batched = detector.run(4)
    for index in range(4):
        detector.input(1).copy_(frames[index:index + 1])
        single = detector.run(1)[0]
        kept_single = single[single[:, 4] >= 0.25]
        kept_batched = batched[index][batched[index, :, 4] >= 0.25]
        assert len(kept_single) >= 3  # the clip has real objects, so this comparison cannot pass on nothing
        assert len(kept_single) == len(kept_batched)
        order_a, order_b = np.lexsort(kept_single[:, :2].T), np.lexsort(kept_batched[:, :2].T)
        assert np.abs(kept_single[order_a] - kept_batched[order_b]).max() < 0.1


def test_a_missing_model_names_the_export_tool(tmp_path):
    """Proves loading from a folder without models fails with an error that names the export tool."""
    with pytest.raises(FileNotFoundError, match="export_models"):
        BatchDetector(tmp_path, (1,), 0)
