# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of the server's preprocessing: the letterbox geometry and the batched NV12 to model-input conversion."""
import numpy as np
import pytest
import torch

from server.letterbox import Letterbox
from server.prepare import PAD_VALUE, BatchPreparer, reference_input


def frame(seed: int, width=64, height=36) -> torch.Tensor:
    """Return a reproducible random NV12 frame (luma plus half-height chroma) for the seed."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(16, 236, (height * 3 // 2, width), generator=generator, dtype=torch.uint8)


def test_letterbox_for_a_wide_target():
    """A 16:9 source in a 640 by 384 target is scaled to 640 by 360 and centred with 12 rows of padding above."""
    box = Letterbox(1920, 1080, 640, 384)
    assert (box.new_width, box.new_height, box.left, box.top) == (640, 360, 0, 12)


def test_letterbox_maps_back_to_the_source_and_clips():
    """to_source maps target boxes back to source pixels and clips boxes that run past the image."""
    box = Letterbox(1920, 1080, 640, 384)
    result = box.to_source(np.array([[0, 12, 640, 372], [-5, 0, 700, 400]], np.float32))
    assert result[0].tolist() == pytest.approx([0, 0, 1920, 1080])
    assert result[1].tolist() == pytest.approx([0, 0, 1920, 1080])


@pytest.mark.parametrize("source,target", [
    ((1920, 1080), (640, 384)),  # the real case: luma exact 3:1, chroma 1.5:1
    ((1920, 1080), (640, 640)),
    ((1280, 720), (640, 384)),
    ((64, 36), (100, 60)),       # upscaling
])
def test_batch_matches_the_single_frame_reference(source, target):
    """The batched conversion matches the one-frame reference within 2e-4 for several sizes, including upscaling."""
    width, height = source
    box = Letterbox(width, height, *target)
    frames = [frame(seed, width, height) for seed in range(3)]
    output = torch.zeros((3, 3, target[1], target[0]))
    BatchPreparer(box, torch.device("cpu")).convert(frames, output)
    for index, nv12 in enumerate(frames):
        expected = torch.zeros((1, 3, target[1], target[0]))
        reference_input(nv12, box, expected)
        assert (output[index] - expected[0]).abs().max() < 2e-4


def test_padding_is_exact():
    """The rows above and below the letterboxed image are exactly PAD_VALUE."""
    box = Letterbox(64, 36, 64, 64)
    output = torch.zeros((1, 3, 64, 64))
    BatchPreparer(box, torch.device("cpu")).convert([frame(0)], output)
    assert (output[..., :box.top, :] == PAD_VALUE).all()
    assert (output[..., box.top + box.new_height:, :] == PAD_VALUE).all()


def test_a_short_batch_leaves_other_rows_alone():
    """Converting fewer frames than the output holds writes only those rows and leaves the rest untouched."""
    box = Letterbox(64, 36, 64, 64)
    output = torch.full((2, 3, 64, 64), 7.0)
    BatchPreparer(box, torch.device("cpu")).convert([frame(0)], output[:1])
    assert (output[1] == 7.0).all()


@pytest.mark.gpu
def test_gpu_matches_cpu():
    """The conversion on a CUDA device matches the CPU result within 2e-4 for real-size frames."""
    box = Letterbox(1920, 1080, 640, 384)
    frames = [frame(seed, 1920, 1080) for seed in range(2)]
    device = torch.device("cuda:0")
    on_gpu = torch.zeros((2, 3, 384, 640), device=device)
    BatchPreparer(box, device).convert([item.to(device) for item in frames], on_gpu)
    on_cpu = torch.zeros((2, 3, 384, 640))
    BatchPreparer(box, torch.device("cpu")).convert(frames, on_cpu)
    assert (on_gpu.cpu() - on_cpu).abs().max() < 2e-4
