# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU tests for the display stage: shader colour conversion, the track overlay, window scaling and shutdown."""
import threading
import time

import numpy as np
import pytest
import torch

from pipeline.bundle import Bundle, BundlePool
from pipeline.config import PipelineConfig
from pipeline.display import DisplayStage
from pipeline.metrics import Metrics
from pipeline.nv12 import Letterbox, nv12_to_input, wrap_nv12

pytestmark = pytest.mark.gpu
EMPTY_TRACKS = np.zeros((0, 8), np.float32)


def make_display(size=(1920, 1080)):
    """Builds a display stage for a 1920x1080 source and a window of the given size, with a one-bundle pool."""
    config = PipelineConfig(display_size=size)
    pool = BundlePool([Bundle(0)], lambda pointer: None, display_enabled=True)
    return DisplayStage(config, pool, (1920, 1080), {2: "car"}, Metrics(), threading.Event(), threading.Event())


def first_frame_pointer(clip_path):
    """Decodes the first frame of the clip on the GPU; returns its device pointer with the decoder and demuxer, which
    the caller must keep alive while the pointer is used."""
    import PyNvVideoCodec as nvc

    demuxer = nvc.CreateDemuxer(str(clip_path))
    decoder = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId(), usedevicememory=True,
                                outputColorType=nvc.OutputColorType.NATIVE)
    pointer = None
    while pointer is None:
        for _ in range(decoder.GetNumDecodedFrame(demuxer.Demux())):
            pointer = pointer or decoder.GetLockedFrame()
    decoder.SyncOnCUStream()
    return pointer, decoder, demuxer


def test_shader_output_matches_cuda_conversion(clip_path):
    """Proves the pixels the display shader draws match the CUDA NV12 conversion to within a small mean error."""
    pointer, decoder, demuxer = first_frame_pointer(clip_path)
    display = make_display()
    display.open()
    display.show(pointer, EMPTY_TRACKS, "")
    shown = display.read_pixels()
    display.close()
    nv12 = wrap_nv12(pointer, 1920, 1080)
    reference = torch.empty((1, 3, 1080, 1920), dtype=torch.float32, device="cuda:0")
    nv12_to_input_full(nv12, reference)
    expected = (reference[0].permute(1, 2, 0) * 255).cpu().numpy()
    assert np.abs(shown.astype(np.float32) - expected).mean() < 2.5


def nv12_to_input_full(nv12, output):
    """Converts an NV12 frame to RGB at full 1920x1080 size with no letterbox padding, as the reference image."""
    letterbox = Letterbox(1920, 1080, (1088, 1920))  # a valid size; the fields below replace its geometry
    letterbox.new_height, letterbox.new_width, letterbox.top, letterbox.left = 1080, 1920, 0, 0
    nv12_to_input(nv12, letterbox, 1920, 1080, output)


def test_overlay_changes_pixels_inside_the_box(clip_path):
    """Proves a track box changes the pixels on its edge and leaves pixels away from the box untouched."""
    pointer, decoder, demuxer = first_frame_pointer(clip_path)
    display = make_display()
    display.open()
    display.show(pointer, EMPTY_TRACKS, "")
    plain = display.read_pixels().copy()
    tracks = np.array([[400, 300, 800, 700, 5, 0.9, 2, 0]], np.float32)
    display.show(pointer, tracks, "status")
    annotated = display.read_pixels()
    display.close()
    assert not np.array_equal(annotated[300, 400:800], plain[300, 400:800])
    assert np.array_equal(annotated[500, 600], plain[500, 600])


def test_downsized_window_scales_the_overlay(clip_path):
    """Proves in a half-size window the box is drawn at half its source coordinates and the image has the window's
    shape."""
    pointer, decoder, demuxer = first_frame_pointer(clip_path)
    display = make_display((960, 540))
    display.open()
    tracks = np.array([[400, 300, 800, 700, 5, 0.9, 2, 0]], np.float32)
    display.show(pointer, np.zeros((0, 8), np.float32), "")
    plain = display.read_pixels()
    display.show(pointer, tracks, "")
    shown = display.read_pixels()
    display.close()
    assert shown.shape == (540, 960, 3)
    assert not np.array_equal(shown[150, 200:400], plain[150, 200:400])  # the top edge at half scale
    assert np.array_equal(shown[300, 480:800], plain[300, 480:800])  # nothing at the unscaled position


def test_closing_the_window_stops_a_running_pipeline(clip_path, model_path, monkeypatch):
    """Proves that when the window is closed a pipeline set to loop forever stops, having displayed frames."""
    from pipeline.app import run_pipeline
    from pipeline.config import PipelineConfig

    started = time.monotonic()
    monkeypatch.setattr("pipeline.display.glfw.window_should_close",
                        lambda window: time.monotonic() - started > 6.0)
    config = PipelineConfig(input=clip_path, model=model_path, loops=0, pace="realtime", display=True)
    summary = run_pipeline(config)
    assert summary["counts"]["displayed"] > 0
    assert summary["counts"]["tracked"] < 611  # the run ended early, not at the end of the clip
