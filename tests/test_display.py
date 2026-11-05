# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU tests for the display stage: colour conversion in the shader, the box overlay, window scaling and
placement, and window cleanup."""
import threading

import glfw
import numpy as np
import pytest
import torch

from pipeline.bundle import Bundle, BundlePool
from pipeline.config import InstanceConfig
from pipeline.display import DisplayStage
from pipeline.metrics import Metrics
from server.imports import _Opened
from server.letterbox import Letterbox
from server.prepare import reference_input

pytestmark = pytest.mark.gpu
EMPTY_TRACKS = np.zeros((0, 8), np.float32)


def make_pool():
    """Build a one-bundle pool with display on and a release callback that does nothing."""
    return BundlePool([Bundle(0)], lambda pointer: None, display_enabled=True)


def make_display(size=(1920, 1080)):
    """Build a display stage for a 1920x1080 source that shows its window at the given size."""
    config = InstanceConfig(address="unused", display_size=size)
    return DisplayStage(config, make_pool(), (1920, 1080), {2: "car"}, Metrics(), threading.Event(),
                        threading.Event())


def first_frame_pointer(clip_path):
    """Decode the first frame of the clip on the GPU; return its device pointer with the decoder and demuxer
    that must stay alive while the pointer is used."""
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
    """Proves the shader's NV12 to RGB output stays within a small mean error of the CUDA conversion."""
    pointer, decoder, demuxer = first_frame_pointer(clip_path)
    display = make_display()
    display.open()
    display.show(pointer, EMPTY_TRACKS, "")
    shown = display.read_pixels()
    display.close()
    nv12 = _Opened(pointer, 1920, 1080, 0).tensor
    reference = torch.empty((1, 3, 1080, 1920), dtype=torch.float32, device="cuda:0")
    reference_input(nv12, Letterbox(1920, 1080, 1920, 1080), reference)
    expected = (reference[0].permute(1, 2, 0) * 255).cpu().numpy()
    assert np.abs(shown.astype(np.float32) - expected).mean() < 2.5


def test_overlay_changes_pixels_inside_the_box(clip_path):
    """Proves drawing a track changes pixels on the box edge and leaves pixels away from it untouched."""
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
    """Proves a half-size window draws the box at half scale, not at the source coordinates."""
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


def test_closing_the_window_ends_run_and_sets_stop(monkeypatch):
    """Proves that when the window reports it should close, run() returns and sets the stop event."""
    stage = make_display((480, 270))
    calls = []

    def should_close(window):
        """Report the window as open for 20 checks, then as closed."""
        calls.append(window)
        return len(calls) > 20

    monkeypatch.setattr("pipeline.display.glfw.window_should_close", should_close)
    stage.run()
    assert stage.stop.is_set()
    assert len(calls) > 20


def test_a_failed_open_does_not_leak_the_window(monkeypatch):
    """Proves that when shader compilation fails during open, the error propagates and no window is left."""
    stage = make_display((480, 270))

    def broken(kind, source):
        """Fail like a shader that does not compile."""
        raise RuntimeError("shader did not compile")

    monkeypatch.setattr("pipeline.display.compile_shader", broken)
    with pytest.raises(RuntimeError, match="shader"):
        stage.run()
    assert stage.window is None


def test_small_window_opens_at_its_position(clip_path):
    """Proves a small window opens at the configured size and position, with the stream in its title and a
    text scale that grows as the window shrinks."""
    config = InstanceConfig(address="unused", stream=3, display_size=(480, 270), window_position=(480, 300))
    stage = DisplayStage(config, make_pool(), (1920, 1080), {}, Metrics(), threading.Event(), threading.Event())
    stage.open()
    try:
        assert glfw.get_window_size(stage.window) == (480, 270)
        assert glfw.get_window_pos(stage.window) == (480, 300)
        assert "3" in stage.title
        assert stage.text_scale == pytest.approx(4.0)
    finally:
        stage.close()
