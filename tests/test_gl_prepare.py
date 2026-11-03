# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU tests for the OpenGL prepare: its output against the CUDA conversion, and a pipeline run that uses it."""
import threading
import time

import pytest
import torch

from pipeline.app import run_pipeline
from pipeline.config import PipelineConfig
from pipeline.nv12 import Letterbox, nv12_to_input, wrap_nv12
from tests.test_display import first_frame_pointer

pytestmark = pytest.mark.gpu
TOP, BOTTOM = 140, 500


class _Frame:
    """Minimal bundle: a decoded NV12 frame and the tensor the converter writes the network input into."""

    def __init__(self, pointer, output):
        """Wraps the device pointer as an NV12 view and keeps the output tensor."""
        self.nv12_pointer = pointer
        self.nv12 = wrap_nv12(pointer, 1920, 1080)
        self.input_tensor = output


def _convert_on_prepare_thread(converter, bundle):
    """Opens the converter and converts one bundle on a separate thread, as the prepare stage does, because the
    OpenGL context belongs to the thread that made it current. Fails if the thread errors or does not finish."""
    errors = []

    def work():
        """Opens, converts and closes the converter; stores any error for the test thread to report."""
        try:
            converter.open()
            try:
                converter.convert(bundle)
            finally:
                converter.close()
        except BaseException as error:  # reported on the test thread
            errors.append(error)

    converter.setup_main()
    try:
        thread = threading.Thread(target=work)
        thread.start()
        thread.join(60)
        assert not thread.is_alive()
    finally:
        converter.teardown_main()
    assert not errors, errors


def test_gl_output_matches_the_cuda_conversion(clip_path):
    """Proves the OpenGL conversion matches the CUDA one to within one level, with padding rows identical and the
    picture the right way up."""
    from pipeline.gl_prepare import GlConverter

    pointer, decoder, demuxer = first_frame_pointer(clip_path)
    letterbox = Letterbox(1920, 1080, 640)
    expected = torch.empty((1, 3, 640, 640), dtype=torch.float32, device="cuda:0")
    actual = torch.empty_like(expected)
    nv12 = wrap_nv12(pointer, 1920, 1080)
    nv12_to_input(nv12, letterbox, 1920, 1080, expected)
    torch.cuda.synchronize()

    _convert_on_prepare_thread(GlConverter(letterbox, (1920, 1080), 0), _Frame(pointer, actual))

    expected, actual = expected[0] * 255, actual[0] * 255
    difference = (actual - expected).abs()
    for channel in range(3):
        assert float(difference[channel].mean()) < 0.05
        assert float(difference[channel].max()) < 1.0
    for rows in (slice(0, TOP), slice(BOTTOM, 640)):
        assert torch.equal(actual[:, rows], expected[:, rows])
    flipped = (actual.flip(1) - expected).abs()
    assert float(flipped[:, TOP:BOTTOM].mean()) > 5.0  # the picture is the right way up


def test_pipeline_runs_with_gl_prepare(clip_path, model_path):
    """Proves a full run with the OpenGL prepare tracks all 611 frames and returns every bundle to the pool."""
    config = PipelineConfig(input=clip_path, model=model_path, loops=1, pace="max", display=False, prepare="gl")
    summary = run_pipeline(config)
    assert summary["counts"]["tracked"] == 611
    assert summary["pool"]["free"] == summary["pool"]["size"]


def test_gl_prepare_and_display_coexist_and_shut_down(clip_path, model_path, monkeypatch):
    """Proves the OpenGL prepare and the display window can run together and shut down cleanly when the window
    closes."""
    started = time.monotonic()
    monkeypatch.setattr("pipeline.display.glfw.window_should_close",
                        lambda window: time.monotonic() - started > 6.0)
    config = PipelineConfig(input=clip_path, model=model_path, loops=0, pace="realtime", display=True,
                            prepare="gl")
    summary = run_pipeline(config)
    assert summary["counts"]["displayed"] > 0
    assert summary["counts"]["tracked"] > 0
    assert summary["counts"]["tracked"] >= summary["counts"]["displayed"]
