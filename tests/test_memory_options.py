# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Pins the options the memory review kept (docs/memory-review.md). These run on the CPU with fakes."""
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import decoder as decoder_module
from pipeline.config import InstanceConfig, parse_instance_config
from pipeline.decoder import DecoderStage
from pipeline.fleet import FleetConfig
from pipeline.metrics import Metrics
from server import detector as detector_module
from server.config import ServerConfig


class Stop(Exception):
    """Raised by a fake to end the code under test once the value of interest has been captured."""


def test_session_disables_the_cpu_arena_and_memory_pattern(monkeypatch):
    """The detector's ONNX Runtime session is created with the CPU memory arena and memory pattern both off."""
    seen = {}

    def fake_session(path, sess_options=None, providers=None):
        """Record the session options and stop before any model is loaded."""
        seen["options"] = sess_options
        raise Stop

    monkeypatch.setattr(detector_module.ort, "InferenceSession", fake_session)
    with pytest.raises(Stop):
        detector_module._Session(Path("b1.onnx"), 1, 0)
    assert seen["options"].enable_cpu_mem_arena is False
    assert seen["options"].enable_mem_pattern is False


def test_decoder_asks_for_low_display_latency(monkeypatch):
    """The decoder stage creates its NVDEC decoder with the LOW display latency setting."""
    seen = {}

    class Fake:
        """A decoder stand-in that accepts frame unlocks and does nothing."""

        def UnlockFrame(self, pointer):
            """Ignore the unlock."""

    def create(**kwargs):
        """Record the decoder creation arguments and return the fake decoder."""
        seen.update(kwargs)
        return Fake()

    monkeypatch.setattr(decoder_module, "use_gpu", lambda gpu_id: None)
    monkeypatch.setattr(decoder_module.nvc, "CreateDecoder", create)
    stop = threading.Event()
    stop.set()
    stage = DecoderStage(SimpleNamespace(codec=None), queue.Queue(), None, queue.SimpleQueue(),
                         InstanceConfig(address="unused"), Metrics(), stop, [])
    stage.work()
    assert seen["latency"] == decoder_module.nvc.DisplayDecodeLatencyType.LOW


def test_smaller_pool_and_batch_sizes_are_the_defaults():
    """The instance pool size defaults to 6 and the server batch sizes to (1, 8)."""
    assert parse_instance_config(["--address", "a"]).pool_size == 6
    assert ServerConfig.__dataclass_fields__["batch_sizes"].default == (1, 8)


def test_every_launcher_defaults_to_batch_sizes_1_and_8():
    """The fleet config, run_fleet and bench all default to batch sizes 1 and 8."""
    import bench
    import run_fleet

    assert FleetConfig(streams=1).batch_sizes == "1,8"
    assert run_fleet.make_parser().parse_args([]).batch_sizes == "1,8"
    assert bench.make_parser().parse_args([]).batch_sizes == "1,8"
