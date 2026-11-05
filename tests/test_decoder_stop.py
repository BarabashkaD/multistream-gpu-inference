# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests that the decoder stage unlocks the frames it can when it stops, without needing a GPU."""
import queue
import threading
from types import SimpleNamespace

from pipeline import decoder as decoder_module
from pipeline.config import InstanceConfig
from pipeline.decoder import DecoderStage
from pipeline.metrics import Metrics


class FakeDecoder:
    """Stand-in for the hardware decoder that only records which frames were unlocked."""

    def __init__(self):
        """Start with no unlocked frames."""
        self.unlocked = []

    def UnlockFrame(self, pointer):
        """Record that the frame at this pointer was unlocked."""
        self.unlocked.append(pointer)


def test_stop_unlocks_frames_that_already_came_back(monkeypatch):
    """Proves that on stop every frame already returned to the decoder is unlocked and none stays counted."""
    fake = FakeDecoder()
    monkeypatch.setattr(decoder_module, "use_gpu", lambda gpu_id: None)
    monkeypatch.setattr(decoder_module.nvc, "CreateDecoder", lambda **kwargs: fake)
    stop = threading.Event()
    stop.set()
    returned = queue.SimpleQueue()
    returned.put(1)
    returned.put(2)
    stage = DecoderStage(SimpleNamespace(codec=None), queue.Queue(), None, returned, InstanceConfig(address="unused"),
                         Metrics(), stop, [])
    stage.locked = 2
    stage.work()
    assert fake.unlocked == [1, 2]
    assert stage.locked == 0


def test_a_frame_still_held_elsewhere_stays_locked(monkeypatch):
    """Proves a frame that has not come back is left locked on stop, because another stage may still read it."""
    fake = FakeDecoder()
    monkeypatch.setattr(decoder_module, "use_gpu", lambda gpu_id: None)
    monkeypatch.setattr(decoder_module.nvc, "CreateDecoder", lambda **kwargs: fake)
    stop = threading.Event()
    stop.set()
    returned = queue.SimpleQueue()
    returned.put(1)
    stage = DecoderStage(SimpleNamespace(codec=None), queue.Queue(), None, returned, InstanceConfig(address="unused"),
                         Metrics(), stop, [])
    stage.locked = 2  # pointer 2 never comes back
    stage.work()
    assert fake.unlocked == [1]
    assert stage.locked == 1
