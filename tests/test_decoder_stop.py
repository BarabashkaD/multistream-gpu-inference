# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests that the decoder stage unlocks the frames that already came back when it stops, without needing a GPU."""
import queue
import threading
from types import SimpleNamespace

from pipeline import decoder as decoder_module
from pipeline.config import PipelineConfig
from pipeline.decoder import DecoderStage
from pipeline.metrics import Metrics


class FakeDecoder:
    """Stand-in for the hardware decoder that only records which frames were unlocked."""

    def __init__(self):
        """Starts with no unlocked frames."""
        self.unlocked = []

    def UnlockFrame(self, pointer):
        """Records that the frame at this pointer was unlocked."""
        self.unlocked.append(pointer)


def test_stop_unlocks_frames_that_already_came_back(monkeypatch):
    """Proves that on stop every frame already returned to the decoder is unlocked and none stays counted."""
    fake = FakeDecoder()
    monkeypatch.setattr(decoder_module.nvc, "CreateDecoder", lambda **kwargs: fake)
    stop = threading.Event()
    stop.set()
    returned = queue.SimpleQueue()
    returned.put(1)
    returned.put(2)
    stage = DecoderStage(SimpleNamespace(codec=None), queue.Queue(), None, returned, PipelineConfig(), Metrics(),
                         stop, [])
    stage.locked = 2
    stage.work()
    assert fake.unlocked == [1, 2]
    assert stage.locked == 0
