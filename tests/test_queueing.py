# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of server.queueing.FairQueue: fair batching of frames across streams."""
import threading
import time

from pipeline.protocol import Frame
from server.queueing import FairQueue


def frame(number):
    """Return a frame message with the given id, no GPU pointer and no handle."""
    return Frame(number, 0, None)


def test_one_stream_stays_in_order():
    """One stream's frames come out oldest first, and a batch takes only as many as the largest size allows."""
    queue = FairQueue()
    for number in range(3):
        queue.add(0, frame(number))
    assert [item.frame_id for _, item in queue.take((1, 2), 0, 0.01, streams=1)] == [0, 1]
    assert len(queue) == 1


def test_a_busy_stream_cannot_fill_every_batch():
    """With one stream holding many frames, other streams that have a frame still appear in the batch."""
    queue = FairQueue()
    for number in range(8):
        queue.add(0, frame(number))
    queue.add(1, frame(100))
    queue.add(2, frame(200))
    streams = [stream for stream, _ in queue.take((1, 4), 0, 0.01, streams=3)]
    assert sorted(set(streams)) == [0, 1, 2]


def test_the_starting_stream_rotates():
    """The stream served first changes from batch to batch, so no stream always has priority."""
    queue = FairQueue()
    starts = []
    for _ in range(4):
        queue.add(0, frame(0))
        queue.add(1, frame(1))
        starts.append(queue.take((1,), 0, 0.01, streams=2)[0][0])
        queue.remove_stream(0)
        queue.remove_stream(1)
    assert set(starts) == {0, 1}


def test_it_takes_every_waiting_frame_up_to_the_largest_size():
    """A batch takes all waiting frames when they fit in the largest size, and only that many when they do not."""
    queue = FairQueue()
    for number in range(5):
        queue.add(number, frame(number))
    assert len(queue.take((1, 8), 0, 0.01, streams=5)) == 5
    assert len(queue) == 0
    for number in range(10):
        queue.add(number, frame(number))
    assert len(queue.take((1, 8), 0, 0.01, streams=10)) == 8
    assert len(queue) == 2


def test_it_gathers_until_every_stream_has_a_frame():
    """take waits for the late stream's frame and returns as soon as every stream has one, well before its limit."""
    queue = FairQueue()
    queue.add(0, frame(0))
    threading.Timer(0.02, lambda: queue.add(1, frame(1))).start()
    started = time.monotonic()
    assert len(queue.take((1, 8), 0.5, 0.01, streams=2)) == 2
    assert time.monotonic() - started < 0.4


def test_a_full_batch_does_not_wait_for_the_other_streams():
    """When the largest batch is already available, take returns at once even though other streams are missing."""
    queue = FairQueue()
    for number in range(8):
        queue.add(0, frame(number))
    started = time.monotonic()
    assert len(queue.take((1, 8), 5.0, 0.01, streams=16)) == 8
    assert time.monotonic() - started < 0.5


def test_gathering_gives_up_after_its_limit():
    """If a stream never delivers, take returns what it has once the gather limit has passed, and not before."""
    queue = FairQueue()
    queue.add(0, frame(0))
    started = time.monotonic()
    assert len(queue.take((1, 8), 0.05, 0.01, streams=4)) == 1
    assert 0.04 < time.monotonic() - started < 0.5


def test_it_records_how_many_frames_were_waiting():
    """last_available holds the number of frames that were queued when the last batch was taken."""
    queue = FairQueue()
    assert queue.last_available == 0
    for number in range(10):
        queue.add(number, frame(number))
    queue.take((1, 8), 0, 0.01, streams=10)
    assert queue.last_available == 10
    queue.take((1, 8), 0, 0.01, streams=2)
    assert queue.last_available == 2
    for number in range(3):
        queue.add(number, frame(number))
    queue.take((1, 8), 0, 0.01, streams=3)
    assert queue.last_available == 3


def test_an_empty_queue_returns_nothing_after_the_timeout():
    """take on an empty queue returns an empty list when its wait ends."""
    assert FairQueue().take((1,), 0, 0.01, streams=1) == []


def test_removing_a_stream_drops_its_frames():
    """remove_stream discards the stream's queued frames, so only the other stream's frame is taken."""
    queue = FairQueue()
    queue.add(0, frame(0))
    queue.add(1, frame(1))
    queue.remove_stream(0)
    assert [stream for stream, _ in queue.take((1,), 0, 0.01, streams=1)] == [1]
