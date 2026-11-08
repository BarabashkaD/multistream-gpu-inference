# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of server.service.InferenceServer: admission, batching, replies, failures and shutdown.

The detector, preparer and frame imports are replaced by fakes, so these run on the CPU without CUDA.
"""
from multiprocessing import AuthenticationError, Pipe
from multiprocessing.connection import Client
import threading
import time

import numpy as np
import pytest
import torch

from pipeline.metrics import Metrics
from pipeline.protocol import Bye, Closed, Frame, Hello, Reject, Result, Start, connect, rows_from_bytes
from server.config import ServerConfig
from server.service import InferenceServer, run


class FakeDetector:
    """A detector that returns two fixed rows per frame, one confident and one below the confidence limit."""

    names = {0: "person"}

    def __init__(self):
        """Start with no input tensors and no recorded runs."""
        self.inputs = {}
        self.runs = []

    def input(self, size):
        """Return the (cached) zero input tensor for a batch of the given size."""
        return self.inputs.setdefault(size, torch.zeros((size, 3, 4, 4)))

    def run(self, size):
        """Record the batch size and return `size` frames of detection rows, 300 rows each."""
        self.runs.append(size)
        rows = np.zeros((size, 300, 6), np.float32)
        rows[:, 0] = [10, 20, 30, 40, 0.9, 1]
        rows[:, 1] = [1, 2, 3, 4, 0.05, 1]  # below min_conf
        return rows


class FakePreparer:
    """A preparer that records the batches it is given and scales boxes back to the source by three."""

    def __init__(self):
        """Start with no recorded batches."""
        self.batches = []

    def convert(self, frames, output):
        """Record the frames of one batch and leave the output as it is."""
        self.batches.append(list(frames))

    def to_source(self, boxes):
        """Map boxes to source coordinates by multiplying by three."""
        return boxes * 3


class FakeImports:
    """A frame-import stand-in that records the requests and the closed streams."""

    def __init__(self):
        """Start with nothing seen or closed."""
        self.closed = []
        self.seen = []

    def view(self, stream, pointer, handle):
        """Record the request and return a placeholder tensor."""
        self.seen.append((stream, pointer, handle))
        return torch.zeros(1)

    def close_stream(self, stream):
        """Record that the stream was closed."""
        self.closed.append(stream)


def build(clients=1, sizes=(1, 2)):
    """Create a server with fakes and admit `clients` streams that have already sent Hello.

    Returns (server, the instance ends of the pipes, detector, preparer, imports).
    """
    preparer, imports, detector = FakePreparer(), FakeImports(), FakeDetector()
    config = ServerConfig(address="unused", clients=clients, batch_sizes=sizes, gather_ms=0.0)
    server = InferenceServer(config, detector, lambda width, height: preparer, imports, Metrics(), threading.Event())
    ends = []
    for stream in range(clients):
        near, far = Pipe()
        near.send(Hello(stream, 64, 36))
        server.admit(far)
        ends.append(near)
    return server, ends, detector, preparer, imports


def test_start_goes_out_only_when_all_have_joined():
    """Start is sent to every stream only after the last expected client has said Hello."""
    near, far = Pipe()
    config = ServerConfig(address="unused", clients=2, batch_sizes=(1,))
    server = InferenceServer(config, FakeDetector(), None, FakeImports(), Metrics(), threading.Event())
    near.send(Hello(0, 64, 36))
    server.admit(far)
    assert not near.poll(0.05)
    other_near, other_far = Pipe()
    other_near.send(Hello(1, 64, 36))
    server.admit(other_far)
    assert near.recv() == Start({0: "person"})
    assert other_near.recv() == Start({0: "person"})


def test_a_frame_of_another_size_is_rejected():
    """A client whose Hello announces a different frame size gets a Reject."""
    server, ends, *_ = build(clients=2)
    other_near, other_far = Pipe()
    other_near.send(Hello(9, 32, 18))
    server.admit(other_far)
    assert isinstance(other_near.recv(), Reject)


def test_a_duplicate_stream_is_rejected():
    """A second Hello for a stream number that is already admitted gets a Reject."""
    server, ends, *_ = build(clients=2)
    near, far = Pipe()
    near.send(Hello(0, 64, 36))
    server.admit(far)
    assert isinstance(near.recv(), Reject)


def test_a_frame_returns_rows_above_min_conf_in_source_coordinates():
    """A frame's Result holds only rows above the confidence limit, mapped to source coordinates."""
    server, [end], detector, preparer, imports = build()
    end.recv()
    server.handle(0, Frame(5, 0x10, b"h"))
    server.step()
    result = end.recv()
    assert isinstance(result, Result) and result.frame_id == 5
    rows = rows_from_bytes(result.rows)
    assert rows.shape == (1, 6)
    assert rows[0].tolist() == pytest.approx([30, 60, 90, 120, 0.9, 1])
    assert imports.seen == [(0, 0x10, b"h")]


def test_frames_from_two_streams_share_a_batch():
    """Frames waiting from two streams run as one batch of two, and each stream gets its own Result."""
    server, ends, detector, preparer, _ = build(clients=2)
    for end in ends:
        end.recv()
    server.handle(0, Frame(1, 0x10, b"a"))
    server.handle(1, Frame(1, 0x20, b"b"))
    server.step()
    assert detector.runs == [2]
    assert len(preparer.batches[0]) == 2
    assert all(isinstance(end.recv(), Result) for end in ends)


def test_a_partial_batch_is_padded_and_only_real_rows_are_sent():
    """Three frames run in a batch of eight with five copies, and only the three real frames get replies."""
    server, ends, detector, preparer, _ = build(clients=3, sizes=(1, 8))
    for end in ends:
        end.recv()
    for stream in range(3):
        server.handle(stream, Frame(stream + 1, 0x10 * (stream + 1), b"h"))
    server.step()
    assert detector.runs == [8]
    assert len(preparer.batches[0]) == 8
    replies = [end.recv() for end in ends]
    assert [reply.frame_id for reply in replies] == [1, 2, 3]
    assert not any(end.poll(0.1) for end in ends)  # no reply for a copy
    snapshot = server.summary()
    assert snapshot["counts"]["padded"] == 5
    assert snapshot["counts"]["inferred"] == 3
    assert snapshot["batches"] == {"8": 1}


def test_a_lone_frame_runs_at_the_smallest_size():
    """A single waiting frame runs at the smallest configured batch size, not the largest."""
    server, [end], detector, _, _ = build(clients=1, sizes=(1, 8))
    end.recv()
    server.handle(0, Frame(1, 0x10, b"h"))
    server.step()
    assert detector.runs == [1]


def test_bye_closes_handles_before_closed_is_sent():
    """On Bye the server closes the stream's mappings first, then sends Closed and records the stream's stats."""
    server, [end], _, _, imports = build()
    end.recv()
    order = []

    def close_stream(stream):
        """Check that Closed has not been sent yet, then record the stream."""
        assert not end.poll()  # the mapping must be gone before Closed is sent
        order.append(stream)

    imports.close_stream = close_stream
    server.handle(0, Bye({"tracked": 3}))
    server.step()
    assert order == [0]
    assert end.recv() == Closed()
    assert server.done()
    assert server.summary()["streams"] == [{"tracked": 3}]


def test_a_dead_stream_is_dropped_and_others_carry_on():
    """Retiring a stream that has a queued frame drops that frame, and the other stream is still served."""
    server, ends, detector, _, imports = build(clients=2)
    for end in ends:
        end.recv()
    server.handle(0, Frame(1, 0x10, b"a"))
    server.handle(1, Frame(1, 0x20, b"b"))
    ends[0].close()
    server.retire(0, None)
    server.step()
    assert imports.closed == [0]
    assert detector.runs == [1]
    assert ends[1].recv().frame_id == 1
    assert server.summary()["dead"] == [0]
    assert not server.done()


def test_a_result_to_a_vanished_stream_does_not_stop_the_server():
    """A failed send to a client that has gone away is ignored, and the server keeps serving the others."""
    server, ends, *_ = build(clients=2)
    for end in ends:
        end.recv()
    server.handle(0, Frame(1, 0x10, b"a"))
    ends[0].close()
    server.step()
    server.handle(1, Frame(2, 0x20, b"b"))
    server.step()
    assert ends[1].recv().frame_id == 2


def test_a_frame_after_retire_is_ignored_and_retiring_twice_is_harmless():
    """Frames for a retired stream are ignored, and a second retire neither raises nor closes it again."""
    server, [end], detector, _, imports = build()
    end.recv()
    server.retire(0, None)
    server.retire(0, None)
    server.handle(0, Frame(1, 0x10, None))
    server.step()
    assert detector.runs == []
    assert imports.closed == [0]
    assert server.summary()["dead"] == [0]


@pytest.mark.parametrize("error", [RuntimeError("cuIpcOpenMemHandle failed"),
                                   BufferError("unsupported device")])
def test_a_handle_that_will_not_open_drops_only_that_stream(error):
    """If importing a stream's frame fails with either error, only that stream is dropped and the batch still runs."""
    server, ends, detector, _, imports = build(clients=2)
    for end in ends:
        end.recv()

    def view(stream, pointer, handle):
        """Fail for stream 0 and succeed for the others."""
        if stream == 0:
            raise error
        return torch.zeros(1)

    imports.view = view
    server.handle(0, Frame(1, 0x10, b"a"))
    server.handle(1, Frame(1, 0x20, b"b"))
    server.step()
    assert detector.runs == [2]
    assert ends[1].recv().frame_id == 1
    server.step()
    assert server.summary()["dead"] == [0]


def test_a_failed_handle_is_reported_and_tells_the_instance(capsys):
    """A failed import is recorded in the summary, printed to stderr and sent to the instance as Reject then Closed."""
    server, ends, detector, _, imports = build(clients=2)
    for end in ends:
        end.recv()

    def view(stream, pointer, handle):
        """Fail for stream 0 and succeed for the others."""
        if stream == 0:
            raise RuntimeError("cuIpcOpenMemHandle failed")
        return torch.zeros(1)

    imports.view = view
    server.handle(0, Frame(1, 0x10, b"a"))
    server.handle(1, Frame(1, 0x20, b"b"))
    server.step()
    server.step()
    assert ends[0].poll(1.0)
    assert isinstance(ends[0].recv(), Reject)
    assert ends[0].poll(1.0)
    assert ends[0].recv() == Closed()
    assert server.summary()["errors"] == [{"stream": 0, "error": "cuIpcOpenMemHandle failed"}]
    assert "cuIpcOpenMemHandle failed" in capsys.readouterr().err


def test_a_failed_close_does_not_stop_the_server():
    """An error while closing a retired stream's mappings is recorded and the other streams keep running."""
    server, ends, _, _, imports = build(clients=2)
    for end in ends:
        end.recv()

    def close_stream(stream):
        """Fail as a broken CUDA close would."""
        raise RuntimeError("cuIpcCloseMemHandle failed")

    imports.close_stream = close_stream
    server.retire(0, None)
    server.step()
    server.handle(1, Frame(2, 0x20, b"b"))
    server.step()
    assert ends[1].recv().frame_id == 2
    assert server.summary()["errors"][0]["stream"] == 0


def test_a_mapping_the_server_kept_is_not_reported_closed():
    """If closing fails on Bye, the instance gets a Reject with the reason and no Closed, so it keeps its memory."""
    server, [end], _, _, imports = build()
    end.recv()

    def close_stream(stream):
        """Fail as a broken CUDA close would."""
        raise RuntimeError("cuIpcCloseMemHandle failed")

    imports.close_stream = close_stream
    server.handle(0, Bye({"tracked": 3}))
    server.step()
    assert end.poll(1.0)
    reply = end.recv()
    assert isinstance(reply, Reject) and "cuIpcCloseMemHandle failed" in reply.reason
    assert not end.poll(0.2)  # no Closed: the instance must keep its decoder memory


def test_a_frame_from_a_stream_retired_after_the_take_is_skipped(capsys):
    """A frame taken just before its stream is retired is skipped quietly, with no error and no output."""
    server, ends, detector, _, imports = build(clients=2)
    for end in ends:
        end.recv()
    take = server.queue.take

    def take_then_retire(*args):
        """Take a batch, then retire stream 0 as the receive thread could while the batch formed."""
        picks = take(*args)
        server.retire(0, None)  # the receive thread saw stream 0 leave while the batch was forming
        return picks

    server.queue.take = take_then_retire
    server.handle(0, Frame(1, 0x10, b"a"))
    server.handle(1, Frame(1, 0x20, b"b"))
    server.step()
    assert [stream for stream, _, _ in imports.seen] == [1]
    assert detector.runs == [2]
    assert ends[1].poll(1.0)
    assert ends[1].recv().frame_id == 1
    assert server.summary()["errors"] == []
    assert capsys.readouterr().err == ""


def test_stop_is_set_when_run_fails(make_address):
    """When run fails with a start timeout, it re-raises TimeoutError and sets the server's stop event."""
    config = ServerConfig(address=make_address("stop"), clients=1, batch_sizes=(1,),
                          start_timeout_s=0.1)
    server = InferenceServer(config, FakeDetector(), None, FakeImports(), Metrics(), threading.Event())
    with pytest.raises(TimeoutError):
        run(server)
    assert server.stop.is_set()


def test_a_stream_that_left_cannot_rejoin():
    """A Hello for a stream number that has already left is answered with a Reject."""
    server, ends, *_ = build(clients=2)
    for end in ends:
        end.recv()
    server.retire(0, None)
    server.step()
    near, far = Pipe()
    near.send(Hello(0, 64, 36))
    server.admit(far)
    assert near.poll(1.0)
    assert isinstance(near.recv(), Reject)


def test_a_first_message_that_is_not_hello_is_dropped():
    """A connection whose first message is not Hello is dropped and does not count as joined."""
    server, ends, *_ = build(clients=2)
    near, far = Pipe()
    near.send(Closed())
    server.admit(far)
    assert server.joined == 2


def test_two_clients_run_through_a_real_listener_to_a_clean_finish(make_address):
    """Two clients connect to a real listener, exchange 20 frames each and close; the run ends with a clean summary."""
    address = make_address("clean")
    config = ServerConfig(address=address, clients=2, batch_sizes=(1, 2), gather_ms=1.0)
    preparer = FakePreparer()
    server = InferenceServer(config, FakeDetector(), lambda width, height: preparer, FakeImports(), Metrics(),
                             threading.Event())
    summaries = []
    thread = threading.Thread(target=lambda: summaries.append(run(server)), daemon=True)
    thread.start()
    conns = []
    for stream in range(2):  # admit() reads the Hello before the next accept, so send it at once
        conns.append(connect(address, 5.0))
        conns[-1].send(Hello(stream, 64, 36))
    for conn in conns:
        assert isinstance(conn.recv(), Start)
    for number in range(20):
        for conn in conns:
            conn.send(Frame(number, 0x10, b"h" if number == 0 else None))
        for conn in conns:
            assert conn.recv().frame_id == number
    for stream, conn in enumerate(conns):
        conn.send(Bye({"stream": stream}))
        assert conn.recv() == Closed()
    thread.join(5.0)
    assert not thread.is_alive()
    assert summaries[0]["counts"]["inferred"] == 40
    assert summaries[0]["dead"] == [] and summaries[0]["errors"] == []
    assert not any(t.name == "report" and t.is_alive() for t in threading.enumerate())


def test_the_summary_records_the_process_memory():
    """The summary carries the server's memory use as a positive whole number of megabytes."""
    memory = build()[0].summary()["memory_mb"]
    assert isinstance(memory, int) and memory > 0


def test_batch_counts_are_a_copy_keyed_by_size():
    """batch_counts returns a snapshot keyed by batch size as text, unaffected by later counting."""
    server = build()[0]
    server.batches[8] += 2
    counts = server.batch_counts()
    server.batches[8] += 1
    assert counts == {"8": 2}


def test_the_frames_available_for_each_batch_are_counted():
    """available_counts tallies how many frames were waiting per batch, grouping large counts as "8+"."""
    server, ends, *_ = build(clients=3, sizes=(1, 8))
    for end in ends:
        end.recv()
    for stream in range(3):
        server.handle(stream, Frame(stream + 1, 0x10, b"h"))
    server.step()
    assert server.available_counts() == {"3": 1}
    for number in range(3):
        for stream in range(3):
            server.handle(stream, Frame(10 + number, 0x10, b"h"))
    server.step()
    counts = server.available_counts()
    assert counts == {"3": 1, "8+": 1}
    assert server.summary()["available"] == counts
    server.available["3"] += 1
    assert counts == {"3": 1, "8+": 1}


def test_a_batch_that_never_ran_is_not_counted_as_available():
    """When every frame of a batch fails to import, no availability is counted for it."""
    server, ends, _, _, imports = build(clients=1)
    ends[0].recv()

    def view(stream, pointer, handle):
        """Fail for every request."""
        raise RuntimeError("cuIpcOpenMemHandle failed")

    imports.view = view
    server.handle(0, Frame(1, 0x10, b"a"))
    server.step()
    assert server.available_counts() == {}


def test_an_instance_that_dies_before_hello_does_not_stop_the_others():
    """A connection that closes before sending Hello is ignored, and the joined count does not change."""
    server, ends, *_ = build(clients=2)
    near, far = Pipe()
    near.close()
    server.admit(far)
    assert server.joined == 2


def test_run_gives_up_when_instances_never_join(make_address):
    """run raises TimeoutError when the expected instances do not join within the start timeout."""
    config = ServerConfig(address=make_address("no-one"), clients=1, batch_sizes=(1,),
                          start_timeout_s=0.1)
    server = InferenceServer(config, FakeDetector(), None, FakeImports(), Metrics(), threading.Event())
    with pytest.raises(TimeoutError):
        run(server)


def test_a_dead_receive_thread_ends_the_run(make_address):
    """If the receive thread fails, run stops and re-raises its error instead of waiting for clients."""
    config = ServerConfig(address=make_address("dead-receive"), clients=1, batch_sizes=(1,))
    server = InferenceServer(config, FakeDetector(), None, FakeImports(), Metrics(), threading.Event())

    def broken():
        """Fail at once, as a crashed receive thread would."""
        raise ValueError("receive failed")

    server.receive = broken
    with pytest.raises(ValueError):
        run(server)


def test_receive_stops_on_the_stop_event_and_finds_a_closed_pipe():
    """The receive thread notices a closed client pipe, which finishes the run, and exits when stop is set."""
    server, [end], *_ = build()
    end.recv()
    thread = threading.Thread(target=server.receive, daemon=True)
    thread.start()
    end.close()
    deadline = threading.Event()
    for _ in range(100):
        server.step()
        if server.done():
            break
        deadline.wait(0.01)
    assert server.done()
    server.stop.set()
    thread.join(2.0)
    assert not thread.is_alive()


def test_a_peer_without_the_key_cannot_connect_and_the_server_keeps_accepting(make_address):
    """A peer using the wrong key is refused, and a client with the right key can still join and finish."""
    address = make_address("auth")
    config = ServerConfig(address=address, clients=1, batch_sizes=(1,), start_timeout_s=20.0)
    server = InferenceServer(config, FakeDetector(), None, FakeImports(), Metrics(), threading.Event())
    thread = threading.Thread(target=run, args=(server,), daemon=True)
    thread.start()
    refused = False
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not refused:
        try:
            Client(address, authkey=b"wrong key")
        except AuthenticationError:
            refused = True
        except OSError:  # the listener is not up yet
            time.sleep(0.02)
    assert refused, "a wrong key was never refused"
    conn = connect(address, 5.0)
    conn.send(Hello(0, 64, 36))
    assert conn.recv() == Start({0: "person"})
    conn.send(Bye({}))
    assert conn.recv() == Closed()
    thread.join(5.0)
    assert not thread.is_alive()
    assert server.joined == 1
