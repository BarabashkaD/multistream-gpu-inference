# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of how an instance leaves the server (pipeline.app): the farewell handshake and stage release.

They run on the CPU with pipes and fake stages.
"""
from multiprocessing import Pipe
import threading
from types import SimpleNamespace

import pytest

import pipeline.app as app
from pipeline.protocol import Bye, Closed, Reject


def test_closed_means_the_server_has_let_go():
    """After the server sends Closed, _leave_server returns True and has sent Bye first."""
    near, far = Pipe()
    near.send(Closed())
    assert app._leave_server(far, {}) is True
    assert isinstance(near.recv(), Bye)


def test_reject_before_closed_is_read_past():
    """A Reject that arrives before Closed is skipped, and the wait still ends with True."""
    near, far = Pipe()
    near.send(Reject("no"))
    near.send(Closed())
    assert app._leave_server(far, {}) is True


def test_a_broken_pipe_means_the_server_has_gone():
    """When the server end is closed, _leave_server returns True, since the server holds nothing any more."""
    near, far = Pipe()
    near.close()
    assert app._leave_server(far, {}) is True


def test_silence_is_not_permission(monkeypatch):
    """If the server says nothing within the farewell timeout, _leave_server returns False."""
    monkeypatch.setattr(app, "_FAREWELL_TIMEOUT_S", 0.05)
    near, far = Pipe()
    assert app._leave_server(far, {}) is False


class FakeStage:
    """A stage that does nothing: it starts, joins and reports as not alive."""

    def __init__(self, *args, **kwargs):
        """Accept any stage arguments and give the stage a name."""
        self.name = "fake"

    def start(self):
        """Do nothing, as there is no thread to start."""

    def join(self, timeout_s=None):
        """Return at once, as there is no thread to wait for."""

    def is_alive(self):
        """Report that the stage has already finished."""
        return False


@pytest.fixture
def fakes(monkeypatch):
    """Replace everything run_instance touches (GPU, source, connection, stages) with fakes.

    Returns a namespace with `calls`, the stats passed to each farewell, and `leave`, the fake farewell whose
    `answer` attribute sets what it returns.
    """
    calls = []
    monkeypatch.setattr(app, "_stuck_stages", [])
    monkeypatch.setattr(app, "use_gpu", lambda gpu_id: None)
    monkeypatch.setattr(app, "MemorySource", lambda path: SimpleNamespace(width=64, height=48))
    monkeypatch.setattr(app, "connect", lambda address, timeout_s: object())
    monkeypatch.setattr(app, "_join_server", lambda conn, config, width, height: {})
    monkeypatch.setattr(app, "TrackerCore", lambda *args, **kwargs: None)
    for name in ("ReaderStage", "DecoderStage", "SubmitStage", "CollectStage", "TrackerStage"):
        monkeypatch.setattr(app, name, FakeStage)

    def leave(conn, stats):
        """Record the stats and return the configured answer."""
        calls.append(stats)
        return leave.answer

    leave.answer = True
    monkeypatch.setattr(app, "_leave_server", leave)
    return SimpleNamespace(calls=calls, leave=leave)


def config():
    """Return an instance config with the display off and a long report interval."""
    return app.InstanceConfig(address="unused", display=False, report_interval_s=60.0)


def test_stages_stay_pinned_unless_the_server_let_go(fakes, monkeypatch):
    """If the server did not confirm the close, all five stages are kept alive in _stuck_stages."""
    monkeypatch.setattr(app, "_wait", lambda *args: None)
    fakes.leave.answer = False
    app.run_instance(config())
    assert len(app._stuck_stages) == 5
    assert len(fakes.calls) == 1


def test_stages_are_released_once_the_server_let_go(fakes, monkeypatch):
    """Once the server confirms the close, nothing is kept, the final snapshot has memory and no reporter lingers."""
    monkeypatch.setattr(app, "_wait", lambda *args: None)
    fakes.leave.answer = True
    snapshot = app.run_instance(config())
    assert isinstance(snapshot["memory_mb"], int)
    assert app._stuck_stages == []
    assert not any(t.name == "reporter" and t.is_alive() for t in threading.enumerate())


def test_a_failing_wait_still_says_goodbye(fakes, monkeypatch):
    """An exception while waiting is re-raised, but the farewell is still sent and the stages stay pinned."""
    def wait(*args):
        """Raise ValueError in place of the real wait."""
        raise ValueError("boom")

    monkeypatch.setattr(app, "_wait", wait)
    fakes.leave.answer = False
    with pytest.raises(ValueError):
        app.run_instance(config())
    assert len(fakes.calls) == 1
    assert len(app._stuck_stages) == 5
