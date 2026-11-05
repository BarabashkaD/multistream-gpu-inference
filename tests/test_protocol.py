# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of pipeline.protocol: the messages between instance and server, row encoding, keys and addresses."""
from multiprocessing import Pipe
import pickle
import socket
import sys

import numpy as np
import pytest

from pipeline.protocol import (Bye, Closed, Frame, Hello, Reject, Result, Start, auth_key, connect, default_address,
                               remove_stale_socket, rows_from_bytes)


@pytest.mark.parametrize("message", [
    Hello(3, 1920, 1080), Start({0: "person"}), Reject("no"), Frame(7, 0x1000, b"h" * 64), Frame(8, 0x1000, None),
    Result(7, b""), Bye({"tracked": 1}), Closed(),
])
def test_messages_survive_a_pipe(message):
    """Every message type arrives through a multiprocessing pipe equal to what was sent and of the same type."""
    left, right = Pipe()
    left.send(message)
    received = right.recv()
    assert received == message
    assert type(received) is type(message)


def test_rows_round_trip():
    """Rows converted to bytes and back are identical."""
    rows = np.arange(12, dtype=np.float32).reshape(2, 6)
    assert np.array_equal(rows_from_bytes(rows.tobytes()), rows)


def test_empty_rows_keep_their_shape():
    """Empty bytes decode to an array of shape (0, 6), not a flat empty array."""
    assert rows_from_bytes(b"").shape == (0, 6)


def test_connect_gives_up_after_its_timeout(make_address):
    """connect raises OSError when nobody listens, once its timeout has passed."""
    with pytest.raises(OSError):
        connect(make_address("nobody-is-listening-here"), timeout_s=0.2)


def test_messages_stay_small():
    """A Frame message pickles to under 120 bytes, so the per-frame message stays cheap."""
    assert len(pickle.dumps(Frame(1, 2 ** 40, None))) < 120

def test_auth_key_comes_from_the_environment(monkeypatch):
    """auth_key returns the value of TRAFFIC_INFERENCE_KEY as bytes."""
    monkeypatch.setenv("TRAFFIC_INFERENCE_KEY", "abc")
    assert auth_key() == b"abc"


def test_a_missing_key_is_a_clear_error(monkeypatch):
    """With the key variable unset, auth_key raises a RuntimeError that names the variable."""
    monkeypatch.delenv("TRAFFIC_INFERENCE_KEY")
    with pytest.raises(RuntimeError, match="TRAFFIC_INFERENCE_KEY"):
        auth_key()


def test_default_address_suits_the_platform(monkeypatch):
    """The default address is a named pipe on Windows and a socket file under /tmp elsewhere."""
    monkeypatch.setattr(sys, "platform", "win32")
    assert default_address().startswith("\\\\.\\pipe\\")
    monkeypatch.setattr(sys, "platform", "linux")
    assert default_address().startswith("/tmp/traffic-inference-") and default_address().endswith(".sock")


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket files are a Linux matter here")
def test_a_stale_socket_file_is_removed(tmp_path):
    """A socket file left behind with no listener is deleted by remove_stale_socket."""
    path = tmp_path / "s.sock"
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(path))
    listener.close()  # the file stays behind, as after a killed server
    remove_stale_socket(str(path))
    assert not path.exists()


def test_other_addresses_and_regular_files_are_left_alone(tmp_path):
    """Pipe names, missing paths and regular files are not touched, and nothing raises."""
    remove_stale_socket(r"\\.\pipe\something")
    remove_stale_socket(str(tmp_path / "missing.sock"))
    regular = tmp_path / "notes.txt"
    regular.write_text("keep")
    remove_stale_socket(str(regular))
    assert regular.exists()
