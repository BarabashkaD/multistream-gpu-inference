# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Messages between instances and the inference server. One named pipe per instance."""
from __future__ import annotations

from multiprocessing.connection import Client, Connection
import os
import stat
import sys
import time
from typing import NamedTuple

import numpy as np

AUTH_KEY_VARIABLE = "TRAFFIC_INFERENCE_KEY"
_RETRY_S = 0.05


class Hello(NamedTuple):
    """First message from an instance: its stream number and the frame size it will send."""

    stream: int
    width: int
    height: int


class Start(NamedTuple):
    """The server's acceptance of a stream, with the class names of the model."""

    names: dict


class Reject(NamedTuple):
    """The server's refusal of a stream, or its dropping of one, with the reason."""

    reason: str


class Frame(NamedTuple):
    """A decoded frame offered for inference: its id and the decoder memory pointer holding it."""

    frame_id: int
    pointer: int
    handle: bytes | None  # the IPC handle, sent only the first time a decoder pointer appears


class Result(NamedTuple):
    """The detections for one frame, matched to its Frame by id."""

    frame_id: int
    rows: bytes  # float32, N x 6: x1, y1, x2, y2, confidence, class, in source coordinates


class Bye(NamedTuple):
    """The instance's last message: it is done and reports its final statistics."""

    stats: dict


class Closed(NamedTuple):
    """The server's reply to Bye: it holds no mapping of the instance's frame memory any more."""

    pass


def default_address() -> str:
    """A Windows named pipe, or a Unix socket path elsewhere."""
    if sys.platform == "win32":
        return rf"\\.\pipe\traffic-inference-{os.getpid()}"
    return f"/tmp/traffic-inference-{os.getpid()}.sock"


def remove_stale_socket(address: str) -> None:
    """A killed server leaves its socket file, and binding the same path then fails. Only a socket is removed."""
    if address.startswith("\\\\.\\pipe\\"):
        return
    try:
        if stat.S_ISSOCK(os.stat(address).st_mode):
            os.unlink(address)
    except FileNotFoundError:
        pass


def auth_key() -> bytes:
    """The per-run secret. It travels by inherited environment, never on a command line, which other users can read."""
    value = os.environ.get(AUTH_KEY_VARIABLE)
    if not value:
        raise RuntimeError(f"{AUTH_KEY_VARIABLE} is not set; run_fleet.py sets it, or set it yourself to a random value")
    return value.encode("ascii")


def connect(address: str, timeout_s: float) -> Connection:
    """Connect to the server at address with the shared key, retrying refused or missing endpoints until timeout_s.

    The last OSError is raised once the time is up. AuthenticationError is not an OSError, so a wrong key is not
    retried.
    """
    key = auth_key()
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return Client(address, authkey=key)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_RETRY_S)


def rows_from_bytes(data: bytes) -> np.ndarray:
    """View result bytes as an N x 6 float32 array of detection rows, without copying."""
    return np.frombuffer(data, np.float32).reshape(-1, 6)
