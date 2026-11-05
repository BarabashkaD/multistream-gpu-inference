# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The base worker thread of a pipeline stage and the stop-aware queue helpers the stages share."""
from __future__ import annotations

import queue
import threading
from typing import Any

END = object()
_WAIT_S = 0.05


def put_until(target: queue.Queue, item: Any, stop: threading.Event) -> bool:
    """Put the item on a bounded queue, retrying while it is full. Return False if stop is set first."""
    while not stop.is_set():
        try:
            target.put(item, timeout=_WAIT_S)
            return True
        except queue.Full:
            continue
    return False


def get_until(source: queue.Queue, stop: threading.Event) -> Any:
    """Take the next item from the queue, waiting while it is empty. Return None if stop is set first."""
    while not stop.is_set():
        try:
            return source.get(timeout=_WAIT_S)
        except queue.Empty:
            continue
    return None


class Stage(threading.Thread):
    """Worker thread that turns any exception into a pipeline-wide stop."""

    def __init__(self, name: str, stop: threading.Event, failures: list):
        """Create a daemon thread; `failures` collects (name, exception) pairs and `stop` is shared by all stages."""
        super().__init__(name=name, daemon=True)
        self.stop_event = stop
        self.failures = failures

    def run(self) -> None:
        """Run work(); on any exception record it in the failures and set the shared stop event."""
        try:
            self.work()
        except BaseException as exc:
            self.failures.append((self.name, exc))
            self.stop_event.set()

    def work(self) -> None:
        """The stage's body, implemented by each subclass."""
        raise NotImplementedError
