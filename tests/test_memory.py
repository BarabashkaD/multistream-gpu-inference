# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of pipeline.memory.private_mb, the per-process memory figure used by the monitor."""
import sys

import psutil
import pytest

from pipeline.memory import private_mb


@pytest.mark.skipif(sys.platform != "win32", reason="Windows reports private bytes")
def test_windows_figure_is_private_bytes():
    """On Windows the figure matches the process's private bytes in megabytes, within 5 percent."""
    process = psutil.Process()
    assert private_mb(process) == pytest.approx(process.memory_info().private / 2**20, rel=0.05)


def test_figure_is_positive_megabytes():
    """The figure for this process is a plausible number of megabytes, not bytes or zero."""
    assert 1 < private_mb(psutil.Process()) < 100_000


def test_linux_falls_back_to_rss_when_uss_is_denied(monkeypatch):
    """On Linux, when the unique set size cannot be read, private_mb returns the resident set size in megabytes."""
    class Denied:
        """A process whose full memory info is denied but whose basic info reports 3 MiB resident."""

        def memory_full_info(self):
            """Refuse, as psutil does for a process owned by another user."""
            raise psutil.AccessDenied(1)

        def memory_info(self):
            """Return an object with only the resident set size, 3 MiB."""
            return type("Info", (), {"rss": 3 * 2**20})()

    monkeypatch.setattr(sys, "platform", "linux")
    assert private_mb(Denied()) == 3.0
