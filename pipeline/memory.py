# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""A process's own memory, in MB, comparable across runs on one platform."""
from __future__ import annotations

import sys

import psutil


def private_mb(process) -> float:
    """Windows: private bytes, as Task Manager's commit figure. Linux: unique set size, or resident size when the
    kernel does not let this process read its unique set size."""
    if sys.platform == "win32":
        return process.memory_info().private / 2**20
    try:
        return process.memory_full_info().uss / 2**20
    except psutil.AccessDenied:
        return process.memory_info().rss / 2**20
