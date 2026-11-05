# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU soak test: a long 16-stream run must not grow the memory of the server or an instance.

It needs a CUDA device and is marked `gpu`.
"""
import json
from pathlib import Path

import pytest

from pipeline.fleet import FleetConfig, run_fleet

pytestmark = pytest.mark.gpu
MAX_GROWTH_MB = 100


def rss_series(log: Path) -> list[float]:
    """Return the memory_mb value of every "period" line in a stats log, in order.

    Lines that are not JSON objects (for example warnings) and lines without a memory figure are skipped.
    """
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
    return [line["memory_mb"] for line in lines if line.get("kind") == "period" and "memory_mb" in line]


def test_sixteen_streams_hold_memory_flat(tmp_path):
    """Sixteen streams for twelve loops finish cleanly, and the server and one instance gain under 100 MB.

    The first quarter of each series is dropped as warm-up, then the peak of the last ten samples is compared
    with the peak of the first ten.
    """
    result = run_fleet(FleetConfig(streams=16, loops=12, display=False, out_dir=tmp_path))
    assert result["stop_reason"] is None and result["failed"] == 0
    for name in ("server", "instance15"):
        series = rss_series(tmp_path / f"{name}.log")
        assert len(series) > 100
        settled = series[len(series) // 4:]
        assert max(settled[-10:]) - max(settled[:10]) < MAX_GROWTH_MB, f"{name} grew: {settled[:3]} {settled[-3:]}"
