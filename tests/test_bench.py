# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for the benchmark's Markdown results table."""
from bench import markdown_table


def row(streams, fps):
    """Build a complete benchmark result row for the given stream count and total frame rate."""
    return {"streams": streams, "total_fps": fps, "min_fps": 25.0, "dropped": 0, "latency_p50_ms": 50.0,
            "latency_p95_ms": 90.0, "gpu_percent": 40, "nvdec_percent": 20, "cpu_percent": 30,
            "infer_ms": 1.5, "prepare_ms": 0.4, "gpu_memory_peak_mb": 4000, "host_available_min_gb": 12.5,
            "bottleneck": "none", "stop_reason": None, "failed": 0, "shown_percent": None}


def test_table_shows_the_share_of_frames_shown_or_a_dash_headless():
    """Proves the Shown % column prints the percentage when known and a dash when the run was headless."""
    lines = markdown_table([row(1, 25.0) | {"shown_percent": 87}, row(2, 50.0)]).splitlines()
    titles = [cell.strip() for cell in lines[0].strip("|").split("|")]
    column = titles.index("Shown %")
    assert [cell.strip() for cell in lines[2].strip("|").split("|")][column] == "87"
    assert [cell.strip() for cell in lines[3].strip("|").split("|")][column] == "-"


def test_table_has_a_row_per_run_and_shows_the_stop_reason():
    """Proves the table has a header, a separator and one line per run, and that a stop reason is printed."""
    stopped = row(16, 100.0) | {"stop_reason": "GPU memory limit"}
    lines = markdown_table([row(1, 25.0), stopped]).splitlines()
    assert lines[0].startswith("| Streams |")
    assert len(lines) == 4
    assert "| 1 |" in lines[2]
    assert "GPU memory limit" in lines[3]
