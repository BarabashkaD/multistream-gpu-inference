#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Scaling table: aggregate throughput against stream count. Run with --help for the options."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pipeline.fleet import FleetConfig, run_fleet

V3 = Path(__file__).resolve().parent
COLUMNS = (("Streams", "streams"), ("Total FPS", "total_fps"), ("Slowest stream", "min_fps"), ("Dropped", "dropped"),
           ("Latency p50 ms", "latency_p50_ms"), ("Latency p95 ms", "latency_p95_ms"), ("GPU %", "gpu_percent"),
           ("NVDEC %", "nvdec_percent"), ("CPU %", "cpu_percent"), ("Infer ms", "infer_ms"),
           ("Prepare ms", "prepare_ms"), ("GPU memory MB", "gpu_memory_peak_mb"),
           ("Host free GB", "host_available_min_gb"), ("Shown %", "shown_percent"), ("Bottleneck", "bottleneck"),
           ("Stop", "stop_reason"))


def _cell(value) -> str:
    """Formats one table cell: a dash for None, one decimal for a float, plain text otherwise."""
    if value is None:
        return "-"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def markdown_table(rows: list[dict]) -> str:
    """Renders fleet result rows as a Markdown table with the COLUMNS headings, one line per row."""
    lines = ["| " + " | ".join(title for title, _ in COLUMNS) + " |",
             "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    lines += ["| " + " | ".join(_cell(row[key]) for _, key in COLUMNS) + " |" for row in rows]
    return "\n".join(lines)


def make_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser; the module docstring is its help text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", default="1,2,4,8,12,16")
    parser.add_argument("--pace", choices=("realtime", "max"), default="realtime")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--batch-sizes", default="1,8")
    return parser


def main() -> None:
    """Runs one fleet per stream count, printing each result row, and stops early when a run reports a stop reason.

    Writes each run's output under results/ and the collected rows to results/bench-<mode>.json. Needs a CUDA GPU
    and the models, like the fleet itself."""
    args = make_parser().parse_args()
    rows = []
    mode = f"{args.pace}-{'windows' if args.display else 'headless'}-b{args.batch_sizes.replace(',', '_')}"
    for streams in (int(item) for item in args.streams.split(",")):
        rows.append(run_fleet(FleetConfig(streams=streams, pace=args.pace, loops=args.loops, display=args.display,
                                          batch_sizes=args.batch_sizes, monitor_period_s=0.0,
                                          out_dir=V3 / "results" / f"n{streams}-{mode}")))
        print(markdown_table(rows[-1:]), flush=True)
        if rows[-1]["stop_reason"]:
            break
    (V3 / "results").mkdir(exist_ok=True)
    (V3 / "results" / f"bench-{mode}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
