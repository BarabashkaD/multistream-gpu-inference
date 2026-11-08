#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Start one inference server and N pipeline instances, then print the combined result.

The result is JSON on standard output and in summary.json in the output folder. The exit status is 1 when a safety
limit stopped the run or any instance failed, and 0 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pipeline.fleet import FleetConfig, run_fleet


def make_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser; the defaults for input and monitor period come from FleetConfig."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--pace", choices=("realtime", "max"), default="realtime")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--slow-stream", type=int, default=None, help="this stream's tracker is delayed")
    parser.add_argument("--slow-ms", type=float, default=0.0)
    parser.add_argument("--screen", default=None, help="WIDTHxHEIGHT, detected if omitted")
    parser.add_argument("--input", type=Path, default=FleetConfig.input, help="the clip every instance plays")
    parser.add_argument("--monitor-period-s", type=float, default=FleetConfig.monitor_period_s,
                        help="live table refresh in seconds, 0 = off")
    return parser


def main() -> None:
    """Runs the fleet from the command line, prints the summary and exits with 0 or 1.

    An invalid option value (such as zero loops) ends with a usage error from the parser, not a traceback."""
    parser = make_parser()
    args = parser.parse_args()
    screen = tuple(int(part) for part in args.screen.lower().split("x")) if args.screen else None
    try:
        config = FleetConfig(streams=args.streams, pace=args.pace, loops=args.loops, screen=screen,
                             display=not args.no_display, batch_sizes=args.batch_sizes,
                             slow_stream=args.slow_stream, slow_ms=args.slow_ms,
                             input=args.input.resolve(), monitor_period_s=args.monitor_period_s)
    except ValueError as error:
        parser.error(str(error))
    result = run_fleet(config)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["stop_reason"] or result["failed"] else 0)


if __name__ == "__main__":
    main()
