#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Live table of a fleet's statistics: reads the newest line of every *.log file in a folder.

It only reads the files, so it can run beside a fleet, in another container, or after the run has ended.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pipeline.monitor import run


def main() -> None:
    """Parses the options and shows the table until the run ends or the user presses Ctrl+C.

    A missing folder or a non-positive period is a usage error. The exit status is 0 on a normal end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True, help="folder with server.log and instance*.log")
    parser.add_argument("--period-s", type=float, default=2.0)
    parser.add_argument("--streams", type=int, default=None, help="exit once this many streams have ended")
    parser.add_argument("--once", action="store_true", help="print one table and exit")
    args = parser.parse_args()
    if not args.logs.is_dir():
        parser.error(f"{args.logs} is not a folder")
    if args.period_s <= 0:
        parser.error("--period-s must be greater than 0")
    try:
        sys.exit(run(args.logs, args.period_s, args.streams, args.once))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
