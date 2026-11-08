#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""One pipeline instance. See docs/design/external-inference.md.

Command line entry point: parses the instance options, runs one stream against an inference server and writes the
periodic and final statistics lines.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # must precede NumPy: each thread otherwise commits memory

from pipeline.app import run_instance
from pipeline.config import parse_instance_config
from pipeline.statslog import StatsLog


def main() -> None:
    """Run one instance from the command line and write its final statistics line; the log is closed even on failure."""
    config = parse_instance_config()
    log = StatsLog(config.stats_file)
    try:
        log.write_final({"kind": "final", **run_instance(config, log)})
    finally:
        log.close()


if __name__ == "__main__":
    main()
