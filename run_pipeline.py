#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Bundle pipeline traffic demo. See docs/design/bundle-pipeline.md."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent / ".ultralytics")

from pipeline.app import run_pipeline
from pipeline.config import parse_config


def main() -> None:
    """Parse the command line, run the pipeline and print the final statistics as JSON."""
    print(json.dumps(run_pipeline(parse_config()), indent=2))


if __name__ == "__main__":
    main()
