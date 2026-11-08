#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Mandatory GPU probe inside the immutable server candidate; no skip-as-pass."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main():
    import pytest
    from server.detector import BatchDetector
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    sizes = tuple(sorted(int(name[1:-5]) for name in manifest["models"]))
    if not sizes or 1 not in sizes:
        raise ValueError("No batch-one model in manifest")
    # BatchDetector warms every model with real CUDA-bound inference and rejects
    # an inactive CUDA provider. This does not prove zero CPU kernels/fallback.
    detector = BatchDetector(ROOT / "models", sizes, 0)
    print(f"CUDA model warmups passed for {sizes}")
    del detector

    class Required:
        skipped = False

        def pytest_collection_finish(self, session):
            required = {"test_a_second_process_reads_identical_frames",
                        "test_an_unknown_pointer_without_a_handle_is_an_error",
                        "test_closing_an_unknown_stream_is_harmless"}
            if not required.issubset({item.name for item in session.items}):
                raise pytest.UsageError("Required GPU IPC tests not collected")

        def pytest_runtest_logreport(self, report):
            if report.skipped:
                self.skipped = True

        def pytest_sessionfinish(self, session, exitstatus):
            if self.skipped:
                session.exitstatus = 1

    return pytest.main(["-q", "-ra", "-o", "addopts=", "-m", "gpu", "tests/test_gpu.py"], plugins=[Required()])


if __name__ == "__main__":
    sys.exit(main())
