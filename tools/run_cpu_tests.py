# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Run a small mandatory CPU suite, failing on absent suites or skipped tests."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import pytest

SUITES = (
    "test_config.py", "test_server_config.py", "test_protocol.py",
    "test_queueing.py", "test_make_compose.py", "test_image_contents.py", "test_model_provenance.py", "test_gpu_smoke_contract.py",
)


class RequiredSuites:
    def __init__(self):
        self.skipped = False

    def pytest_collection_finish(self, session):
        collected = {Path(item.path).name for item in session.items}
        missing = set(SUITES) - collected
        if missing:
            raise pytest.UsageError(f"Required CPU suites were not collected: {sorted(missing)}")

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped = True

    def pytest_sessionfinish(self, session, exitstatus):
        if self.skipped:
            session.exitstatus = 1
            print("Mandatory CPU tests must not be skipped", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "-ra", "-m", "not gpu", *[f"tests/{name}" for name in SUITES],
                          *sys.argv[1:]], plugins=[RequiredSuites()]))
