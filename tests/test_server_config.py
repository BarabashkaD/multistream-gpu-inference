# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of server.config.parse_server_config: the statistics and batch-gathering command-line options."""
import pytest

from server.config import parse_server_config


def test_statistics_options(tmp_path):
    """--report-interval-s and --stats-file are parsed into a float and a Path."""
    config = parse_server_config(["--address", "a", "--report-interval-s", "2", "--stats-file", str(tmp_path / "s.log")])
    assert config.report_interval_s == 2.0
    assert config.stats_file == tmp_path / "s.log"


def test_report_interval_must_be_positive():
    """A report interval of zero is refused with a usage error (SystemExit)."""
    with pytest.raises(SystemExit):
        parse_server_config(["--address", "a", "--report-interval-s", "0"])


def test_gather_option():
    """--gather-ms defaults to 10 milliseconds and can be set to a fractional value."""
    assert parse_server_config(["--address", "a"]).gather_ms == 10.0
    assert parse_server_config(["--address", "a", "--gather-ms", "2.5"]).gather_ms == 2.5


def test_gather_must_not_be_negative():
    """A negative --gather-ms is refused with a usage error (SystemExit)."""
    with pytest.raises(SystemExit):
        parse_server_config(["--address", "a", "--gather-ms", "-1"])
