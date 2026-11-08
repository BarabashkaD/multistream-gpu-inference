# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for command-line parsing of the instance configuration."""
import pytest

from pipeline.config import InstanceConfig, parse_instance_config


def test_defaults():
    """Proves the defaults for pool size, drop policy, display and window position."""
    config = parse_instance_config(["--address", "pipe"])
    assert config.address == "pipe"
    assert config.pool_size == 6
    assert config.drop_policy == "oldest"
    assert config.display is True
    assert config.window_position is None


def test_window_options():
    """Proves the display size and window position options are parsed into integer pairs."""
    config = parse_instance_config(["--address", "a", "--display-size", "480x270", "--window-position", "480,270"])
    assert config.display_size == (480, 270)
    assert config.window_position == (480, 270)


@pytest.mark.parametrize("arguments", [
    ["--address", "a", "--pool-size", "3"],
    ["--address", "a", "--pool-size", "9"],
    ["--address", "a", "--loops", "-1"],
    ["--address", "a", "--packet-queue", "0"],
    ["--address", "a", "--display-fps", "0"],
    ["--address", "a", "--server-timeout-s", "0"],
    [],
])
def test_bad_arguments_exit(arguments):
    """Proves out-of-range or missing arguments make the parser exit."""
    with pytest.raises(SystemExit):
        parse_instance_config(arguments)


def test_a_pool_above_8_is_refused_because_of_the_pipe_buffer(capsys):
    """Proves a pool of 8 is accepted, 9 is refused, and the error names the 64 KiB pipe buffer limit."""
    assert parse_instance_config(["--address", "a", "--pool-size", "8"]).pool_size == 8
    with pytest.raises(SystemExit):
        parse_instance_config(["--address", "a", "--pool-size", "9"])
    assert "64 KiB" in capsys.readouterr().err


def test_config_is_frozen():
    """Proves a configuration cannot be changed after it is created."""
    with pytest.raises(AttributeError):
        InstanceConfig(address="a").loops = 2


def test_statistics_options(tmp_path):
    """Proves the report interval and statistics file are parsed, and the file is optional."""
    config = parse_instance_config(["--address", "a", "--report-interval-s", "2.5",
                                    "--stats-file", str(tmp_path / "i.log")])
    assert config.report_interval_s == 2.5
    assert config.stats_file == tmp_path / "i.log"
    assert parse_instance_config(["--address", "a"]).stats_file is None


@pytest.mark.parametrize("value", ["0", "-1"])
def test_report_interval_must_be_positive(value):
    """Proves a zero or negative report interval makes the parser exit."""
    with pytest.raises(SystemExit):
        parse_instance_config(["--address", "a", "--report-interval-s", value])
