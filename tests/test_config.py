# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for command-line parsing of the pipeline configuration: defaults, overrides and rejected values."""
import pytest

from pipeline.config import PipelineConfig, parse_config


def test_defaults_match_the_design():
    """Proves the defaults for pool size, pace, drop policy, confidence, display and lag limits."""
    config = parse_config([])
    assert config.pool_size == 16
    assert config.pace == "realtime"
    assert config.drop_policy == "none"
    assert config.conf == 0.1
    assert config.display is True
    assert config.display_size == (1920, 1080)
    assert config.display_grace_s == 0.2
    assert config.max_lag_s == 0.5


def test_cli_overrides():
    """Proves each command-line option overrides its default and is converted to the right type."""
    config = parse_config(["--no-display", "--pace", "max", "--loops", "3",
                           "--display-size", "960x540", "--pool-size", "8",
                           "--display-grace-s", "0.1", "--max-lag-s", "1.5"])
    assert config.display is False
    assert config.pace == "max"
    assert config.loops == 3
    assert config.display_size == (960, 540)
    assert config.pool_size == 8
    assert config.display_grace_s == 0.1
    assert config.max_lag_s == 1.5


def test_rejects_pool_below_minimum():
    """Proves a pool size below the minimum makes the parser exit."""
    with pytest.raises(SystemExit):
        parse_config(["--pool-size", "2"])


def test_config_is_immutable():
    """Proves a configuration cannot be changed after it is created."""
    config = PipelineConfig()
    with pytest.raises(Exception):
        config.loops = 2


@pytest.mark.parametrize("argv", [
    ["--packet-queue", "0"],
    ["--display-fps", "0"],
    ["--display-fps", "-5"],
    ["--drop-policy", "oldest", "--pace", "max"],
])
def test_rejects_invalid_values(argv):
    """Proves a zero queue size, a non-positive display rate and a drop policy with max pace each make the parser
    exit."""
    with pytest.raises(SystemExit):
        parse_config(argv)


def test_prepare_defaults_to_cuda():
    """Proves frames are prepared with CUDA unless another mode is asked for."""
    assert parse_config([]).prepare == "cuda"


def test_prepare_gl_is_accepted():
    """Proves the OpenGL prepare mode can be selected."""
    assert parse_config(["--prepare", "gl"]).prepare == "gl"


def test_rejects_unknown_prepare_mode():
    """Proves an unknown prepare mode makes the parser exit."""
    with pytest.raises(SystemExit):
        parse_config(["--prepare", "vulkan"])


def test_rejects_gl_prepare_on_another_gpu():
    """Proves the OpenGL prepare mode is refused for a GPU other than the default one."""
    with pytest.raises(SystemExit):
        parse_config(["--prepare", "gl", "--gpu-id", "1"])
