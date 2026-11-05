# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Settings of one pipeline instance and the command line parser that builds them."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_ROOT
MIN_POOL_SIZE = 4
# The server sends results without a timeout. That is safe only while one instance's results in flight, at most
# one per pooled frame and 7.3 KB each in the worst case, fit the 64 KiB pipe buffer: 8 x 7.3 KB does.
MAX_POOL_SIZE = 8


@dataclass(frozen=True)
class InstanceConfig:
    """Immutable settings for one instance: where the server is, which stream and input, and pacing, queue, display
    and timeout options. The defaults here are also the command line defaults."""

    address: str
    stream: int = 0
    input: Path = REPO_ROOT / "data" / "pexels-36304379.mp4"
    loops: int = 1  # 0 repeats until stopped
    pool_size: int = 6
    packet_queue: int = 64
    pace: str = "realtime"
    drop_policy: str = "oldest"
    max_age_s: float = 2.0
    max_lag_s: float = 0.5
    display: bool = True
    display_fps: float = 25.0
    display_grace_s: float = 0.2
    display_size: tuple[int, int] = (1920, 1080)
    window_position: tuple[int, int] | None = None
    conf: float = 0.1
    gpu_id: int = 0
    report_interval_s: float = 1.0
    connect_timeout_s: float = 120.0
    server_timeout_s: float = 10.0
    track_delay_ms: float = 0.0  # simulates a slow consumer, for the benchmark
    stats_file: Path | None = None


def _size(text: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT into a pair of integers."""
    width, height = text.lower().split("x")
    return int(width), int(height)


def _position(text: str) -> tuple[int, int]:
    """Parse X,Y into a pair of integers."""
    x, y = text.split(",")
    return int(x), int(y)


def parse_instance_config(argv: list[str] | None = None) -> InstanceConfig:
    """Parse the instance options from argv (the process arguments when None) into an InstanceConfig.

    Exits through the argument parser with a usage error for a missing --address or an out-of-range value, such as
    a pool size outside the limits above.
    """
    defaults = InstanceConfig(address="")
    parser = argparse.ArgumentParser(description="One pipeline instance: decodes, tracks and shows one stream")
    parser.add_argument("--address", required=True)
    parser.add_argument("--stream", type=int, default=defaults.stream)
    parser.add_argument("--input", type=Path, default=defaults.input)
    parser.add_argument("--loops", type=int, default=defaults.loops, help="1 or N passes, 0 until stopped")
    parser.add_argument("--pool-size", type=int, default=defaults.pool_size)
    parser.add_argument("--packet-queue", type=int, default=defaults.packet_queue)
    parser.add_argument("--pace", choices=("realtime", "max"), default=defaults.pace)
    parser.add_argument("--drop-policy", choices=("none", "oldest"), default=defaults.drop_policy)
    parser.add_argument("--max-age-s", type=float, default=defaults.max_age_s)
    parser.add_argument("--max-lag-s", type=float, default=defaults.max_lag_s)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--display-fps", type=float, default=defaults.display_fps)
    parser.add_argument("--display-grace-s", type=float, default=defaults.display_grace_s)
    parser.add_argument("--display-size", type=_size, default=defaults.display_size, help="WIDTHxHEIGHT")
    parser.add_argument("--window-position", type=_position, default=None, help="X,Y")
    parser.add_argument("--conf", type=float, default=defaults.conf)
    parser.add_argument("--gpu-id", type=int, default=defaults.gpu_id)
    parser.add_argument("--connect-timeout-s", type=float, default=defaults.connect_timeout_s)
    parser.add_argument("--server-timeout-s", type=float, default=defaults.server_timeout_s)
    parser.add_argument("--track-delay-ms", type=float, default=defaults.track_delay_ms)
    parser.add_argument("--report-interval-s", type=float, default=defaults.report_interval_s,
                        help="seconds between statistics lines")
    parser.add_argument("--stats-file", type=Path, default=None, help="also write the statistics lines here")
    args = parser.parse_args(argv)
    if args.report_interval_s <= 0:
        parser.error("--report-interval-s must be greater than 0")
    if args.pool_size < MIN_POOL_SIZE:
        parser.error(f"--pool-size must be at least {MIN_POOL_SIZE}")
    if args.pool_size > MAX_POOL_SIZE:
        parser.error(f"--pool-size must be at most {MAX_POOL_SIZE}: the server sends results without a timeout, "
                     f"which relies on a pool's results (7.3 KB each at worst) fitting the 64 KiB pipe buffer")
    if args.loops < 0:
        parser.error("--loops must be 0 or more")
    if args.packet_queue < 1:
        parser.error("--packet-queue must be at least 1 (0 would make the queue unbounded)")
    if args.display_fps <= 0:
        parser.error("--display-fps must be greater than 0")
    if args.server_timeout_s <= 0:
        parser.error("--server-timeout-s must be greater than 0")
    return InstanceConfig(
        address=args.address, stream=args.stream, input=args.input, loops=args.loops, pool_size=args.pool_size,
        packet_queue=args.packet_queue, pace=args.pace, drop_policy=args.drop_policy, max_age_s=args.max_age_s,
        max_lag_s=args.max_lag_s, display=not args.no_display, display_fps=args.display_fps,
        display_grace_s=args.display_grace_s, display_size=args.display_size,
        window_position=args.window_position, conf=args.conf, gpu_id=args.gpu_id,
        connect_timeout_s=args.connect_timeout_s, server_timeout_s=args.server_timeout_s,
        track_delay_ms=args.track_delay_ms, report_interval_s=args.report_interval_s, stats_file=args.stats_file,
    )
