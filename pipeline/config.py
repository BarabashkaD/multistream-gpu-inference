# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Settings of one pipeline run and the command line parser that builds them."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_POOL_SIZE = 4


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable settings of one run: input and model files, pool and queue sizes, pacing, drop policy, display,
    detection and tracking options and the frame preparation method. The defaults here are also the command line
    defaults."""

    input: Path = REPO_ROOT / "data" / "pexels-36304379.mp4"
    model: Path = REPO_ROOT / "yolo26n.onnx"
    loops: int = 1  # 0 repeats until stopped
    pool_size: int = 16
    batch_size: int = 1
    packet_queue: int = 64
    pace: str = "realtime"
    drop_policy: str = "none"
    max_age_s: float = 2.0  # reclaim stuck bundles after this long
    max_lag_s: float = 0.5  # --drop-policy oldest skips frames this far behind their source time
    display: bool = True
    display_fps: float = 25.0
    display_grace_s: float = 0.2  # how long display waits for a complete frame before using old boxes
    display_size: tuple[int, int] = (1920, 1080)
    conf: float = 0.1
    tracker: str = "bytetrack.yaml"
    detector_size: int = 640
    gpu_id: int = 0
    prepare: str = "cuda"  # "gl" runs frame preparation as an OpenGL shader
    report_interval_s: float = 1.0


def validate_batch_pool(batch_size: int, pool_size: int) -> None:
    """Reject configurations that cannot fill a batch while keeping decoder progress possible."""
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    if pool_size < max(MIN_POOL_SIZE, batch_size + 2):
        raise ValueError(f"pool size must be at least {max(MIN_POOL_SIZE, batch_size + 2)} for batch {batch_size}")


def _size(text: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT into a pair of integers."""
    width, height = text.lower().split("x")
    return int(width), int(height)


def parse_config(argv: list[str] | None = None) -> PipelineConfig:
    """Parse the run options from argv (the process arguments when None) into a PipelineConfig.

    Exits through the argument parser with a usage error for an out-of-range value or an option combination that
    cannot work: a pool smaller than the minimum, a negative loop count, a packet queue below 1, a display rate of
    0 or less, the oldest-drop policy with maximum pace, or OpenGL preparation on a GPU other than 0.
    """
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Bundle pipeline traffic demo")
    parser.add_argument("--input", type=Path, default=defaults.input)
    parser.add_argument("--model", type=Path, default=defaults.model)
    parser.add_argument("--loops", type=int, default=defaults.loops, help="1 or N passes, 0 until stopped")
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--packet-queue", type=int, default=defaults.packet_queue)
    parser.add_argument("--pace", choices=("realtime", "max"), default=defaults.pace)
    parser.add_argument("--drop-policy", choices=("none", "oldest"), default=defaults.drop_policy)
    parser.add_argument("--max-age-s", type=float, default=defaults.max_age_s)
    parser.add_argument("--max-lag-s", type=float, default=defaults.max_lag_s)
    parser.add_argument("--display-grace-s", type=float, default=defaults.display_grace_s)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--display-fps", type=float, default=defaults.display_fps)
    parser.add_argument("--display-size", type=_size, default=defaults.display_size, help="WIDTHxHEIGHT")
    parser.add_argument("--conf", type=float, default=defaults.conf)
    parser.add_argument("--tracker", default=defaults.tracker)
    parser.add_argument("--gpu-id", type=int, default=defaults.gpu_id)
    parser.add_argument("--prepare", choices=("cuda", "gl"), default=defaults.prepare)
    args = parser.parse_args(argv)
    if args.pool_size is None:
        args.pool_size = max(defaults.pool_size, args.batch_size + 2)
    try:
        validate_batch_pool(args.batch_size, args.pool_size)
    except ValueError as exc:
        parser.error(str(exc))
    if args.loops < 0:
        parser.error("--loops must be 0 or more")
    if args.packet_queue < 1:
        parser.error("--packet-queue must be at least 1 (0 would make the queue unbounded)")
    if args.display_fps <= 0:
        parser.error("--display-fps must be greater than 0")
    if args.drop_policy == "oldest" and args.pace == "max":
        parser.error("--drop-policy oldest has no effect with --pace max; use --pace realtime")
    if args.prepare == "gl" and args.gpu_id != 0:
        parser.error("--prepare gl cannot choose a GPU; use --gpu-id 0")
    return PipelineConfig(
        input=args.input, model=args.model, loops=args.loops, pool_size=args.pool_size,
        batch_size=args.batch_size, packet_queue=args.packet_queue, pace=args.pace, drop_policy=args.drop_policy,
        max_age_s=args.max_age_s, max_lag_s=args.max_lag_s, display=not args.no_display,
        display_fps=args.display_fps, display_grace_s=args.display_grace_s,
        display_size=args.display_size, conf=args.conf, tracker=args.tracker, gpu_id=args.gpu_id,
        prepare=args.prepare,
    )
