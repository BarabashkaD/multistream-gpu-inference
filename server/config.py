# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Settings of the inference server and their command-line parsing."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ServerConfig:
    """Immutable settings of one server run. Only the address has no default.

    The defaults are what parse_server_config uses for options left out of the command line."""

    address: str
    clients: int = 1
    models: Path = V3_ROOT / "models"
    batch_sizes: tuple[int, ...] = (1, 8)
    gather_ms: float = 10.0
    min_conf: float = 0.1
    gpu_id: int = 0
    report_interval_s: float = 1.0
    stats_file: Path | None = None
    start_timeout_s: float = 150.0


def _sizes(text: str) -> tuple[int, ...]:
    """Parses a comma list of batch sizes into a sorted tuple without duplicates."""
    return tuple(sorted({int(item) for item in text.split(",")}))


def parse_server_config(argv: list[str] | None = None) -> ServerConfig:
    """Parses the server options from argv, or from sys.argv when argv is None, into a ServerConfig.

    Exits with a usage error, through argparse, when a value is out of range: a report interval of zero or less,
    fewer than one client, batch sizes that omit 1 or hold a non-positive size, or a negative gather time."""
    defaults = ServerConfig(address="")
    parser = argparse.ArgumentParser(description="Inference server: batches frames from pipeline instances")
    parser.add_argument("--address", required=True)
    parser.add_argument("--clients", type=int, default=defaults.clients)
    parser.add_argument("--models", type=Path, default=defaults.models)
    parser.add_argument("--batch-sizes", type=_sizes, default=defaults.batch_sizes, help="comma list, must include 1")
    parser.add_argument("--gather-ms", type=float, default=defaults.gather_ms,
                        help="ms to wait for a frame from every stream")
    parser.add_argument("--min-conf", type=float, default=defaults.min_conf)
    parser.add_argument("--gpu-id", type=int, default=defaults.gpu_id)
    parser.add_argument("--report-interval-s", type=float, default=defaults.report_interval_s,
                        help="seconds between statistics lines")
    parser.add_argument("--stats-file", type=Path, default=None, help="also write the statistics lines here")
    args = parser.parse_args(argv)
    if args.report_interval_s <= 0:
        parser.error("--report-interval-s must be greater than 0")
    if args.clients < 1:
        parser.error("--clients must be at least 1")
    if 1 not in args.batch_sizes or min(args.batch_sizes) < 1:
        parser.error("--batch-sizes must include 1 and hold positive sizes")
    if args.gather_ms < 0:
        parser.error("--gather-ms must not be negative")
    return ServerConfig(address=args.address, clients=args.clients, models=args.models, batch_sizes=args.batch_sizes,
                        gather_ms=args.gather_ms, min_conf=args.min_conf, gpu_id=args.gpu_id,
                        report_interval_s=args.report_interval_s, stats_file=args.stats_file)
