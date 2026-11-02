#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Launch independent YOLO memory-loop demo processes with isolated artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "pexels-36304379.mp4"
DEFAULT_MODEL = PROJECT_ROOT / "yolo26n.onnx"


def parse_args() -> argparse.Namespace:
    """Parses the command line; most options mirror run_demo.py and are forwarded to every child process."""
    parser = argparse.ArgumentParser(
        description="Launch independent YOLO26 memory-loop demo processes."
    )
    parser.add_argument("--instances", type=int, default=1, help="Number of child processes (default: 1)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--decoder", choices=("opencv", "nvdec"), default="nvdec")
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--loops", type=int, default=1, help="1/N finite passes, 0 until user stop")
    parser.add_argument("--max-frames", type=int, default=0, help="Per-loop test bound; zero means full source")
    parser.add_argument("--headless", action="store_true", help="Pass --no-display to every child")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Artifact root; default is a unique runs/launcher-* directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Enable recording into unique instance subdirectories; omitted means no MP4",
    )
    return parser.parse_args()


def unique_run_dir(requested: Path | None) -> Path:
    """Creates and returns the artifact folder: `requested` if given, else runs/launcher-<UTC time>-<pid>.
    Raises FileExistsError if the folder already exists, so one launch never writes into another's artifacts."""
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=False)
        return requested
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = PROJECT_ROOT / "runs" / f"launcher-{stamp}-{os.getpid()}"
    base.mkdir(parents=True, exist_ok=False)
    return base


def child_command(args: argparse.Namespace, instance_id: str, output_path: Path | None) -> list[str]:
    """Builds the run_demo.py command line for one child, run with the current interpreter. Adds --no-display when
    headless and --output only when a recording path is given."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_demo.py"),
        "--instance-id", instance_id,
        "--input", str(args.input),
        "--model", str(args.model),
        "--decoder", args.decoder,
        "--backend", args.backend,
        "--tracker", args.tracker,
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--imgsz", str(args.imgsz),
        "--loops", str(args.loops),
        "--max-frames", str(args.max_frames),
    ]
    if args.headless:
        command.append("--no-display")
    if output_path is not None:
        command.extend(["--output", str(output_path)])
    return command


def stop_children(children: list[tuple[subprocess.Popen[str], object]]) -> None:
    """Stops the children that are still running, escalating from an interrupt (Ctrl-Break on Windows, SIGINT
    elsewhere) to terminate after 8 seconds and kill after 3 more. Only touches processes this launcher started."""
    alive = [proc for proc, _ in children if proc.poll() is None]
    for proc in alive:
        try:
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except OSError:
            pass
    deadline = time.monotonic() + 8.0
    for proc in alive:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.terminate()
    for proc in alive:
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    """Starts --instances run_demo.py children, each with its own log (and recording folder if --output-dir is
    set), and waits for all of them. Returns 0 if every child succeeded, 1 if any failed, 130 after Ctrl-C.
    Raises ValueError for invalid counts or an output folder with --loops 0, FileNotFoundError for a missing input
    or model, and FileExistsError if the run or output folder already exists."""
    args = parse_args()
    if args.instances < 1:
        raise ValueError("--instances must be >= 1")
    if args.loops < 0:
        raise ValueError("--loops must be >= 0")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")
    if args.loops == 0 and args.output_dir is not None:
        raise ValueError("--output-dir cannot be used with --loops 0; choose finite loops for recording")
    input_path = args.input.resolve()
    model_path = args.model.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    run_dir = unique_run_dir(args.run_dir.resolve() if args.run_dir else None)
    log_dir = run_dir / "logs"
    log_dir.mkdir()
    output_root = args.output_dir.resolve() if args.output_dir else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=False)

    children: list[tuple[subprocess.Popen[str], object]] = []
    try:
        for index in range(1, args.instances + 1):
            instance_id = f"instance-{index:02d}"
            instance_dir = output_root / instance_id if output_root is not None else None
            if instance_dir is not None:
                instance_dir.mkdir()
                output_path = instance_dir / "result.mp4"
            else:
                output_path = None
            log_path = log_dir / f"{instance_id}.log"
            log_file = log_path.open("w", encoding="utf-8")
            command = child_command(args, instance_id, output_path)
            log_file.write("$ " + subprocess.list2cmdline(command) + "\n")
            log_file.flush()
            kwargs = {
                "cwd": str(PROJECT_ROOT),
                "env": os.environ.copy(),
                "stdout": log_file,
                "stderr": subprocess.STDOUT,
                "text": True,
            }
            # A separate session or process group lets Ctrl-C reach only the launcher, which then stops the
            # children itself.
            if os.name != "nt":
                kwargs["start_new_session"] = True
            elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(command, **kwargs)
            children.append((proc, log_file))
            print(f"[launcher] started {instance_id} pid={proc.pid} log={log_path}")
        statuses = []
        for proc, _ in children:
            statuses.append(proc.wait())
        failed = sum(status != 0 for status in statuses)
        print(f"[launcher] completed={len(statuses)} failed={failed} run_dir={run_dir}")
        return 1 if failed else 0
    except KeyboardInterrupt:
        print("[launcher] stopping only children owned by this launcher", file=sys.stderr)
        stop_children(children)
        return 130
    finally:
        for _, log_file in children:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
