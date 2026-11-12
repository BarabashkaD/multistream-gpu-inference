#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Scaling benchmark for v1: run 1..N independent run_demo.py processes and summarize throughput, latency and load.

For each process count ("point") it starts launch_instances.py (headless, NVDEC, CUDA), samples the GPU (NVML: SM
busy, NVDEC, memory) and the host (psutil: CPU, available memory) once a second, and summarizes the JSON each process
prints at the end. Loop 1 includes loading the model and warm-up, so only loops 2..N are measured:

- FPS of a process: the mean processing_fps of its loops 2..N. The table shows the total and the slowest process.
- Latency of a process: p50 = mean of its loop p50s, p95 = max of its loop p95s, of due_to_tracked_ms (from the
  frame's due time to the end of its tracking; at --pace max the due time is the read time). The table shows the
  slowest process: the max over processes of p50 and, separately, of p95.
- Dropped: frames skipped by --drop-policy oldest in loops 2..N, summed over processes (loop 1 is kept separately).
- GPU busy, NVDEC and CPU: means over the window in which every process is in loop 2 or later, until the first
  process finishes. GPU memory: the peak during the point minus the memory in use before it started, read once that
  has settled (memory of processes that just ended can take seconds to be released).

A point is stopped (the whole process tree is killed) after --timeout-s or when free GPU memory or available host
memory stays below its guard for 3 samples in a row; that point is recorded as an error and the configuration
stops. A completed point whose minimum free memory fell below a guard is kept, but no larger point is started.

Example, from the repository root with the project venv:
  python bench.py --model models\\b1.onnx --imgsz 384 640 --batch-size 1 --instances 1,2,3,4,5,6,7,8 --pace realtime

Writes <out>/nN/{logs/, launcher.out, summary.json, samples.json}, <out>/bench.json and <out>/table.md. Absolute
paths of the repository, the output folder and the home folder are replaced with <repo>, <out> and <home> in
everything it writes, including the process logs. Results depend on the machine.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

from input_shape import add_imgsz_argument, normalize_imgsz

REPO = Path(__file__).resolve().parent
LAUNCHER = REPO / "launch_instances.py"
RUN_DEMO = REPO / "run_demo.py"
GUARD_SAMPLES = 3
"""Consecutive samples below a memory guard that stop a running point."""


def parse_instances(text: str) -> list[int]:
    """Parses a comma-separated list of positive process counts, e.g. "1,2,4". Raises ValueError otherwise."""
    try:
        counts = [int(item) for item in text.split(",") if item.strip()]
    except ValueError:
        raise ValueError(f"--instances must be comma-separated integers, got {text!r}") from None
    if not counts or any(n < 1 for n in counts):
        raise ValueError(f"--instances needs one or more counts >= 1, got {text!r}")
    return counts


def out_dir_name(pace: str, drop_policy: str, imgsz, batch_size: int) -> str:
    """The default result folder name: v1-<pace>[-drop]-<W>x<H>-b<N>, e.g. v1-realtime-drop-640x384-b8."""
    height, width = normalize_imgsz(imgsz)
    drop = "-drop" if drop_policy == "oldest" else ""
    return f"v1-{pace}{drop}-{width}x{height}-b{batch_size}"


def launcher_command(python: str, model: str, imgsz, batch_size: int, instances: int, loops: int, pace: str,
                     drop_policy: str, max_lag_s: float | None, run_dir: str) -> list[str]:
    """The launch_instances.py command line for one point. --max-lag-s is passed only when given, so by default every
    process uses its own frame-based limit."""
    command = [python, str(LAUNCHER), "--instances", str(instances), "--headless", "--loops", str(loops),
               "--decoder", "nvdec", "--backend", "cuda", "--model", model,
               "--imgsz", *map(str, normalize_imgsz(imgsz)), "--batch-size", str(batch_size),
               "--pace", pace, "--drop-policy", drop_policy, "--run-dir", run_dir]
    if max_lag_s is not None:
        command += ["--max-lag-s", str(max_lag_s)]
    return command


def path_replacements(out_dir: Path | None = None) -> list[tuple[str, str]]:
    """Machine-specific path prefixes and their placeholders, longest first: the output folder, the repository and
    the home folder, each with both slash styles and both drive-letter cases."""
    pairs = []
    for path, label in ((out_dir, "<out>"), (REPO, "<repo>"), (Path.home(), "<home>")):
        if path is None:
            continue
        text = str(path)
        variants = {text, text.replace("\\", "/"), text.replace("\\", "\\\\")}
        if len(text) > 1 and text[1] == ":":
            variants |= {v[0].swapcase() + v[1:] for v in variants}
        pairs += [(variant, label) for variant in variants]
    return sorted(set(pairs), key=lambda pair: -len(pair[0]))


def portable(text: str, replacements: list[tuple[str, str]]) -> str:
    """Replaces every machine-specific path prefix in `text` with its placeholder."""
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def scrub_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    """Rewrites a log in place with machine paths replaced, in its UTF-8 and UTF-16-LE parts alike (onnxruntime
    writes UTF-16 to the same log). Nothing else in the file changes."""
    data = original = path.read_bytes()
    for old, new in replacements:
        for encoding in ("utf-8", "utf-16-le"):
            data = data.replace(old.encode(encoding), new.encode(encoding))
    if data != original:
        path.write_bytes(data)


def log_text(data: bytes) -> str:
    """Decodes a process log that mixes UTF-8 with UTF-16-LE text: NUL bytes are dropped, so ASCII written in either
    encoding reads as text, and undecodable bytes are replaced."""
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def final_json(text: str) -> dict:
    """Returns the last top-level JSON object in a log that is a run_demo.py summary (has "loops_completed").
    Raises ValueError if there is none."""
    decoder = json.JSONDecoder()
    index = text.rfind("\n{")
    while index >= 0:
        try:
            value = decoder.raw_decode(text[index + 1:])[0]
            if isinstance(value, dict) and "loops_completed" in value:
                return value
        except json.JSONDecodeError:
            pass
        index = text.rfind("\n{", 0, index)
    raise ValueError("no run_demo.py JSON summary in log")


def _mean(values):
    """The mean of the values that are not None, or None if there are none."""
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _round(value, digits=1):
    """Rounds a number to `digits` decimals (to an int for 0), passing None through."""
    if value is None:
        return None
    return int(round(value)) if digits == 0 else round(value, digits)


def summarize_point(n: int, reports: list[dict], samples: list[dict], steady_from: float | None,
                    steady_to: float | None, base_mem_mb: float | None) -> dict:
    """Summarizes one point from the processes' JSON reports and the 1-second samples (dicts with t, sm, dec,
    mem_mb, free_mb, cpu, host_free_gb). Loops 2..N are the steady state; loop 1 counts only as loop1_dropped.
    Raises ValueError if a report has no loop 2 or a process was stopped early."""
    if len(reports) != n:
        raise ValueError(f"expected {n} reports, got {len(reports)}")
    per_process = []
    for report in reports:
        steady = [loop for loop in report["loops"] if loop["loop"] >= 2]
        if not steady:
            raise ValueError(f"{report.get('instance_id')}: no loop 2 or later to measure")
        if report.get("stopped_by_user") or any(loop.get("stopped_by_user") for loop in report["loops"]):
            raise ValueError(f"{report.get('instance_id')}: stopped before the end")
        per_process.append(dict(
            instance_id=report.get("instance_id"),
            fps=statistics.mean(loop["processing_fps"] for loop in steady),
            p50_ms=statistics.mean(loop["due_to_tracked_ms"]["p50"] for loop in steady),
            p95_ms=max(loop["due_to_tracked_ms"]["p95"] for loop in steady),
            read_to_tracked_p95_ms=max(loop["read_to_batch_complete_ms"]["p95"] for loop in steady),
            dropped=sum(loop.get("frames_dropped", 0) for loop in steady),
            loop1_dropped=sum(loop.get("frames_dropped", 0) for loop in report["loops"] if loop["loop"] == 1),
            frames_processed=sum(loop["frames_processed"] for loop in steady),
            frames_read=sum(loop["frames_read"] for loop in steady),
            max_lag_ms=_round(steady[0].get("max_lag_ms"), 3),
        ))
    in_window = [s for s in samples if steady_from is not None and steady_to is not None
                 and steady_from <= s["t"] <= steady_to]
    window = in_window or samples
    mems = [s["mem_mb"] for s in samples if s.get("mem_mb") is not None]
    frees = [s["free_mb"] for s in samples if s.get("free_mb") is not None]
    hosts = [s["host_free_gb"] for s in samples if s.get("host_free_gb") is not None]
    peak = max(mems) if mems else None
    return dict(
        streams=n,
        total_fps=round(sum(p["fps"] for p in per_process), 1),
        slowest_fps=round(min(p["fps"] for p in per_process), 1),
        latency_p50_ms=round(max(p["p50_ms"] for p in per_process), 1),
        latency_p95_ms=round(max(p["p95_ms"] for p in per_process), 1),
        latency_basis="due_to_tracked_ms",
        read_to_tracked_p95_ms=round(max(p["read_to_tracked_p95_ms"] for p in per_process), 1),
        dropped=sum(p["dropped"] for p in per_process),
        loop1_dropped=sum(p["loop1_dropped"] for p in per_process),
        max_lag_ms=sorted({p["max_lag_ms"] for p in per_process}, key=lambda v: (v is None, v)),
        gpu_busy_pct=_round(_mean(s.get("sm") for s in window), 0),
        nvdec_pct=_round(_mean(s.get("dec") for s in window), 0),
        cpu_pct=_round(_mean(s.get("cpu") for s in window), 0),
        load_window="steady" if in_window else "whole run",
        load_window_samples=len(window),
        steady_window_s=_round(steady_to - steady_from) if in_window else None,
        gpu_mem_baseline_mb=_round(base_mem_mb, 0),
        gpu_mem_peak_mb=_round(peak, 0),
        gpu_mem_app_mb=_round(peak - base_mem_mb, 0) if peak is not None and base_mem_mb is not None else None,
        min_free_gpu_mb=_round(min(frees), 0) if frees else None,
        min_free_host_gb=_round(min(hosts)) if hosts else None,
        per_process=[{k: (_round(v) if isinstance(v, float) else v) for k, v in p.items()} for p in per_process],
    )


def _cell(value) -> str:
    """One table cell: a dash for None."""
    return "-" if value is None else str(value)


TABLE_COLUMNS = (("Processes", "streams"), ("Total FPS", "total_fps"), ("Slowest process FPS", "slowest_fps"),
                 ("Dropped", "dropped"), ("Latency p50, ms", "latency_p50_ms"), ("Latency p95, ms", "latency_p95_ms"),
                 ("GPU busy, %", "gpu_busy_pct"), ("NVDEC, %", "nvdec_pct"), ("CPU, %", "cpu_pct"),
                 ("GPU memory (app), MB", "gpu_mem_app_mb"))


def render_table(points: list[dict], stop_reason: str | None = None) -> str:
    """Renders the measured points as a Markdown table, followed by one line per failed point and the stop reason."""
    lines = ["| " + " | ".join(title for title, _ in TABLE_COLUMNS) + " |",
             "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"]
    lines += ["| " + " | ".join(_cell(p.get(key)) for _, key in TABLE_COLUMNS) + " |"
              for p in points if "error" not in p]
    notes = [f"- {p['streams']} processes: not measured, {p['error']}" for p in points if "error" in p]
    if stop_reason:
        notes.append(f"- Stopped: {stop_reason}")
    return "\n".join(lines + ([""] + notes if notes else []))


class Sampler(threading.Thread):
    """Samples once a second: time, SM busy %, NVDEC %, GPU memory used and free (MB), CPU % (all logical CPUs) and
    available host memory (GB). A metric NVML cannot read is recorded as None."""

    def __init__(self, nvml, psutil, handle):
        """Prepares sampling of the NVML device `handle`; call start() to begin."""
        super().__init__(daemon=True)
        self.nvml, self.psutil, self.handle = nvml, psutil, handle
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        psutil.cpu_percent(None)

    def _read(self, function):
        """Calls an NVML reader, returning None if the device does not support it."""
        try:
            return function()
        except self.nvml.NVMLError:
            return None

    def run(self):
        """Appends one sample a second until stop_event is set."""
        nvml = self.nvml
        while not self.stop_event.wait(1.0):
            util = self._read(lambda: nvml.nvmlDeviceGetUtilizationRates(self.handle))
            dec = self._read(lambda: nvml.nvmlDeviceGetDecoderUtilization(self.handle))
            mem = self._read(lambda: nvml.nvmlDeviceGetMemoryInfo(self.handle))
            self.samples.append(dict(
                t=time.monotonic(), sm=util.gpu if util else None, dec=dec[0] if dec else None,
                mem_mb=mem.used / 2**20 if mem else None, free_mb=mem.free / 2**20 if mem else None,
                cpu=self.psutil.cpu_percent(None), host_free_gb=self.psutil.virtual_memory().available / 2**30))


def guard_breach(samples: list[dict], min_free_gpu_mb: float, min_free_host_gb: float) -> str | None:
    """A reason to stop the running point if the last GUARD_SAMPLES samples are all below a memory guard, else None."""
    last = samples[-GUARD_SAMPLES:]
    if len(last) < GUARD_SAMPLES:
        return None
    if all(s.get("free_mb") is not None and s["free_mb"] < min_free_gpu_mb for s in last):
        return f"memory guard: free GPU memory {last[-1]['free_mb']:.0f} MB < {min_free_gpu_mb} MB"
    if all(s["host_free_gb"] < min_free_host_gb for s in last):
        return f"memory guard: available host memory {last[-1]['host_free_gb']:.1f} GB < {min_free_host_gb} GB"
    return None


def settled_memory_mb(read_used_mb, sleep=time.sleep, tolerance_mb: float = 32, timeout_s: float = 30) -> float:
    """GPU memory in use once it has settled: reads `read_used_mb()` once a second until two readings in a row differ
    by less than `tolerance_mb`, or `timeout_s` passes, and returns the last reading. Processes that just ended
    (above all killed ones) can hold their memory for a few seconds, which would inflate the next baseline."""
    previous = read_used_mb()
    for _ in range(max(int(timeout_s), 1)):
        sleep(1.0)
        current = read_used_mb()
        if abs(current - previous) < tolerance_mb:
            return current
        previous = current
    return previous


def kill_tree(psutil, root_pid: int, known: dict) -> list[int]:
    """Kills the launcher `root_pid`, all its descendants and every process in `known` (pid -> psutil.Process seen
    as a descendant earlier). Returns the pids still alive 10 s later."""
    procs = dict(known)
    try:
        root = psutil.Process(root_pid)
        for child in root.children(recursive=True):
            procs[child.pid] = child
        procs[root.pid] = root
    except psutil.Error:
        pass
    for proc in procs.values():
        try:
            proc.kill()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(list(procs.values()), timeout=10)
    return [proc.pid for proc in alive]


def leftover_run_demo(psutil) -> list[int]:
    """Pids of running processes whose command line runs this repository's run_demo.py."""
    target = os.path.normcase(str(RUN_DEMO))
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if any(os.path.normcase(arg) == target for arg in proc.info["cmdline"] or ()):
                found.append(proc.info["pid"])
        except (psutil.Error, TypeError):
            pass
    return found


def run_point(args, n: int, nvml, psutil, handle, out_dir: Path, replacements) -> dict:
    """Runs one point with `n` processes and returns its summary, or a dict with "error". Writes nN/launcher.out,
    nN/summary.json and nN/samples.json. Kills the whole process tree on timeout, memory guard or Ctrl-C (re-raised
    after cleanup)."""
    run_dir = out_dir / f"n{n}"
    logs = [run_dir / "logs" / f"instance-{i:02d}.log" for i in range(1, n + 1)]
    command = launcher_command(sys.executable, str(args.model), args.imgsz, args.batch_size, n, args.loops,
                               args.pace, args.drop_policy, args.max_lag_s, str(run_dir))
    base_mem = settled_memory_mb(lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    kwargs = {}
    # Its own process group (session) keeps Ctrl-C away from the launcher: the benchmark stops the tree itself.
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    launcher_out = out_dir / f".n{n}-launcher.out"
    sampler = Sampler(nvml, psutil, handle)
    sampler.start()
    started = time.monotonic()
    known: dict = {}
    error = None
    finished = False
    steady_from = steady_to = None
    with launcher_out.open("wb") as out:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, **kwargs)
        try:
            while proc.poll() is None:
                time.sleep(0.5)
                now = time.monotonic()
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        known.setdefault(child.pid, child)
                except psutil.Error:
                    pass
                if now - started > args.timeout_s:
                    error = f"timeout after {args.timeout_s:g} s"
                else:
                    error = guard_breach(sampler.samples, args.min_free_gpu_mb, args.min_free_host_gb)
                if error:
                    break
                if steady_to is None:
                    texts = [log_text(p.read_bytes()) if p.exists() else "" for p in logs]
                    if steady_from is None and all("[loop 2]" in t for t in texts):
                        steady_from = now
                    if steady_from is not None and any('"loops_completed"' in t for t in texts):
                        steady_to = now
            finished = error is None
        except KeyboardInterrupt:
            error = "interrupted by user"
        finally:
            # Anything but a normal exit of the launcher (timeout, guard, Ctrl-C, an error here) kills the tree.
            alive = [] if finished else kill_tree(psutil, proc.pid, known)
            proc.wait()
            ended = time.monotonic()
            sampler.stop_event.set()
            sampler.join()
    if not error and steady_from is not None and steady_to is None:
        steady_to = ended  # every process finished between two polls
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(launcher_out), str(run_dir / "launcher.out"))
    for path in [run_dir / "launcher.out", *logs]:
        if path.exists():
            scrub_file(path, replacements)
    samples = [dict(s, t=round(s["t"] - started, 1)) for s in sampler.samples]
    (run_dir / "samples.json").write_text(json.dumps(samples, indent=1) + "\n", encoding="utf-8")
    orphans = sorted(set(alive) | set(leftover_run_demo(psutil)))
    if error == "interrupted by user":
        raise KeyboardInterrupt
    base = dict(streams=n, command=[portable(c, replacements) for c in [Path(sys.executable).name, *command[1:]]],
                wall_s=round(ended - started, 1))
    if orphans:
        error = (error + "; " if error else "") + f"processes left running: {orphans}"
    if not error and proc.returncode != 0:
        tail = log_text((run_dir / "launcher.out").read_bytes())[-1000:].strip()
        error = f"launcher exit code {proc.returncode}: {tail}"
    if not error:
        try:
            reports = [final_json(log_text(p.read_bytes())) for p in logs]
            point = dict(base, **summarize_point(n, reports, sampler.samples, steady_from, steady_to, base_mem))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            error = f"could not summarize: {exc}"
    if error:
        point = dict(base, error=error)
    (run_dir / "summary.json").write_text(json.dumps(point, indent=2) + "\n", encoding="utf-8")
    return point


def git_commit() -> str | None:
    """The repository's current commit, with "-dirty" if the working tree has changes, or None without git."""
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=7"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def make_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser; the module docstring is its help text."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="Static ONNX model matching --imgsz and --batch-size")
    add_imgsz_argument(p)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--instances", default="1,2,3,4,5,6,7,8", help="Comma-separated process counts, one point each")
    p.add_argument("--loops", type=int, default=3, help="Loops per process, >= 2; loop 1 is not measured")
    p.add_argument("--pace", choices=("max", "realtime"), default="max")
    p.add_argument("--drop-policy", choices=("none", "oldest"), default="none")
    p.add_argument("--max-lag-s", type=float, default=None,
                   help="Passed to the processes only when given (their default: 1.1 frame periods)")
    p.add_argument("--out", type=Path, default=None,
                   help="Result folder; must not exist (default: results/v1-<pace>[-drop]-<W>x<H>-b<N>)")
    p.add_argument("--min-free-gpu-mb", type=float, default=400)
    p.add_argument("--min-free-host-gb", type=float, default=2.0)
    p.add_argument("--timeout-s", type=float, default=600.0, help="Per point")
    return p


def main() -> int:
    """Runs the points one after another and writes bench.json and table.md. Returns 0 when every point was measured
    (a memory-guard stop after a measured point included), 1 if a point failed, 130 after Ctrl-C."""
    parser = make_parser()
    args = parser.parse_args()
    try:
        counts = parse_instances(args.instances)
    except ValueError as exc:
        parser.error(str(exc))
    if args.loops < 2:
        parser.error("--loops must be >= 2: loop 1 is not measured")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.drop_policy == "oldest" and args.pace != "realtime":
        parser.error("--drop-policy oldest needs --pace realtime")
    if args.max_lag_s is not None and args.max_lag_s <= 0:
        parser.error("--max-lag-s must be > 0")
    args.model = args.model.resolve()
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    height, width = normalize_imgsz(args.imgsz)
    out_dir = (args.out or REPO / "results" / out_dir_name(args.pace, args.drop_policy, args.imgsz,
                                                          args.batch_size)).resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"{out_dir} already exists; choose another --out")
    replacements = path_replacements(out_dir)

    import psutil
    import pynvml as nvml

    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)  # the device the processes use (CUDA device 0)
    config = dict(
        version="v1", commit=git_commit(), started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        model=portable(str(args.model), replacements), imgsz_hw=[height, width], batch_size=args.batch_size,
        instances=counts, loops=args.loops, measured_loops=f"2..{args.loops}", pace=args.pace,
        drop_policy=args.drop_policy, max_lag_s=args.max_lag_s,
        max_lag_note=None if args.max_lag_s is not None else "process default: 1.1 frame periods of the clip",
        decoder="nvdec", backend="cuda", headless=True, timeout_s=args.timeout_s,
        min_free_gpu_mb=args.min_free_gpu_mb, min_free_host_gb=args.min_free_host_gb,
        gpu=str(nvml.nvmlDeviceGetName(handle)), driver=str(nvml.nvmlSystemGetDriverVersion()),
        logical_cpus=psutil.cpu_count(), host_memory_gb=round(psutil.virtual_memory().total / 2**30, 1),
    )
    points, stop_reason, status = [], None, 0
    try:
        for n in counts:
            point = run_point(args, n, nvml, psutil, handle, out_dir, replacements)
            points.append(point)
            print(json.dumps({k: v for k, v in point.items() if k not in ("per_process", "command")}), flush=True)
            if "error" in point:
                stop_reason, status = f"{n} processes: {point['error']}", 1
                break
            if point["min_free_gpu_mb"] is not None and point["min_free_gpu_mb"] < args.min_free_gpu_mb:
                stop_reason = f"free GPU memory fell to {point['min_free_gpu_mb']:.0f} MB at {n} processes"
                break
            if point["min_free_host_gb"] is not None and point["min_free_host_gb"] < args.min_free_host_gb:
                stop_reason = f"available host memory fell to {point['min_free_host_gb']} GB at {n} processes"
                break
    except KeyboardInterrupt:
        stop_reason, status = "interrupted by user", 130
    finally:
        nvml.nvmlShutdown()
        (out_dir / "bench.json").write_text(json.dumps(dict(config=config, stop_reason=stop_reason, points=points),
                                                       indent=2) + "\n", encoding="utf-8")
        table = render_table(points, stop_reason)
        (out_dir / "table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
