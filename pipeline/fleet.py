# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Starts one server and N instances, watches the safety limits and adds up the results.

run_fleet() launches the processes with their output redirected to log files, samples GPU and host load once a
second, stops everything when a limit is reached, and writes summary.json from the final line each process printed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys
import threading
import time

import psutil
import pynvml

from pipeline.config import REPO_ROOT, V3_ROOT
from pipeline.memory import private_mb
from pipeline.protocol import AUTH_KEY_VARIABLE, default_address

MIN_GPU_FREE_MB = 300
MIN_HOST_FREE_GB = 1.0
REALTIME_FPS = 24.5
_START_LIMIT_S = 180.0
_POLL_S = 1.0
_MEMORY_AFTER_SAMPLES = 20
_EXIT_WAIT_S = 30.0
_MAX_NVML_FAILURES = 5
_MEMORY_KEYS = ("server_gpu_mb", "server_private_mb", "instance_gpu_mb", "instance_private_mb")


@dataclass(frozen=True)
class FleetConfig:
    """Settings for one fleet run: how many streams, how they are paced and shown, and where the logs go.

    Raises ValueError from construction when monitor_period_s is negative or loops is below 1 (the time limit
    is derived from the loops, so zero loops would give a meaningless limit).
    """

    streams: int
    pace: str = "realtime"
    loops: int = 2
    display: bool = True
    screen: tuple[int, int] | None = None  # detected if None
    batch_sizes: str = "1,8"
    input: Path = REPO_ROOT / "data" / "pexels-36304379.mp4"
    out_dir: Path = V3_ROOT / "results" / "latest"
    slow_stream: int | None = None
    slow_ms: float = 0.0
    monitor_period_s: float = 2.0

    def __post_init__(self) -> None:
        """Rejects values that would break the run's time limit or the live table's refresh period."""
        if self.monitor_period_s < 0:
            raise ValueError(f"monitor_period_s must be 0 (off) or more, got {self.monitor_period_s}")
        if self.loops < 1:
            raise ValueError(f"loops must be at least 1, got {self.loops}: the time limit is computed from the loops")


def grid_position(stream: int, streams: int, screen: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Window position and size for this stream in a grid that fills the screen.

    Returns ((x, y), (width, height)). The grid is as square as the stream count allows, and the height is capped
    at the 16:9 height for the cell width, so cells may leave empty space at the bottom of the screen."""
    columns = math.ceil(math.sqrt(streams))
    rows = math.ceil(streams / columns)
    width = screen[0] // columns
    height = min(screen[1] // rows, width * 9 // 16)  # keeps 16:9 video undistorted
    return (width * (stream % columns), height * (stream // columns)), (width, height)


def screen_size() -> tuple[int, int]:
    """Primary screen width and height in pixels; a fixed 1920x1080 off Windows, where there is no query."""
    if sys.platform == "win32":
        import ctypes

        return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    return 1920, 1080


def _last_json(path: Path) -> dict | None:
    """The process's final summary: the last line with "kind": "final". None when the process never wrote one
    (it was killed or failed)."""
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("kind") == "final":
            return item
    return None


def _stderr_is_terminal() -> bool:
    """True when standard error is an interactive terminal; a closed or missing stream counts as not one."""
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _start_live_table(config: FleetConfig, out: Path) -> tuple[threading.Event, threading.Thread] | None:
    """The monitor writes to standard error, so the final JSON on standard output stays parseable.

    Starts a daemon thread that redraws the live table until the returned event is set. Returns (event, thread),
    or None when the table is switched off or standard error is not a terminal."""
    if config.monitor_period_s <= 0 or not _stderr_is_terminal():
        return None
    stop = threading.Event()

    def watch() -> None:
        """Thread body: runs the monitor and reports any failure on standard error instead of raising."""
        try:
            from rich.console import Console  # only here: the fleet must not load rich at import

            from pipeline import monitor
            monitor.run(out, config.monitor_period_s, console=Console(stderr=True), stop=stop)
        except Exception as error:  # the table is a convenience; it must never end the run
            print(f"live table stopped: {error}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=watch, name="live-table", daemon=True)
    thread.start()
    return stop, thread


def _gpu_memory_mb(pid: int) -> float | None:
    """This process's GPU memory, or None if Windows cannot say."""
    counter = rf"\GPU Process Memory(pid_{pid}_*)\Dedicated Usage"
    try:
        output = subprocess.run(["typeperf", counter, "-sc", "1"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [line for line in output.splitlines() if line.startswith('"') and not line.startswith('"(PDH')]
    if not rows:
        return None
    try:
        return sum(float(value.strip('"')) for value in rows[0].split(",")[1:] if value.strip('"')) / 2**20
    except ValueError:
        return None


def _real_process(process: subprocess.Popen) -> psutil.Process:
    """A virtual environment's python.exe is a launcher, and the work happens in its child."""
    launcher = psutil.Process(process.pid)
    children = launcher.children()
    return children[0] if children else launcher


def _command(script: str, arguments: list[str]) -> list[str]:
    """Command line that runs a script with the interpreter that runs the fleet."""
    return [sys.executable, script, *arguments]


def _kill_all(processes: list[subprocess.Popen]) -> list[str]:
    """Kills every process tree. The real worker is a child of the virtual environment's launcher, so one
    failed kill must not skip the rest. Returns what could not be killed."""
    errors = []
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        except psutil.Error as error:
            errors.append(f"listing children of {process.pid}: {error}")
            children = []
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass  # it exited between listing and killing
            except psutil.Error as error:
                errors.append(f"killing {child.pid}: {error}")
        try:
            process.kill()
        except OSError as error:
            if process.poll() is None:
                errors.append(f"killing {process.pid}: {error}")
    return errors


def bottleneck(gpu_percent: int, nvdec_percent: int, cpu_percent: int, min_fps: float) -> str:
    """The resource nearest its limit. "none" means the streams kept up and nothing is near one.

    Inputs are average percentages over the busy part of the run and the slowest stream's frame rate. A resource
    counts once it reaches its limit (85 GPU, 80 NVDEC, 90 CPU); the one furthest over wins."""
    limits = [(gpu_percent, 85, "GPU compute"), (nvdec_percent, 80, "NVDEC"), (cpu_percent, 90, "CPU")]
    reached = [(value - limit, name) for value, limit, name in limits if value >= limit]
    if reached:
        return max(reached)[1]
    return "none" if min_fps >= REALTIME_FPS else "unknown, below 25 FPS with no resource near its limit"


def summarise(server: dict | None, instances: list[dict | None], samples: list[tuple], stop_reason: str | None,
              memory: dict | None = None) -> dict:
    """Adds up the final lines of the server and the instances and the load samples into one summary dict.

    samples are (gpu %, decoder %, GPU MB used, CPU %, host GB free) tuples. Load averages skip the first third
    and the last three samples, where the fleet is starting or draining. A missing or killed process contributes
    nothing and is counted in "failed" (instances) or leaves empty server fields. Never raises on missing data."""
    finished = [item for item in instances if item and "counts" in item]  # only final lines reach here; "counts" guards against killed or failed processes
    rates = sorted(item["steady_fps"].get("tracked", 0.0) for item in finished)
    busy = samples[len(samples) // 3: -3 or None] or samples
    if not server or "median_ms" not in server:  # a killed server wrote no final line
        server = {"batches": {}, "available": {}, "median_ms": {}, "dead": []}
    gpu_percent = round(statistics.mean(s[0] for s in busy)) if busy else 0
    nvdec_percent = round(statistics.mean(s[1] for s in busy)) if busy else 0
    cpu_percent = round(statistics.mean(s[3] for s in busy)) if busy else 0
    frames = sum(item["counts"].get("tracked", 0) for item in finished)
    displayed = sum(item["counts"].get("displayed", 0) for item in finished)
    empty_ticks = sum(item["counts"].get("ticks_without_frame", 0) for item in finished)
    # A window counts a tick every display period, shown or empty, so a run with neither had no windows.
    windowed = any("displayed" in item["counts"] or "ticks_without_frame" in item["counts"] for item in finished)
    return {
        "streams": len(instances),
        "total_fps": round(sum(rates), 1),
        "min_fps": rates[0] if rates else 0.0,
        "median_fps": statistics.median(rates) if rates else 0.0,
        "frames": frames,
        "dropped": sum(item["counts"].get("dropped", 0) for item in finished),
        "displayed": displayed,
        "ticks_without_frame": empty_ticks,
        "shown_percent": round(displayed / frames * 100) if windowed and frames else None,
        "latency_p50_ms": max((item["median_ms"].get("latency", 0.0) for item in finished), default=0.0),
        "latency_p95_ms": max((item["p95_ms"].get("latency", 0.0) for item in finished), default=0.0),
        "gpu_percent": gpu_percent,
        "nvdec_percent": nvdec_percent,
        "cpu_percent": cpu_percent,
        "bottleneck": bottleneck(gpu_percent, nvdec_percent, cpu_percent, rates[0] if rates else 0.0),
        "gpu_memory_peak_mb": round(max((s[2] for s in samples), default=0)),
        "host_available_min_gb": min((s[4] for s in samples), default=0.0),
        "batches": server["batches"],
        "available": server.get("available", {}),
        "infer_ms": server["median_ms"].get("infer"),
        "prepare_ms": server["median_ms"].get("prepare"),
        "dead": server.get("dead", []),
        "errors": server.get("errors", []),
        "stop_reason": stop_reason,
        "failed": len(instances) - len(finished),
        **dict.fromkeys(_MEMORY_KEYS), **(memory or {}),
    }


def run_fleet(config: FleetConfig) -> dict:
    """Runs one server and config.streams instances to completion or until a safety limit stops them.

    Writes server.log, instanceN.log and summary.json in config.out_dir (removing the previous run's files first)
    and returns the summary. Stops early on low GPU or host memory, the time limit, a server exit, a monitoring
    failure or Ctrl+C, and always kills the process trees; the reason is in "stop_reason". Launch failures are
    reported the same way instead of being raised."""
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in [out / "server.log", out / "summary.json", *out.glob("instance*.log")]:
        stale.unlink(missing_ok=True)  # a run that stops early must not read an earlier run's results
    address = default_address()
    environment = {**os.environ, AUTH_KEY_VARIABLE: secrets.token_hex(32)}  # never on a command line
    logs = []
    everyone: list[subprocess.Popen] = []
    names: list[str] = []

    def launch(name: str, arguments: list[str]) -> subprocess.Popen:
        """Starts a script with its output in <name>.log and records the process and log for cleanup."""
        log = (out / f"{name}.log").open("w", encoding="utf-8", errors="replace")
        logs.append(log)
        process = subprocess.Popen(_command(arguments[0], arguments[1:]), cwd=V3_ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, env=environment)
        everyone.append(process)
        names.append(name)
        return process

    samples: list[tuple] = []
    memory: dict = dict.fromkeys(_MEMORY_KEYS)
    probed = False
    stop_reason = None
    live = None
    try:
        try:
            pynvml.nvmlInit()
            gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            screen = config.screen or screen_size()
            server = launch("server", ["run_server.py", "--address", address, "--clients", str(config.streams),
                                       "--batch-sizes", config.batch_sizes])
            instances = []
            for stream in range(config.streams):
                arguments = ["run_instance.py", "--address", address, "--stream", str(stream),
                             "--input", str(config.input), "--pace", config.pace, "--loops", str(config.loops)]
                if config.display:
                    (x, y), (width, height) = grid_position(stream, config.streams, screen)
                    arguments += ["--display-size", f"{width}x{height}", "--window-position", f"{x},{y}"]
                else:
                    arguments.append("--no-display")
                if stream == config.slow_stream:
                    arguments += ["--track-delay-ms", str(config.slow_ms)]
                instances.append(launch(f"instance{stream}", arguments))
        except OSError as error:
            stop_reason = f"launch failed: {error}"
        if stop_reason is None:
            live = _start_live_table(config, out)
            started = time.monotonic()
            psutil.cpu_percent(None)
            expected_s = 30.0 * config.loops
            limit_s = _START_LIMIT_S + expected_s * 2
            nvml_failures = 0
            while any(process.poll() is None for process in everyone):
                time.sleep(_POLL_S)
                try:
                    state = pynvml.nvmlDeviceGetMemoryInfo(gpu)
                    rates = pynvml.nvmlDeviceGetUtilizationRates(gpu)
                    decoder = pynvml.nvmlDeviceGetDecoderUtilization(gpu)[0]
                except pynvml.NVMLError as error:
                    nvml_failures += 1
                    if nvml_failures >= _MAX_NVML_FAILURES:
                        stop_reason = f"monitor error: {error}"
                        break
                    continue
                nvml_failures = 0
                host_free = psutil.virtual_memory().available / 2**30
                samples.append((rates.gpu, decoder, state.used / 2**20, psutil.cpu_percent(None), host_free))
                if (not probed and len(samples) > min(_MEMORY_AFTER_SAMPLES, expected_s / 2)
                        and server.poll() is None and instances[-1].poll() is None):
                    probed = True
                    try:  # a child may exit between poll() and the probe
                        worker, client = _real_process(server), _real_process(instances[-1])
                        memory = {"server_gpu_mb": _gpu_memory_mb(worker.pid),
                                  "server_private_mb": private_mb(worker),
                                  "instance_gpu_mb": _gpu_memory_mb(client.pid),
                                  "instance_private_mb": private_mb(client)}
                    except (psutil.Error, OSError):
                        pass
                if state.free / 2**20 < MIN_GPU_FREE_MB:
                    stop_reason = f"GPU memory limit: {state.free / 2**20:.0f} MB free"
                elif host_free < MIN_HOST_FREE_GB:
                    stop_reason = f"host memory limit: {host_free:.1f} GB available"
                elif time.monotonic() - started > limit_s:
                    stop_reason = "time limit"
                elif server.poll() not in (None, 0):
                    stop_reason = f"server exited with {server.returncode}"
                if stop_reason:
                    break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as error:  # last resort: the summary must still be written
        stop_reason = f"monitor error: {error}"
    finally:
        if live is not None:
            live[0].set()
            live[1].join(5.0)
        stuck: list[str] = []
        try:
            kill_errors = _kill_all(everyone)
        except Exception as error:  # the summary must still be written
            kill_errors = [f"killing processes: {error}"]
        for name, process in zip(names, everyone):
            try:
                process.wait(timeout=_EXIT_WAIT_S)
            except subprocess.TimeoutExpired:
                stuck.append(name)
            except OSError:
                pass
        for log in logs:
            try:
                log.close()
            except OSError:
                pass
    if stuck and not stop_reason:
        stop_reason = f"processes did not exit: {', '.join(stuck)}"
    if kill_errors and not stop_reason:
        stop_reason = f"cleanup failed: {kill_errors[0]}"
    result = summarise(_last_json(out / "server.log"),
                       [_last_json(out / f"instance{index}.log") for index in range(config.streams)],
                       samples, stop_reason, memory)
    result["stuck"] = stuck
    result["kill_errors"] = kill_errors
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
