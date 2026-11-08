# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for the fleet runner: screen grid layout, summary arithmetic, the live table, and process handling
with fake child processes, so no GPU is used."""
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pynvml
import pytest

from pipeline import fleet
from pipeline.fleet import FleetConfig, bottleneck, grid_position, run_fleet, summarise
from pipeline.protocol import AUTH_KEY_VARIABLE


def test_grid_is_square_enough():
    """Proves 16 streams are laid out as a 4x4 grid of equal cells on the screen."""
    assert grid_position(0, 16, (1920, 1080)) == ((0, 0), (480, 270))
    assert grid_position(5, 16, (1920, 1080)) == ((480, 270), (480, 270))
    assert grid_position(15, 16, (1920, 1080)) == ((1440, 810), (480, 270))


def test_one_stream_fills_the_screen_cell():
    """Proves a single stream gets the whole screen."""
    assert grid_position(0, 1, (1920, 1080))[1] == (1920, 1080)


def test_two_streams_share_a_row():
    """Proves two streams sit side by side, each taking half the screen width."""
    assert grid_position(1, 2, (1920, 1080)) == ((960, 0), (960, 540))


def instance(stream, tracked_fps, dropped=0):
    """Build the final report of one instance, with fixed latency figures and a full buffer pool."""
    return {"stream": stream, "steady_fps": {"tracked": tracked_fps}, "counts": {"tracked": 100, "dropped": dropped},
            "median_ms": {"latency": 40.0}, "p95_ms": {"latency": 80.0 + stream}, "pool": {"free": 8, "size": 8}}


def test_summary_adds_up_streams_and_keeps_the_worst_latency():
    """Proves the summary sums stream rates, takes the lowest rate, the worst latency and the peak resource use."""
    server = {"batches": {"8": 10}, "median_ms": {"infer": 1.5, "prepare": 0.4}, "dead": []}
    samples = [(50, 10, 3000, 40, 12.0)] * 3 + [(90, 30, 5000, 60, 10.0)] * 9 + [(10, 0, 2000, 5, 14.0)] * 3
    result = summarise(server, [instance(0, 25.0), instance(1, 24.0, dropped=2)], samples, None)
    assert result["total_fps"] == pytest.approx(49.0)
    assert result["min_fps"] == 24.0
    assert result["dropped"] == 2
    assert result["latency_p95_ms"] == 81.0
    assert result["gpu_percent"] == 90
    assert result["gpu_memory_peak_mb"] == 5000
    assert result["host_available_min_gb"] == 10.0
    assert result["failed"] == 0


def test_a_killed_server_leaves_only_a_report_line():
    """Proves a server that never gave a final summary yields no batch data, and the stop reason is kept."""
    result = summarise({"current": {}, "queued": 0}, [instance(0, 25.0)], [], "limit")
    assert result["batches"] == {}
    assert result["stop_reason"] == "limit"


def test_the_summary_carries_the_frames_available_per_batch():
    """Proves the server's frames-available histogram reaches the summary, and is empty without a server report."""
    server = {"batches": {"8": 10}, "median_ms": {}, "available": {"3": 1, "8+": 9}}
    assert summarise(server, [instance(0, 25.0)], [], None)["available"] == {"3": 1, "8+": 9}
    assert summarise(None, [instance(0, 25.0)], [], None)["available"] == {}


def test_the_last_final_line_is_read_even_when_period_lines_follow(tmp_path):
    """Proves the final line of a log is found even when periodic lines come after it."""
    path = tmp_path / "instance0.log"
    path.write_text("\n".join(json.dumps(line) for line in [
        {"kind": "period", "counts": {"tracked": 1}},
        {"kind": "final", "counts": {"tracked": 611}, "steady_fps": {"tracked": 25.0}},
        {"kind": "period", "counts": {"tracked": 611}},
    ]) + "\n", encoding="utf-8")
    assert fleet._last_json(path)["counts"]["tracked"] == 611


def test_a_log_with_only_period_lines_has_no_result(tmp_path):
    """Proves a log without a final line gives no result."""
    path = tmp_path / "server.log"
    path.write_text(json.dumps({"kind": "period", "batches": {"8": 1}}) + "\n", encoding="utf-8")
    assert fleet._last_json(path) is None


def test_a_server_line_without_medians_is_not_taken_for_a_summary():
    """Proves a periodic server line, which has no medians, is not used as the server's summary."""
    result = summarise({"kind": "period", "batches": {"8": 1}, "current": {}}, [instance(0, 25.0)], [], None)
    assert result["batches"] == {}


def test_negative_monitor_period_is_rejected():
    """Proves a negative monitor period is refused when the fleet configuration is built."""
    with pytest.raises(ValueError):
        FleetConfig(streams=1, monitor_period_s=-1.0)


@pytest.mark.parametrize("terminal,period_s", [(False, 2.0), (True, 0.0)])
def test_no_live_table_when_turned_off_or_not_a_terminal(tmp_path, monkeypatch, terminal, period_s):
    """Proves no live table starts when its period is zero or when stderr is not a terminal."""
    monkeypatch.setattr(fleet, "_stderr_is_terminal", lambda: terminal)
    config = FleetConfig(streams=1, display=False, monitor_period_s=period_s, out_dir=tmp_path)
    assert fleet._start_live_table(config, tmp_path) is None


@pytest.mark.parametrize("gpu,nvdec,cpu,fps,expected", [
    (95, 30, 50, 22.0, "GPU compute"),
    (40, 90, 50, 22.0, "NVDEC"),
    (40, 30, 99, 22.0, "CPU"),
    (40, 30, 50, 25.0, "none"),
])
def test_bottleneck_names_the_resource_nearest_its_limit(gpu, nvdec, cpu, fps, expected):
    """Proves the bottleneck is the resource near its limit, or none when the frame rate is met."""
    assert bottleneck(gpu, nvdec, cpu, fps) == expected


def test_summary_counts_the_frames_shown_in_windows():
    """Proves the summary adds up displayed frames and empty ticks, and reports the share of tracked frames shown."""
    shown = [instance(0, 25.0), instance(1, 25.0)]
    shown[0]["counts"] |= {"displayed": 80, "ticks_without_frame": 5}
    shown[1]["counts"] |= {"displayed": 70, "ticks_without_frame": 2}
    result = summarise(None, shown, [], None)
    assert result["displayed"] == 150
    assert result["ticks_without_frame"] == 7
    assert result["shown_percent"] == 75  # 150 of 200 tracked


def test_a_headless_summary_has_no_shown_share():
    """Proves a run without windows reports zero displayed frames and no shown percentage."""
    result = summarise(None, [instance(0, 25.0)], [], None)
    assert result["displayed"] == 0
    assert result["ticks_without_frame"] == 0
    assert result["shown_percent"] is None


def test_a_missing_instance_counts_as_failed():
    """Proves an instance that left no report is counted as failed."""
    result = summarise({"batches": {}, "median_ms": {}, "dead": [1]}, [instance(0, 25.0), None], [], "limit")
    assert result["failed"] == 1
    assert result["stop_reason"] == "limit"


FAKE_CHILD = '''
import json, os, sys, time
mode = os.environ["FAKE_MODE"]
if mode == "record":
    name = sys.argv[1] + "-" + str(os.getpid())
    with open(os.path.join(os.environ["FAKE_DIR"], name + ".json"), "w") as handle:
        json.dump({"argv": sys.argv, "key": os.environ.get("TRAFFIC_INFERENCE_KEY")}, handle)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("argv", *sys.argv)  # everything a child could log, but the key itself
    for variable, value in os.environ.items():
        if variable != "TRAFFIC_INFERENCE_KEY":
            print(variable, value)
    print("started", flush=True)
elif mode == "brief":
    time.sleep(1.5)
else:
    time.sleep(60)
'''


@pytest.fixture
def fake_fleet(tmp_path, monkeypatch):
    """Replace the server and instance programs with a fake child script and the GPU monitor with fixed readings;
    return the configuration, the list of started processes and the temporary folder."""
    script = tmp_path / "fake_child.py"
    script.write_text(FAKE_CHILD, encoding="utf-8")
    monkeypatch.setattr(fleet, "_command", lambda name, arguments: [sys.executable, str(script), name, *arguments])
    monkeypatch.setenv("FAKE_DIR", str(tmp_path))
    monkeypatch.setenv("FAKE_MODE", "record")
    monkeypatch.setattr(fleet, "_POLL_S", 0.05)
    monkeypatch.setattr(fleet, "MIN_HOST_FREE_GB", 0.0)
    monkeypatch.setattr(fleet, "_gpu_memory_mb", lambda pid: None)
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", lambda index: object())
    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo",
                        lambda handle: SimpleNamespace(used=2 * 2**30, free=8 * 2**30))
    monkeypatch.setattr(pynvml, "nvmlDeviceGetUtilizationRates", lambda handle: SimpleNamespace(gpu=50))
    monkeypatch.setattr(pynvml, "nvmlDeviceGetDecoderUtilization", lambda handle: (10, 1000))
    started = []
    real_popen = subprocess.Popen

    class Recording(real_popen):
        """Process class that remembers every process the fleet starts."""

        def __init__(self, *args, **kwargs):
            """Start the process as usual and record it."""
            super().__init__(*args, **kwargs)
            started.append(self)

    monkeypatch.setattr(subprocess, "Popen", Recording)
    config = FleetConfig(streams=2, display=False, screen=(1920, 1080), loops=1, out_dir=tmp_path / "out")
    return SimpleNamespace(config=config, started=started, tmp=tmp_path)


def test_the_key_reaches_children_only_through_their_environment(fake_fleet):
    """Proves all children get the same random key in their environment, and it is never in an argument or log."""
    result = run_fleet(fake_fleet.config)
    records = [json.loads(path.read_text()) for path in fake_fleet.tmp.glob("*.json")]
    assert len(records) == 3
    keys = {record["key"] for record in records}
    assert len(keys) == 1
    key = keys.pop()
    assert key and re.fullmatch("[0-9a-f]{64}", key)
    assert AUTH_KEY_VARIABLE == "TRAFFIC_INFERENCE_KEY"
    assert not any(key in argument for record in records for argument in record["argv"])
    logs = {path.name: path.read_text(encoding="utf-8", errors="replace") for path in (fake_fleet.tmp / "out").glob("*")}
    assert all(f"FAKE_DIR {fake_fleet.tmp}" in logs[f"{name}.log"] for name in ("server", "instance0", "instance1"))
    assert not any(key in text for text in logs.values())
    assert result["stop_reason"] is None


def test_the_live_table_runs_beside_the_fleet_and_stops_before_the_summary(fake_fleet, monkeypatch):
    """Proves the live table starts with the configured folder and period and is stopped before the run ends."""
    calls = []

    def fake_run(folder, period_s, streams=None, once=False, console=None, stop=None):
        """Record the call, wait for the stop event, record the stop and return success."""
        calls.append((folder, period_s))
        stop.wait(30)
        calls.append("stopped")
        return 0

    import pipeline.monitor
    monkeypatch.setattr(pipeline.monitor, "run", fake_run)
    monkeypatch.setattr(fleet, "_stderr_is_terminal", lambda: True)
    monkeypatch.setenv("FAKE_MODE", "brief")
    result = run_fleet(fake_fleet.config)
    assert calls == [(fake_fleet.config.out_dir, 2.0), "stopped"]
    assert result["stop_reason"] is None


def test_a_broken_live_table_does_not_end_the_run(fake_fleet, monkeypatch, capsys):
    """Proves an exception in the live table is reported on stderr and does not stop the run."""
    import pipeline.monitor
    monkeypatch.setattr(pipeline.monitor, "run", lambda *args, **kwargs: 1 / 0)
    monkeypatch.setattr(fleet, "_stderr_is_terminal", lambda: True)
    monkeypatch.setenv("FAKE_MODE", "brief")
    result = run_fleet(fake_fleet.config)
    assert result["stop_reason"] is None
    assert "live table stopped" in capsys.readouterr().err


def test_a_child_that_will_not_die_still_gives_a_summary(fake_fleet, monkeypatch):
    """Proves a child that never exits is named as stuck and the summary file is still written."""
    monkeypatch.setenv("FAKE_MODE", "brief")
    real_wait = subprocess.Popen.wait

    def wait(self, timeout=None):
        """Time out when waiting for instance 0, and wait normally for every other process."""
        arguments = list(self.args)
        if "--stream" in arguments and arguments[arguments.index("--stream") + 1] == "0":
            raise subprocess.TimeoutExpired(self.args, timeout)
        return real_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    result = run_fleet(fake_fleet.config)
    assert (fake_fleet.tmp / "out" / "summary.json").exists()
    assert result["stuck"] == ["instance0"]
    assert result["stop_reason"].startswith("processes did not exit")


def test_a_monitor_error_still_gives_a_summary(fake_fleet, monkeypatch):
    """Proves a GPU monitor error stops the run with a reason, writes the summary and leaves no process running."""
    monkeypatch.setenv("FAKE_MODE", "sleep")

    def broken(handle):
        """Fail like an unreadable GPU utilisation counter."""
        raise pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)

    monkeypatch.setattr(pynvml, "nvmlDeviceGetUtilizationRates", broken)
    result = run_fleet(fake_fleet.config)
    assert result["stop_reason"].startswith("monitor error")
    assert (fake_fleet.tmp / "out" / "summary.json").exists()
    assert len(fake_fleet.started) == 3
    assert all(process.poll() is not None for process in fake_fleet.started)
    assert result["stuck"] == []


def test_a_probe_of_an_exited_child_is_not_fatal(fake_fleet, monkeypatch):
    """Proves a memory probe of a child that has already exited does not fail the run."""
    monkeypatch.setenv("FAKE_MODE", "brief")
    monkeypatch.setattr(fleet, "_real_process", lambda process: (_ for _ in ()).throw(psutil.NoSuchProcess(1)))
    monkeypatch.setattr(fleet, "_MEMORY_AFTER_SAMPLES", 2)
    result = run_fleet(fake_fleet.config)
    assert result["stop_reason"] is None
    assert result["server_private_mb"] is None
    assert (fake_fleet.tmp / "out" / "summary.json").exists()


def test_a_failed_launch_still_gives_a_summary(fake_fleet, monkeypatch):
    """Proves a program that cannot be started gives a launch-failed reason and a written summary."""
    monkeypatch.setattr(fleet, "_command", lambda name, arguments: [str(fake_fleet.tmp / "no_such_program")])
    result = run_fleet(fake_fleet.config)
    assert result["stop_reason"].startswith("launch failed")
    assert (fake_fleet.tmp / "out" / "summary.json").exists()


def test_zero_loops_is_rejected():
    """Proves a loop count of zero is refused when the fleet configuration is built."""
    with pytest.raises(ValueError):
        FleetConfig(streams=2, loops=0)


def test_one_failed_kill_does_not_skip_the_rest_of_the_tree(monkeypatch):
    """Proves that when killing one child fails, the others and the parent are still killed and the error is
    reported."""
    killed = []

    class Child:
        """Child process that either records its kill or raises the given error."""

        def __init__(self, pid, error=None):
            """Store the pid and the error to raise on kill, if any."""
            self.pid, self.error = pid, error

        def kill(self):
            """Raise the configured error, or record the pid as killed."""
            if self.error:
                raise self.error
            killed.append(self.pid)

    class Parent:
        """Stand-in for a process handle whose children are one gone, one protected and one killable."""

        def __init__(self, pid):
            """Accept the pid and ignore it."""
            pass

        def children(self, recursive):
            """Return the three children."""
            return [Child(1, psutil.NoSuchProcess(1)), Child(2, psutil.AccessDenied(2)), Child(3)]

    class Process:
        """Running process whose kill is recorded."""

        pid = 10

        def poll(self):
            """Report the process as still running."""
            return None

        def kill(self):
            """Record the process as killed."""
            killed.append(self.pid)

    monkeypatch.setattr(fleet.psutil, "Process", Parent)
    errors = fleet._kill_all([Process()])
    assert killed == [3, 10]
    assert len(errors) == 1 and "2" in errors[0]


def test_results_of_an_earlier_run_are_cleared(tmp_path, monkeypatch):
    """Proves reports left in the output folder by an earlier run are not mistaken for this run's results."""
    (tmp_path / "instance0.log").write_text('{"counts": {"tracked": 611}, "steady_fps": {"tracked": 25.0}}\n')
    (tmp_path / "summary.json").write_text("{}")
    monkeypatch.setattr(fleet, "_command", lambda script, arguments: [str(tmp_path / "missing.exe")])
    result = run_fleet(FleetConfig(streams=1, loops=1, display=False, out_dir=tmp_path))
    assert result["stop_reason"].startswith("launch failed")
    assert result["failed"] == 1
