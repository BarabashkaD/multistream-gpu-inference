# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU failure tests: a killed instance, a killed server and a slow instance each leave the rest of the
system working or stopping with a clear error."""
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

V3 = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.gpu


def spawn(processes, script, *arguments):
    """Run a repository script in a child process, output and errors merged, and register it for cleanup."""
    process = subprocess.Popen([sys.executable, script, *arguments], cwd=V3, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    processes.append(process)
    return process


def kill_tree(process):
    """Kill a process and all its descendants, so no child keeps holding the GPU."""
    for child in psutil.Process(process.pid).children(recursive=True):
        child.kill()
    process.kill()


def wait_for_progress(instance):
    """Read the instance's output until it reports tracked frames; raise if it ends without doing so."""
    for line in instance.stdout:
        if '"current"' in line and '"tracked"' in line:
            return
    raise AssertionError("the instance made no progress")


def instance(processes, address, stream, clip):
    """Start a headless instance that loops the clip until it is killed."""
    return spawn(processes, "run_instance.py", "--address", address, "--stream", str(stream), "--input", str(clip),
                 "--no-display", "--loops", "0")


def test_killing_an_instance_leaves_the_others_running(clip_path, models_dir, processes, pipe_address):
    """Proves that killing one instance does not stop the other, and the server reports the killed stream as dead."""
    server = spawn(processes, "run_server.py", "--address", pipe_address, "--clients", "2", "--models", str(models_dir))
    survivor = spawn(processes, "run_instance.py", "--address", pipe_address, "--stream", "0",
                     "--input", str(clip_path), "--no-display", "--loops", "2")
    victim = instance(processes, pipe_address, 1, clip_path)
    wait_for_progress(victim)
    kill_tree(victim)
    output, _ = survivor.communicate(timeout=120)
    assert survivor.returncode == 0, output[-1000:]
    assert json.loads(output.strip().splitlines()[-1])["counts"]["tracked"] > 600
    summary, _ = server.communicate(timeout=60)
    assert server.returncode == 0
    assert json.loads(summary.strip().splitlines()[-1])["dead"] == [1]


def test_killing_the_server_stops_instances_with_an_error(clip_path, models_dir, processes, pipe_address):
    """Proves that killing the server makes a running instance exit promptly with a nonzero code and a failure
    message."""
    server = spawn(processes, "run_server.py", "--address", pipe_address, "--clients", "1", "--models", str(models_dir))
    worker = instance(processes, pipe_address, 0, clip_path)
    wait_for_progress(worker)
    kill_tree(server)
    started = time.monotonic()
    output, _ = worker.communicate(timeout=30)
    assert worker.returncode != 0
    assert time.monotonic() - started < 20
    assert "failed" in output


def test_a_slow_instance_drops_frames_and_the_others_keep_up(clip_path, models_dir, processes, pipe_address):
    """Proves an instance made slow drops its own frames while a healthy one drops none and keeps real time."""
    server = spawn(processes, "run_server.py", "--address", pipe_address, "--clients", "2", "--models", str(models_dir))
    healthy = spawn(processes, "run_instance.py", "--address", pipe_address, "--stream", "0",
                    "--input", str(clip_path), "--no-display")
    slow = spawn(processes, "run_instance.py", "--address", pipe_address, "--stream", "1", "--input", str(clip_path),
                 "--no-display", "--track-delay-ms", "120")
    results = [json.loads(process.communicate(timeout=120)[0].strip().splitlines()[-1]) for process in (healthy, slow)]
    server.communicate(timeout=60)
    assert results[0]["counts"].get("dropped", 0) == 0
    assert results[0]["steady_fps"]["tracked"] > 24.0
    assert results[1]["counts"]["dropped"] > 0
