# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU end-to-end test: a real server and two real instances process the sample clip, and every frame is
tracked."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

V3 = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.gpu


def start(processes, script, *arguments):
    """Run a repository script in a child process with its output piped, and register it for cleanup."""
    process = subprocess.Popen([sys.executable, script, *arguments], cwd=V3, stdout=subprocess.PIPE, text=True)
    processes.append(process)
    return process


def final(process):
    """Wait for a child to exit with success and return the JSON summary on its last output line."""
    output, _ = process.communicate(timeout=240)
    assert process.returncode == 0, output[-1000:]
    return json.loads(output.strip().splitlines()[-1])


def test_two_instances_track_every_frame(clip_path, models_dir, processes, pipe_address):
    """Proves two headless instances sharing one server track every clip frame, get real detections, return
    all their buffers, and leave no dead stream."""
    server = start(processes, "run_server.py", "--address", pipe_address, "--clients", "2", "--models", str(models_dir))
    instances = [start(processes, "run_instance.py", "--address", pipe_address, "--stream", str(stream),
                       "--input", str(clip_path), "--no-display", "--pace", "max") for stream in range(2)]
    results = [final(process) for process in instances]
    summary = final(server)
    for result in results:
        assert result["counts"]["tracked"] == 611
        assert result["counts"]["rows"] > 611 * 3  # real objects were found, so frames really reached the model
        assert result["pool"]["free"] == result["pool"]["size"]
    assert summary["counts"]["inferred"] == 1222
    assert summary["dead"] == []
    assert len(summary["streams"]) == 2
