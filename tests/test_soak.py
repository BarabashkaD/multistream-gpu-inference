# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU soak test: a long run must not grow the process's host or GPU memory."""
import os
import subprocess
import sys
import threading

import psutil
import pytest

from pipeline.app import run_pipeline
from pipeline.config import PipelineConfig

pytestmark = pytest.mark.gpu
ALLOWED_GROWTH_MB = 64


def process_gpu_memory_mb() -> float:
    """This process's GPU memory. The whole-GPU figure also moves with other programs.

    On Windows it reads the process's dedicated GPU memory counter with `typeperf`; elsewhere it asks NVML for this
    process's usage on GPU 0.
    """
    if sys.platform == "win32":
        counter = rf"\GPU Process Memory(pid_{os.getpid()}_*)\Dedicated Usage"
        output = subprocess.run(["typeperf", counter, "-sc", "1"], capture_output=True, text=True, timeout=30).stdout
        rows = [line for line in output.splitlines() if line.startswith('"') and not line.startswith('"(PDH')]
        return sum(float(value.strip('"')) for value in rows[0].split(",")[1:] if value.strip('"')) / 2**20
    import pynvml

    pynvml.nvmlInit()
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(0))
    return next(p.usedGpuMemory for p in processes if p.pid == os.getpid()) / 2**20


def test_twenty_loops_keep_memory_flat(clip_path, model_path):
    """Proves twenty loops over the clip track every frame, return every bundle, and grow neither host memory nor
    this process's GPU memory by more than the allowed amount once the first quarter of the samples has passed."""
    process = psutil.Process()
    samples = []
    done = threading.Event()

    def sample():
        """Every two seconds until done, records the process's resident host memory and its GPU memory."""
        while not done.wait(2.0):
            samples.append((process.memory_info().rss, process_gpu_memory_mb()))

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    config = PipelineConfig(input=clip_path, model=model_path, loops=20, pace="max", display=False)
    summary = run_pipeline(config)
    done.set()
    sampler.join(35)

    assert summary["counts"]["tracked"] == 611 * 20
    assert summary["pool"]["free"] == summary["pool"]["size"]
    settled = samples[len(samples) // 4:]
    host_growth = (max(s[0] for s in settled) - settled[0][0]) / 2**20
    gpu_growth = max(s[1] for s in settled) - settled[0][1]
    assert host_growth < ALLOWED_GROWTH_MB, f"host memory grew {host_growth:.0f} MB; settled samples (rss, gpu MB): {settled}"
    assert gpu_growth < ALLOWED_GROWTH_MB, f"GPU memory grew {gpu_growth:.0f} MB; settled samples (rss, gpu MB): {settled}"
