# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU tests for the whole pipeline: frame counts per stage, bundle return, pacing and stage failure."""
import pytest

from pipeline.app import run_pipeline
from pipeline.config import PipelineConfig

pytestmark = pytest.mark.gpu


def test_every_frame_reaches_every_stage_and_all_bundles_return(clip_path, model_path):
    """Proves one pass over the clip decodes, infers and tracks all 611 frames and returns every bundle to the pool."""
    config = PipelineConfig(input=clip_path, model=model_path, loops=1, pace="max", display=False)
    summary = run_pipeline(config)
    assert summary["counts"]["decoded"] == 611
    assert summary["counts"]["inferred"] == 611
    assert summary["counts"]["tracked"] == 611
    assert summary["pool"]["free"] == summary["pool"]["size"]
    assert summary["pool"]["leased"] == 0


def test_two_loops(clip_path, model_path):
    """Proves two loops over the clip track twice as many frames."""
    config = PipelineConfig(input=clip_path, model=model_path, loops=2, pace="max", display=False)
    assert run_pipeline(config)["counts"]["tracked"] == 1222


def test_realtime_pace_runs_near_source_speed(clip_path, model_path):
    """Proves realtime pacing decodes at about the clip's own frame rate (23 to 27 fps) without dropping frames."""
    config = PipelineConfig(input=clip_path, model=model_path, loops=1, pace="realtime", display=False)
    summary = run_pipeline(config)
    assert summary["counts"]["tracked"] == 611
    assert 23.0 < summary["fps"]["decoded"] < 27.0


def test_stage_failure_is_raised(clip_path, model_path, monkeypatch):
    """Proves an exception inside a stage thread comes out of run_pipeline as a RuntimeError naming the stage."""
    def fail(self, rows):
        """Stand-in tracker update that always fails."""
        raise ValueError("tracker broke")

    monkeypatch.setattr("pipeline.tracking.TrackerCore.update", fail)
    with pytest.raises(RuntimeError, match="stage tracker failed"):
        run_pipeline(PipelineConfig(input=clip_path, model=model_path, pace="max", display=False))
