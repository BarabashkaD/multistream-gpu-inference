# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests that joining the stage threads at shutdown reports a stuck stage and keeps it referenced, without a GPU."""
import pytest

from pipeline import app


class _StuckStage:
    """Stand-in stage whose thread never finishes."""
    name = "stuck"

    def join(self, timeout=None):
        """Returns at once, as a join that timed out would."""
        pass

    def is_alive(self):
        """Reports the thread as still running."""
        return True


class _DoneStage:
    """Stand-in stage whose thread has already finished."""
    name = "done"

    def join(self, timeout=None):
        """Returns at once."""
        pass

    def is_alive(self):
        """Reports the thread as finished."""
        return False


def test_stuck_stage_is_reported_and_kept_referenced():
    """Proves a stage still alive after the join raises a RuntimeError naming it and stays in the module's list of
    stuck stages, so a reference to it is kept."""
    stuck = _StuckStage()
    try:
        with pytest.raises(RuntimeError, match="stuck"):
            app._join_stages([_DoneStage(), stuck], 0.01)
        assert stuck in app._stuck_stages
    finally:
        app._stuck_stages.clear()


def test_finished_stages_raise_nothing():
    """Proves joining stages that have all finished raises nothing and records no stuck stage."""
    app._join_stages([_DoneStage()], 0.01)
    assert app._stuck_stages == []
