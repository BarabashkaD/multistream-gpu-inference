# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of pipeline.display.status_text, the status line shown in the display window."""
from pipeline.display import status_text


def test_status_text_shows_current_average_and_lag():
    """The full text shows the display rate, the current and average tracked rate, and the overlay lag."""
    text = status_text({"displayed": 25.0, "tracked": 25.1}, {"tracked": 25.0}, 0)
    assert text == "display 25.0 fps | tracked 25.1 fps (avg 25.0) | overlay_lag=0"


def test_compact_status_text_is_short_enough_for_small_windows():
    """The compact form uses abbreviations to fit a small window."""
    text = status_text({"displayed": 25.0, "tracked": 25.1}, {"tracked": 25.0}, 0, compact=True)
    assert text == "25.0 fps | trk 25.1 (25.0) | lag 0"


def test_status_text_treats_missing_keys_as_zero():
    """Rates that are absent from the inputs are shown as 0.0 instead of raising."""
    assert status_text({}, {}, 3) == "display 0.0 fps | tracked 0.0 fps (avg 0.0) | overlay_lag=3"
