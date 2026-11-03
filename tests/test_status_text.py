# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for the status line the display draws over the video."""
from pipeline.display import status_text


def test_status_text_shows_current_average_and_lag():
    """Proves the status line shows the display rate, the tracked rate with its average, and the overlay lag."""
    text = status_text({"displayed": 25.0, "tracked": 25.1}, {"tracked": 25.0}, 0)
    assert text == "display 25.0 fps | tracked 25.1 fps (avg 25.0) | overlay_lag=0"


def test_status_text_treats_missing_keys_as_zero():
    """Proves rates that are not reported yet are shown as zero instead of raising."""
    assert status_text({}, {}, 3) == "display 0.0 fps | tracked 0.0 fps (avg 0.0) | overlay_lag=3"
