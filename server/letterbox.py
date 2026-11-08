# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Letterbox geometry: how a source frame is scaled and centred in the detector input, and the way back."""
from __future__ import annotations

import numpy as np


class Letterbox:
    """Fits a source frame into the detector input at its aspect ratio, padded on the short side.

    Takes the source and target sizes in pixels and computes the scale gain, the scaled size and the top and left
    offsets. The offsets are rounded as the usual YOLO letterbox does, so detections match that reference. Source
    and target heights and widths must be positive; a zero size raises ZeroDivisionError."""

    def __init__(self, source_width: int, source_height: int, target_width: int, target_height: int):
        """Computes the gain, the scaled size and the offsets for the given source and target sizes."""
        self.source_width = source_width
        self.source_height = source_height
        self.gain = min(target_height / source_height, target_width / source_width)
        self.new_height = max(round(source_height * self.gain), 1)
        self.new_width = max(round(source_width * self.gain), 1)
        self.top = round((target_height - self.new_height) / 2 - 0.1)
        self.left = round((target_width - self.new_width) / 2 - 0.1)

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        """Maps x1, y1, x2, y2 boxes from detector input pixels to source pixels, clipped to the source frame.

        Returns a float32 copy; the argument is not modified."""
        result = boxes.astype(np.float32).copy()
        result[..., [0, 2]] = ((result[..., [0, 2]] - self.left) / self.gain).clip(0, self.source_width)
        result[..., [1, 3]] = ((result[..., [1, 3]] - self.top) / self.gain).clip(0, self.source_height)
        return result
