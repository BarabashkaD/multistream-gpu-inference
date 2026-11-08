# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""NV12 frames to the detector input, one set of kernels per batch.

Bilinear sampling works on index tables, so no full-resolution float copy of a frame is ever made.
"""
from __future__ import annotations

import numpy as np
import torch

from server.letterbox import Letterbox

PAD_VALUE = 114.0 / 255.0


def _axis(source: int, target: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The two source lines and the weight for each target line, as bilinear with align_corners=False.

    Returns the lower and upper source indices and the float weight of the upper one, all on `device`."""
    position = ((torch.arange(target, dtype=torch.float64, device=device) + 0.5) * source / target - 0.5)
    position = position.clamp(0, source - 1)
    low = position.floor().long()
    high = (low + 1).clamp(max=source - 1)
    return low, high, (position - low).float()


def _sample(plane: torch.Tensor, rows: tuple, columns: tuple) -> torch.Tensor:
    """Bilinear resize of a uint8 plane with the row and column tables from _axis; returns a float tensor.

    plane is (batch, height, width) or (batch, height, width, 2), uint8. A pass whose weights are all zero is
    skipped, since it would only copy."""
    row_low, row_high, row_weight = rows
    column_low, column_high, column_weight = columns
    trailing = [1] * (plane.dim() - 3)
    row_weight = row_weight.view(1, -1, 1, *trailing)
    column_weight = column_weight.view(1, 1, -1, *trailing)

    def line(selected: torch.Tensor) -> torch.Tensor:
        """Resamples the columns of already row-selected lines."""
        result = selected[:, :, column_low].float()
        if column_weight.any():
            result.lerp_(selected[:, :, column_high].float(), column_weight)
        return result

    result = line(plane[:, row_low])
    if row_weight.any():
        result.lerp_(line(plane[:, row_high]), row_weight)
    return result


class BatchPreparer:
    """Converts batches of NV12 frames of one source size into the letterboxed RGB detector input.

    The sampling tables depend only on the sizes, so they are built once here and reused for every batch."""

    def __init__(self, letterbox: Letterbox, device: torch.device):
        """Builds the luma and chroma sampling tables on `device` for the letterbox's source and scaled sizes."""
        self.letterbox = letterbox
        size = (letterbox.new_height, letterbox.new_width)
        width, height = letterbox.source_width, letterbox.source_height
        self.width, self.height = width, height
        self.luma_rows, self.luma_columns = _axis(height, size[0], device), _axis(width, size[1], device)
        self.chroma_rows = _axis(height // 2, size[0], device)
        self.chroma_columns = _axis(width // 2, size[1], device)

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        """Maps boxes from detector input pixels back to source pixels; see Letterbox.to_source."""
        return self.letterbox.to_source(boxes)

    def convert(self, frames: list[torch.Tensor], output: torch.Tensor) -> None:
        """Fills `output`, shaped (batch, 3, input height, input width), from one NV12 tensor per batch row.

        Resizes luma and chroma, converts limited-range BT.709 YUV to RGB in 0..1, and pads the border with the
        grey pad value. On a CUDA output it synchronises the stream before returning, because ONNX Runtime reads
        the tensor by raw pointer. Every frame must be a uint8 (height * 3 / 2, width) tensor of the source size."""
        count, width, height = len(frames), self.width, self.height
        nv12 = torch.stack(frames)
        luma = _sample(nv12[:, :height], self.luma_rows, self.luma_columns)
        chroma = _sample(nv12[:, height:].view(count, height // 2, width // 2, 2),
                         self.chroma_rows, self.chroma_columns)
        y = (luma - 16.0) * (255.0 / 219.0)
        u = (chroma[..., 0] - 128.0) * (255.0 / 224.0)
        v = (chroma[..., 1] - 128.0) * (255.0 / 224.0)
        rgb = torch.stack((y + 1.5748 * v, y - 0.1873 * u - 0.4681 * v, y + 1.8556 * u), dim=1)
        output.fill_(PAD_VALUE)
        box = self.letterbox
        output[..., box.top:box.top + box.new_height, box.left:box.left + box.new_width] = (rgb / 255.0).clamp(0.0, 1.0)
        if output.device.type == "cuda":
            torch.cuda.current_stream(output.device).synchronize()


def reference_input(nv12: torch.Tensor, letterbox: Letterbox, output: torch.Tensor) -> None:
    """The v2 single-frame conversion, kept as the reference for BatchPreparer's tests.

    Fills the (1, 3, input height, input width) `output` from one NV12 tensor with torch's own interpolation."""
    width, height = letterbox.source_width, letterbox.source_height
    size = (letterbox.new_height, letterbox.new_width)
    luma = nv12[:height].float()[None, None]
    chroma = nv12[height:].view(height // 2, width // 2, 2).permute(2, 0, 1).float()[None]
    luma = torch.nn.functional.interpolate(luma, size=size, mode="bilinear", align_corners=False)
    chroma = torch.nn.functional.interpolate(chroma, size=size, mode="bilinear", align_corners=False)
    y = (luma - 16.0) * (255.0 / 219.0)
    u, v = ((chroma - 128.0) * (255.0 / 224.0)).split(1, dim=1)
    rgb = torch.cat((y + 1.5748 * v, y - 0.1873 * u - 0.4681 * v, y + 1.8556 * u), dim=1)
    output.fill_(PAD_VALUE)
    top, left = letterbox.top, letterbox.left
    output[..., top:top + size[0], left:left + size[1]] = (rgb / 255.0).clamp(0.0, 1.0)
