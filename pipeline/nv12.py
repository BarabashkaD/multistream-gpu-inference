# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""NV12 frames in decoder memory: a zero-copy tensor view, the letterbox geometry and the CUDA conversion to input.

The ctypes structures mirror the DLPack C structures so a raw device pointer can be handed to torch without a copy.
"""
from __future__ import annotations

import ctypes

import numpy as np
import torch

PAD_VALUE = 114.0 / 255.0
_KDL_CUDA = 2
_KDL_UINT = 1


class _DLDevice(ctypes.Structure):
    """DLPack device: the device type code and device index."""

    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    """DLPack element type: type code, bits per lane and lane count."""

    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    """DLPack tensor description: data pointer, device, shape, strides and element type."""

    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    """DLPack tensor plus the owner's context and deleter; the deleter stays null, so torch never frees the memory."""

    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_descriptors: dict[tuple[int, int, int, int], tuple] = {}


def wrap_nv12(pointer: int, width: int, height: int, gpu_id: int = 0) -> torch.Tensor:
    """Tensor view over decoder memory. It must be dropped before the buffer is unlocked.

    Assumes a tight NV12 layout: row pitch equal to width, with chroma directly after luma.
    That layout was measured on the demo clip. The GPU test test_input_matches_the_library_rgb_path
    fails if the decoder returns a different one.
    """
    key = (pointer, width, height, gpu_id)
    if key not in _descriptors:
        shape = (ctypes.c_int64 * 2)(height * 3 // 2, width)
        managed = _DLManagedTensor()
        managed.dl_tensor.data = pointer
        managed.dl_tensor.device = _DLDevice(_KDL_CUDA, gpu_id)
        managed.dl_tensor.ndim = 2
        managed.dl_tensor.dtype = _DLDataType(_KDL_UINT, 8, 1)
        managed.dl_tensor.shape = shape
        _descriptors[key] = (managed, shape)
    capsule = ctypes.pythonapi.PyCapsule_New(ctypes.addressof(_descriptors[key][0]), b"dltensor", None)
    return torch.utils.dlpack.from_dlpack(capsule)


class Letterbox:
    """Geometry of fitting a source frame into a square detector input, keeping the aspect ratio and padding the rest."""

    def __init__(self, source_width: int, source_height: int, target: int):
        """Compute the scale gain, the resized size and the top and left padding for a target-sized square."""
        self.source_width = source_width
        self.source_height = source_height
        self.target = target
        self.gain = min(target / source_height, target / source_width)
        self.new_height = max(round(source_height * self.gain), 1)
        self.new_width = max(round(source_width * self.gain), 1)
        self.top = round((target - self.new_height) / 2 - 0.1)
        self.left = round((target - self.new_width) / 2 - 0.1)

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        """Map x1, y1, x2, y2 boxes from detector input pixels back to source pixels, clipped to the frame.

        Returns a new float32 array; the input is not modified.
        """
        result = boxes.astype(np.float32).copy()
        result[..., [0, 2]] = ((result[..., [0, 2]] - self.left) / self.gain).clip(0, self.source_width)
        result[..., [1, 3]] = ((result[..., [1, 3]] - self.top) / self.gain).clip(0, self.source_height)
        return result


def nv12_to_input(nv12: torch.Tensor, letterbox: Letterbox, width: int, height: int, output: torch.Tensor) -> None:
    """Resize, then convert BT.709 limited-range NV12 to padded RGB in [0, 1]."""
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
