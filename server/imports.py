# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Decoder frames of other processes, seen as tensors. Used only from the batch thread."""
from __future__ import annotations

import ctypes

import torch
import torch.utils.dlpack

from pipeline.gpu import close_handle, open_handle

_KDL_CUDA = 2
_KDL_UINT = 1


class _DLDevice(ctypes.Structure):
    """DLPack's DLDevice: the device type and index that own a tensor's memory."""

    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    """DLPack's DLDataType: the element type code, its width in bits and the number of lanes."""

    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    """DLPack's DLTensor: a data pointer with its device, element type, shape, strides and byte offset."""

    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    """DLPack's DLManagedTensor: a DLTensor plus the deleter that the consumer calls when it is done."""

    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_live: dict[int, "_Opened"] = {}  # by struct address: a struct must outlive every view torch made from it


@_DELETER
def _released(managed, live=_live, addressof=ctypes.addressof) -> None:
    """Torch calls this when the last view of the storage dies, which may be long after opened.tensor went.

    Its names are bound here, not looked up, because a view can die after shutdown cleared this module's globals."""
    opened = live.pop(addressof(managed.contents), None)
    if opened is not None:
        opened.released = True


# Never freed, and through its default neither is _live: at shutdown, a struct whose views are still alive must not
# be freed under them, and torch must still find a callback to call.
ctypes.pythonapi.Py_IncRef(ctypes.py_object(_released))


class _Opened:
    """One imported decoder frame buffer: its mapped device pointer and a torch tensor viewing it.

    The tensor is an NV12 frame of height * 3 / 2 rows by width bytes. `released` turns true once torch has dropped
    the last view, which is when the mapping may be closed."""

    def __init__(self, pointer: int, width: int, height: int, gpu_id: int):
        """Wraps the mapped pointer in a DLPack struct and builds the tensor from it, without copying.

        Raises whatever torch raises when it rejects the capsule; the struct is then forgotten again."""
        self.pointer = pointer
        self.released = False
        self.shape = (ctypes.c_int64 * 2)(height * 3 // 2, width)
        self.managed = _DLManagedTensor()
        self.managed.dl_tensor.data = pointer
        self.managed.dl_tensor.device = _DLDevice(_KDL_CUDA, gpu_id)
        self.managed.dl_tensor.ndim = 2
        self.managed.dl_tensor.dtype = _DLDataType(_KDL_UINT, 8, 1)
        self.managed.dl_tensor.shape = self.shape
        self.managed.deleter = ctypes.cast(_released, ctypes.c_void_p)
        address = ctypes.addressof(self.managed)
        capsule = ctypes.pythonapi.PyCapsule_New(address, b"dltensor", None)
        _live[address] = self
        try:
            self.tensor = torch.utils.dlpack.from_dlpack(capsule)
        except BaseException:
            _live.pop(address, None)  # no view exists, so the deleter will never fire
            raise


class FrameImports:
    """Per-stream tables of imported frame buffers, keyed by the exporter's pointer.

    A buffer is imported once, on the first frame that carries its handle, and reused for every later frame that
    names the same pointer. Not thread safe: use it from the batch thread only."""

    def __init__(self, width: int, height: int, gpu_id: int):
        """Sets the frame size and GPU of the buffers to import; the size may be changed before the first view."""
        self.width = width
        self.height = height
        self.gpu_id = gpu_id
        self._streams: dict[int, dict[int, _Opened]] = {}

    def view(self, stream: int, pointer: int, handle: bytes | None) -> torch.Tensor:
        """Returns the tensor for a stream's frame buffer, importing it from the handle the first time.

        Raises KeyError when the buffer is new and no handle came with it, and whatever opening the handle raises
        (RuntimeError, BufferError); a failed import leaves nothing mapped."""
        table = self._streams.setdefault(stream, {})
        if pointer not in table:
            if handle is None:
                raise KeyError(f"stream {stream} sent pointer {pointer:#x} with no handle")
            mapped = open_handle(handle)
            try:
                table[pointer] = _Opened(mapped, self.width, self.height, self.gpu_id)
            except BaseException:
                close_handle(mapped)
                raise
        return table[pointer].tensor

    def close_stream(self, stream: int) -> None:
        """Drops every buffer of a stream and closes its mappings.

        A mapping whose view is still alive is left open, and a failing close does not stop the others; the first
        error is raised as RuntimeError after all buffers were handled."""
        errors = []
        for opened in self._streams.pop(stream, {}).values():
            del opened.tensor  # the view must go before the mapping
            if not opened.released:  # a leak is safe and an unmapped read is not, so this mapping stays
                errors.append(RuntimeError(f"a view of stream {stream} is still alive"))
                continue
            try:
                close_handle(opened.pointer)
            except RuntimeError as error:
                errors.append(error)
        if errors:
            raise errors[0]
