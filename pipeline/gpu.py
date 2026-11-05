# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""CUDA context and IPC handles through ctypes, so an instance never loads PyTorch."""
from __future__ import annotations

import ctypes
import sys

_library = None
_contexts: dict[int, ctypes.c_void_p] = {}


class _IpcHandle(ctypes.Structure):
    """The 64-byte opaque CUDA IPC memory handle."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


def _cuda():
    """Load the CUDA driver library once, declare the argument types of the calls used here and initialise CUDA."""
    global _library
    if _library is None:
        library = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        library.cuIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_uint64]
        library.cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), _IpcHandle, ctypes.c_uint]
        library.cuIpcCloseMemHandle.argtypes = [ctypes.c_uint64]
        library.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        _check(library.cuInit(0), "cuInit")
        _library = library
    return _library


def _check(status: int, call: str) -> None:
    """Raise RuntimeError naming the call when a CUDA driver status is not zero."""
    if status != 0:
        raise RuntimeError(f"{call} failed with CUDA status {status}")


def use_gpu(gpu_id: int) -> None:
    """Make the GPU's primary context current on this thread."""
    library = _cuda()
    if gpu_id not in _contexts:
        device = ctypes.c_int()
        _check(library.cuDeviceGet(ctypes.byref(device), gpu_id), "cuDeviceGet")
        context = ctypes.c_void_p()
        _check(library.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
        _contexts[gpu_id] = context
    _check(library.cuCtxSetCurrent(_contexts[gpu_id]), "cuCtxSetCurrent")


def export_handle(pointer: int) -> bytes:
    """Return the IPC handle for device memory at `pointer`, so another process can map it; raise RuntimeError on failure."""
    handle = _IpcHandle()
    _check(_cuda().cuIpcGetMemHandle(ctypes.byref(handle), pointer), "cuIpcGetMemHandle")
    return bytes(handle)


def open_handle(handle: bytes) -> int:
    """Map memory from another process's IPC handle and return the local device pointer.

    The mapping must be released with close_handle. Raises RuntimeError if CUDA refuses the handle.
    """
    local = ctypes.c_uint64()
    status = _cuda().cuIpcOpenMemHandle_v2(ctypes.byref(local), _IpcHandle.from_buffer_copy(handle), 1)
    _check(status, "cuIpcOpenMemHandle")
    return local.value


def close_handle(pointer: int) -> None:
    """Release a mapping made by open_handle; raise RuntimeError on failure."""
    _check(_cuda().cuIpcCloseMemHandle(pointer), "cuIpcCloseMemHandle")
