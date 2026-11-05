# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Test helper. Decodes frames, exports each new pointer from another thread, prints a line per frame,
and unlocks on a line of input."""
import ctypes
import hashlib
import sys
import threading

import PyNvVideoCodec as nvc

from pipeline.gpu import export_handle, use_gpu


def export_elsewhere(pointer: int) -> bytes:
    """Export an IPC handle for a device pointer from a new thread, to prove the handle does not depend on the
    thread that decoded the frame."""
    result = {}

    def work():
        """Select the GPU on this thread and export the handle into the shared result."""
        use_gpu(0)
        result["handle"] = export_handle(pointer)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(10)
    return result["handle"]


clip, count = sys.argv[1], int(sys.argv[2])
use_gpu(0)
library = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
library.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
demuxer = nvc.CreateDemuxer(clip)
decoder = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId(), usedevicememory=True,
                            outputColorType=nvc.OutputColorType.NATIVE)
size = 1920 * 1620
seen = set()
sent = 0
while sent < count:
    for _ in range(decoder.GetNumDecodedFrame(demuxer.Demux())):
        pointer = decoder.GetLockedFrame()
        decoder.SyncOnCUStream()
        host = ctypes.create_string_buffer(size)
        assert library.cuMemcpyDtoH_v2(host, pointer, size) == 0
        handle = "-" if pointer in seen else export_elsewhere(pointer).hex()
        seen.add(pointer)
        print(pointer, handle, hashlib.sha256(host.raw).hexdigest(), flush=True)
        sys.stdin.readline()
        decoder.UnlockFrame(pointer)
        sent += 1
