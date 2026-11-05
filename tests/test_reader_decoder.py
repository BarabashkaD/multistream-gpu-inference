# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU tests of the reader and decoder stages: frame counts, display order, looping and clean shutdown.

These need a CUDA device and NVDEC and are marked `gpu`.
"""
import ctypes
import hashlib
import queue
import sys
import threading

import PyNvVideoCodec as nvc
import pytest
import torch

from pipeline.bundle import Bundle, BundlePool
from pipeline.config import InstanceConfig
from pipeline.decoder import DecoderStage
from pipeline.gpu import use_gpu
from pipeline.metrics import Metrics
from pipeline.source import MemorySource, ReaderStage
from pipeline.stage import END, get_until

pytestmark = pytest.mark.gpu
NV12_BYTES = 1920 * 1620  # tight NV12: 1080 luma rows plus 540 chroma rows
_cuda = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
_cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]


def digest_of(pointer):
    """Copy one tight NV12 frame from the GPU address to the host and return its SHA-256 hex digest."""
    host = ctypes.create_string_buffer(NV12_BYTES)
    assert _cuda.cuMemcpyDtoH_v2(host, pointer, NV12_BYTES) == 0
    return hashlib.sha256(host.raw).hexdigest()


def run_decode(clip_path, loops, pool_size=16, digests=None, stop_after=None):
    """Run the reader and decoder stages on the clip and collect what reaches the inference queue.

    Frames are abandoned back to the pool as they arrive. With `digests` given, the first 30 frames are hashed
    into it; with `stop_after` given, the run is stopped after that many frames. Returns the list of
    (loop, pts, frame_id), the set of distinct buffer pointers and the pool statistics. Fails if a stage
    does not stop within 10 s or reports a failure, and (for a full run) if the decoder leaves frames locked.
    """
    use_gpu(0)  # this thread copies frames to the host
    config = InstanceConfig(address="unused", input=clip_path, loops=loops, pool_size=pool_size, pace="max",
                            display=False)
    stop, failures = threading.Event(), []
    source = MemorySource(clip_path)
    packets = queue.Queue(maxsize=config.packet_queue)
    returned = queue.SimpleQueue()
    pool = BundlePool([Bundle(i) for i in range(pool_size)], returned.put, display_enabled=False)
    metrics = Metrics()
    stages = [ReaderStage(source, packets, config, metrics, stop, failures),
              DecoderStage(source, packets, pool, returned, config, metrics, stop, failures)]
    for stage in stages:
        stage.start()
    seen, buffers = [], set()
    while True:
        item = get_until(pool.to_infer, stop)
        if item is None or item is END:
            break
        seen.append((item.loop, item.pts, item.frame_id))
        buffers.add(item.nv12_pointer)
        if digests is not None and len(digests) < 30:
            digests.append(digest_of(item.nv12_pointer))
        pool.abandon(item)
        if stop_after is not None and len(seen) >= stop_after:
            stop.set()
    for stage in stages:
        stage.join(10)
    assert not any(stage.is_alive() for stage in stages), "a stage did not stop"
    assert not failures, failures
    if stop_after is None:
        assert stages[1].locked == 0, "every locked frame must be unlocked before the decoder thread ends"
    return seen, buffers, pool.stats()


def first_frames_from_simple_decoder(clip_path, count):
    """SHA-256 of the first frames from `nvc.SimpleDecoder`, in native NV12 with tight stride.

    A DecodedFrame exposes DLPack as one tight (height * 3 / 2, width) uint8 plane, which is the layout
    the server's frame import assumes. The assertion keeps this helper honest if the library changes.
    """
    decoder = nvc.SimpleDecoder(clip_path, gpu_id=0, use_device_memory=True,
                                output_color_type=nvc.OutputColorType.NATIVE)
    digests = []
    for index in range(count):
        frame = torch.utils.dlpack.from_dlpack(decoder[index])
        assert (tuple(frame.shape), frame.stride()) == ((1620, 1920), (1920, 1))
        digests.append(hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest())
    return digests


def test_source_metadata(clip_path):
    """The in-memory source reports the clip's size as 1920 by 1080 at 25 frames per second."""
    source = MemorySource(clip_path)
    assert (source.width, source.height, source.fps) == (1920, 1080, 25.0)


def test_one_pass_delivers_every_frame_in_display_order(clip_path):
    """One pass yields all 611 frames with rising timestamps and consecutive ids, and returns every pool slot."""
    seen, buffers, stats = run_decode(clip_path, 1)
    assert len(seen) == 611
    pts = [item[1] for item in seen]
    assert pts == sorted(pts)
    assert [item[2] for item in seen] == list(range(611))
    assert stats["free"] == stats["size"]


def test_frames_match_simple_decoder_in_order(clip_path):
    """Pixels prove the order. `pts` proves nothing, because it comes from this stage's own heap."""
    digests = []
    run_decode(clip_path, 1, digests=digests)
    assert digests == first_frames_from_simple_decoder(clip_path, 30)


def test_two_loops_with_one_decoder(clip_path):
    """Two loops over the clip deliver 611 frames per loop from one decoder and return every pool slot."""
    seen, _, stats = run_decode(clip_path, 2)
    assert len(seen) == 1222
    assert sum(item[0] == 0 for item in seen) == 611
    assert sum(item[0] == 1 for item in seen) == 611
    assert stats["free"] == stats["size"]


def test_stop_mid_stream_ends_both_stages(clip_path):
    """Setting stop after 20 frames ends both stages before the clip is finished."""
    seen, _, _ = run_decode(clip_path, 1, stop_after=20)
    assert 20 <= len(seen) < 611


def test_small_pool_bounds_decoder_buffers(clip_path):
    """A pool of four bundles keeps the number of distinct decoder buffers at twelve or fewer."""
    _, buffers, _ = run_decode(clip_path, 1, pool_size=4)
    assert len(buffers) <= 12
