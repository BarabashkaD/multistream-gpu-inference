# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""GPU tests for the reader and decoder stages: source metadata, frame order and count, looping, stopping and
decoder buffer use."""
import hashlib
import queue
import threading

import PyNvVideoCodec as nvc
import pytest
import torch

from pipeline.bundle import Bundle, BundlePool
from pipeline.config import PipelineConfig
from pipeline.decoder import DecoderStage
from pipeline.metrics import Metrics
from pipeline.source import MemorySource, ReaderStage
from pipeline.stage import END, get_until

pytestmark = pytest.mark.gpu


def run_decode(clip_path, loops, pool_size=16, digests=None, stop_after=None):
    """Runs the reader and decoder stages over the clip and drains the frames they publish, abandoning each one.
    Returns the (loop, pts, frame id) of every frame, the set of decoder buffer pointers seen and the pool
    statistics. Appends up to 30 frame digests to `digests` if given, and sets stop after `stop_after` frames. Fails
    if a stage does not stop or fails, or, for a full run, if the decoder ends with frames still locked."""
    config = PipelineConfig(input=clip_path, loops=loops, pool_size=pool_size, pace="max", display=False)
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
            digests.append(hashlib.sha256(item.nv12.cpu().numpy().tobytes()).hexdigest())
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
    `wrap_nv12` assumes. The assertion keeps this helper honest if the library changes.
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
    """Proves the source reports the clip's width, height and frame rate."""
    source = MemorySource(clip_path)
    assert (source.width, source.height, source.fps) == (1920, 1080, 25.0)


def test_one_pass_delivers_every_frame_in_display_order(clip_path):
    """Proves one pass delivers all 611 frames with rising timestamps and consecutive frame ids, and every bundle
    comes back to the pool."""
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
    """Proves two loops deliver the clip twice, 611 frames per loop, through one decoder, with every bundle
    returned."""
    seen, _, stats = run_decode(clip_path, 2)
    assert len(seen) == 1222
    assert sum(item[0] == 0 for item in seen) == 611
    assert sum(item[0] == 1 for item in seen) == 611
    assert stats["free"] == stats["size"]


def test_stop_mid_stream_ends_both_stages(clip_path):
    """Proves setting the stop event partway through ends both stages without a failure, before the clip is done."""
    seen, _, _ = run_decode(clip_path, 1, stop_after=20)
    assert 20 <= len(seen) < 611


def test_small_pool_bounds_decoder_buffers(clip_path):
    """Proves a pool of four bundles keeps the decoder to at most twelve distinct buffers."""
    _, buffers, _ = run_decode(clip_path, 1, pool_size=4)
    assert len(buffers) <= 12
