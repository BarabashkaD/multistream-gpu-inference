# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The decoder stage: decodes packets on the GPU and publishes each frame into a bundle at its due time."""
from __future__ import annotations

import heapq
import queue
import threading
import time

import PyNvVideoCodec as nvc

from pipeline.bundle import BundlePool
from pipeline.config import PipelineConfig
from pipeline.metrics import Metrics
from pipeline.nv12 import wrap_nv12
from pipeline.source import MemorySource
from pipeline.stage import END, Stage, get_until, put_until


class DecoderStage(Stage):
    """Only this thread touches the decoder, including unlocking returned buffers."""

    def __init__(self, source: MemorySource, packets: queue.Queue, pool: BundlePool, returned: queue.SimpleQueue,
                 config: PipelineConfig, metrics: Metrics, stop: threading.Event, failures: list):
        """Wire the stage to its source, packet queue, bundle pool and the queue of frame pointers to unlock."""
        super().__init__("decoder", stop, failures)
        self.source = source
        self.packets = packets
        self.pool = pool
        self.returned = returned
        self.config = config
        self.metrics = metrics
        self.pending: list[tuple[int, int]] = []
        self.frame_id = 0
        self.started_at = 0.0
        self.decoder = None  # kept on the stage so it outlives every view of its memory
        self.locked = 0

    def work(self) -> None:
        """Create the decoder on the configured GPU and decode until the stream ends or the stage is stopped.

        Output stays in device memory in the native NV12 layout. Frames that already came back are unlocked on
        every exit path.
        """
        self.decoder = nvc.CreateDecoder(gpuid=self.config.gpu_id, codec=self.source.codec, usedevicememory=True,
                                         outputColorType=nvc.OutputColorType.NATIVE)
        self.started_at = time.monotonic()
        try:
            self._decode_packets()
        finally:
            # A stop or failure must not strand frames that already came back. Frames still held
            # elsewhere stay locked until this stage is dropped, which happens when run_pipeline returns.
            self._unlock_returned()

    def _decode_packets(self) -> None:
        """Feed queued packets to the decoder; at END flush it, signal the prepare queue and wait for frames to return.

        Each packet's loop and timestamp are queued in a heap so they can be matched to the frames the decoder emits,
        which may come out in a different order.
        """
        while True:
            item = get_until(self.packets, self.stop_event)
            if item is None:
                return
            if item is END:
                self._emit(nvc.PacketData())
                put_until(self.pool.to_infer, END, self.stop_event)
                self._drain()
                return
            heapq.heappush(self.pending, (item.loop, item.packet.pts))
            self._emit(item.packet)

    def _emit(self, packet) -> None:
        """Decode one packet and publish each frame it yields, paced to the source frame rate.

        Each frame needs a free bundle first; if the pool gives none because the stage is stopping, the rest are
        skipped. Frame ids are assigned in output order. An empty packet flushes the frames the decoder still holds.
        """
        started = time.perf_counter()
        count = self.decoder.GetNumDecodedFrame(packet)
        self.metrics.record_ms("decode", (time.perf_counter() - started) * 1e3)
        for _ in range(count):
            bundle = self.pool.acquire(self.stop_event)
            if bundle is None:
                return
            self._unlock_returned()
            started = time.perf_counter()
            pointer = self.decoder.GetLockedFrame()
            self.locked += 1
            self.decoder.SyncOnCUStream()
            self.metrics.record_ms("lock", (time.perf_counter() - started) * 1e3)
            loop, pts = heapq.heappop(self.pending)
            due = self.started_at + self.frame_id / self.source.fps
            self._pace(due)
            nv12 = wrap_nv12(pointer, self.source.width, self.source.height, self.config.gpu_id)
            self.pool.publish_frame(bundle, pointer, nv12, self.frame_id, loop, pts, due_at=due)
            self.frame_id += 1
            self.metrics.count("decoded")

    def _unlock_returned(self) -> None:
        """Unlock every frame whose pointer has been returned by the pool, without waiting for more."""
        while True:
            try:
                pointer = self.returned.get_nowait()
            except queue.Empty:
                return
            self.decoder.UnlockFrame(pointer)
            self.locked -= 1

    def _drain(self) -> None:
        """Frames in flight at end of stream come back here, because only this thread may unlock."""
        while self.locked and not self.stop_event.is_set():
            self._unlock_returned()
            self.stop_event.wait(0.005)
        self._unlock_returned()

    def _pace(self, due: float) -> None:
        """In realtime mode, sleep until the frame's due time; a stop wakes the wait early."""
        if self.config.pace != "realtime":
            return
        delay = due - time.monotonic()
        if delay > 0:
            self.stop_event.wait(delay)
