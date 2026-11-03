# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The input side: a clip held in memory, owned copies of its packets, and the stage that reads them in a loop."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time

import PyNvVideoCodec as nvc

from pipeline.config import PipelineConfig
from pipeline.metrics import Metrics
from pipeline.stage import END, Stage, put_until


class MemorySource:
    """Holds the compressed clip in RAM and creates demuxers over it."""

    def __init__(self, path: Path):
        """Read the whole file into memory and probe it for frame size, frame rate and codec."""
        self.data = Path(path).read_bytes()
        probe = self.demuxer()
        self.width = int(probe.Width())
        self.height = int(probe.Height())
        self.fps = float(probe.FrameRate())
        self.codec = probe.GetNvCodecId()

    def demuxer(self):
        """Create a new demuxer reading the in-memory clip from its start, so each pass has its own position."""
        data = self.data
        position = [0]

        def feed(buffer):
            """Fill the demuxer's buffer with the next bytes of the clip and return how many were written."""
            chunk = data[position[0]:position[0] + len(buffer)]
            buffer[:len(chunk)] = chunk
            position[0] += len(chunk)
            return len(chunk)

        return nvc.CreateDemuxer(feed)


@dataclass
class OwnedPacket:
    """A packet with the memory that backs it, and the pass of the clip it came from."""

    packet: object
    storage: object
    loop: int


def own_packet(source, loop: int) -> OwnedPacket:
    """Copy a demuxed packet. Its memory is freed by the next Demux() call."""
    storage = ctypes.create_string_buffer(ctypes.string_at(source.bsl_data, source.bsl), source.bsl)
    packet = nvc.PacketData()
    packet.bsl = source.bsl
    packet.bsl_data = ctypes.addressof(storage)
    packet.pts = source.pts
    packet.dts = source.dts
    packet.key = source.key
    packet.duration = source.duration
    packet.pos = source.pos
    packet.decode_flag = source.decode_flag
    return OwnedPacket(packet, storage, loop)


class ReaderStage(Stage):
    """Demuxes the clip and queues owned packets for the decoder, repeating the clip as many times as configured."""

    def __init__(self, source: MemorySource, packets: queue.Queue, config: PipelineConfig,
                 metrics: Metrics, stop: threading.Event, failures: list):
        """Wire the stage to its source and the packet queue it fills."""
        super().__init__("reader", stop, failures)
        self.source = source
        self.packets = packets
        self.config = config
        self.metrics = metrics

    def work(self) -> None:
        """Queue every packet of each pass, then the end marker; return early if the stage is stopped.

        The queue is bounded, so this thread waits when the decoder falls behind. A loops value of 0 repeats the
        clip until stopped, in which case the end marker is never queued.
        """
        loop = 0
        while self.config.loops == 0 or loop < self.config.loops:
            demuxer = self.source.demuxer()
            while True:
                started = time.perf_counter()
                packet = demuxer.Demux()
                if packet.bsl == 0:
                    break
                owned = own_packet(packet, loop)
                self.metrics.record_ms("read", (time.perf_counter() - started) * 1e3)
                if not put_until(self.packets, owned, self.stop_event):
                    return
                self.metrics.count("packets")
            loop += 1
        put_until(self.packets, END, self.stop_event)
