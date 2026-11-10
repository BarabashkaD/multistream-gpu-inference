# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The detector and the inference stage: the ONNX model runs on GPU tensors through IO binding, with no host copy."""
from __future__ import annotations

import ast
from model_shape import normalize_size, validate_metadata
from pathlib import Path
import threading
import time

import numpy as np
import onnxruntime as ort
import torch

from pipeline.bundle import BundlePool, Part
from pipeline.metrics import Metrics
from pipeline.stage import END, Stage, get_until, put_until


class Detector:
    """Runs the ONNX model on GPU tensors through IO binding, with no host copy."""

    def __init__(self, model_path: Path, gpu_id: int, batch_size: int = 1, size: int | tuple[int, int] = 640):
        """Load the model on the given GPU and read its input and output names, output shape and class names.

        Raises RuntimeError if the CUDA execution provider is not active, because a silent CPU fallback would be
        far slower. The class names come from the model's "names" metadata entry.
        """
        self.gpu_id = gpu_id
        self.session = ort.InferenceSession(
            str(model_path),
            providers=[("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"],
        )
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(f"CUDA provider inactive: {self.session.get_providers()}")
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        height, width = normalize_size(size)
        expected = (batch_size, 3, height, width)
        if tuple(input_info.shape) != expected or input_info.type != "tensor(float)":
            raise ValueError(f"Expected static FP32 ONNX input {expected}, got {input_info.shape} {input_info.type}; re-export with --batch-size {batch_size} --imgsz {height} {width}")
        if tuple(output_info.shape) != (batch_size, 300, 6) or output_info.type != "tensor(float)":
            raise ValueError(f"Expected static FP32 output {(batch_size, 300, 6)}, got {output_info.shape} {output_info.type}")
        self.batch_size = batch_size
        self.input_shape = expected
        self.input_name = input_info.name
        output = self.session.get_outputs()[0]
        self.output_name = output.name
        self.output_shape = tuple(output.shape)
        metadata = self.session.get_modelmeta().custom_metadata_map
        validate_metadata(metadata, (height, width), batch_size)
        self.names = ast.literal_eval(metadata["names"])
        self.binding = self.session.io_binding()

    def run(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        """Run the model on the input tensor and write the result into the output tensor, both on this GPU.

        Both tensors must be contiguous float32 and keep their memory until the call returns, which waits for the
        outputs to be complete.
        """
        self.binding.bind_input(self.input_name, "cuda", self.gpu_id, np.float32,
                                tuple(input_tensor.shape), input_tensor.data_ptr())
        self.binding.bind_output(self.output_name, "cuda", self.gpu_id, np.float32,
                                 tuple(output_tensor.shape), output_tensor.data_ptr())
        try:
            self.session.run_with_iobinding(self.binding)
            self.binding.synchronize_outputs()
        except BaseException:
            # ORT may use its own CUDA stream. Never release/reuse leased pointers
            # until all device work has stopped, including a failed provider call.
            torch.cuda.synchronize(self.gpu_id)
            raise


class InferenceStage(Stage):
    """Collect ordered prepared frames, infer once, then transfer each real frame to tracking.

    EOF/loop boundaries flush a partial batch using duplicate-last padding. Stop/error
    cancels unsubmitted frames; it never waits for a full batch during shutdown.
    """

    def __init__(self, detector: Detector, pool: BundlePool, metrics: Metrics,
                 stop: threading.Event, failures: list, batch_size: int = 1):
        super().__init__("inference", stop, failures)
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.detector, self.pool, self.metrics = detector, pool, metrics
        self.batch_size = batch_size
        self.batch_input = self.batch_output = None

    def _synchronize(self, tensor) -> None:
        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()

    def _infer(self, pending: list) -> None:
        count = len(pending)
        if not count:
            return
        started = time.perf_counter()
        if self.batch_size == 1:
            self.detector.run(pending[0].input_tensor, pending[0].detections)
        else:
            sample = pending[0]
            if self.batch_input is None:
                self.batch_input = sample.input_tensor.new_empty((self.batch_size, *sample.input_tensor.shape[1:]))
                self.batch_output = sample.detections.new_empty((self.batch_size, *sample.detections.shape[1:]))
            for index in range(self.batch_size):
                self.batch_input[index:index+1].copy_(pending[min(index, count-1)].input_tensor)
            # ORT owns a different stream; complete staging before handing over its pointer.
            self._synchronize(self.batch_input)
            self.detector.run(self.batch_input, self.batch_output)
            for index, bundle in enumerate(pending):
                bundle.detections.copy_(self.batch_output[index:index+1])
            # No bundle can be reclaimed or read by Tracker before scatter completes.
            self._synchronize(self.batch_output)
        elapsed = (time.perf_counter() - started) * 1e3
        self.metrics.record_ms("infer", elapsed)
        self.metrics.record_ms("infer_per_frame", elapsed / count)
        self.metrics.count("inference_calls")
        self.metrics.count("full_batches" if count == self.batch_size else "partial_batches")
        self.metrics.count("padding_slots", self.batch_size - count)
        self.metrics.count(f"batch_frames_{count}")
        # Remove a lease from pending only after it has been transferred successfully.
        while pending:
            bundle = pending[0]
            self.pool.mark(bundle, Part.DETECTIONS)
            self.metrics.count("inferred")
            self.pool.to_track.put(bundle)
            pending.pop(0)

    def work(self) -> None:
        pending = []
        received = None
        first_at = 0.0
        try:
            while True:
                received = get_until(self.pool.to_detect, self.stop_event)
                if received is None:
                    return
                if received is END:
                    self._flush(pending, first_at)
                    put_until(self.pool.to_track, END, self.stop_event)
                    received = None
                    return
                if pending and received.loop != pending[0].loop:
                    self._flush(pending, first_at)
                if not pending:
                    first_at = time.perf_counter()
                pending.append(received)
                received = None
                if len(pending) == self.batch_size:
                    self._flush(pending, first_at)
        finally:
            if received is not None and received is not END:
                pending.append(received)
            # Synchronize outstanding PyTorch copies before leases can be reclaimed on error.
            if pending:
                if pending[0].input_tensor.is_cuda:
                    torch.cuda.synchronize(pending[0].input_tensor.device)
                else:
                    self._synchronize(pending[0].input_tensor)
            for bundle in pending:
                self.pool.abandon(bundle)
                self.metrics.count("inference_cancelled")

    def _flush(self, pending: list, first_at: float) -> None:
        if pending:
            self.metrics.record_ms("batch_fill_wait", (time.perf_counter() - first_at) * 1e3)
            if self.stop_event.is_set():
                return
            self._infer(pending)
