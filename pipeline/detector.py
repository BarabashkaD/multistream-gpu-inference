# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The detector and the inference stage: the ONNX model runs on GPU tensors through IO binding, with no host copy."""
from __future__ import annotations

import ast
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

    def __init__(self, model_path: Path, gpu_id: int):
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
        self.input_name = self.session.get_inputs()[0].name
        output = self.session.get_outputs()[0]
        self.output_name = output.name
        self.output_shape = tuple(output.shape)
        self.names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map["names"])
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
        self.session.run_with_iobinding(self.binding)
        self.binding.synchronize_outputs()


class InferenceStage(Stage):
    """Runs the detector on each prepared bundle and passes it on to the tracker."""

    def __init__(self, detector: Detector, pool: BundlePool, metrics: Metrics,
                 stop: threading.Event, failures: list):
        """Wire the stage to the detector and the bundle pool."""
        super().__init__("inference", stop, failures)
        self.detector = detector
        self.pool = pool
        self.metrics = metrics

    def work(self) -> None:
        """Detect on each queued bundle, mark its detections done and queue it for tracking, until the end marker.

        Passes the end marker on to the tracker. The bundle's lease moves to the tracker queue with it.
        """
        while True:
            bundle = get_until(self.pool.to_detect, self.stop_event)
            if bundle is None:
                return
            if bundle is END:
                put_until(self.pool.to_track, END, self.stop_event)
                return
            started = time.perf_counter()
            self.detector.run(bundle.input_tensor, bundle.detections)
            self.metrics.record_ms("infer", (time.perf_counter() - started) * 1e3)
            self.pool.mark(bundle, Part.DETECTIONS)
            self.metrics.count("inferred")
            self.pool.to_track.put(bundle)  # the lease moves with the bundle
