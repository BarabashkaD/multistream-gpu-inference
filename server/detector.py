# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The object detector: ONNX Runtime sessions that read and write GPU tensors directly."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class _Session:
    """One ONNX Runtime session for a fixed batch size, bound to preallocated input and output GPU tensors."""

    def __init__(self, path: Path, size: int, gpu_id: int):
        """Loads the model on the given GPU and binds its input and output to tensors allocated once.

        Raises RuntimeError when the CUDA provider is not active or the model's batch dimension is not `size`.
        The arena and the memory pattern are off so the session holds no more GPU memory than its tensors need."""
        provider = ("CUDAExecutionProvider", {"device_id": gpu_id, "arena_extend_strategy": "kSameAsRequested"})
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=[provider, "CPUExecutionProvider"])
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(f"CUDA provider inactive: {self.session.get_providers()}")
        node = self.session.get_inputs()[0]
        output = self.session.get_outputs()[0]
        if node.shape[0] != size or output.shape[0] != size:
            raise RuntimeError(f"{path.name} is not a batch {size} model: {node.shape} {output.shape}")
        self.input_name, self.output_name = node.name, output.name
        self.height, self.width = node.shape[2], node.shape[3]
        device = f"cuda:{gpu_id}"
        self.input = torch.zeros(tuple(node.shape), dtype=torch.float32, device=device)
        self.output = torch.zeros(tuple(output.shape), dtype=torch.float32, device=device)
        self.binding = self.session.io_binding()
        self.binding.bind_input(self.input_name, "cuda", gpu_id, np.float32, tuple(self.input.shape),
                                self.input.data_ptr())
        self.binding.bind_output(self.output_name, "cuda", gpu_id, np.float32, tuple(self.output.shape),
                                 self.output.data_ptr())

    def run(self) -> np.ndarray:
        """Runs the model on the bound input and returns a host copy of the output once the GPU has finished."""
        self.session.run_with_iobinding(self.binding)
        self.binding.synchronize_outputs()
        return self.output.cpu().numpy()


class BatchDetector:
    """One ONNX session per batch size. Each is warmed up, so the first real frame is not slow."""

    def __init__(self, models: Path, sizes: tuple[int, ...], gpu_id: int):
        """Loads and warms up models/b<size>.onnx for every size, then reads the input size and class names.

        Raises FileNotFoundError when a model file is missing and RuntimeError when a model does not fit its size
        or the CUDA provider is unavailable. Takes the input size and class names from the first model."""
        self.sessions = {}
        for size in sizes:
            path = Path(models) / f"b{size}.onnx"
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; run tools/export_models.py")
            self.sessions[size] = _Session(path, size, gpu_id)
            self.sessions[size].run()
        first = next(iter(self.sessions.values()))
        self.input_size = (first.width, first.height)
        metadata = first.session.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(metadata["names"])

    def input(self, size: int) -> torch.Tensor:
        """The GPU tensor the model of this batch size reads its input from; callers fill it in place."""
        return self.sessions[size].input

    def run(self, size: int) -> np.ndarray:
        """Runs the model of this batch size on its input tensor and returns the output as a host array.

        ONNX Runtime reads the bound input by raw pointer, so the caller must have synchronised any GPU work
        that fills it."""
        return self.sessions[size].run()
