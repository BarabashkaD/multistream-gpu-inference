# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
"""Shared static YOLO input-size contract (height, width)."""
import ast


def normalize_size(value):
    values = (value, value) if isinstance(value, int) else tuple(value)
    if len(values) == 1:
        values = values * 2
    if len(values) != 2 or any(type(v) is not int or v <= 0 or v % 32 for v in values):
        raise ValueError("imgsz needs one or two positive multiples of 32 (H W)")
    return values


def model_suffix(size, batch):
    height, width = normalize_size(size)
    return ("" if (height, width) == (640, 640) else f"-{width}x{height}") + ("" if batch == 1 else f"-b{batch}")


def validate_metadata(metadata, size, batch):
    """Accept legacy missing metadata; reject contradictory export declarations."""
    if "imgsz" in metadata and normalize_size(ast.literal_eval(metadata["imgsz"])) != normalize_size(size):
        raise ValueError("ONNX imgsz metadata disagrees with input shape")
    if "batch" in metadata and int(metadata["batch"]) != batch:
        raise ValueError("ONNX batch metadata disagrees with input shape")
    if "dynamic" in metadata and str(metadata["dynamic"]).lower() not in ("false", "0"):
        raise ValueError("Only static ONNX models are supported")
