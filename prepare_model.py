#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Export YOLO weights to a static ONNX model for one temporal batch.

Default 640x640 batch 1 retains checkpoint.onnx. Nondefault shapes/batches use
distinct names. Isolated export preserves existing models if export/validation fails.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from input_shape import add_imgsz_argument, normalize_imgsz, model_filename, validate_model_shape

os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent / ".ultralytics")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n.pt")
    add_imgsz_argument(parser)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    return parser.parse_args(argv)


def export_model(args, model_factory=None):
    """Export on CPU, retaining distinct filenames for each static batch size."""
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if model_factory is None:
        from ultralytics import YOLO
        model_factory = YOLO
    model = model_factory(args.model)
    imgsz = normalize_imgsz(args.imgsz)
    options = dict(format="onnx", imgsz=imgsz, opset=args.opset,
                   batch=args.batch_size, dynamic=False, simplify=False,
                   nms=False, device="cpu")
    checkpoint = Path(model.ckpt_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = model_filename(checkpoint, args.batch_size, imgsz)
    # Always isolate export: failed validation must not replace a working model.
    with tempfile.TemporaryDirectory(prefix=".batch-export-", dir=checkpoint.parent) as directory:
        temporary_checkpoint = Path(directory) / checkpoint.name
        shutil.copy2(checkpoint, temporary_checkpoint)
        temporary_model = model_factory(str(temporary_checkpoint))
        exported = Path(str(temporary_model.export(**options))).resolve()
        if exported.parent != Path(directory).resolve():
            raise RuntimeError(f"Exporter wrote outside its isolated directory: {exported}")
        validate_model_shape(exported, args.batch_size, imgsz)
        exported.replace(output)
    return {"checkpoint": args.model, "onnx": str(output), "imgsz": args.imgsz,
            "opset": args.opset, "batch_size": args.batch_size}


def main():
    print(json.dumps(export_model(parse_args()), indent=2))


if __name__ == "__main__":
    main()
