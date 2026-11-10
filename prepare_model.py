#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Export YOLO weights to a static ONNX model for one temporal batch.

Batch 1 retains checkpoint.onnx. Larger batches use checkpoint-bN.onnx and
export in an isolated directory, preserving batch 1 even if export fails.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent / ".ultralytics")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
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
    options = dict(format="onnx", imgsz=args.imgsz, opset=args.opset,
                   batch=args.batch_size, dynamic=False, simplify=False,
                   nms=False, device="cpu")
    if args.batch_size == 1:
        output = Path(str(model.export(**options))).resolve()
    else:
        # YOLO has resolved/downloaded the checkpoint. Reload from its copy:
        # the exporter derives its output path from the loaded checkpoint.
        checkpoint = Path(model.ckpt_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = checkpoint.with_name(f"{checkpoint.stem}-b{args.batch_size}.onnx")
        with tempfile.TemporaryDirectory(prefix=".batch-export-", dir=checkpoint.parent) as directory:
            temporary_checkpoint = Path(directory) / checkpoint.name
            shutil.copy2(checkpoint, temporary_checkpoint)
            temporary_model = model_factory(str(temporary_checkpoint))
            exported = Path(str(temporary_model.export(**options))).resolve()
            if exported.parent != Path(directory).resolve():
                raise RuntimeError(f"Exporter wrote outside its isolated directory: {exported}")
            exported.replace(output)
    return {"checkpoint": args.model, "onnx": str(output), "imgsz": args.imgsz,
            "opset": args.opset, "batch_size": args.batch_size}


def main():
    print(json.dumps(export_model(parse_args()), indent=2))


if __name__ == "__main__":
    main()
