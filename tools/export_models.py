#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Export fixed-batch detector models. Each file is flushed to disk, because a crash once left zeroed files.

Usage: export_models.py --weights yolo26n.pt --sizes 1,4,8,16
The export settings matter: other opsets or a dynamic export produce raw predictions, not [batch, 300, 6].
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before NumPy loads, through onnx or ultralytics

V3 = Path(__file__).resolve().parents[1]


def main() -> None:
    """Exports one ONNX file per batch size (b<size>.onnx) into the output folder and prints each name and size.

    Runs on the CPU. Raises RuntimeError when an exported model's output shape is not [size, 300, 6], so a wrong
    export is never written. Needs onnx and ultralytics; Ultralytics may download the weights by name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n.pt", help="a .pt file, or a name Ultralytics can download")
    parser.add_argument("--out", type=Path, default=V3 / "models")
    parser.add_argument("--sizes", default="1,4,8,16")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights).resolve() if Path(args.weights).exists() else args.weights
    os.environ["YOLO_CONFIG_DIR"] = str(V3 / ".ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(args.out)
    import onnx
    from ultralytics import YOLO

    for size in (int(text) for text in args.sizes.split(",")):
        target = args.out / f"b{size}.onnx"
        work = args.out / "export-work.pt"
        if isinstance(weights, Path):
            shutil.copyfile(weights, work)
        exported = Path(YOLO(str(work) if isinstance(weights, Path) else weights).export(
            format="onnx", imgsz=(args.height, args.width), opset=12, batch=size, dynamic=False, simplify=False,
            nms=False, device="cpu", verbose=False))
        data = exported.read_bytes()
        model = onnx.load_from_string(data)
        shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
        if shape != [size, 300, 6]:
            raise RuntimeError(f"unexpected output shape {shape}")
        with target.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        print(target.name, len(data), flush=True)


if __name__ == "__main__":
    main()