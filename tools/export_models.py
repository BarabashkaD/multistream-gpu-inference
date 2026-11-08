#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Export fixed-batch detector models. Each file is flushed to disk, because a crash once left zeroed files.

Usage: export_models.py --weights yolo26n.pt --sizes 1,8,16
The export settings matter: other opsets or a dynamic export produce raw predictions, not [batch, 300, 6].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import urllib.request
import os
from pathlib import Path
import shutil

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before NumPy loads, through onnx or ultralytics

V3 = Path(__file__).resolve().parents[1]
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
WEIGHTS_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_weights(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, temporary)
            if sha256(temporary) != WEIGHTS_SHA256:
                raise ValueError("Downloaded weights SHA-256 mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(path) != WEIGHTS_SHA256:
        raise ValueError("Weights SHA-256 mismatch; use the pinned YOLO26n asset")
    return path



def main() -> None:
    """Exports one ONNX file per batch size (b<size>.onnx) into the output folder and prints each name and size.

    Runs on the CPU. Raises RuntimeError when an exported model's output shape is not [size, 300, 6], so a wrong
    export is never written. Needs onnx and ultralytics; only the pinned, hash-verified weights are accepted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n.pt", help="path to the pinned YOLO26n weights (downloaded if missing)")
    parser.add_argument("--out", type=Path, default=V3 / "models")
    parser.add_argument("--sizes", default="1,8,16")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = verified_weights(Path(args.weights).resolve())
    os.environ["YOLO_CONFIG_DIR"] = str(V3 / ".ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(args.out)
    import onnx
    from ultralytics import YOLO

    sizes = [int(text) for text in args.sizes.split(",")]
    if not sizes or any(size <= 0 for size in sizes) or len(sizes) != len(set(sizes)):
        parser.error("sizes must be distinct positive integers")
    manifest = {"schema": 1, "weights": {"url": WEIGHTS_URL, "sha256": WEIGHTS_SHA256,
        "license": "AGPL-3.0 (Ultralytics); distribution review required"},
        "export": {"width": args.width, "height": args.height, "opset": 12,
                   "dynamic": False, "simplify": False, "nms": False, "device": "cpu"},
        "packages": {name: importlib.metadata.version(name) for name in ("ultralytics", "onnx", "torch")},
        "models": {}}
    # Remove stale proof before writing any output; a failed export must not look verified.
    (args.out / "manifest.json").unlink(missing_ok=True)
    for size in sizes:
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
        manifest["models"][target.name] = {"sha256": sha256(target), "shape": shape}
        print(target.name, len(data), flush=True)
    temporary = args.out / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.out / "manifest.json")


if __name__ == "__main__":
    main()
