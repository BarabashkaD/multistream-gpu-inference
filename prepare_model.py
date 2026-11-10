#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Export a separate static ONNX model for each temporal batch size."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from model_shape import normalize_size, model_suffix, validate_metadata

os.environ['YOLO_CONFIG_DIR'] = str(Path(__file__).resolve().parent / '.ultralytics')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--imgsz', type=positive_int, nargs='+', default=[640], help='S or H W')
    parser.add_argument('--opset', type=positive_int, default=12)
    parser.add_argument('--batch-size', type=positive_int, default=1)
    args = parser.parse_args(argv)
    try:
        args.imgsz = normalize_size(args.imgsz)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def export_model(args, model_factory=None, validator=None):
    if model_factory is None:
        from ultralytics import YOLO
        model_factory = YOLO
    if validator is None:
        validator = validate_export
    model = model_factory(args.model)
    checkpoint = Path(model.ckpt_path).resolve()
    args.imgsz = normalize_size(args.imgsz)
    suffix = model_suffix(args.imgsz, args.batch_size)
    output = checkpoint.with_name(checkpoint.stem + suffix + '.onnx')
    with tempfile.TemporaryDirectory(prefix='.batch-export-', dir=checkpoint.parent) as directory:
        temporary = Path(directory) / checkpoint.name
        shutil.copy2(checkpoint, temporary)
        exported = Path(str(model_factory(str(temporary)).export(
            format='onnx', imgsz=args.imgsz, opset=args.opset, batch=args.batch_size,
            dynamic=False, simplify=False, nms=False, device='cpu'))).resolve()
        if exported.parent != Path(directory).resolve():
            raise RuntimeError('Exporter wrote outside its isolated directory')
        validator(exported, args.batch_size, args.imgsz)
        exported.replace(output)
    return {'checkpoint': str(checkpoint), 'onnx': str(output), 'imgsz': args.imgsz,
            'batch_size': args.batch_size, 'opset': args.opset}


def validate_export(path, batch, size):
    import onnx
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    height, width = normalize_size(size)
    validate_metadata({p.key: p.value for p in model.metadata_props}, size, batch)
    for node, shape in ((model.graph.input[0], (batch, 3, height, width)),
                        (model.graph.output[0], (batch, 300, 6))):
        tensor = node.type.tensor_type
        actual = tuple(d.dim_value for d in tensor.shape.dim)
        if actual != shape or tensor.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f'Expected static FP32 {shape}; got {actual}')


if __name__ == '__main__':
    print(json.dumps(export_model(parse_args()), indent=2))
