# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Shared static model shape and letterbox geometry contracts (H, W)."""
import argparse
import ast


def normalize_imgsz(value):
    values = [value] if isinstance(value, int) else list(value)
    if len(values) == 1:
        values *= 2
    if len(values) != 2 or any(type(v) is not int or v <= 0 or v % 32 for v in values):
        raise ValueError('--imgsz requires one or two positive multiples of 32, in H W order')
    return tuple(values)


class ImageSizeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, normalize_imgsz(values))
        except ValueError as exc:
            parser.error(str(exc))


def add_imgsz_argument(parser):
    parser.add_argument('--imgsz', type=int, nargs='+', action=ImageSizeAction,
                        default=(640, 640), metavar='N',
                        help='One square size or H W, multiples of 32 (default: 640 640)')


def model_filename(checkpoint, batch, imgsz):
    h, w = normalize_imgsz(imgsz)
    suffix = '' if (h, w) == (640, 640) else f'-{w}x{h}'
    if batch != 1:
        suffix += f'-b{batch}'
    return checkpoint.with_name(f'{checkpoint.stem}{suffix}.onnx')


def letterbox_geometry(orig_h, orig_w, imgsz):
    h, w = normalize_imgsz(imgsz)
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError('Source dimensions must be positive')
    ratio = min(h / orig_h, w / orig_w)
    new_h, new_w = max(round(orig_h * ratio), 1), max(round(orig_w * ratio), 1)
    top, left = round((h-new_h)/2-.1), round((w-new_w)/2-.1)
    return dict(orig_h=orig_h, orig_w=orig_w, target_h=h, target_w=w,
                new_h=new_h, new_w=new_w, pad_top=top, pad_left=left,
                pad_bottom=h-new_h-top, pad_right=w-new_w-left, ratio=ratio,
                gain_x=new_w/orig_w, gain_y=new_h/orig_h)


def validate_model_shape(path, batch_size, imgsz):
    """Verify graph AND metadata used by Ultralytics before decoding/export commit."""
    import onnx
    h, w = normalize_imgsz(imgsz)
    model = onnx.load(str(path), load_external_data=False)
    expected = [batch_size, 3, h, w]
    hint = f'export with --batch-size {batch_size} --imgsz {h} {w}'
    if len(model.graph.input) != 1:
        raise ValueError(f'Expected one ONNX input; {hint}')
    tensor = model.graph.input[0].type.tensor_type
    shape = [d.dim_value if d.HasField('dim_value') else None for d in tensor.shape.dim]
    if shape != expected or tensor.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(f'ONNX input {shape} type {tensor.elem_type} does not match FP32 {expected}; {hint}')
    metadata = {p.key: p.value for p in model.metadata_props}
    try:
        metadata_size = normalize_imgsz(ast.literal_eval(metadata['imgsz']))
        metadata_batch = ast.literal_eval(metadata['batch'])
        dynamic = ast.literal_eval(metadata.get('args', '{}')).get('dynamic', False)
        if 'dynamic' in metadata:
            dynamic = ast.literal_eval(metadata['dynamic'])
    except (KeyError, ValueError, SyntaxError, TypeError, AttributeError) as exc:
        raise ValueError(f'Missing or invalid ONNX batch/imgsz metadata; {hint}') from exc
    if (metadata_size != (h,w) or type(metadata_batch) is not int
            or metadata_batch != batch_size or dynamic is not False):
        raise ValueError(f'ONNX metadata disagrees with static input {expected}; {hint}')
    return shape
