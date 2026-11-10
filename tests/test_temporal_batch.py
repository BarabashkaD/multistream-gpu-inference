# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
"""CPU-only contracts: real detector stage with fake ORT and NumPy-backed tensors."""
import importlib.util
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pipeline.bundle import Bundle, BundlePool, Part
from pipeline.metrics import Metrics
from pipeline.stage import END
from pipeline.config import parse_config, validate_batch_pool


def load_detector():
    spec = importlib.util.spec_from_file_location('batch_detector_test', Path(__file__).parents[1] / 'pipeline/detector.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': SimpleNamespace(), 'onnxruntime': SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


class Tensor:
    is_cuda = False
    def __init__(self, values): self.values = np.asarray(values, dtype=np.float32)
    @property
    def shape(self): return self.values.shape
    def new_empty(self, shape): return Tensor(np.empty(shape, np.float32))
    def copy_(self, other): self.values[...] = other.values
    def __getitem__(self, item): return TensorView(self.values[item])


class TensorView(Tensor):
    def __init__(self, values): self.values = values


class Detector:
    def __init__(self, fail=False): self.calls = []; self.fail = fail
    def run(self, inp, out):
        self.calls.append(inp.values.copy())
        if self.fail: raise RuntimeError('inference failed')
        out.values[...] = inp.values.reshape(len(inp.values), -1)[:, :1, None]


def make_pool(count):
    return BundlePool([Bundle(i, Tensor(np.zeros((1, 3, 2, 2))), Tensor(np.zeros((1, 300, 6))))
                       for i in range(count)], lambda pointer: None, display_enabled=False)


def add_frames(pool, loops):
    stop = threading.Event()
    for frame, loop in enumerate(loops):
        bundle = pool.acquire(stop)
        pool.publish_frame(bundle, frame, None, frame, loop, frame)
        assert pool.to_infer.get_nowait() is bundle
        bundle.input_tensor.values.fill(frame + 1)
        pool.mark(bundle, Part.INPUT)
        pool.to_detect.put(bundle)


@pytest.mark.parametrize('batch', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('length', [0, 1, 3, 4, 5])
def test_full_tail_and_duplicate_padding(batch, length):
    module = load_detector(); pool = make_pool(max(1, length)); add_frames(pool, [0]*length)
    pool.to_detect.put(END)
    detector = Detector(); metrics = Metrics()
    module.InferenceStage(detector, pool, metrics, threading.Event(), [], batch).work()
    result = []
    while True:
        item = pool.to_track.get_nowait()
        if item is END: break
        result.append(item)
    assert [x.frame_id for x in result] == list(range(length))
    assert len(detector.calls) == (length + batch - 1)//batch
    for item in result:
        assert np.all(item.detections.values == item.frame_id + 1)
        pool.set_tracks(item, []); pool.release(item)
    if length % batch:
        assert np.all(detector.calls[-1][length % batch:] == detector.calls[-1][length % batch - 1])
    assert pool.stats()['leased'] == 0
    assert metrics.snapshot()['counts'].get('inferred', 0) == length


def test_loop_boundary_flushes_before_next_loop():
    module = load_detector(); pool = make_pool(6); add_frames(pool, [0, 0, 0, 1, 1, 1]); pool.to_detect.put(END)
    detector = Detector()
    module.InferenceStage(detector, pool, Metrics(), threading.Event(), [], 4).work()
    assert [call[:, 0, 0, 0].tolist() for call in detector.calls] == [[1,2,3,3], [4,5,6,6]]


def test_exception_abandons_batch_and_boundary_lookahead():
    module = load_detector(); pool = make_pool(3); add_frames(pool, [0, 0, 1]); detector = Detector(True)
    with pytest.raises(RuntimeError, match='inference failed'):
        module.InferenceStage(detector, pool, Metrics(), threading.Event(), [], 4).work()
    assert pool.stats()['leased'] == 0
    assert pool.stats()['abandoned'] == 3


def test_stop_cancels_pending_without_inference():
    module = load_detector(); pool = make_pool(2); add_frames(pool, [0, 0]); stop = threading.Event()
    real_get = module.get_until; taken = 0
    def get(*args):
        nonlocal taken
        if taken == 2:
            stop.set(); return None
        taken += 1
        return real_get(*args)
    module.get_until = get
    detector = Detector(); metrics = Metrics()
    module.InferenceStage(detector, pool, metrics, stop, [], 4).work()
    assert not detector.calls
    assert pool.stats()['leased'] == 0
    assert metrics.snapshot()['counts']['inference_cancelled'] == 2


def test_configuration():
    assert parse_config([]).batch_size == 1
    assert parse_config(['--batch-size','16']).pool_size == 18
    with pytest.raises(SystemExit): parse_config(['--batch-size','4','--pool-size','4'])
    with pytest.raises(ValueError): validate_batch_pool(0, 16)


def test_export_isolated_names_and_failure(tmp_path):
    import prepare_model
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'weights')
    old = tmp_path / 'model.onnx'; old.write_bytes(b'old')
    class Model:
        def __init__(self, path): self.ckpt_path = path
        def export(self, **kwargs):
            assert kwargs['batch'] == 4 and not kwargs['dynamic']
            out = Path(self.ckpt_path).with_suffix('.onnx'); out.write_bytes(b'new'); return out
    args = prepare_model.parse_args(['--model',str(checkpoint),'--batch-size','4'])
    result = prepare_model.export_model(args, Model, lambda *args: None)
    assert Path(result['onnx']).name == 'model-b4.onnx'
    assert old.read_bytes() == b'old'
    def fail(*args): raise ValueError('bad export')
    with pytest.raises(ValueError): prepare_model.export_model(args, Model, fail)
    assert Path(result['onnx']).read_bytes() == b'new'


def load_decoder(nvc):
    spec = importlib.util.spec_from_file_location('batch_decoder_test', Path(__file__).parents[1] / 'pipeline/decoder.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'PyNvVideoCodec': nvc,
                                 'pipeline.nv12': SimpleNamespace(wrap_nv12=lambda *args: None),
                                 'pipeline.source': SimpleNamespace(MemorySource=object)}):
        spec.loader.exec_module(module)
    return module


def test_decoder_error_signals_stop_before_waiting_for_cleanup():
    decoder = SimpleNamespace(UnlockFrame=lambda pointer: None)
    nvc = SimpleNamespace(CreateDecoder=lambda **kwargs: decoder, OutputColorType=SimpleNamespace(NATIVE=0))
    module = load_decoder(nvc)
    stop = threading.Event(); failures = []; cleanup = threading.Event()
    stage = module.DecoderStage(SimpleNamespace(codec=0), queue.Queue(), None, queue.SimpleQueue(),
                                parse_config([]), Metrics(), stop, failures)
    stage.cleanup_done = cleanup
    def fail(): raise ValueError('decode error')
    stage._decode_packets = fail
    stage.start()
    try:
        assert stop.wait(1)
        assert stage.production_done.wait(1)
        assert stage.is_alive()  # decoder lifetime retained until consumers are gone
    finally:
        cleanup.set(); stage.join(1)
    assert not stage.is_alive()
    assert len(failures) == 1 and str(failures[0][1]) == 'decode error'


def test_decoder_production_done_after_last_publication():
    decoder = SimpleNamespace(UnlockFrame=lambda pointer: None)
    nvc = SimpleNamespace(CreateDecoder=lambda **kwargs: decoder, OutputColorType=SimpleNamespace(NATIVE=0))
    module = load_decoder(nvc)
    stop = threading.Event(); returned = queue.SimpleQueue(); seen = []
    decoder.UnlockFrame = seen.append
    stage = module.DecoderStage(SimpleNamespace(codec=0), queue.Queue(), None, returned,
                                parse_config([]), Metrics(), stop, [])
    cleanup = threading.Event(); stage.cleanup_done = cleanup
    published = queue.Queue()
    def produce():
        stop.set()
        published.put('late frame')
    stage._decode_packets = produce
    stage.start()
    try:
        assert stage.production_done.wait(1)
        assert published.get_nowait() == 'late frame'
        stage.locked = 1; returned.put(9)
    finally:
        cleanup.set(); stage.join(1)
    assert seen == [9]
    assert not stage.is_alive()


def test_cancel_unpublished_reservation():
    pool = make_pool(1); bundle = pool.acquire(threading.Event())
    pool.cancel_acquired(bundle)
    assert pool.stats()['free'] == 1
    assert bundle.leases == 0


@pytest.mark.parametrize('input_shape,output_shape,ok', [
    ((4,3,640,640),(4,300,6), True),
    ((1,3,640,640),(1,300,6), False),
    (('batch',3,640,640),(4,300,6), False),
    ((4,3,640,640),(4,6,300), False),
])
def test_static_detector_contract(input_shape, output_shape, ok):
    module = load_detector()
    class Session:
        def get_providers(self): return ['CUDAExecutionProvider']
        def get_inputs(self): return [SimpleNamespace(name='input', shape=input_shape, type='tensor(float)')]
        def get_outputs(self): return [SimpleNamespace(name='output', shape=output_shape, type='tensor(float)')]
        def get_modelmeta(self): return SimpleNamespace(custom_metadata_map={'names':"{0:'object'}"})
        def io_binding(self): return object()
    module.ort.InferenceSession = lambda *args, **kwargs: Session()
    if ok:
        assert module.Detector(Path('model.onnx'), 0, 4).batch_size == 4
    else:
        with pytest.raises(ValueError): module.Detector(Path('model.onnx'), 0, 4)
