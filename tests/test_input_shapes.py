"""Input shape contracts; no GPU, torch or model download required."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from model_shape import normalize_size, model_suffix, validate_metadata
from pipeline.config import parse_config
import prepare_model


@pytest.mark.parametrize('value,expected', [(640,(640,640)), ([640],(640,640)), ([384,640],(384,640))])
def test_size(value, expected):
    assert normalize_size(value) == expected


@pytest.mark.parametrize('value', [[], [1,2,3], [0], [360,640], [-32], [True]])
def test_bad_size(value):
    with pytest.raises(ValueError): normalize_size(value)


def test_cli_and_names():
    assert parse_config([]).detector_size == (640,640)
    assert parse_config(['--imgsz','384','640','--batch-size','4']).detector_size == (384,640)
    assert model_suffix(640,1) == ''
    assert model_suffix(640,4) == '-b4'
    assert model_suffix((384,640),1) == '-640x384'
    assert model_suffix((384,640),4) == '-640x384-b4'


@pytest.mark.parametrize('metadata', [{'imgsz':'[640,640]'}, {'batch':'1'}, {'dynamic':'True'}])
def test_bad_metadata(metadata):
    with pytest.raises(ValueError): validate_metadata(metadata,(384,640),4)


def test_metadata():
    validate_metadata({'imgsz':'[384,640]', 'batch':'4', 'dynamic':'False'},(384,640),4)
    validate_metadata({},(384,640),4)  # legacy exports without metadata


def letterbox_class():
    spec=importlib.util.spec_from_file_location('shape_nv12', Path(__file__).parents[1]/'pipeline/nv12.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch':SimpleNamespace()}): spec.loader.exec_module(module)
    return module.Letterbox


@pytest.mark.parametrize('shape,top', [((640,640),140),((384,640),12)])
def test_geometry(shape, top):
    box=letterbox_class()(1920,1080,shape)
    assert (box.new_width,box.new_height,box.left,box.top)==(640,360,0,top)
    assert box.to_source(np.array([[0,top,640,top+360]])).tolist()[0] == pytest.approx([0,0,1920,1080])


@pytest.mark.parametrize('source', [(1080,1920),(1280,960),(1919,1079)])
def test_other_aspects(source):
    box=letterbox_class()(*source,(384,640))
    assert box.new_width+2*box.left <= 641
    assert box.new_height+2*box.top <= 385
    assert box.to_source(np.array([[-100,-100,1000,1000]])).tolist()[0] == pytest.approx([0,0,*source])


@pytest.mark.parametrize('batch', [1,4])
def test_export_rectangular(tmp_path, batch):
    checkpoint=tmp_path/'model.pt'; checkpoint.write_bytes(b'weights')
    old=tmp_path/'model.onnx'; old.write_bytes(b'old')
    class Model:
        def __init__(self,path): self.ckpt_path=path
        def export(self,**kwargs):
            assert kwargs['imgsz']==(384,640) and kwargs['batch']==batch
            path=Path(self.ckpt_path).with_suffix('.onnx'); path.write_bytes(b'new'); return path
    args=prepare_model.parse_args(['--model',str(checkpoint),'--imgsz','384','640','--batch-size',str(batch)])
    result=prepare_model.export_model(args,Model,lambda *args:None)
    assert Path(result['onnx']).name == 'model'+model_suffix((384,640),batch)+'.onnx'
    assert old.read_bytes()==b'old'


def test_gl_rectangular_dimensions():
    import types
    box=letterbox_class()(1920,1080,(384,640))
    stubs={name:types.ModuleType(name) for name in ['cuda','cuda.bindings','cuda.bindings.driver','glfw','torch','OpenGL']}
    stubs['cuda'].bindings=stubs['cuda.bindings']; stubs['cuda.bindings'].driver=stubs['cuda.bindings.driver']
    stubs['OpenGL'].GL=SimpleNamespace()
    stubs['pipeline.gl_util']=SimpleNamespace(compile_shader=None,cuda_ok=None,make_texture=None)
    stubs['pipeline.nv12']=SimpleNamespace(Letterbox=type(box),PAD_VALUE=114/255)
    spec=importlib.util.spec_from_file_location('shape_gl',Path(__file__).parents[1]/'pipeline/gl_prepare.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,stubs): spec.loader.exec_module(module)
    converter=module.GlConverter(box,(1920,1080))
    assert (converter.target_height,converter.target_width)==(384,640)
    assert converter.plane_bytes==384*640*4


@pytest.mark.parametrize('graph,metadata,ok', [
    ((4,3,384,640),{'imgsz':'[384,640]','batch':'4'},True),
    ((4,3,640,384),{'imgsz':'[384,640]','batch':'4'},False),
    ((4,3,384,640),{'imgsz':'[640,640]','batch':'4'},False),
])
def test_detector_rectangular_contract(graph, metadata, ok):
    spec=importlib.util.spec_from_file_location('shape_detector',Path(__file__).parents[1]/'pipeline/detector.py')
    module=importlib.util.module_from_spec(spec)
    class Session:
        def get_providers(self): return ['CUDAExecutionProvider']
        def get_inputs(self): return [SimpleNamespace(name='input',shape=graph,type='tensor(float)')]
        def get_outputs(self): return [SimpleNamespace(name='out',shape=(4,300,6),type='tensor(float)')]
        def get_modelmeta(self): return SimpleNamespace(custom_metadata_map={'names':"{0:'object'}",**metadata})
        def io_binding(self): return object()
    with patch.dict(sys.modules,{'torch':SimpleNamespace(),'onnxruntime':SimpleNamespace(InferenceSession=lambda *a,**k:Session())}):
        spec.loader.exec_module(module)
    if ok:
        assert module.Detector(Path('model.onnx'),0,4,(384,640)).input_shape==(4,3,384,640)
    else:
        with pytest.raises(ValueError): module.Detector(Path('model.onnx'),0,4,(384,640))
