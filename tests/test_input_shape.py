# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import onnx
from onnx import helper, TensorProto
import prepare_model
import launch_instances
from input_shape import *


def write_model(path, n=1, hw=(640,640), metadata=None, dtype=TensorProto.FLOAT):
    x = helper.make_tensor_value_info('images', dtype, [n,3,*hw])
    model = helper.make_model(helper.make_graph([], 'shape', [x], [x]))
    helper.set_model_props(model, metadata if metadata is not None else {'batch':str(n), 'imgsz':str(list(hw)), 'args':"{'dynamic': False}"})
    onnx.save(model,path)


class ShapeTests(unittest.TestCase):
    def test_cli(self):
        self.assertEqual(prepare_model.parse_args([]).imgsz,(640,640))
        for raw,expected in [(['640'],(640,640)),(['384','640'],(384,640))]:
            self.assertEqual(prepare_model.parse_args(['--imgsz',*raw]).imgsz,expected)
            with patch('sys.argv',['launcher','--imgsz',*raw]):
                args=launch_instances.parse_args()
            cmd=launch_instances.child_command(args,'test',None)
            i=cmd.index('--imgsz')
            self.assertEqual(cmd[i+1:i+3],list(map(str,expected)))
        for invalid in ([],[0],[-32],[360,640],[32,32,32],[True],['640']):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): normalize_imgsz(invalid)

    def test_geometry(self):
        for size,pad in [(640,140),((384,640),12)]:
            g=letterbox_geometry(1080,1920,size)
            self.assertEqual((g['new_h'],g['new_w']),(360,640))
            self.assertEqual((g['pad_top'],g['pad_bottom']),(pad,pad))
        for source in [(720,1280),(480,640),(1920,1080),(721,1281)]:
            for target in [(640,640),(384,640)]:
                g=letterbox_geometry(*source,target)
                self.assertEqual(g['new_h']+g['pad_top']+g['pad_bottom'],target[0])
                self.assertEqual(g['new_w']+g['pad_left']+g['pad_right'],target[1])
                self.assertLessEqual(abs(g['pad_top']-g['pad_bottom']),1)
                # Synthetic source coordinates recover exactly using rounded geometry.
                for x,y in [(0,0),(source[1]-1,source[0]-1)]:
                    px=x*g['gain_x']+g['pad_left']; py=y*g['gain_y']+g['pad_top']
                    self.assertAlmostEqual((px-g['pad_left'])/g['gain_x'],x)
                    self.assertAlmostEqual((py-g['pad_top'])/g['gain_y'],y)

    def test_contract_matrix_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'m.onnx'
            for n in [1,2,4,8,16]:
                for hw in [(640,640),(384,640)]:
                    write_model(p,n,hw)
                    self.assertEqual(validate_model_shape(p,n,hw),[n,3,*hw])
                    with self.assertRaises(ValueError): validate_model_shape(p,n+1,hw)
            for meta in [{},{'batch':'1','imgsz':'[640,640]'}, {'batch':'True','imgsz':'[384,640]'}, {'batch':'1','imgsz':'[384,640]','args':"{'dynamic': True}"}]:
                write_model(p,1,(384,640),meta)
                with self.assertRaises(ValueError): validate_model_shape(p,1,(384,640))
            write_model(p,1,(384,640),dtype=TensorProto.FLOAT16)
            with self.assertRaises(ValueError): validate_model_shape(p,1,(384,640))
            write_model(p,'N',(384,640))
            with self.assertRaises(ValueError): validate_model_shape(p,1,(384,640))

    def test_real_export_validation_and_isolation(self):
        for n in [1,4]:
            for hw in [(640,640),(384,640)]:
                with tempfile.TemporaryDirectory() as tmp:
                    ckpt=Path(tmp)/'yolo26n.pt'; ckpt.write_bytes(b'weights')
                    square=ckpt.with_suffix('.onnx');square.write_bytes(b'old-square')
                    class FakeModel:
                        def __init__(self,path): self.ckpt_path=path
                        def export(self,**kw):
                            path=Path(self.ckpt_path).with_suffix('.onnx')
                            write_model(path,kw['batch'],kw['imgsz'])
                            return path
                    args=prepare_model.parse_args(['--model',str(ckpt),'--batch-size',str(n),'--imgsz',*map(str,hw)])
                    result=prepare_model.export_model(args,FakeModel)
                    target=model_filename(ckpt,n,hw)
                    self.assertEqual(Path(result['onnx']),target)
                    self.assertEqual(validate_model_shape(target,n,hw),[n,3,*hw])
                    if (n,hw)!=(1,(640,640)): self.assertEqual(square.read_bytes(),b'old-square')
                    before=target.read_bytes()
                    class WrongModel(FakeModel):
                        def export(self,**kw):
                            path=Path(self.ckpt_path).with_suffix('.onnx');write_model(path,99,kw['imgsz']);return path
                    with self.assertRaises(ValueError): prepare_model.export_model(args,WrongModel)
                    self.assertEqual(target.read_bytes(),before)
                    self.assertFalse(list(Path(tmp).glob('.batch-export-*')))

if __name__=='__main__': unittest.main()
