# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import threading
import numpy as np
import run_demo as demo

class Reader:
    backend='opencv'; height=12; width=16; fps=25.; device_verified=False
    def __init__(self, n, stop=None): self.frame_count=n; self.i=0; self.stop=stop
    def read(self, **kw):
        if self.i == self.frame_count: return False,None,None
        self.i+=1
        if self.stop and self.i==2: self.stop.set()
        return True,np.full((12,16,3),self.i,dtype=np.uint8),self.i/25

class Model:
    def __init__(self): self.calls=[]; self.ids=[]
    def track(self, source, **kwargs):
        frames=source if isinstance(source,list) else [source]
        self.calls.append(len(frames)); self.ids.extend(int(f[0,0,0]) for f in frames)
        return [SimpleNamespace(boxes=None,names={}) for f in frames]

class BatchingTests(unittest.TestCase):
    def run_case(self,n,b,max_frames=0,stop=None,writer=False):
        args=SimpleNamespace(batch_size=b,tracker='bytetrack.yaml',conf=.25,iou=.7,max_frames=max_frames)
        model=Model(); reader=Reader(n,stop)
        from pathlib import Path
        class Writer:
            frames=[]
            def isOpened(self): return True
            def write(self,f): self.frames.append(int(f[0,0,0]))
            def release(self): pass
        w=Writer()
        with patch.object(demo.cv2,'VideoWriter',return_value=w), patch.object(demo,'annotate',side_effect=lambda f,*a,**k:f):
            stats=demo.run_one_loop(args,reader,model,'cpu',640,1,Path('results/test.mp4') if writer else None,None,stop,{'actual':None})
        return stats,model,w
    def test_sizes_tails_and_order(self):
        for b in [1,2,4,8,16]:
            for n in [1,b,max(1,b-1),b+1,2*b+1]:
                with self.subTest(n=n,b=b):
                    s,m,w=self.run_case(n,b,writer=True)
                    self.assertEqual(s['frames_processed'],n)
                    self.assertEqual(s['inference_batches'],(n+b-1)//b)
                    self.assertEqual(m.ids,list(range(1,n+1)))
                    self.assertEqual(w.frames,list(range(1,n+1)))
                    self.assertEqual(s['frames_written'],n)
    def test_max_frames_tail(self):
        s,m,_=self.run_case(12,4,max_frames=5)
        self.assertEqual(m.calls,[4,1]); self.assertEqual(s['frames_read'],5)
    def test_stop_flush(self):
        s,m,_=self.run_case(12,4,stop=threading.Event())
        self.assertEqual(m.calls,[2]); self.assertTrue(s['stopped_by_user'])
    def test_empty_raises(self):
        with self.assertRaisesRegex(RuntimeError,'decoded no frames'): self.run_case(0,4)
    def test_shape_contract(self):
        import onnx
        from onnx import helper,TensorProto
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as temp:
            p=Path(temp)/'m.onnx'
            x=helper.make_tensor_value_info('x',TensorProto.FLOAT,[4,3,640,640])
            onnx.save(helper.make_model(helper.make_graph([], 'shape', [x], [x])),p)
            self.assertEqual(demo.validate_model_shape(p,4,640),[4,3,640,640])
            with self.assertRaises(ValueError): demo.validate_model_shape(p,1,640)

if __name__=='__main__': unittest.main()
