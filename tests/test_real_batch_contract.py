"""Opt-in real dependency contract: set V1_BATCH_MODEL and V1_BATCH_CLIP."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_demo
import onnxruntime as ort
from ultralytics.trackers.byte_tracker import BYTETracker

@unittest.skipUnless(os.getenv('V1_BATCH_MODEL') and os.getenv('V1_BATCH_CLIP'), 'opt-in GPU contract')
class RealBatchContract(unittest.TestCase):
    def test_calls_and_tail_tracking(self):
        size=int(os.getenv('V1_BATCH_SIZE','4'))
        calls=[]; updates=[]
        run=ort.InferenceSession.run_with_iobinding
        update=BYTETracker.update
        def record_run(session,*a,**kw):
            calls.append(1); return run(session,*a,**kw)
        def record_update(tracker,*a,**kw):
            updates.append((id(tracker),tracker.frame_id)); return update(tracker,*a,**kw)
        imgsz=tuple(map(int,os.getenv('V1_BATCH_IMGSZ','640 640').split()))
        args=SimpleNamespace(batch_size=size,imgsz=imgsz,input=Path(os.environ['V1_BATCH_CLIP']),
             model=Path(os.environ['V1_BATCH_MODEL']),decoder='nvdec',backend='cuda',loops=2,
             max_frames=size*2+1,tracker='bytetrack.yaml',conf=.25,iou=.7,output=None,instance_id='contract')
        with patch.object(ort.InferenceSession,'run_with_iobinding',record_run), patch.object(BYTETracker,'update',record_update):
            stats=run_demo.process_stream(args)
        self.assertEqual(len(calls),7) # one warmup, three real batches per loop
        self.assertEqual(len(updates),2*(size*2+1))
        self.assertEqual(len({item[0] for item in updates}),1)
        self.assertEqual([item[1] for item in updates],list(range(size*2+1))*2)
        self.assertEqual([s['batch_sizes'] for s in stats['loops']],[{size:2,1:1}]*2)
