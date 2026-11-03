#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Export a YOLO checkpoint to a static-shape ONNX file and print a JSON summary of the result.

Takes --model (default yolo26n.pt, downloaded by Ultralytics if absent), --imgsz and --opset. Exports on the CPU
without NMS, simplification or dynamic axes, so the ONNX runtime loop can run the tracker itself. Writes the .onnx
file next to the checkpoint and uses .ultralytics in this folder as the Ultralytics config directory.
"""
import argparse,json,os
from pathlib import Path
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent / ".ultralytics")
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument("--model",default="yolo26n.pt"); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--opset",type=int,default=12); a=p.parse_args()
m=YOLO(a.model); out=m.export(format="onnx",imgsz=a.imgsz,opset=a.opset,dynamic=False,simplify=False,nms=False,device="cpu")
print(json.dumps({"checkpoint":a.model,"onnx":str(Path(str(out)).resolve()),"imgsz":a.imgsz,"opset":a.opset},indent=2))
