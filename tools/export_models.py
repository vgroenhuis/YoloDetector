"""Exports the YOLOv8 models in models/ to ONNX (any input size that is a
multiple of 32, so the page can pick the input size and keep the camera's
aspect ratio).

    pip install ultralytics onnx onnxslim     # with torch (the CPU build is enough)
    python tools/export_models.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

OUT = Path(__file__).resolve().parent.parent / "models"

for name in ("yolov8n", "yolov8s"):
    onnx = YOLO(f"{name}.pt").export(format="onnx", dynamic=True, opset=17, simplify=True)
    shutil.move(onnx, OUT / f"{name}.onnx")
    print("wrote", OUT / f"{name}.onnx")
