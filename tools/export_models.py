"""Exports the YOLOv8 models in models/ to ONNX (any input size that is a
multiple of 32, so the page can pick the input size and keep the camera's
aspect ratio).

YOLOv8m's file would be 104 MB, over GitHub's 100 MB limit: its weights
are stored as float16, each cast back to float32 when the model is loaded
(the model still computes in float32; the outputs differ from the float32
model by about 0.01 pixel and 0.01 in the scores).

    pip install ultralytics onnx onnxslim     # with torch (the CPU build is enough)
    python tools/export_models.py
"""

import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from ultralytics import YOLO

OUT = Path(__file__).resolve().parent.parent / "models"

FP16_WEIGHTS = {"yolov8m"}


def fp16_weights(path: Path) -> None:
    """Stores the float32 weights as float16, with a Cast back to float32."""
    model = onnx.load(path)
    graph = model.graph
    casts, inits = [], []
    for init in graph.initializer:
        a = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and a.size >= 256:
            half = numpy_helper.from_array(a.astype(np.float16), init.name + "_fp16")
            inits.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "_to_fp32"))
        else:
            inits.append(init)
    del graph.initializer[:]
    graph.initializer.extend(inits)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, path)


for name in ("yolov8n", "yolov8s", "yolov8m"):
    exported = YOLO(f"{name}.pt").export(format="onnx", dynamic=True, opset=17, simplify=True)
    shutil.move(exported, OUT / f"{name}.onnx")
    if name in FP16_WEIGHTS:
        fp16_weights(OUT / f"{name}.onnx")
    print("wrote", OUT / f"{name}.onnx")
