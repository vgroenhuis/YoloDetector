"""YOLOv8 detection with onnxruntime (on the CPU, or a GPU): the same pre- and
postprocessing as the page (app.js), so both give the same boxes."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
MODELS = ("yolov8n", "yolov8s", "yolov8m")
SIZES = (320, 480, 640)
MAX_CANDIDATES, MAX_DETECTIONS = 3000, 300


class Detector:
    """One onnxruntime session per model, loaded on first use. Not
    thread-safe for running: the server runs one detection at a time."""

    def __init__(self, model_dir: Path, threads: int = 1, providers: tuple[str, ...] = ("CPUExecutionProvider",)) -> None:
        self.model_dir = model_dir
        self.threads = threads
        # e.g. ("CUDAExecutionProvider", "CPUExecutionProvider") with onnxruntime-gpu
        self.providers = [p for p in providers if p in ort.get_available_providers()] or ["CPUExecutionProvider"]
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._load_lock = threading.Lock()

    def session(self, model: str) -> ort.InferenceSession:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}")
        with self._load_lock:
            if model not in self._sessions:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = self.threads
                opts.inter_op_num_threads = 1
                opts.log_severity_level = 3
                self._sessions[model] = ort.InferenceSession(
                    str(self.model_dir / f"{model}.onnx"), opts, providers=self.providers)
            return self._sessions[model]

    def detect(self, image: np.ndarray, model: str = "yolov8n", size: int = 480,
               conf: float = 0.25, iou: float = 0.45) -> tuple[list[dict], tuple[int, int]]:
        """Detections in `image` (BGR, as from cv2.imdecode), in its pixels:
        dicts with x1, y1, x2, y2, score, cls and name. Also returns the
        model's input size (width, height)."""
        if size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}")
        session = self.session(model)
        x, (in_w, in_h), (r, left, top) = preprocess(image, size)
        out = session.run(None, {session.get_inputs()[0].name: x})[0][0]
        h, w = image.shape[:2]
        return postprocess(out, conf, iou, r, left, top, w, h), (in_w, in_h)


def preprocess(image: np.ndarray, size: int):
    """Scales the image (longest side = size, keeping the aspect ratio) and
    pads it with grey to a multiple of 32, centred, as Ultralytics does."""
    h, w = image.shape[:2]
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    in_w, in_h = -(-nw // 32) * 32, -(-nh // 32) * 32
    left, top = (in_w - nw) // 2, (in_h - nh) // 2
    canvas = np.full((in_h, in_w, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0   # BGR -> RGB, NCHW
    return np.ascontiguousarray(x), (in_w, in_h), (r, left, top)


def postprocess(out: np.ndarray, conf: float, iou: float, r: float, left: int, top: int,
                w: int, h: int) -> list[dict]:
    """out: [84, n] (box centre and size in input pixels, 80 class scores).
    Boxes above `conf` after per-class non-maximum suppression."""
    scores = out[4:]
    cls = scores.argmax(0)
    best = scores[cls, np.arange(scores.shape[1])]
    keep = np.nonzero(best >= conf)[0]
    keep = keep[np.argsort(-best[keep], kind="stable")][:MAX_CANDIDATES]
    cx, cy, bw, bh = out[0, keep], out[1, keep], out[2, keep], out[3, keep]
    boxes = np.stack([
        np.clip((cx - bw / 2 - left) / r, 0, w), np.clip((cy - bh / 2 - top) / r, 0, h),
        np.clip((cx + bw / 2 - left) / r, 0, w), np.clip((cy + bh / 2 - top) / r, 0, h)], axis=1)
    cls, best = cls[keep], best[keep]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    alive = np.ones(len(keep), bool)
    result = []
    for i in range(len(keep)):
        if not alive[i]:
            continue
        result.append({
            "x1": round(float(boxes[i, 0]), 1), "y1": round(float(boxes[i, 1]), 1),
            "x2": round(float(boxes[i, 2]), 1), "y2": round(float(boxes[i, 3]), 1),
            "score": round(float(best[i]), 4), "cls": int(cls[i]), "name": COCO[int(cls[i])],
        })
        if len(result) >= MAX_DETECTIONS:
            break
        rest = np.nonzero(alive[i + 1:] & (cls[i + 1:] == cls[i]))[0] + i + 1
        if len(rest):
            iw = np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0])
            ih = np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1])
            inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
            alive[rest[inter / (areas[i] + areas[rest] - inter + 1e-9) > iou]] = False
    return result
