# YOLO Object Detector

Live object detection on your device's camera with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), running entirely in the browser: no image leaves the device.

**Live page:** https://vgroenhuis.github.io/YoloDetector/ (needs HTTPS for camera access; open it on your phone and tap *Start camera*).

- Detects the 80 [COCO](https://cocodataset.org/) classes: people, vehicles, animals, food, furniture, ...
- Models: YOLOv8n (fast, 13 MB, default) and YOLOv8s (more accurate, 45 MB). The model is downloaded once and then cached by the browser.
- Runs on the GPU with WebGPU where available, otherwise on the CPU with WebAssembly ([onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)). A software (emulated) GPU, or one that is slower than about 1.5 s per frame, makes it switch to WebAssembly. Force a backend with `?backend=wasm` or `?backend=webgpu`, and the input size with `?size=320`.
- **Input size**: the frame's longest side as the model sees it (320, 480 or 640 pixels; the other side keeps the aspect ratio). Smaller is faster, larger finds smaller objects. Default 640 on WebGPU, 480 on WebAssembly.
- **Min confidence** and **Overlap (NMS IoU)**: the detection threshold, and how much two boxes of the same class may overlap before the weaker one is dropped (non-maximum suppression).
- **Boxes in sync**: shows the frame the boxes were computed on, instead of the live video. The live video is smoother, but its boxes lag by one detection.
- **Open image** runs the detector on a photo instead of the camera.

## Run locally

```
python -m http.server 8000
```
then open http://localhost:8000 (opening `index.html` directly does not work). Camera access works on `localhost` without HTTPS.

## Models

`models/*.onnx` are the official YOLOv8n/s COCO weights (ultralytics 8.4.175), exported with `tools/export_models.py`: dynamic input size, opset 17, output `[1, 84, anchors]` (box centre, size and 80 class scores per anchor), without NMS. The page letterboxes the frame like Ultralytics (grey padding to a multiple of 32), and does the NMS itself.

## Licence

AGPL-3.0 (see `LICENSE`), as are YOLOv8 and its weights.
