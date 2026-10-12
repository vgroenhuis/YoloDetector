# YOLO Object Detector

Live object detection on your device's camera with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), running entirely in the browser: no image leaves the device.

**Live page:** https://vgroenhuis.github.io/YoloDetector/ (needs HTTPS for camera access; open it on your phone and tap *Start camera*).

- Detects the 80 [COCO](https://cocodataset.org/) classes: people, vehicles, animals, food, furniture, ...
- Models: YOLOv8n (fast, 13 MB, default), YOLOv8s (more accurate, 45 MB) and YOLOv8m (most accurate, about 2.5x slower than YOLOv8s, 52 MB). The model is downloaded once and then cached by the browser.
- Runs on the GPU with WebGPU where available, otherwise on the CPU with WebAssembly ([onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)). Choose it under **Run on** (Auto, WebGPU, WebAssembly), or with `?backend=webgpu` / `?backend=wasm` in the URL; `?size=320` sets the input size. In Auto mode it switches to WebAssembly when WebGPU:
  - is a software (emulated) GPU;
  - gives wrong results: before a model is used on WebGPU at an input size, it detects `selftest.jpg` and compares the result with what WebAssembly finds there (`selftest.json`, made with `tools/make_selftest.py`; some phone GPUs return garbage, such as dozens of "boats" at the top of the image);
  - takes more than 1.5 s per frame (after the first frame, which compiles the GPU code).
- **Input size**: the frame's longest side as the model sees it (320, 480 or 640 pixels; the other side keeps the aspect ratio). Smaller is faster, larger finds smaller objects. Default 640 on WebGPU, 480 on WebAssembly.
- **Min confidence** and **Overlap (NMS IoU)**: the detection threshold, and how much two boxes of the same class may overlap before the weaker one is dropped (non-maximum suppression).
- **Boxes in sync**: shows the frame the boxes were computed on, instead of the live video. The live video is smoother, but its boxes lag by one detection.
- **Open image** runs the detector on a photo instead of the camera.

## Detection on the server

**Process on: the server** sends the frames to the YOLO server on the CloSer server (needs its access code), which also passes the detections on to a robot controller or other program subscribed to the page's **Stream**. See [server/README.md](server/README.md).

## Run locally

```
python -m http.server 8000
```
then open http://localhost:8000 (opening `index.html` directly does not work). Camera access works on `localhost` without HTTPS.

## Models

`models/*.onnx` are the official YOLOv8n/s/m COCO weights (ultralytics 8.4.175), exported with `tools/export_models.py`: dynamic input size, opset 17, output `[1, 84, anchors]` (box centre, size and 80 class scores per anchor), without NMS. YOLOv8m's weights are stored as float16 (cast back to float32 on loading; the model still computes in float32), to stay under GitHub's 100 MB file limit: its results differ from the float32 model by about 0.01 pixel. The page letterboxes the frame like Ultralytics (grey padding to a multiple of 32), and does the NMS itself.

## Licence

AGPL-3.0 (see `LICENSE`), as are YOLOv8 and its weights.
