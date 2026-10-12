# YOLO detector server

Runs the same YOLOv8 models as the page, on the server: for devices too slow to run them, and for projects in which a program (a robot controller, say) acts on what a camera sees.

Live at https://closer.ram.eemcs.utwente.nl/yolo/ (the page, served by this server). The page on GitHub Pages uses it too: set **Process on** to *the server* and enter the access code.

## How it works

- A **camera** (the page, or any program) sends JPEG frames; each comes back with its detections. One detection runs at a time; waiting cameras take turns, and each can be capped at `YOLO_MAX_FPS` frames per second (0, the default: no cap).
- Every camera publishes to a **stream** (the *Stream* name in the page's settings). **Subscribers** to a stream receive each frame's detections, and optionally the frame itself.
- A **hook** (optional) is Python code on the server, called with every frame and its detections; what it returns goes to the camera's page (its `text` is shown there) and to the subscribers. See `hooks/example.py`, which steers towards the largest person.

Everything needs the access code (`YOLO_ACCESS_CODE` in `server/.env`): the `X-Access-Code` header or a `code` parameter. Ten wrong codes from one address in 10 minutes block it for a while.

## API

All paths are relative to the server's address (`https://closer.ram.eemcs.utwente.nl/yolo/`).

| | |
|---|---|
| `GET api/info` | models, input sizes, limits, hook (no code needed) |
| `POST api/detect?model=yolov8n&size=480&conf=0.25&iou=0.45&stream=` | body: a JPEG or PNG. Returns its detections. With `stream=name`, it also goes to that stream's subscribers and the hook (a camera that isn't a browser). |
| `GET api/activity` | counts for the server statistics page (cameras, subscribers, detections, busy time); local only, refused when it comes through Apache |
| `GET api/streams` | the streams: name, camera connected, frame count, subscribers |
| `GET api/streams/<name>/latest` | the stream's latest result |
| `GET api/streams/<name>/latest.jpg` | its frame |
| WebSocket `api/streams/<name>/ws?frames=0` | subscribe: a JSON message per result (and per hook message); with `frames=1` each result is preceded by its JPEG as a binary message |
| WebSocket `ws` | a camera: `{"type": "hello", "code": ..., "stream": ...}`, then per frame `{"type": "frame", "seq": n, "model": ..., "size": ..., "conf": ..., "iou": ...}` followed by the JPEG as a binary message; send the next frame after the result |

A result:

```json
{"type": "result", "stream": "my-phone", "seq": 17, "t": 1791838539.5, "width": 640, "height": 480,
 "model": "yolov8n", "input": [480, 384], "conf": 0.25, "iou": 0.45, "ms": 44.0, "wait_ms": 0.0,
 "detections": [{"x1": 50.2, "y1": 39.0, "x2": 243.1, "y2": 476.5, "score": 0.87, "cls": 0, "name": "person"}]}
```

Coordinates are pixels of the image as sent. Examples (in `examples/`): `subscribe.py` (receive a stream, `pip install websockets`) and `detect_image.py` (send an image; standard library only).

## Writing a hook

A module with `on_frame(stream, image, detections, info)` (and optionally `on_start()`), set with `YOLO_HOOK=server.hooks.my_robot` in `server/.env`. `image` is the frame as a numpy array (BGR); `info` has `seq`, `t`, `width`, `height` and `model`. It runs in its own thread: frames that arrive while it's busy are skipped, so a slow hook doesn't hold up detection. Return a JSON-able dict (or `None`); a `text` field shows up on the camera's page.

## Running it

```
uv venv --python 3.12 server/.venv
uv pip install --python server/.venv/bin/python -r server/requirements.txt
cp server/.env.example server/.env        # and set YOLO_ACCESS_CODE
server/.venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 8110     # from the repository root
```

On the CloSer server it runs as `yolo-server.service` (`~/closer-deploy/yolo-server.service`, installed by `~/closer-deploy/setup-yolo.sh`), behind Apache at `/yolo/` (`~/closer-deploy/apache/50-yolo.conf`), with `Nice=10` so the SOFA simulations come first. After a code change: `git pull` in `~/YoloDetector`, then `sudo systemctl restart yolo-server`.

On a server with an NVIDIA GPU: install `onnxruntime-gpu` instead of `onnxruntime` and set `YOLO_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`.
