"""YOLOv8 detection on the server (FastAPI).

Cameras (the page in "Process on: server" mode, or any program) send JPEG
frames; each frame's detections go back to the sender and to everyone
subscribed to its stream (a robot controller, say), and to an optional
Python hook running next to the detector. See server/README.md.

    uvicorn server.app:app --host 127.0.0.1 --port 8110      # from the repository root
"""

from __future__ import annotations

import asyncio
import hmac
import importlib
import json
import logging
import os
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from .detector import MODELS, SIZES, Detector

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("yolo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _load_env(path: Path) -> None:
    """KEY=value lines of server/.env, unless already in the environment
    (systemd's EnvironmentFile sets them too)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(Path(__file__).resolve().parent / ".env")

ACCESS_CODE = os.environ.get("YOLO_ACCESS_CODE", "")
THREADS = int(os.environ.get("YOLO_THREADS", "1"))          # onnxruntime threads (the CPU is shared)
MAX_FPS = float(os.environ.get("YOLO_MAX_FPS", "10"))       # per camera
MAX_CLIENTS = int(os.environ.get("YOLO_MAX_CLIENTS", "8"))  # cameras + subscribers
HOOK = os.environ.get("YOLO_HOOK", "")                      # e.g. server.hooks.example
PROVIDERS = tuple(p.strip() for p in os.environ.get("YOLO_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip())
MAX_IMAGE_BYTES = 4 * 2**20
STREAM_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")

if not ACCESS_CODE:
    log.warning("YOLO_ACCESS_CODE is not set: every request is refused")

app = FastAPI(title="YOLO detector server", docs_url=None, redoc_url=None)
detector = Detector(ROOT / "models", threads=THREADS, providers=PROVIDERS)
log.info("onnxruntime providers: %s", ", ".join(detector.providers))
infer_pool = ThreadPoolExecutor(1, thread_name_prefix="infer")
hook_pool = ThreadPoolExecutor(1, thread_name_prefix="hook")
infer_lock = asyncio.Lock()   # one detection at a time; waiters are served in turn (FIFO)
clients = 0


# ---------------------------------------------------------------- access

_failures: dict[str, list[float]] = {}
FAILURE_WINDOW, MAX_FAILURES = 600.0, 10


def client_ip(headers, fallback: Optional[str]) -> str:
    # Only reachable through Apache on this host, which sets X-Forwarded-For.
    forwarded = headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or fallback or "?"


def check_code(code: Optional[str], ip: str) -> Optional[str]:
    """None when the code is right, else why not."""
    now = time.time()
    recent = [t for t in _failures.get(ip, []) if now - t < FAILURE_WINDOW]
    _failures[ip] = recent
    if len(recent) >= MAX_FAILURES:
        return "too many wrong access codes, try again later"
    if not ACCESS_CODE:
        return "the server has no access code set"
    if code and hmac.compare_digest(code.encode(), ACCESS_CODE.encode()):
        return None
    recent.append(now)
    return "wrong access code"


def require_code(request: Request) -> None:
    code = request.headers.get("x-access-code") or request.query_params.get("code")
    problem = check_code(code, client_ip(request.headers, request.client.host if request.client else None))
    if problem:
        raise HTTPException(429 if problem.startswith("too many") else 401, problem)


# ---------------------------------------------------------------- streams

@dataclass
class Subscriber:
    ws: WebSocket
    frames: bool
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=4))


@dataclass
class Stream:
    name: str
    publisher: bool = False   # a camera is connected to it
    seq: int = 0
    latest: Optional[dict] = None
    latest_jpeg: Optional[bytes] = None
    subscribers: list[Subscriber] = field(default_factory=list)

    def publish(self, message: dict, jpeg: Optional[bytes] = None) -> None:
        """To every subscriber; a slow one loses its oldest messages."""
        for sub in self.subscribers:
            items = ([jpeg] if jpeg is not None and sub.frames else []) + [message]
            for item in items:
                if sub.queue.full():
                    try:
                        sub.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                sub.queue.put_nowait(item)


streams: dict[str, Stream] = {}


STREAM_KEEP_SECONDS = 3600   # an idle stream's last frame is kept this long


def get_stream(name: str) -> Stream:
    now = time.time()
    for old in [s for s in streams.values() if not s.publisher and not s.subscribers
                and (s.latest is None or now - s.latest["t"] > STREAM_KEEP_SECONDS)]:
        del streams[old.name]
    if not STREAM_NAME.match(name):
        raise ValueError("a stream name is 1-40 letters, digits, '.', '_' or '-'")
    if name not in streams:
        streams[name] = Stream(name)
    return streams[name]


def forget_if_idle(stream: Stream) -> None:
    if not stream.publisher and not stream.subscribers and stream.latest is None:
        streams.pop(stream.name, None)


# ---------------------------------------------------------------- hook

class Hook:
    """The YOLO_HOOK module's on_frame(stream, image, detections, info),
    called for every detected frame in its own thread. When it is still
    busy with an earlier frame, a frame is skipped rather than queued.
    What it returns (a JSON-able dict, or None) goes to the stream's camera
    and subscribers as a "hook" message."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.module = importlib.import_module(name) if name else None
        self.busy = False
        self.errors = 0
        if self.module is not None and hasattr(self.module, "on_start"):
            self.module.on_start()
        if self.module is not None:
            log.info("Hook %s loaded", name)

    def offer(self, stream: Stream, image: np.ndarray, detections: list[dict], info: dict) -> None:
        if self.module is None or self.busy:
            return
        self.busy = True
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(hook_pool, self._call, stream.name, image, detections, info)
        future.add_done_callback(lambda f: self._done(f, stream, info))

    def _call(self, name, image, detections, info):
        return self.module.on_frame(name, image, detections, info)

    def _done(self, future, stream: Stream, info: dict) -> None:
        self.busy = False
        try:
            data = future.result()
        except Exception:
            self.errors += 1
            if self.errors <= 10 or self.errors % 100 == 0:
                log.exception("Hook %s failed (%d times)", self.name, self.errors)
            return
        if data is not None:
            message = {"type": "hook", "stream": stream.name, "seq": info["seq"], "data": data}
            stream.publish(message)
            for cam in cameras.get(stream.name, []):
                if not cam.full():
                    cam.put_nowait(message)


hook = Hook(HOOK)
cameras: dict[str, list[asyncio.Queue]] = {}   # stream -> its camera connections' outgoing hook messages


# ---------------------------------------------------------------- detection

def _decode_and_detect(data: bytes, model: str, size: int, conf: float, iou: float):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not an image (JPEG or PNG)")
    t0 = time.perf_counter()
    detections, input_size = detector.detect(image, model, size, conf, iou)
    return image, detections, input_size, (time.perf_counter() - t0) * 1000


async def detect(stream: Optional[Stream], data: bytes, model: str, size: int, conf: float, iou: float,
                 extra: Optional[dict] = None) -> dict:
    if model not in MODELS:
        raise ValueError(f"model must be one of {', '.join(MODELS)}")
    if size not in SIZES:
        raise ValueError(f"size must be one of {', '.join(map(str, SIZES))}")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image too large")
    conf = min(max(conf, 0.01), 0.99)
    iou = min(max(iou, 0.05), 0.95)
    t_wait = time.perf_counter()
    async with infer_lock:
        wait_ms = (time.perf_counter() - t_wait) * 1000
        image, detections, (in_w, in_h), ms = await asyncio.get_running_loop().run_in_executor(
            infer_pool, _decode_and_detect, data, model, size, conf, iou)
    h, w = image.shape[:2]
    result = {"type": "result", "stream": stream.name if stream else None, "seq": 0, "t": time.time(),
              "width": w, "height": h, "model": model, "input": [in_w, in_h], "conf": conf, "iou": iou,
              "ms": round(ms, 1), "wait_ms": round(wait_ms, 1), "detections": detections, **(extra or {})}
    if stream is not None:
        stream.seq += 1
        result["seq"] = stream.seq
        stream.latest, stream.latest_jpeg = result, data
        stream.publish(result, data)
        hook.offer(stream, image, detections, {k: result[k] for k in ("seq", "t", "width", "height", "model")})
    return result


# ---------------------------------------------------------------- camera WebSocket

@app.websocket("/ws")
async def camera_ws(ws: WebSocket) -> None:
    """A camera: hello, then per frame a JSON {"type": "frame", ...} message
    followed by the JPEG as a binary message; one frame at a time (send
    the next after its result)."""
    global clients
    await ws.accept()
    ip = client_ip(ws.headers, ws.client.host if ws.client else None)
    try:
        hello = json.loads(await asyncio.wait_for(ws.receive_text(), 15))
    except (asyncio.TimeoutError, ValueError, WebSocketDisconnect, KeyError):
        await ws.close(4000)
        return
    problem = check_code(hello.get("code"), ip)
    if problem:
        await ws.send_json({"type": "error", "message": problem})
        await ws.close(4001)
        return
    if clients >= MAX_CLIENTS:
        await ws.send_json({"type": "error", "message": "the server is busy (too many connections), try again later"})
        await ws.close(4003)
        return
    name = hello.get("stream") or f"cam-{secrets.token_hex(3)}"
    try:
        stream = get_stream(name)
    except ValueError as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close(4000)
        return
    if stream.publisher:
        await ws.send_json({"type": "error", "message": f"stream {name!r} already has a camera"})
        await ws.close(4009)
        return
    stream.publisher = True
    clients += 1
    outbox: asyncio.Queue = asyncio.Queue(maxsize=16)   # hook messages for this camera
    cameras.setdefault(name, []).append(outbox)
    log.info("Camera %s connected from %s", name, ip)

    async def forward_hook_messages():
        while True:
            await ws.send_json(await outbox.get())

    forwarder = asyncio.create_task(forward_hook_messages())
    try:
        await ws.send_json({"type": "welcome", "stream": name, "models": list(MODELS), "sizes": list(SIZES),
                            "max_fps": MAX_FPS, "hook": HOOK or None})
        meta: Optional[dict] = None
        next_allowed = 0.0
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                meta = json.loads(msg["text"])
                continue
            data = msg.get("bytes")
            if not data or not meta or meta.get("type") != "frame":
                continue
            delay = next_allowed - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            next_allowed = time.monotonic() + 1.0 / MAX_FPS
            try:
                result = await detect(stream, data, str(meta.get("model", "yolov8n")), int(meta.get("size", 480)),
                                      float(meta.get("conf", 0.25)), float(meta.get("iou", 0.45)),
                                      {"client_seq": meta.get("seq")})
                await ws.send_json(result)
            except ValueError as exc:
                await ws.send_json({"type": "error", "message": str(exc), "client_seq": meta.get("seq")})
            meta = None
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        forwarder.cancel()
        cameras[name].remove(outbox)
        if not cameras[name]:
            del cameras[name]
        stream.publisher = False
        clients -= 1
        log.info("Camera %s disconnected", name)


# ---------------------------------------------------------------- subscribers

@app.websocket("/api/streams/{name}/ws")
async def subscribe_ws(ws: WebSocket, name: str, code: str = "", frames: int = 0) -> None:
    """A stream's results (and hook messages) as JSON text messages; with
    frames=1, each result is preceded by its JPEG as a binary message."""
    global clients
    await ws.accept()
    problem = check_code(code or ws.headers.get("x-access-code"), client_ip(ws.headers, ws.client.host if ws.client else None))
    if problem:
        await ws.send_json({"type": "error", "message": problem})
        await ws.close(4001)
        return
    if clients >= MAX_CLIENTS:
        await ws.send_json({"type": "error", "message": "the server is busy (too many connections)"})
        await ws.close(4003)
        return
    try:
        stream = get_stream(name)
    except ValueError as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close(4000)
        return
    sub = Subscriber(ws, bool(frames))
    stream.subscribers.append(sub)
    clients += 1

    async def watch_disconnect():
        while (await ws.receive())["type"] != "websocket.disconnect":
            pass

    watcher = asyncio.create_task(watch_disconnect())
    try:
        while not watcher.done():
            getter = asyncio.create_task(sub.queue.get())
            done, _ = await asyncio.wait({getter, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if getter not in done:
                getter.cancel()
                break
            item = getter.result()
            if isinstance(item, bytes):
                await ws.send_bytes(item)
            else:
                await ws.send_json(item)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        watcher.cancel()
        stream.subscribers.remove(sub)
        clients -= 1
        forget_if_idle(stream)


# ---------------------------------------------------------------- HTTP API

@app.get("/api/info")
def info() -> dict:
    return {"models": list(MODELS), "sizes": list(SIZES), "max_fps": MAX_FPS, "hook": HOOK or None,
            "providers": detector.providers, "access_code_required": True}


@app.get("/api/streams")
def list_streams(request: Request) -> list[dict]:
    require_code(request)
    return [{"name": s.name, "camera": s.publisher, "seq": s.seq, "subscribers": len(s.subscribers),
             "last_t": s.latest["t"] if s.latest else None} for s in streams.values()]


@app.get("/api/streams/{name}/latest")
def latest(name: str, request: Request) -> dict:
    require_code(request)
    stream = streams.get(name)
    if stream is None or stream.latest is None:
        raise HTTPException(404, "no frames in this stream yet")
    return stream.latest


@app.get("/api/streams/{name}/latest.jpg")
def latest_jpeg(name: str, request: Request) -> Response:
    require_code(request)
    stream = streams.get(name)
    if stream is None or stream.latest_jpeg is None:
        raise HTTPException(404, "no frames in this stream yet")
    return Response(stream.latest_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


_post_next: dict[str, float] = {}


@app.post("/api/detect")
async def detect_http(request: Request, model: str = "yolov8n", size: int = 480, conf: float = 0.25,
                      iou: float = 0.45, stream: str = "") -> JSONResponse:
    """The detections in the request body (a JPEG or PNG). With
    stream=name, the frame also goes to that stream's subscribers and the
    hook (a camera that isn't a browser)."""
    require_code(request)
    key = client_ip(request.headers, request.client.host if request.client else None) + "|" + stream
    delay = _post_next.get(key, 0.0) - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    _post_next[key] = time.monotonic() + 1.0 / MAX_FPS
    data = await request.body()
    try:
        target = get_stream(stream) if stream else None
        return JSONResponse(await detect(target, data, model, size, conf, iou))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ---------------------------------------------------------------- the page

STATIC = {"app.js": "text/javascript", "selftest.jpg": "image/jpeg", "selftest.json": "application/json",
          "favicon.svg": "image/svg+xml"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/models/{name}.onnx")
def model_file(name: str) -> FileResponse:
    if name not in MODELS:
        raise HTTPException(404)
    return FileResponse(ROOT / "models" / f"{name}.onnx", media_type="application/octet-stream")


@app.get("/{name}")
def static_file(name: str) -> FileResponse:
    if name not in STATIC or not (ROOT / name).exists():
        raise HTTPException(404)
    return FileResponse(ROOT / name, media_type=STATIC[name], headers={"Cache-Control": "no-cache"})
