// YOLOv8 object detection in the browser with onnxruntime-web (WebGPU when
// available, else WebAssembly). Nothing is sent to a server.
'use strict';

const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

const COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
  'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
  'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
  'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];
// Ultralytics' box colours.
const PALETTE = ['#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86',
  '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085',
  '#CB38FF', '#FF95C8', '#FF37C7'];
const MAX_CANDIDATES = 3000, MAX_DETECTIONS = 100;

const $ = id => document.getElementById(id);
const video = $('cam'), canvas = $('out'), ctx = canvas.getContext('2d');
// The detector's input: the frame, scaled and padded to a multiple of 32.
const inCanvas = document.createElement('canvas');
const inCtx = inCanvas.getContext('2d', {willReadFrequently: true});
// The frame the latest boxes belong to ("boxes in sync", pause).
const snapCanvas = document.createElement('canvas');
const snapCtx = snapCanvas.getContext('2d');

const params = new URLSearchParams(location.search);
let backend = null;          // 'webgpu' | 'wasm'
const sessions = {};         // model name -> Promise<InferenceSession>
let stream = null, facing = 'environment';
let image = null;            // an opened image instead of the camera
let paused = false, running = false;
let last = null;             // {raw, n, map, frameW, frameH, inW, inH, ms}
let detections = [];
let busy = false;            // an inference is running
let fpsTimes = [];
const gpuRuns = {};          // model@input size -> inferences on WebGPU (see detect)
const gpuChecked = {};       // model@input size -> passed the WebGPU self-test
let gpuName = '';            // the WebGPU adapter, for the HUD
let notice = '';             // why WebGPU isn't used, for the HUD

// ---------------------------------------------------------------- loading

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src; s.crossOrigin = 'anonymous';
    s.onload = resolve; s.onerror = () => reject(new Error('Could not load ' + src));
    document.head.appendChild(s);
  });
}

async function pickBackend() {
  const want = params.get('backend');   // ?backend=wasm or ?backend=webgpu
  if (want !== 'wasm' && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      // A software (CPU) WebGPU adapter is far slower than WebAssembly.
      if (adapter && (want === 'webgpu' || !(adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter))) {
        const info = adapter.info || {};
        gpuName = [info.vendor, info.architecture].filter(Boolean).join(' ');
        return 'webgpu';
      }
    } catch (e) { /* no WebGPU */ }
  }
  return 'wasm';
}

async function initOrt() {
  backend = await pickBackend();
  await loadScript(ORT_URL + (backend === 'webgpu' ? 'ort.webgpu.min.js' : 'ort.wasm.min.js'));
  ort.env.wasm.wasmPaths = ORT_URL;
  ort.env.logLevel = 'error';
  // Threads need a cross-origin isolated page, which GitHub Pages doesn't serve.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  $('size').value = params.get('size') || (backend === 'webgpu' ? '640' : '480');
  $('backend').value = params.get('backend') || 'auto';
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  // content-length may be the compressed size: the progress is approximate.
  const total = +res.headers.get('content-length') || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress(Math.min(got / total, 1));
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

function getSession(name) {
  if (!sessions[name]) {
    sessions[name] = (async () => {
      const bytes = await fetchWithProgress(`models/${name}.onnx`,
        f => setStatus(`Downloading ${name}: ${Math.round(f * 100)}%`));
      setStatus(`Preparing ${name} (${backend === 'webgpu' ? 'WebGPU' : 'WebAssembly'})...`);
      try {
        return await ort.InferenceSession.create(bytes, {executionProviders: [backend], graphOptimizationLevel: 'all', logSeverityLevel: 3});
      } catch (e) {
        if (backend !== 'webgpu') throw e;
        console.warn('WebGPU failed, using WebAssembly', e);
        backend = 'wasm';
        return await ort.InferenceSession.create(bytes, {executionProviders: ['wasm'], logSeverityLevel: 3});
      }
    })();
    sessions[name].catch(() => delete sessions[name]);
  }
  return sessions[name];
}

function setStatus(text) {
  $('status').textContent = text;
  if (running) $('hud').textContent = text;
}

// ---------------------------------------------------------------- detection

// Scales the frame (longest side = input size, keeping its aspect ratio),
// pads it with grey to a multiple of 32 on both sides, centred, as
// Ultralytics does, and returns the NCHW float tensor data.
function preprocess(source, frameW, frameH, size) {
  const r = Math.min(size / frameW, size / frameH);
  const w = Math.round(frameW * r), h = Math.round(frameH * r);
  const inW = Math.ceil(w / 32) * 32, inH = Math.ceil(h / 32) * 32;
  const left = Math.floor((inW - w) / 2), top = Math.floor((inH - h) / 2);
  if (inCanvas.width !== inW || inCanvas.height !== inH) { inCanvas.width = inW; inCanvas.height = inH; }
  inCtx.fillStyle = 'rgb(114,114,114)';
  inCtx.fillRect(0, 0, inW, inH);
  inCtx.drawImage(source, left, top, w, h);
  const px = inCtx.getImageData(0, 0, inW, inH).data;
  const plane = inW * inH, data = new Float32Array(3 * plane);
  for (let i = 0, j = 0; i < plane; i++, j += 4) {
    data[i] = px[j] / 255;
    data[i + plane] = px[j + 1] / 255;
    data[i + 2 * plane] = px[j + 2] / 255;
  }
  return {data, inW, inH, map: {r, left, top}};
}

// Output [1, 84, n]: per anchor the box (cx, cy, w, h, in input pixels)
// and the 80 class scores. Returns the boxes above `conf` after per-class
// non-maximum suppression, in frame pixels.
function postprocess(res, conf, iou) {
  const {raw, n, map, frameW, frameH} = res;
  let cand = [];
  for (let i = 0; i < n; i++) {
    let best = 0, cls = -1;
    for (let c = 0, k = 4 * n + i; c < 80; c++, k += n) {
      if (raw[k] > best) { best = raw[k]; cls = c; }
    }
    if (best < conf) continue;
    const cx = raw[i], cy = raw[n + i], w = raw[2 * n + i], h = raw[3 * n + i];
    cand.push({
      cls, score: best,
      x1: Math.max(0, (cx - w / 2 - map.left) / map.r), y1: Math.max(0, (cy - h / 2 - map.top) / map.r),
      x2: Math.min(frameW, (cx + w / 2 - map.left) / map.r), y2: Math.min(frameH, (cy + h / 2 - map.top) / map.r),
    });
  }
  cand.sort((a, b) => b.score - a.score);
  if (cand.length > MAX_CANDIDATES) cand = cand.slice(0, MAX_CANDIDATES);
  const keep = [];
  for (const d of cand) {
    let ok = true;
    for (const k of keep) {
      if (k.cls === d.cls && overlap(k, d) > iou) { ok = false; break; }
    }
    if (ok) { keep.push(d); if (keep.length >= MAX_DETECTIONS) break; }
  }
  return keep;
}

function overlap(a, b) {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1), h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

function thresholds() {
  return [+$('conf').value / 100, +$('iou').value / 100];
}

function refilter() {
  if (last) detections = postprocess(last, ...thresholds());
  updateHud();
}

function switchToWasm(reason) {
  console.warn(reason + ', using WebAssembly');
  notice = reason + ': using WebAssembly.';
  backend = 'wasm';
  for (const k in sessions) delete sessions[k];
  $('size').value = '480';
}

// Some GPUs (seen on phones) run the model on WebGPU but return wrong
// numbers. So before a model is used on WebGPU at an input size, it
// detects selftest.jpg, and the result is compared with what WebAssembly
// finds there (selftest.json, made with tools/make_selftest.py).
let selftest = null;
function loadSelftest() {
  if (!selftest) {
    selftest = Promise.all([
      fetch('selftest.jpg').then(r => r.blob()).then(b => createImageBitmap(b)),
      fetch('selftest.json').then(r => r.json()),
    ]);
    selftest.catch(() => { selftest = null; });
  }
  return selftest;
}

async function gpuSelfTest(session, modelName, size) {
  let img, expected;
  try { [img, expected] = await loadSelftest(); } catch (e) { console.warn('No self-test', e); return true; }
  const want = expected[modelName]?.[size];
  if (!want) return true;
  const {data, inW, inH, map} = preprocess(img, img.width, img.height, size);
  const out = await session.run({[session.inputNames[0]]: new ort.Tensor('float32', data, [1, 3, inH, inW])});
  const tensor = out[session.outputNames[0]];
  const raw = tensor.data;
  for (let i = 0; i < raw.length; i++) if (!Number.isFinite(raw[i])) return false;
  const found = postprocess({raw, n: tensor.dims[2], map, frameW: img.width, frameH: img.height}, 0.25, 0.45);
  // Each confident detection of WebAssembly found again, and not many more.
  const ok = want.every(e => found.some(d => d.cls === e.cls && overlap(d, e) > 0.6)) &&
    found.filter(d => d.score >= 0.5).length <= want.length + 3;
  if (!ok) console.warn(`WebGPU self-test of ${modelName}@${size} failed`, {want, found});
  return ok;
}

async function detect(source, frameW, frameH) {
  const modelName = $('model').value, size = +$('size').value;
  const session = await getSession(modelName);
  const checkKey = `${modelName}@${size}`;
  if (backend === 'webgpu' && !(checkKey in gpuChecked)) {
    setStatus(`Checking ${modelName} on WebGPU...`);
    gpuChecked[checkKey] = await gpuSelfTest(session, modelName, size);
    if (!gpuChecked[checkKey]) {
      if (params.get('backend') !== 'webgpu') {
        switchToWasm('WebGPU gave wrong results on this device');
        return detect(source, frameW, frameH);
      }
      notice = 'WebGPU gave wrong results in the self-test (forced WebGPU).';
    }
  }
  const t0 = performance.now();
  const {data, inW, inH, map} = preprocess(source, frameW, frameH, size);
  // The analysed frame, for "boxes in sync" and pause.
  if (snapCanvas.width !== frameW || snapCanvas.height !== frameH) { snapCanvas.width = frameW; snapCanvas.height = frameH; }
  snapCtx.drawImage(source, 0, 0, frameW, frameH);
  const input = new ort.Tensor('float32', data, [1, 3, inH, inW]);
  const out = await session.run({[session.inputNames[0]]: input});
  const tensor = out[session.outputNames[0]];
  const raw = tensor.data;   // copied to the CPU
  input.dispose?.();
  const ms = performance.now() - t0;
  const used = backend;
  // The first run of a model on WebGPU, at each input size, compiles its
  // shaders; if the next one is still slow, the GPU is a poor one (or
  // emulated): WebAssembly does better.
  const runKey = `${modelName}@${inW}x${inH}`;
  if (backend === 'webgpu' && (gpuRuns[runKey] = (gpuRuns[runKey] || 0) + 1) >= 2 && ms > 1500 &&
      params.get('backend') !== 'webgpu') {
    switchToWasm(`WebGPU took ${(ms / 1000).toFixed(1)} s per frame`);
  }
  last = {raw, n: tensor.dims[2], map, frameW, frameH, inW, inH, ms, model: modelName, backend: used};
  detections = postprocess(last, ...thresholds());
  const now = performance.now();
  fpsTimes.push(now);
  while (fpsTimes.length && now - fpsTimes[0] > 2000) fpsTimes.shift();
  updateHud();
}

async function detectLoop() {
  while (running) {
    if (!paused && !image && video.readyState >= 2 && video.videoWidth) {
      busy = true;
      try {
        await detect(video, video.videoWidth, video.videoHeight);
      } catch (e) {
        showError(e);
        await new Promise(r => setTimeout(r, 1000));
      }
      busy = false;
    }
    // Let the page draw (and the browser breathe) between inferences.
    await new Promise(r => requestAnimationFrame(r));
  }
}

// ---------------------------------------------------------------- drawing

function draw() {
  const live = !image && !paused && !$('sync').checked;
  const src = image ? snapCanvas : live ? video : snapCanvas;
  const w = image ? snapCanvas.width : live ? video.videoWidth : snapCanvas.width;
  const h = image ? snapCanvas.height : live ? video.videoHeight : snapCanvas.height;
  if (w && h) {
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const mirror = $('mirror').checked && !image;
    ctx.setTransform(mirror ? -1 : 1, 0, 0, 1, mirror ? w : 0, 0);
    ctx.drawImage(src, 0, 0, w, h);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    // Boxes of a frame of another size (camera switched) are dropped.
    if (last && last.frameW === w && last.frameH === h) drawBoxes(w, h, mirror);
  }
  requestAnimationFrame(draw);
}

function drawBoxes(w, h, mirror) {
  // Canvas pixels per screen pixel (the canvas is scaled to fit the window).
  const scale = Math.max(w / (canvas.clientWidth || w), h / (canvas.clientHeight || h));
  const lw = Math.max(2 * scale, Math.round((w + h) / 2 * 0.003));
  const fs = Math.round(Math.max(13 * scale, lw * 5.5));
  ctx.lineWidth = lw;
  ctx.font = `600 ${fs}px system-ui,sans-serif`;
  ctx.textBaseline = 'top';
  for (const d of detections) {
    const x1 = mirror ? w - d.x2 : d.x1, x2 = mirror ? w - d.x1 : d.x2;
    const color = PALETTE[d.cls % PALETTE.length];
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, d.y1, x2 - x1, d.y2 - d.y1);
    const text = `${COCO[d.cls]} ${Math.round(d.score * 100)}%`;
    const tw = ctx.measureText(text).width + fs * 0.5, th = fs * 1.3;
    const ty = d.y1 - th >= 0 ? d.y1 - th : d.y1;   // above the box, or inside when at the top edge
    const tx = Math.min(Math.max(0, x1 - lw / 2), w - tw);
    ctx.fillStyle = color;
    ctx.fillRect(tx, ty, tw, th);
    ctx.fillStyle = textColor(color);
    ctx.fillText(text, tx + fs * 0.25, ty + fs * 0.15);
  }
}

function textColor(hex) {
  const v = parseInt(hex.slice(1), 16);
  const y = 0.299 * (v >> 16) + 0.587 * ((v >> 8) & 255) + 0.114 * (v & 255);
  return y > 150 ? '#000' : '#fff';
}

// ---------------------------------------------------------------- HUD and UI

function updateHud() {
  if (!last) return;
  const counts = new Map();
  for (const d of detections) counts.set(d.cls, (counts.get(d.cls) || 0) + 1);
  const list = [...counts].sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .map(([c, k]) => k > 1 ? `${COCO[c]} x${k}` : COCO[c]).join(', ');
  const fps = fpsTimes.length > 1 ? (fpsTimes.length - 1) / ((fpsTimes[fpsTimes.length - 1] - fpsTimes[0]) / 1000) : 0;
  const head = `${last.model} | ${last.backend === 'webgpu' ? `WebGPU${gpuName ? ` (${gpuName})` : ''}` : 'WebAssembly'} | input ${last.inW}x${last.inH}` +
    ` | ${last.ms.toFixed(0)} ms` + (image || paused ? '' : ` | ${fps.toFixed(1)} fps`) + (paused ? ' | paused' : '');
  $('hud').textContent = `${head}\n${detections.length ? list : 'nothing detected'}` + (notice ? `\n${notice}` : '');
}

function showError(e) {
  console.error(e);
  const msg = e && e.message ? e.message : String(e);
  $('err').textContent = msg;
  $('hud').textContent = 'Error: ' + msg;
}

async function startCamera() {
  $('err').textContent = '';
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(location.protocol === 'https:' || location.hostname === 'localhost'
        ? 'This browser has no camera access.' : 'Camera access needs HTTPS.');
    }
    if (stream) stream.getTracks().forEach(t => t.stop());
    const [cw, ch] = $('camres').value.split('x').map(Number);
    stream = await navigator.mediaDevices.getUserMedia({
      video: {facingMode: facing, width: {ideal: cw}, height: {ideal: ch}}, audio: false});
    video.srcObject = stream;
    await video.play();
    const actual = stream.getVideoTracks()[0].getSettings().facingMode;
    $('mirror').checked = (actual || facing) === 'user';
    image = null; paused = false; $('pause').textContent = 'Pause';
    last = null; detections = []; fpsTimes = [];
    enterApp();
  } catch (e) {
    $('start').classList.remove('hidden');
    $('err').textContent = 'Camera: ' + (e.message || e) + ' You can still open an image.';
    console.error(e);
  }
}

function enterApp() {
  $('start').classList.add('hidden');
  if (!running) { running = true; detectLoop(); requestAnimationFrame(draw); }
}

async function openImage(file) {
  if (!file) return;
  try {
    const bitmap = await createImageBitmap(file);
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    image = bitmap; paused = false; $('pause').textContent = 'Camera';
    enterApp();
    await detectImage();
  } catch (e) {
    showError(e);
  }
}

async function detectImage() {
  if (!image) return;
  $('hud').textContent = 'Detecting...';
  while (busy) await new Promise(r => setTimeout(r, 20));
  busy = true;
  try { await detect(image, image.width, image.height); } catch (e) { showError(e); }
  busy = false;
}

$('conf').oninput = () => { $('confv').textContent = $('conf').value + '%'; refilter(); };
$('iou').oninput = () => { $('iouv').textContent = (+$('iou').value / 100).toFixed(2); refilter(); };
$('model').onchange = () => { fpsTimes = []; detectImage(); };
$('size').onchange = () => { fpsTimes = []; detectImage(); };
$('flip').onclick = () => { facing = facing === 'environment' ? 'user' : 'environment'; startCamera(); };
$('camres').onchange = () => { if (!image) startCamera(); };
$('pause').onclick = () => {
  if (image) { startCamera(); return; }   // back to the camera
  paused = !paused;
  $('pause').textContent = paused ? 'Resume' : 'Pause';
  fpsTimes = [];
  updateHud();
};
$('openimg').onclick = $('go-img').onclick = () => $('file').click();
$('file').onchange = () => { openImage($('file').files[0]); $('file').value = ''; };
$('go').onclick = startCamera;
$('backend').onchange = () => {
  const q = new URLSearchParams(location.search);
  if ($('backend').value === 'auto') q.delete('backend'); else q.set('backend', $('backend').value);
  q.delete('size');
  location.search = q.toString();
};
$('more').onclick = () => document.body.classList.toggle('showopts');

(async () => {
  try {
    await initOrt();
    await getSession($('model').value);
    setStatus(`Ready (${backend === 'webgpu' ? 'WebGPU' : 'WebAssembly'}).`);
    $('go').disabled = $('go-img').disabled = false;
  } catch (e) {
    $('status').textContent = '';
    showError(e);
  }
})();
