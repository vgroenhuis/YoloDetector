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

// Detection on the server (server/ in the repository): the page on GitHub
// Pages uses the CloSer server; the page served by the server, itself.
const SERVER_URL = params.get('server') ||
  (location.hostname.endsWith('github.io') ? 'https://closer.ram.eemcs.utwente.nl/yolo/' : new URL('.', location.href).href);
let mode = 'device';         // 'device' | 'server' (Process on)
let srv = null;              // the server connection: {ws, ready, pending, name}
let srvBlocked = '';         // why not to reconnect (wrong code, ...) until the settings change
let srvSeq = 0;
let hookText = '';           // the server hook's latest message
const sendCanvas = document.createElement('canvas');
const sendCtx = sendCanvas.getContext('2d');

function store(key, value) { try { localStorage.setItem('yolo.' + key, value); } catch (e) { /* no storage */ } }
function stored(key) { try { return localStorage.getItem('yolo.' + key); } catch (e) { return null; } }

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

let ortReady = null;
function ensureOrt() {
  if (!ortReady) {
    ortReady = (async () => { await initOrt(); await getSession($('model').value); })();
    ortReady.catch(() => { ortReady = null; });
  }
  return ortReady;
}

async function initOrt() {
  backend = await pickBackend();
  await loadScript(ORT_URL + (backend === 'webgpu' ? 'ort.webgpu.min.js' : 'ort.wasm.min.js'));
  ort.env.wasm.wasmPaths = ORT_URL;
  ort.env.logLevel = 'error';
  // Threads need a cross-origin isolated page, which GitHub Pages doesn't serve.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  if (!params.get('size') && backend === 'webgpu' && mode === 'device') $('size').value = '640';
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
  if (last) detections = last.server ? filterServer(last) : postprocess(last, ...thresholds());
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
  if (mode === 'server') return serverDetect(source, frameW, frameH);
  await ensureOrt();
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

// ---------------------------------------------------------------- the server

function connectServer() {
  if (srv) return srv.ready;
  if (srvBlocked) return Promise.reject(new Error(srvBlocked));
  const url = SERVER_URL.replace(/^http/, 'ws').replace(/\/?$/, '/') + 'ws';
  const ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  const conn = {ws, pending: new Map(), name: ''};
  srv = conn;
  setStatus('Connecting to the server...');
  conn.ready = new Promise((resolve, reject) => {
    ws.onopen = () => ws.send(JSON.stringify({type: 'hello', code: $('code').value, stream: $('streamname').value.trim()}));
    ws.onmessage = ev => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'welcome') { conn.name = msg.stream; resolve(conn); }
      else if (msg.type === 'result') {
        conn.pending.get(msg.client_seq)?.resolve(msg);
        conn.pending.delete(msg.client_seq);
      } else if (msg.type === 'hook') {
        const d = msg.data;
        hookText = d && typeof d === 'object' && 'text' in d ? String(d.text) : JSON.stringify(d);
        updateHud();
      } else if (msg.type === 'error') {
        if (msg.client_seq != null && conn.pending.has(msg.client_seq)) {
          conn.pending.get(msg.client_seq).reject(new Error(msg.message));
          conn.pending.delete(msg.client_seq);
        } else {
          conn.error = msg.message;
        }
      }
    };
    ws.onclose = ev => {
      const why = conn.error || `the connection to the server closed (${ev.code})`;
      // Wrong code, stream in use, server full: don't retry until the settings change.
      if ([4000, 4001, 4003, 4009].includes(ev.code)) srvBlocked = 'Server: ' + why;
      for (const p of conn.pending.values()) p.reject(new Error('Server: ' + why));
      conn.pending.clear();
      if (srv === conn) srv = null;
      reject(new Error('Server: ' + why));
    };
  });
  conn.ready.catch(() => {});
  return conn.ready;
}

function disconnectServer() {
  srvBlocked = '';
  hookText = '';
  if (srv) { const ws = srv.ws; srv = null; ws.close(); }
}

// Sends the frame (as a JPEG, longest side the input size) and waits for its
// detections. The server sends them in the JPEG's pixels.
async function serverDetect(source, frameW, frameH) {
  const conn = await connectServer();
  const modelName = $('model').value, size = +$('size').value;
  const r = Math.min(1, size / Math.max(frameW, frameH));
  const w = Math.round(frameW * r), h = Math.round(frameH * r);
  if (sendCanvas.width !== w || sendCanvas.height !== h) { sendCanvas.width = w; sendCanvas.height = h; }
  sendCtx.drawImage(source, 0, 0, w, h);
  if (snapCanvas.width !== frameW || snapCanvas.height !== frameH) { snapCanvas.width = frameW; snapCanvas.height = frameH; }
  snapCtx.drawImage(source, 0, 0, frameW, frameH);
  const blob = await new Promise(res => sendCanvas.toBlob(res, 'image/jpeg', 0.85));
  const seq = ++srvSeq, t0 = performance.now();
  const reply = new Promise((resolve, reject) => {
    conn.pending.set(seq, {resolve, reject});
    setTimeout(() => { if (conn.pending.delete(seq)) reject(new Error('Server: no answer')); }, 20000);
  });
  // The thresholds are applied on the server: its stream's subscribers and
  // hook get what this page shows.
  conn.ws.send(JSON.stringify({type: 'frame', seq, model: modelName, size, conf: +$('conf').value / 100, iou: +$('iou').value / 100}));
  conn.ws.send(await blob.arrayBuffer());
  const res = await reply;
  const all = res.detections.map(d => ({cls: d.cls, score: d.score, x1: d.x1 / r, y1: d.y1 / r, x2: d.x2 / r, y2: d.y2 / r}));
  last = {server: true, all, frameW, frameH, inW: res.input[0], inH: res.input[1], ms: res.ms,
          rtt: performance.now() - t0, model: res.model, backend: 'server', stream: conn.name};
  detections = filterServer(last);
  const now = performance.now();
  fpsTimes.push(now);
  while (fpsTimes.length && now - fpsTimes[0] > 2000) fpsTimes.shift();
  updateHud();
}

function filterServer(res) {
  const conf = +$('conf').value / 100;
  return res.all.filter(d => d.score >= conf).slice(0, MAX_DETECTIONS);
}

async function detectLoop() {
  while (running) {
    if (!paused && !image && video.readyState >= 2 && video.videoWidth) {
      busy = true;
      try {
        await detect(video, video.videoWidth, video.videoHeight);
      } catch (e) {
        showError(e);
        await new Promise(r => setTimeout(r, mode === 'server' ? 2000 : 1000));
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
  const where = last.server ? 'server' : last.backend === 'webgpu' ? `WebGPU${gpuName ? ` (${gpuName})` : ''}` : 'WebAssembly';
  const time = last.server ? `${last.ms.toFixed(0)} ms (${last.rtt.toFixed(0)} ms round trip)` : `${last.ms.toFixed(0)} ms`;
  const head = `${last.model} | ${where} | input ${last.inW}x${last.inH} | ${time}` +
    (image || paused ? '' : ` | ${fps.toFixed(1)} fps`) + (paused ? ' | paused' : '') +
    (last.server ? ` | stream ${last.stream}` : '');
  const extra = last.server ? (hookText ? `\n> ${hookText}` : '') : (notice ? `\n${notice}` : '');
  $('hud').textContent = `${head}\n${detections.length ? list : 'nothing detected'}${extra}`;
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
// On the server the thresholds are applied there: detect an image again
// (a lower confidence threshold shows more boxes only after that).
$('iou').onchange = $('conf').onchange = () => { if (mode === 'server') detectImage(); };
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

// Served by the CloSer server (server/): link back to its project page.
if (location.hostname === 'closer.ram.eemcs.utwente.nl') {
  for (const a of document.querySelectorAll('a.home')) { a.href = '/'; a.textContent = '\u2190 CloSer projects'; }
}

// Process on, access code and stream: on the start screen and in the settings.
function setMode(value) {
  mode = value === 'server' ? 'server' : 'device';
  $('where').value = $('where0').value = mode;
  document.body.classList.toggle('server', mode === 'server');
  store('where', mode);
  fpsTimes = [];
  if (mode === 'device') {
    disconnectServer();
    $('go').disabled = $('go-img').disabled = true;
    ensureOrt().then(() => {
      setStatus(`Ready (${backend === 'webgpu' ? 'WebGPU' : 'WebAssembly'}).`);
      $('go').disabled = $('go-img').disabled = false;
      detectImage();
    }).catch(e => { $('status').textContent = ''; showError(e); });
  } else {
    setStatus('Ready: detection on the server.');
    $('go').disabled = $('go-img').disabled = false;
    detectImage();
  }
}

function pair(a, b, key, onChange) {
  for (const [x, y] of [[a, b], [b, a]]) {
    $(x).addEventListener('change', () => { $(y).value = $(x).value; store(key, $(x).value); onChange(); });
  }
}

$('where0').onchange = () => setMode($('where0').value);
$('where').onchange = () => setMode($('where').value);
const serverSettingsChanged = () => { disconnectServer(); if (mode === 'server') detectImage(); };
pair('code0', 'code', 'code', serverSettingsChanged);
pair('streamname0', 'streamname', 'stream', serverSettingsChanged);
$('code').value = $('code0').value = params.get('code') || stored('code') || '';
$('streamname').value = $('streamname0').value = stored('stream') ||
  'cam-' + Math.random().toString(36).slice(2, 6);
store('stream', $('streamname').value);
if (params.get('size')) $('size').value = params.get('size');
setMode(params.get('where') || stored('where') || 'device');
