"""Writes selftest.json: what WebAssembly detects in selftest.jpg with each
model and input size (detections with a score of at least 0.5). The page
compares WebGPU's result with it (see gpuSelfTest in app.js).

Needs Playwright with Chromium (pip install playwright; playwright install chromium):

    python -m http.server 8000      # in the repository
    python tools/make_selftest.py http://localhost:8000/
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/"
OUT = Path(__file__).resolve().parent.parent / "selftest.json"

SCRIPT = """async ([models, sizes]) => {
  const blob = await (await fetch('selftest.jpg', {cache: 'no-store'})).blob();
  const img = await createImageBitmap(blob);
  const result = {};
  for (const name of models) {
    const session = await getSession(name);
    result[name] = {};
    for (const size of sizes) {
      const {data, inW, inH, map} = preprocess(img, img.width, img.height, size);
      const out = await session.run({[session.inputNames[0]]: new ort.Tensor('float32', data, [1, 3, inH, inW])});
      const t = out[session.outputNames[0]];
      result[name][size] = postprocess({raw: t.data, n: t.dims[2], map, frameW: img.width, frameH: img.height}, 0.5, 0.45)
        .map(d => ({cls: d.cls, score: +d.score.toFixed(3),
                    x1: +d.x1.toFixed(1), y1: +d.y1.toFixed(1), x2: +d.x2.toFixed(1), y2: +d.y2.toFixed(1)}));
    }
  }
  return result;
}"""

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL + "?backend=wasm")
    page.wait_for_function("!document.getElementById('go').disabled", timeout=300000)
    result = page.evaluate(SCRIPT, [["yolov8n", "yolov8s", "yolov8m"], [320, 480, 640]])
    browser.close()

OUT.write_text(json.dumps(result, indent=1) + "\n")
for name, sizes in result.items():
    print(name, {size: len(dets) for size, dets in sizes.items()})
