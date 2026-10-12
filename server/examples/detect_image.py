"""Sends an image to the server and prints its detections: a camera that
isn't a browser (a robot's own camera, say). Standard library only.

    python detect_image.py --code ACCESS_CODE photo.jpg
    python detect_image.py --code ACCESS_CODE --stream robot-cam photo.jpg   # also to the stream's subscribers and hook
"""

import argparse
import json
import urllib.request
from urllib.parse import urlencode

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--server", default="https://closer.ram.eemcs.utwente.nl/yolo/")
parser.add_argument("--code", required=True, help="the access code")
parser.add_argument("--model", default="yolov8n", choices=["yolov8n", "yolov8s", "yolov8m"])
parser.add_argument("--size", default=480, type=int, choices=[320, 480, 640])
parser.add_argument("--stream", default="")
args = parser.parse_args()

query = urlencode({"model": args.model, "size": args.size, "stream": args.stream})
with open(args.image, "rb") as f:
    request = urllib.request.Request(f"{args.server.rstrip('/')}/api/detect?{query}", data=f.read(), method="POST",
                                     headers={"X-Access-Code": args.code, "Content-Type": "application/octet-stream"})
with urllib.request.urlopen(request) as response:
    result = json.load(response)
print(f"{result['width']}x{result['height']}, input {result['input']}, {result['ms']:.0f} ms "
      f"(+{result['wait_ms']:.0f} ms waiting)")
for d in result["detections"]:
    print(f"  {d['name']:<14} {d['score']:.2f}  ({d['x1']:.0f}, {d['y1']:.0f}) - ({d['x2']:.0f}, {d['y2']:.0f})")
