"""Receives a stream's detections (what a robot controller would do).

    pip install websockets
    python subscribe.py --code ACCESS_CODE --stream my-phone
    python subscribe.py --code ACCESS_CODE --stream my-phone --frames   # also save the frames

The stream name is the one in the page's settings (or the "stream"
parameter of POST /api/detect).
"""

import argparse
import asyncio
import json
from urllib.parse import quote

import websockets

parser = argparse.ArgumentParser()
parser.add_argument("--server", default="wss://closer.ram.eemcs.utwente.nl/yolo/")
parser.add_argument("--code", required=True, help="the access code")
parser.add_argument("--stream", required=True)
parser.add_argument("--frames", action="store_true", help="also receive the JPEG frames (saved as latest.jpg)")
args = parser.parse_args()


async def main():
    url = f"{args.server.rstrip('/')}/api/streams/{quote(args.stream)}/ws?frames={int(args.frames)}"
    async with websockets.connect(url, additional_headers={"X-Access-Code": args.code}, max_size=8 * 2**20) as ws:
        print("connected, waiting for", args.stream)
        async for message in ws:
            if isinstance(message, bytes):          # the frame of the next result
                with open("latest.jpg", "wb") as f:
                    f.write(message)
                continue
            msg = json.loads(message)
            if msg["type"] == "result":
                found = ", ".join(f"{d['name']} {d['score']:.2f}" for d in msg["detections"]) or "nothing"
                print(f"#{msg['seq']} {msg['width']}x{msg['height']} {msg['ms']:.0f} ms: {found}")
                # ... decide here what the robot should do
            elif msg["type"] == "hook":
                print("  hook:", msg["data"])
            elif msg["type"] == "error":
                print("error:", msg["message"])


try:
    asyncio.run(main())
except websockets.ConnectionClosed as exc:
    print("connection closed:", exc.rcvd.reason if exc.rcvd else exc)
except KeyboardInterrupt:
    pass
