"""Example hook: steer towards the largest person in view.

Enable it with YOLO_HOOK=server.hooks.example in server/.env. on_frame is
called in its own thread for each detected frame (frames that arrive while
it's still busy are skipped). What it returns goes to the camera's page
(its "text" is shown there) and to the stream's subscribers. A real hook
would send a command to a robot here (serial port, socket, ROS, ...).
"""

import logging

log = logging.getLogger("yolo.hook")


def on_start():
    """Called once, when the server starts (optional)."""
    log.info("example hook ready")


def on_frame(stream, image, detections, info):
    """stream: the stream's name; image: the frame (numpy, BGR, height x
    width x 3); detections: dicts with x1, y1, x2, y2 (pixels), score, cls
    and name; info: seq, t (Unix time), width, height, model."""
    people = [d for d in detections if d["name"] == "person" and d["score"] >= 0.5]
    if not people:
        return {"text": "no person: stop", "steer": 0.0, "speed": 0.0}
    p = max(people, key=lambda d: (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]))
    # -1 (far left) ... +1 (far right) of the image's centre
    steer = ((p["x1"] + p["x2"]) / 2 / info["width"]) * 2 - 1
    # how much of the image's height the person fills: close up -> slow down
    fill = (p["y2"] - p["y1"]) / info["height"]
    speed = round(max(0.0, 1.0 - fill / 0.8), 2)
    side = "left" if steer < -0.15 else "right" if steer > 0.15 else "straight ahead"
    return {"text": f"person {side} (steer {steer:+.2f}), speed {speed:.2f}", "steer": round(steer, 3), "speed": speed}
