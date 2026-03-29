#!/usr/bin/env python3
from __future__ import annotations

import atexit
import os
import time

import cv2
from flask import Flask, Response, render_template_string

from app.weed_spray import (
    FRONT_CROP_BOTTOM_RATIO,
    FRONT_CROP_TOP_RATIO,
    FRONT_MIN_GREEN_RATIO,
    FRONT_MIN_PATH_RATIO,
    FRONT_ZONE_WIDTH_RATIO,
    SIDE_CROP_BOTTOM_RATIO,
    SIDE_CROP_TOP_RATIO,
    SIDE_MIN_GREEN_RATIO,
    SIDE_MIN_PATH_RATIO,
    SIDE_ZONE_WIDTH_RATIO,
    WeedDetector,
    open_front_usb,
    open_side_cameras,
)


JPEG_QUALITY = 75
app = Flask(__name__)

class Source:
    def __init__(self, label: str, camera, detector: WeedDetector) -> None:
        self.popisek = label
        self.kamera = camera
        self.detektor = detector

    def frame(self):
        obrazek = self.kamera.read()
        vysledek = self.detektor.analyze(obrazek)
        return vysledek.obrazek

    def close(self) -> None:
        self.kamera.close()


zdroje = []


def _init_sources():
    predni = open_front_usb()
    leva, prava = open_side_cameras()
    return [
        Source(
            f"Predni USB (stred {int(FRONT_ZONE_WIDTH_RATIO * 100)}%)",
            predni,
            WeedDetector(
                min_path_ratio=FRONT_MIN_PATH_RATIO,
                min_green_ratio=FRONT_MIN_GREEN_RATIO,
                crop_top_ratio=FRONT_CROP_TOP_RATIO,
                crop_bottom_ratio=FRONT_CROP_BOTTOM_RATIO,
                zone_width_ratio=FRONT_ZONE_WIDTH_RATIO,
            ),
        ),
        Source(
            f"Leva CSI (stred {int(SIDE_ZONE_WIDTH_RATIO * 100)}%)",
            leva,
            WeedDetector(
                min_path_ratio=SIDE_MIN_PATH_RATIO,
                min_green_ratio=SIDE_MIN_GREEN_RATIO,
                crop_top_ratio=SIDE_CROP_TOP_RATIO,
                crop_bottom_ratio=SIDE_CROP_BOTTOM_RATIO,
                zone_width_ratio=SIDE_ZONE_WIDTH_RATIO,
            ),
        ),
        Source(
            f"Prava CSI (stred {int(SIDE_ZONE_WIDTH_RATIO * 100)}%)",
            prava,
            WeedDetector(
                min_path_ratio=SIDE_MIN_PATH_RATIO,
                min_green_ratio=SIDE_MIN_GREEN_RATIO,
                crop_top_ratio=SIDE_CROP_TOP_RATIO,
                crop_bottom_ratio=SIDE_CROP_BOTTOM_RATIO,
                zone_width_ratio=SIDE_ZONE_WIDTH_RATIO,
            ),
        ),
    ]


HTML_INDEX = """
<!doctype html>
<title>Weed Spray Preview</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; }
  .grid { display:flex; flex-wrap:wrap; gap:10px; padding:10px; }
  .card { border:1px solid #333; padding:6px; background:#222; }
  img { display:block; max-width:100%; height:auto; }
</style>
<div class="grid">
  {% for idx, s in sources %}
  <div class="card">
    <h3>{{ s.popisek }}</h3>
    <img src="/cam/{{ idx }}">
  </div>
  {% endfor %}
</div>
"""


@atexit.register
def _cleanup() -> None:
    for zdroj in zdroje:
        try:
            zdroj.close()
        except Exception:
            pass


def _mjpeg_stream(source_idx: int):
    if source_idx >= len(zdroje):
        return
    zdroj = zdroje[source_idx]
    while True:
        try:
            obrazek = zdroj.frame()
            if obrazek is None:
                time.sleep(0.1)
                continue
            ok, jpeg = cv2.imencode(".jpg", obrazek, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        except Exception:
            time.sleep(0.2)


@app.route("/")
def index():
    return render_template_string(HTML_INDEX, sources=list(enumerate(zdroje)))


@app.route("/cam/<int:idx>")
def cam(idx):
    return Response(_mjpeg_stream(idx), mimetype="multipart/x-mixed-replace; boundary=frame")


def main() -> None:
    global zdroje
    zdroje = _init_sources()
    app.run(host="0.0.0.0", port=int(os.environ.get("WEED_SPRAY_PREVIEW_PORT", "5050")), threaded=True)


if __name__ == "__main__":
    main()
