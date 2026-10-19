"""
fastsam_demo.py  --  detector bake-off: FastSAM "segment everything".

The CPU-viable stand-in for MobileSAM (MobileSAM is ~24 s/image on CPU;
FastSAM-s is ~58 ms). This is the CLASS-AGNOSTIC option: it finds object
regions but gives NO labels and NO objectness-vs-class score the way a
detector does, and it over-segments (expect background blobs and parts).
Here to see how noisy a pure "detect any blob" front-end is vs the detectors.

Run:  python fastsam_demo.py data/frames/frame_00000.jpg
"""
import sys
import time

import cv2
from ultralytics import FastSAM

path = sys.argv[1] if len(sys.argv) > 1 else "tests/data/sample.jpeg"

model = FastSAM("FastSAM-s.pt")   # small variant; the one timed at ~58 ms CPU

IMGSZ = 640      # drop to 512 / 448 if too slow
CONF = 0.5       # keep only regions >= this confidence
IOU = 0.5        # drop a region overlapping a kept one by more than this (was 0.9)

# Warm-up (download + graph build) — not timed.
_ = model.predict(path, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)

t = time.perf_counter()
res = model.predict(path, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]
dt = (time.perf_counter() - t) * 1000

boxes = res.boxes
n = 0 if boxes is None else len(boxes)
print(f"FastSAM detect: {dt:.0f} ms, {n} regions (class-agnostic, no labels)")

bgr = cv2.imread(path)
if boxes is not None:
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        score = float(b.conf[0]) if b.conf is not None else 0.0
        print(f"  region{i:<3d} conf={score:.3f} box=({x1}, {y1}, {x2}, {y2})")
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 180, 0), 2)
cv2.imwrite("fastsam_out.jpg", bgr)
print("wrote fastsam_out.jpg")