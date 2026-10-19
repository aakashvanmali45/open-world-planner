"""
yoloe_demo.py  --  detector bake-off: YOLOE in PROMPT-FREE mode.

Mirrors your YOLO-World detect demo so the outputs are comparable.
Prompt-free = no vocabulary to supply; labels come from YOLOE's built-in
4,585-name vocabulary. (Do NOT call set_classes() on a -pf checkpoint; it
raises AssertionError.)

Run:  python perception/yoloe_demo.py data/frames/frame_00000.jpg
"""
import sys
import time

import cv2
from ultralytics import YOLOE

path = sys.argv[1] if len(sys.argv) > 1 else "tests/data/sample.jpeg"

# yoloe-11s-seg-pf.pt: safe default, works on older ultralytics too.
# If ultralytics >= 8.4.0, yoloe-26n-seg-pf.pt is smaller/faster on CPU.
model = YOLOE("yoloe-11s-seg-pf.pt")

CONF = 0.5       # keep only boxes >= this. NOTE: open-vocab conf runs LOW;
                 # 0.5 may drop your novel objects. Try 0.25 if output looks empty.
IOU = 0.5        # class-agnostic NMS: drop a box overlapping a kept one by > this
IMGSZ = 640      # drop to 512 / 448 if CPU latency is too high

# Warm-up: first call downloads weights + builds the graph. Don't time it.
_ = model.predict(path, conf=CONF, iou=IOU, agnostic_nms=True, imgsz=IMGSZ, verbose=False)

t = time.perf_counter()
res = model.predict(path, conf=CONF, iou=IOU, agnostic_nms=True, imgsz=IMGSZ, verbose=False)[0]
dt = (time.perf_counter() - t) * 1000

boxes = res.boxes
n = 0 if boxes is None else len(boxes)
print(f"YOLOE-pf detect: {dt:.0f} ms, {n} objects")

bgr = cv2.imread(path)
if boxes is not None:
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        score = float(b.conf[0])
        label = model.names[int(b.cls[0])]
        print(f"  {label:14s} conf={score:.3f} box=({x1}, {y1}, {x2}, {y2})")
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(bgr, f"{label} {score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
cv2.imwrite("yoloe_out.jpg", bgr)
print("wrote yoloe_out.jpg")