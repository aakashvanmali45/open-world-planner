"""
perception/demo_track.py -- eyeball the tracker on a real keyframe folder.

Runs: detector -> N-of-M tracker, frame by frame, over data/frames/*.jpg.
Draws each live track's id + status (green=confirmed, orange=candidate) and
writes annotated frames to outputs/track/. Prints a per-frame line and a final
summary: how many tracks were created vs how many ever reached 'confirmed'.

Usage:
    python perception/demo_track.py
    python perception/demo_track.py --frames data/frames --backend yoloe --conf 0.1
"""
import argparse
import glob
import os

import cv2

import argparse, glob, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from perception.detector import Detector
from perception.track import Tracker

GREEN  = (0, 180, 0)      # confirmed
ORANGE = (0, 140, 255)    # candidate


def main():
    ap = argparse.ArgumentParser(description="Run detector + tracker over a frame folder.")
    ap.add_argument("--frames", default="data/frames", help="folder of frame_*.jpg")
    ap.add_argument("--backend", default="yoloe", choices=["yoloe", "yoloworld"])
    ap.add_argument("--conf", type=float, default=None, help="override detector min conf")
    ap.add_argument("--out", default="outputs/track", help="where annotated frames go")
    # tracker knobs (same defaults as Tracker)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--max-age", type=int, default=3)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames, "frame_*.jpg")))
    if not frames:
        raise SystemExit(f"no frames found in {args.frames}")

    det = Detector(backend=args.backend)
    if args.backend == "yoloworld":
        det.set_vocabulary(["bottle", "mug", "cup", "book"])   # demo vocab; yoloe ignores
    if args.conf is not None:
        det.min_conf = args.conf                               # lower to catch marginal objects
    trk = Tracker(iou_thresh=args.iou, n=args.n, m=args.m, max_age=args.max_age)

    os.makedirs(args.out, exist_ok=True)
    det.warmup(frames[0])

    all_ids, confirmed_ids = set(), set()
    for i, path in enumerate(frames):
        dets = det.detect(path)
        trk.update(dets, i)

        bgr = cv2.imread(path)
        for t in trk.tracks:                       # draw ALL live tracks, not just confirmed
            all_ids.add(t.id)
            x1, y1, x2, y2 = t.box
            color = GREEN if t.status == "confirmed" else ORANGE
            if t.status == "confirmed":
                confirmed_ids.add(t.id)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(bgr, f"T{t.id} {t.status[:4]}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(os.path.join(args.out, os.path.basename(path)), bgr)

        n_conf = sum(1 for t in trk.tracks if t.status == "confirmed")
        print(f"frame {i:03d}: {len(dets):2d} dets | live {len(trk.tracks):2d} | confirmed {n_conf}")

    print(f"\n{len(frames)} frames processed")
    print(f"tracks created : {len(all_ids)}")
    print(f"ever confirmed : {len(confirmed_ids)}  (ids: {sorted(confirmed_ids)})")
    print(f"never confirmed: {len(all_ids) - len(confirmed_ids)}  <- flickers / junk the gate dropped")
    print(f"annotated frames -> {args.out}/")


if __name__ == "__main__":
    main()