"""
perception/demo_loop_viz.py -- run the loop and DRAW its decisions.
Box colour is keyed to the memory object_id: two DIFFERENT objects in the SAME
colour = a false merge. Writes annotated frames to outputs/loop/ and an mp4.

Run from repo root:
    python perception/demo_loop_viz.py --frames data/frames
"""
import argparse, glob, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.detector import Detector
from perception.track import Tracker
from perception.recognizer import Recognizer
from perception.embedder import Embedder
from perception.loop import PerceptionLoop
from memory.store import MemoryStore


def color_for(object_id: str):
    h = abs(hash(object_id))
    return (50 + h % 180, 50 + (h // 180) % 180, 50 + (h // 32400) % 180)  # BGR, not too dark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--thresh", type=float, default=0.80)
    ap.add_argument("--mem", default="memory_data/viz")
    ap.add_argument("--out", default="outputs/loop")
    ap.add_argument("--fps", type=float, default=5.0, help="output video fps")
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames, "frame_*.jpg")))
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")

    detector = Detector(backend="yoloe")
    detector.min_conf = args.conf
    tracker = Tracker()
    store = MemoryStore(local_path=args.mem)
    loop = PerceptionLoop(detector, tracker, Recognizer(Embedder(), store, threshold=args.thresh))
    detector.warmup(frames[0])

    os.makedirs(args.out, exist_ok=True)
    info = {}                                   # track_id -> (status, score) from decision events
    h0, w0 = cv2.imread(frames[0]).shape[:2]
    writer = cv2.VideoWriter(os.path.join("outputs", "loop_demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w0, h0))
    try:
        for i, path in enumerate(frames):
            for e in loop.step(path, i):
                info[e.track_id] = (e.result.status, e.result.score)

            bgr = cv2.imread(path)
            for t in tracker.tracks:
                if t.status != "confirmed":
                    continue
                oid = loop._bound.get(t.id)
                if oid is None:
                    continue
                status, score = info.get(t.id, ("?", 0.0))
                x1, y1, x2, y2 = t.box
                col = color_for(oid)
                tag = "NEW" if status == "novel" else "KNOWN"
                cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 3)
                cv2.putText(bgr, f"{oid[:4]} {tag} {score:.2f}", (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.imwrite(os.path.join(args.out, os.path.basename(path)), bgr)
            writer.write(bgr)
    finally:
        writer.release()
        store.close()

    print(f"frames -> {args.out}/   video -> outputs/loop_demo.mp4")
    print("Colour = memory object_id. DIFFERENT objects in the SAME colour = false merge.")


if __name__ == "__main__":
    main()