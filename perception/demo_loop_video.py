"""
perception/demo_loop_video.py -- run the loop directly on a video file.
Samples the video to ~proc_fps, feeds frames to the loop, draws decisions
(colour = memory object_id; same colour on different objects = false merge),
writes an annotated mp4, prints a summary.

Run from repo root:
    python perception/demo_loop_video.py --video data/raw/walkthrough_rotated.mp4
"""
import argparse, os, sys, time
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
    return (50 + h % 180, 50 + (h // 180) % 180, 50 + (h // 32400) % 180)   # BGR, not too dark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--proc-fps", type=float, default=3.0, help="frames/sec to actually process")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--thresh", type=float, default=0.80)
    ap.add_argument("--mem", default="memory_data/video")
    ap.add_argument("--out", default="outputs/loop_video.mp4")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / args.proc_fps))          # process every Nth source frame
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = Detector(backend="yoloe")
    detector.min_conf = args.conf
    tracker = Tracker()
    store = MemoryStore(local_path=args.mem)
    loop = PerceptionLoop(detector, tracker, Recognizer(Embedder(), store, threshold=args.thresh))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.proc_fps, (w, h))

    info = {}
    warmed = False
    src_idx = step_idx = 0
    t_total = n_novel = n_known = 0.0
    try:
        while True:
            ok, frame_bgr = cap.read()                       # OpenCV gives BGR
            if not ok:
                break
            if src_idx % stride == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # detector expects RGB arrays
                if not warmed:
                    detector.warmup(rgb); warmed = True
                t0 = time.perf_counter()
                events = loop.step(rgb, step_idx)
                t_total += (time.perf_counter() - t0) * 1000
                for e in events:
                    info[e.track_id] = (e.result.status, e.result.score)
                    if e.result.status == "novel": n_novel += 1
                    else: n_known += 1

                for t in tracker.tracks:                     # draw on the ORIGINAL bgr frame
                    if t.status != "confirmed":
                        continue
                    oid = loop._bound.get(t.id)
                    if oid is None:
                        continue
                    status, score = info.get(t.id, ("?", 0.0))
                    x1, y1, x2, y2 = t.box
                    col = color_for(oid)
                    tag = "NEW" if status == "novel" else "KNOWN"
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), col, 3)
                    cv2.putText(frame_bgr, f"{oid[:4]} {tag} {score:.2f}", (x1, max(y1 - 8, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                writer.write(frame_bgr)
                step_idx += 1
            src_idx += 1
    finally:
        cap.release(); writer.release(); store.close()

    if step_idx:
        print(f"processed {step_idx} frames (every {stride} of {src_fps:.0f}fps) | "
              f"avg {t_total/step_idx:.0f} ms/frame | {1000*step_idx/t_total:.1f} fps")
    print(f"learned (novel): {int(n_novel)} | known: {int(n_known)}")
    print(f"video -> {args.out}  (colour = object_id; same colour on different objects = false merge)")


if __name__ == "__main__":
    main()