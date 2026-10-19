"""
perception/render_demo.py -- two-pass HUD render for the LinkedIn demo.

Pass 1: run detect->track->recognize + async VLM over the clip, collect per-frame
        track state and per-object thumbnails, then wait for all names.
Pass 2: re-read the clip and draw the split-screen (feed + memory grid + counters)
        with clean pacing, writing outputs/demo_hud.mp4.

Needs Ollama running (gemma). Run from repo root:
    python perception/render_demo.py --video data/raw/white_bg_test.mp4
"""
import argparse, os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.detector import Detector
from perception.track import Tracker
from perception.recognizer import Recognizer
from perception.embedder import Embedder
from perception.describer import Describer
from perception.vlm_worker import VLMWorker
from perception.loop import PerceptionLoop
from memory.store import MemoryStore

FEED_W = 960                      # feed scaled to this width
PANEL_W = 520
PANEL_BG = (28, 28, 28)
WHITE, GREEN, AMBER, BLUE = (240, 240, 240), (80, 220, 80), (60, 170, 255), (255, 170, 60)


def color_for(oid):
    h = abs(hash(oid))
    return (60 + h % 180, 60 + (h // 180) % 180, 60 + (h // 32400) % 180)


def badge(img, text, org, color, scale=0.7):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 10, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def frame_iter(path, proc_fps):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    stride = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30.0) / proc_fps))
    src = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if src % stride == 0:
            yield frame                       # BGR
        src += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw/white_bg_test.mp4")
    ap.add_argument("--proc-fps", type=float, default=3.0)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--thresh", type=float, default=0.80)
    ap.add_argument("--beat", type=int, default=5, help="frames to show 'describing...'")
    ap.add_argument("--mem", default="memory_data/demo")
    ap.add_argument("--out", default="outputs/demo_hud.mp4")
    args = ap.parse_args()

    store = MemoryStore(local_path=args.mem)
    detector = Detector(backend="yoloe"); detector.min_conf = args.conf
    worker = VLMWorker(Describer(), store)
    loop = PerceptionLoop(detector, Tracker(), Recognizer(Embedder(), store, threshold=args.thresh),
                          vlm_worker=worker)

    # ---------- PASS 1: process + collect ----------
    frame_tracks = {}       # step -> [(box, oid)]
    obj_meta = {}           # oid -> {first, status, score, thumb, order}
    warmed = False
    for step, bgr in enumerate(frame_iter(args.video, args.proc_fps)):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not warmed:
            detector.warmup(rgb); warmed = True
        for e in loop.step(rgb, step):
            r = e.result
            if r.object_id not in obj_meta:
                crop_bgr = cv2.cvtColor(r.prototype_crop, cv2.COLOR_RGB2BGR) \
                    if hasattr(r, "prototype_crop") else None
                obj_meta[r.object_id] = {"first": step, "status": r.status,
                                         "score": r.score, "order": len(obj_meta),
                                         "thumb": None}
        state = []
        for t in loop.tracker.tracks:
            if t.status == "confirmed" and t.id in loop._bound:
                oid = loop._bound[t.id]
                state.append((t.box, oid))
                if obj_meta.get(oid, {}).get("thumb") is None and t.best_crop is not None:
                    obj_meta[oid]["thumb"] = cv2.cvtColor(t.best_crop, cv2.COLOR_RGB2BGR)
        frame_tracks[step] = state
        print(f"pass1 frame {step:03d}: {len(state)} confirmed")
    worker.join(); worker.stop()
    print("names resolved, rendering...")

    # ---------- PASS 2: render ----------
    def name_of(oid):
        p = store.get(oid)
        return (p.canonical_label or "?") if p else "?"

    def count_of(oid):
        p = store.get(oid)
        return p.observation_count if p else 0

    first_frame = cv2.VideoCapture(args.video); _, f0 = first_frame.read(); first_frame.release()
    fh, fw = f0.shape[:2]
    feed_h = round(FEED_W * fh / fw)
    out_w, out_h = FEED_W + PANEL_W, feed_h
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.proc_fps, (out_w, out_h))
    sx, sy = FEED_W / fw, feed_h / fh

    for step, bgr in enumerate(frame_iter(args.video, args.proc_fps)):
        feed = cv2.resize(bgr, (FEED_W, feed_h))
        for box, oid in frame_tracks.get(step, []):
            x1, y1, x2, y2 = [int(v) for v in (box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy)]
            m = obj_meta[oid]; age = step - m["first"]; col = color_for(oid)
            cv2.rectangle(feed, (x1, y1), (x2, y2), col, 3)
            if m["status"] == "novel":
                txt = "NEW  describing..." if age < args.beat else f"LEARNED: {name_of(oid)}"
                bcol = AMBER if age < args.beat else GREEN
            else:
                txt = f"RECOGNISED: {name_of(oid)}  {m['score']:.2f}"; bcol = BLUE
            badge(feed, txt, (x1, max(y1 - 10, 18)), bcol)

        panel = np.full((feed_h, PANEL_W, 3), PANEL_BG, np.uint8)
        learned = [o for o, m in obj_meta.items() if m["first"] <= step and m["status"] == "novel"]
        vlm_calls = len(learned)
        cv2.putText(panel, f"Objects learned: {len(learned)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        cv2.putText(panel, f"VLM calls: {vlm_calls}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        cv2.putText(panel, f"Frame: {step}", (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        for oid in learned:
            m = obj_meta[oid]; i = m["order"]
            col_i, row_i = i % 2, i // 2
            x, y = 20 + col_i * 255, 130 + row_i * 100
            if m["thumb"] is not None:
                panel[y:y+80, x:x+80] = cv2.resize(m["thumb"], (80, 80))
            cv2.rectangle(panel, (x, y), (x + 80, y + 80), color_for(oid), 2)
            nm = "describing..." if (step - m["first"]) < args.beat else name_of(oid)
            cv2.putText(panel, nm[:14], (x + 88, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            cv2.putText(panel, f"seen x{count_of(oid)}", (x + 88, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        writer.write(np.hstack([feed, panel]))
    writer.release(); store.close()
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()