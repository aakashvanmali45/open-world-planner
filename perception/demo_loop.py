"""
perception/demo_loop.py -- run the full loop over a frame folder, TIMED.
Prints each NEW/KNOWN decision with its similarity score, and avg ms/frame + fps
so we can see whether it's real-time on this machine.

Run from repo root:
    python perception/demo_loop.py --frames data/frames
Delete the --mem folder to start from an empty memory.
"""
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from perception.detector import Detector
from perception.track import Tracker
from perception.recognizer import Recognizer
from perception.embedder import Embedder
from perception.loop import PerceptionLoop
from memory.store import MemoryStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--backend", default="yoloe", choices=["yoloe", "yoloworld"])
    ap.add_argument("--conf", type=float, default=0.1, help="detector min conf (low = catch marginal)")
    ap.add_argument("--thresh", type=float, default=0.80, help="novelty threshold T")
    ap.add_argument("--mem", default="memory_data/demo", help="memory dir (delete to reset)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames, "frame_*.jpg")))
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")

    detector = Detector(backend=args.backend)
    detector.min_conf = args.conf
    if args.backend == "yoloworld":
        detector.set_vocabulary(["bottle", "mug", "cup", "book"])
    tracker = Tracker(n=args.n, m=args.m)
    embedder = Embedder()                                   # frozen CLIP (cached from Phase 0)
    store = MemoryStore(local_path=args.mem)
    loop = PerceptionLoop(detector, tracker, Recognizer(embedder, store, threshold=args.thresh))

    detector.warmup(frames[0])

    t_total = n_novel = n_known = 0
    try:
        for i, path in enumerate(frames):
            t0 = time.perf_counter()
            events = loop.step(path, i)
            t_total += (time.perf_counter() - t0) * 1000
            for e in events:
                r = e.result
                if r.status == "novel":
                    n_novel += 1
                    print(f"frame {i:03d}  T{e.track_id:<3d}  NEW object   (best match {r.score:.2f} < {args.thresh})")
                else:
                    n_known += 1
                    print(f"frame {i:03d}  T{e.track_id:<3d}  KNOWN: {r.label or '(unnamed)'}  "
                          f"(match {r.score:.2f}, seen x{r.prototype.observation_count})")
    finally:
        store.close()

    n = len(frames)
    print(f"\n{n} frames | avg {t_total/n:.0f} ms/frame | {1000*n/t_total:.1f} fps")
    print(f"learned (novel): {n_novel}   |   re-recognitions (known): {n_known}")
    print("note: VLM naming not wired yet -> novel objects are stored UNNAMED. That's build #2.")

if __name__ == "__main__":
    main()