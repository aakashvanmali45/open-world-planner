"""
perception/eval_recognition.py -- measure recognition separation + calibrate T.

Two modes:
  --scan          : find on-screen time-segments, write outputs/labels_template.csv
                    (fill the 'instance' column; same name = same object's repeats)
  (default)       : read --labels CSV, enroll each instance from its FIRST segment,
                    score later views -> genuine vs impostor distributions,
                    suggested threshold, error table, histogram PNG, scores CSV.

Run from repo root:
    python perception/eval_recognition.py --video data/raw/white_bg_test.mp4 --scan
    python perception/eval_recognition.py --video data/raw/white_bg_test.mp4 --labels outputs/labels.csv
"""
import argparse, csv, os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.detector import Detector
from perception.track import Tracker
from perception.embedder import Embedder


def frames(path, proc_fps):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / proc_fps))
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield i / fps, bgr          # (timestamp_sec, BGR frame)
        i += 1
    cap.release()


def collect(video, proc_fps, conf):
    """Run detect->track; return list of (t_sec, embedding) for every confirmed-track crop."""
    det = Detector(backend="yoloe"); det.min_conf = conf
    trk = Tracker()
    emb = Embedder(pad_color=(255, 255, 255))      # white pad matches the white-bg scene
    out, warmed = [], False
    for t, bgr in frames(video, proc_fps):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not warmed:
            det.warmup(rgb); warmed = True
        dets = det.detect(rgb)
        trk.update(dets, int(t * proc_fps))
        for tk in trk.tracks:
            if tk.status == "confirmed" and tk.best_crop is not None:
                out.append((t, emb.embed_image(tk.best_crop)))
    return out


def scan(video, proc_fps, conf, out_dir, gap=1.0):
    """Find contiguous on-screen segments; write a labels template to fill in."""
    times = [t for t, _ in collect(video, proc_fps, conf)]
    segs, start, prev = [], None, None
    for t in times:
        if start is None:
            start = prev = t
        elif t - prev <= gap:
            prev = t
        else:
            segs.append((start, prev)); start = prev = t
    if start is not None:
        segs.append((start, prev))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "labels_template.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start_sec", "end_sec", "instance"])
        for s, e in segs:
            w.writerow([f"{s:.2f}", f"{e:.2f}", ""])
    print(f"found {len(segs)} segments -> {path}")
    for i, (s, e) in enumerate(segs):
        print(f"  seg {i}: {s:5.1f}s - {e:5.1f}s")
    print("\nFill the 'instance' column (same name for an object's repeat appearances), "
          "save as outputs/labels.csv, then re-run without --scan.")


def evaluate(video, labels_csv, proc_fps, conf, out_dir):
    with open(labels_csv) as f:
        rows = [(float(r["start_sec"]), float(r["end_sec"]), r["instance"].strip())
                for r in csv.DictReader(f) if r["instance"].strip()]
    if not rows:
        raise SystemExit("no labeled rows in the CSV")

    def label_at(t):
        for s, e, name in rows:
            if s <= t <= e:
                return name, s
        return None, None

    # gather embeddings tagged with (instance, segment_start)
    data = []   # (instance, seg_start, emb)
    for t, e in collect(video, proc_fps, conf):
        name, seg_start = label_at(t)
        if name is not None:
            data.append((name, seg_start, e))
    if not data:
        raise SystemExit("no crops fell inside labeled segments")

    instances = sorted(set(d[0] for d in data))
    # enroll: prototype = mean unit vector of each instance's EARLIEST segment
    prototypes = {}
    for inst in instances:
        starts = [d[1] for d in data if d[0] == inst]
        first = min(starts)
        vecs = [d[2] for d in data if d[0] == inst and d[1] == first]
        p = np.mean(vecs, axis=0); prototypes[inst] = p / (np.linalg.norm(p) + 1e-9)
        n_q = sum(1 for d in data if d[0] == inst and d[1] != first)
        print(f"  {inst}: enrolled from {len(vecs)} crops, {n_q} query views")

    # queries = all later-segment views; score genuine vs impostor + nearest prototype
    proto_mat = np.stack([prototypes[i] for i in instances])     # (I, dim)
    genuine, impostor, per_query = [], [], []
    for inst, seg_start, e in data:
        if seg_start == min(d[1] for d in data if d[0] == inst):
            continue                                              # skip enroll views
        e = e / (np.linalg.norm(e) + 1e-9)
        sims = proto_mat @ e                                      # vs every prototype
        own = sims[instances.index(inst)]
        others = np.delete(sims, instances.index(inst))
        genuine.append(float(own))
        impostor.append(float(others.max()) if len(others) else 0.0)
        nn = int(np.argmax(sims))
        per_query.append((inst, instances[nn], float(sims[nn]), float(own)))

    genuine, impostor = np.array(genuine), np.array(impostor)
    if len(genuine) == 0:
        raise SystemExit("no query views (need instances that appear more than once)")

    # sweep T: false-novel = query of known inst scored below T; false-match = nearest is wrong inst, >=T
    Ts = np.arange(0.50, 0.99, 0.01)
    fn = np.array([(np.array([n[2] for n in per_query]) < T).mean() for T in Ts])      # nearest sim < T
    fm = np.array([np.mean([(n[1] != n[0] and n[2] >= T) for n in per_query]) for T in Ts])
    eer_i = int(np.argmin(np.abs(fn - fm)))
    T_star = float(Ts[eer_i])

    os.makedirs(out_dir, exist_ok=True)
    # histogram
    plt.figure(figsize=(9, 5))
    plt.hist(genuine, bins=30, alpha=.6, label=f"genuine (same instance)  n={len(genuine)}", color="#16a34a")
    plt.hist(impostor, bins=30, alpha=.6, label=f"impostor (diff instance)  n={len(impostor)}", color="#dc2626")
    plt.axvline(T_star, color="#111827", ls="--", label=f"suggested T = {T_star:.2f} (EER)")
    plt.xlabel("cosine similarity"); plt.ylabel("count"); plt.legend()
    plt.title("Recognition separation — genuine vs impostor")
    hist_path = os.path.join(out_dir, "eval_recognition_hist.png")
    plt.tight_layout(); plt.savefig(hist_path, dpi=130); plt.close()

    with open(os.path.join(out_dir, "eval_scores.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true_instance", "nearest_instance", "nearest_sim", "own_sim"])
        w.writerows(per_query)

    print("\n--- separation ---")
    print(f"genuine  : mean {genuine.mean():.3f}  min {genuine.min():.3f}")
    print(f"impostor : mean {impostor.mean():.3f}  max {impostor.max():.3f}")
    print(f"gap (genuine.min - impostor.max): {genuine.min() - impostor.max():+.3f}   (positive = clean separation)")
    print(f"\nsuggested T = {T_star:.2f}  (equal-error point)")
    print(f"{'T':>5} {'false-novel%':>13} {'false-match%':>13}")
    for T, a, b in zip(Ts, fn, fm):
        if round(T, 2) in (0.75, 0.80, 0.85, round(T_star, 2), 0.90):
            print(f"{T:5.2f} {100*a:13.1f} {100*b:13.1f}")
    print(f"\nhistogram -> {hist_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--labels", default="outputs/labels.csv")
    ap.add_argument("--proc-fps", type=float, default=3.0)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    if args.scan:
        scan(args.video, args.proc_fps, args.conf, args.out_dir)
    else:
        evaluate(args.video, args.labels, args.proc_fps, args.conf, args.out_dir)


if __name__ == "__main__":
    main()