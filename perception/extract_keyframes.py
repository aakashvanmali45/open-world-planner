import argparse
import csv
import sys
from pathlib import Path

import cv2

def extract(video_path:Path, out_dir: Path, target_fps: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if src_fps <= 0:
        sys.exit("Could not read source fps -- is this a valid video file")

    stride = max(1, round(src_fps/target_fps))
    out_dir.mkdir(parents=True, exist_ok = True)

    kept, idx = 0, 0

    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keyframe_id", "filename", "src_frame_idx", "timestamp_s"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                name = f"frame_{kept:05d}.jpg"
                cv2.imwrite(str(out_dir/name), frame)
                writer.writerow([kept, name, idx, round(idx/src_fps, 3)])
                kept += 1
            idx += 1

    cap.release()
    print(f"Source    : {src_fps:.1f} fps, {total} frames")
    print(f"Sampling    : every {stride} frames -> ~{target_fps} keyframes/sec")
    print(f"Wrote    : {kept} keyframes + manifest.csv -> {out_dir}/")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract keyframes at a fixed low fps")
    p.add_argument("video", type= Path, help="path to the walkthrough clip")
    p.add_argument("--out", type=Path, default = Path("data/frames"), help="output directory")
    p.add_argument("--fps", type=float, default=2.0, help="target keyframes per second")

    args = p.parse_args()
    extract(args.video, args.out, args.fps)