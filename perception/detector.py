from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Detection:
    box: tuple          # (x1, y1, x2, y2) in pixels
    score: float        # match/objectness proxy — "is it a thing?", NOT identity
    label: str          # detector's guess (a hint only; CLIP/VLM decide real identity)
    crop: np.ndarray    # HxWx3 RGB crop, ready to hand straight to Embedder.embed_image


def load_lvis_names() -> list[str]:
    """The LVIS ~1,203 class names, from the ultralytics-bundled lvis.yaml.
    Used as the broad open-vocab list for the yoloworld backend."""
    import yaml
    from ultralytics.utils import ROOT          # ROOT still exported in 8.4.x
    path = ROOT / "cfg" / "datasets" / "lvis.yaml"
    if not path.exists():
        raise RuntimeError(
            f"LVIS names not found at {path}. Pass vocab=[...] to Detector instead.")
    with open(path, encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):               # {0: 'aerosol can', ...}
        return [names[i] for i in sorted(names)]
    return list(names)

def _containment(inner: tuple, outer:tuple)->float:
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)   
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))

    return inter/a_area


class Detector:
    """Open-vocabulary detector with two selectable backends.

        backend="yoloe"      YOLOE prompt-free, built-in 4,585-name vocabulary.
                             The end-to-end default. No vocab to manage. Labels
                             it prints are throwaway — identity is CLIP's job.
        backend="yoloworld"  YOLO-World + LVIS ~1,200 vocab. Kept for comparison;
                             call set_vocabulary() to narrow it.

    Answers WHERE (a box + a match/objectness score) only — never WHAT.
    Both backends return the same Detection objects (box, score, label, RGB crop),
    so nothing downstream (embedder, tracker, loop) knows which detector ran.
    """

    def __init__(
        self,
        backend: str = "yoloe",
        conf: float = 0.05,               # cast wide; stability gate + embedding do the filtering
        min_conf: float = 0.5,
        iou: float = 0.5,                 # class-agnostic NMS: drop overlaps above this
        imgsz: int = 640,                 # lower to 512/448 if CPU latency is high
        containment_thresh: float = 0.9,
        big_box_area_frac: float = 0.70,  # a box bigger than this share of the frame...
        big_box_min_conf: float = 0.75,   # ...AND below this confidence is dropped as junk
        yoloe_weights: str = "yoloe-11s-seg-pf.pt",
        yoloworld_weights: str = "yolov8s-world.pt",
        vocab: list[str] | None = None,   # yoloworld only: override the LVIS vocab
    ):
        if backend not in ("yoloe", "yoloworld"):
            raise ValueError(f"unknown backend {backend!r}; use 'yoloe' or 'yoloworld'")
        self.backend = backend
        self.conf, self.min_conf, self.iou, self.imgsz = conf, min_conf, iou, imgsz
        self.containment_thresh = containment_thresh
        self.big_box_area_frac = big_box_area_frac
        self.big_box_min_conf = big_box_min_conf

        if backend == "yoloe":
            from ultralytics import YOLOE
            self.model = YOLOE(yoloe_weights)          # prompt-free: no set_classes()
        else:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(yoloworld_weights)
            self.set_vocabulary(vocab if vocab is not None else load_lvis_names())

    def set_vocabulary(self, prompts: list[str]):
        """Set the open-vocab prompt list. yoloworld only — YOLOE-pf resolves
        classes from its own built-in vocabulary and rejects external prompts."""
        if self.backend != "yoloworld":
            raise RuntimeError(
                "set_vocabulary() applies only to the 'yoloworld' backend; "
                "the yoloe backend is prompt-free and uses its built-in vocab.")
        self.model.set_classes(prompts)

    def warmup(self, image) -> None:
        """One throwaway pass so later detect() timings exclude load/graph-build."""
        self.detect(image)

    def detect(self, image, conf: float | None = None) -> list[Detection]:

        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None:
                raise FileNotFoundError(f"Couldnt read image: {image}")
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        results = self.model.predict(
            bgr, conf=conf or self.conf, iou = self.iou,
            agnostic_nms = True, imgsz=self.imgsz, verbose=False
        )

        if not results:
            return []

        results = results[0]
        boxes = results.boxes
        if boxes is None:
            return []

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = results.orig_shape
        frame_area = float(h * w)
        names = results.names

        raw = []
        for b in boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            score = float(b.conf[0])
            box=(x1, y1, x2, y2)
            if score < self.min_conf:
                continue
            area_frac = (x2 - x1)*(y2-y1) / frame_area
            if area_frac > self.big_box_area_frac and score < self.big_box_min_conf:
                continue

            raw.append((score, box, int(b.cls[0])))

        raw.sort(key=lambda t: t[0], reverse = True)
        kept = []

        for score, box, cls in raw:
            if any(_containment(box, k[1]) > self.containment_thresh for k in kept):
                continue
            kept.append((score, box, cls))

        dets = []

        for score, (x1, y1, x2, y2), cls in kept:
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            dets.append(Detection(box=(x1,y1,x2,y2),
                                  score = score,
                                  label = names[int(cls)],
                                  crop = crop))

        return dets
