from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Detection:
    box: tuple          # (x1, y1, x2, y2) in pixels
    score: float        # objectness/confidence — "is it a thing?", NOT identity
    label: str          # which prompt matched (a hint only; CLIP/VLM decide real identity)
    crop: np.ndarray    # HxWx3 RGB crop, ready to hand straight to Embedder.embed_image


class Detector:
    """Open-vocabulary detector (YOLO-World). Localizes objects from a free-text
    prompt list with no training, so genuinely unknown things still get boxed.
    Answers WHERE (objectness) only — never WHAT. Identity is CLIP's job."""

    def __init__(self, model_name: str = "yolov8s-world.pt", conf: float = 0.10):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(model_name)
        self.conf = conf                      # low default: catch unknowns, filter junk later

    def set_vocabulary(self, prompts: list[str]):
        """The 'open-vocab' part: hand it any words. Broad list => unknowns still localize."""
        self.model.set_classes(prompts)

    def detect(self, image, conf: float | None = None) -> list[Detection]:
        """image: path or an HxWx3 RGB array. Returns detections with RGB crops.
        The model is fed BGR (ultralytics' convention); crops are RGB (CLIP's)."""
        # Load/convert to BGR for the model
        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None:
                raise FileNotFoundError(f"could not read image: {image}")
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   # caller gives RGB by convention

        results = self.model.predict(bgr, conf=conf or self.conf, verbose=False)
        if not results:
            return []
        results = results[0]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)         # crops for CLIP
        boxes = results.boxes
        if boxes is None:                                   # guard: no detections
            return []

        dets = []
        names = results.names
        for b in boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            dets.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(b.conf[0]),
                label=names[int(b.cls[0])],
                crop=crop))
        return dets