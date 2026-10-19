from __future__ import annotations
import numpy as np


# The mask-selection and application logic is kept as pure functions so it can
# be tested without loading FastSAM. Only _infer() touches ultralytics.

def _touches_all_edges(mask: np.ndarray) -> bool:
    """A blob spanning all four borders is almost always background, not the object."""
    return bool(mask[0, :].any() and mask[-1, :].any()
                and mask[:, 0].any() and mask[:, -1].any())

def _select_mask(masks, hw, min_area_frac: float, max_area_frac: float):
    """Pick the largest plausible object mask, or None to signal 'use the plain crop'.
    masks: (N, H, W) float or bool. Returns a bool (H, W) mask or None."""
    if masks is None or len(masks) == 0:
        return None
    h, w = hw
    total = float(h * w)
    best, best_area = None, 0.0
    for m in masks:
        mb = np.asarray(m) > 0.5
        area = float(mb.sum())
        frac = area / total
        if frac < min_area_frac or frac > max_area_frac:   # too small / basically the whole box
            continue
        if _touches_all_edges(mb):                         # background blob
            continue
        if area > best_area:
            best, best_area = mb, area
    return best

def _apply_mask(crop: np.ndarray, mask: np.ndarray, pad_color) -> np.ndarray:
    """Set background (mask == 0) to pad_color so the embedding sees the object,
    not the surface behind it. Keeps the crop's size and scale."""
    h, w = crop.shape[:2]
    if mask.shape != (h, w):
        import cv2
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    out = crop.copy()
    out[~mask] = pad_color
    return np.ascontiguousarray(out)


class Masker:
    """FastSAM-in-box background removal. Segments the object inside an already-
    cropped box and neutralizes the background before embedding. Fails SAFE:
    on no/too-small/too-large/edge-spanning mask, or any inference error, it
    returns the original crop unchanged -- a bad mask is never worse than none.

    pad_color defaults to 114 to match the embedder's letterbox padding, so a
    neutralized background blends with the pad instead of adding a hard edge."""

    def __init__(self, model_path: str = "weights/FastSAM-s.pt", device: str = "cpu",
                 imgsz: int = 512, conf: float = 0.4, iou: float = 0.9,
                 min_side: int = 16, min_area_frac: float = 0.05,
                 max_area_frac: float = 0.95, pad_color=(114, 114, 114)):
        from ultralytics import FastSAM
        self._model = FastSAM(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.min_side = min_side            # skip FastSAM on tiny crops (unreliable + its weak spot)
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.pad_color = pad_color

    def _infer(self, crop) -> np.ndarray:
        res = self._model(crop, device=self.device, retina_masks=True,
                          imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)
        r = res[0]
        if r.masks is None:
            return np.zeros((0, crop.shape[0], crop.shape[1]), dtype=bool)
        return r.masks.data.cpu().numpy()

    def apply(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if min(h, w) < self.min_side:            # too small to segment reliably
            return crop
        try:
            masks = self._infer(crop)
        except Exception:
            return crop                          # inference failed -> plain crop
        m = _select_mask(masks, (h, w), self.min_area_frac, self.max_area_frac)
        if m is None:
            return crop                          # nothing usable -> plain crop
        return _apply_mask(crop, m, self.pad_color)