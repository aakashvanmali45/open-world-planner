"""
Groups per-frame detection, into persistent tracks and promotes a track to confirmed once its 
seen in N of M frames
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from perception.detector import Detection

@dataclass
class Track:
    id: int
    box:tuple
    score:float
    status: str = "candidate"
    hits: int = 0
    misses: int = 0
    last_frame: int = -1
    history: deque = field(default_factory=deque)
    best_crop: np.ndarray = None
    best_score: float = -1.0

def iou(box_a: tuple, box_b:tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter / (area_a + area_b - inter)

@dataclass
class Track:
    id: int
    box: tuple                        # most recent box (x1, y1, x2, y2)
    score: float                      # most recent detection score
    status: str = "candidate"         # "candidate" -> "confirmed"
    hits: int = 0                     # total frames matched
    misses: int = 0                   # consecutive frames missed (for max_age)
    last_frame: int = -1
    history: deque = field(default_factory=deque)   # recent True(hit)/False(miss); maxlen=M
    best_crop: np.ndarray = None      # crop from highest-score sighting (fed to embedder)
    best_score: float = -1.0


def iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-Union of two (x1, y1, x2, y2) boxes. 0.0 if no overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    # overlap rectangle
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    def __init__(self, iou_thresh: float = 0.3, n: int = 3, m: int = 5, max_age: int = 3):
        self.iou_thresh = iou_thresh
        self.n, self.m = n, m             # N-of-M gate
        self.max_age = max_age
        self._tracks: list[Track] = []
        self._next_id = 0

    # ---- bookkeeping -----------------------------------------------------
    def _new_track(self, det: Detection, frame_idx: int) -> Track:
        t = Track(id=self._next_id, box=det.box, score=det.score)
        t.history = deque(maxlen=self.m)
        self._next_id += 1
        self._record_hit(t, det, frame_idx)
        self._tracks.append(t)
        return t

    def _record_hit(self, t: Track, det: Detection, frame_idx: int) -> None:
        t.box, t.score = det.box, det.score
        t.hits += 1
        t.misses = 0
        t.last_frame = frame_idx
        t.history.append(True)
        if det.score > t.best_score:          # keep best view for the embedder
            t.best_score, t.best_crop = det.score, det.crop

    def _record_miss(self, t: Track) -> None:
        t.misses += 1
        t.history.append(False)

    # ---- the three core pieces ------------------------------------------
    def _is_confirmed(self, t: Track) -> bool:
        """N-of-M gate: at least N hits among the last M frames.
        t.history is a deque(maxlen=M) of booleans; sum() counts the True (hit) ones."""
        return sum(t.history) >= self.n

    def _associate(self, detections: list[Detection]):
        """Greedily match detections to existing tracks by IoU.
        For each track, take the best-overlapping detection not already taken,
        accepting the pair only if IoU >= iou_thresh. Each det/track used once."""
        matches = []
        used = set()                              # indices of detections already matched
        for track in self._tracks:
            best_iou, best_j = self.iou_thresh, -1
            for j, det in enumerate(detections):
                if j in used:
                    continue
                score = iou(track.box, det.box)
                if score >= best_iou:             # >= so a det exactly at threshold still matches
                    best_iou, best_j = score, j
            if best_j >= 0:
                matches.append((track, detections[best_j]))
                used.add(best_j)
        unmatched = [d for j, d in enumerate(detections) if j not in used]
        return matches, unmatched

    # ---- orchestration ---------------------------------------------------
    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        """Process one frame. Returns the currently CONFIRMED tracks."""
        matches, unmatched = self._associate(detections)

        matched_ids = set()
        for track, det in matches:
            self._record_hit(track, det, frame_idx)
            matched_ids.add(track.id)

        for t in self._tracks:                      # tracks that got no detection this frame
            if t.id not in matched_ids:
                self._record_miss(t)

        for det in unmatched:                       # brand-new tracks
            self._new_track(det, frame_idx)

        for t in self._tracks:                      # promote on the gate
            if t.status != "confirmed" and self._is_confirmed(t):
                t.status = "confirmed"

        self._tracks = [t for t in self._tracks if t.misses <= self.max_age]  # retire stale
        return [t for t in self._tracks if t.status == "confirmed"]

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    