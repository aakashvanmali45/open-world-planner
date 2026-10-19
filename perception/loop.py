"""
each confirmed tracked detection gets a name
"""

from __future__ import annotations
from dataclasses import dataclass
from perception.detector import Detector
from perception.track import Tracker
from perception.recognizer import Recognizer, MatchResult

@dataclass
class LoopEvent: 
    frame_idx: int
    track_id: int
    result: MatchResult # status (known/novel), object_id, score, label

class PerceptionLoop:
    def __init__(self, detector: Detector, tracker: Tracker, recognizer: Recognizer, vlm_worker=None):
        self.detector = detector
        self.tracker = tracker
        self.recognizer = recognizer
        self.vlm_worker = vlm_worker    
        self._bound: dict[int, str] = {} #track_id -> object_id

    def step(self, image, frame_idx: int) -> list[LoopEvent]:
        "Process one frame"

        dets = self.detector.detect(image)
        confirmed = self.tracker.update(dets, frame_idx)

        events = []

        for trk in confirmed:
            if trk.id in self._bound or trk.best_crop is None:
                continue # already detected or no crops yet
            result = self.recognizer.recognize(trk.best_crop)
            self._bound[trk.id] = result.object_id
            if result.status == "novel" and self.vlm_worker is not None:
                self.vlm_worker.enqueue(result.object_id, trk.best_crop) 
            events.append(LoopEvent(frame_idx, trk.id, result))

        return events
