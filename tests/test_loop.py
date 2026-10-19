import numpy as np
from perception.detector import Detection
from perception.track import Tracker
from perception.recognizer import Recognizer
from perception.loop import PerceptionLoop
from memory.store import MemoryStore


class FakeEmbedder:
    def embed_image(self, v):
        v = np.asarray(v, np.float32); return v / np.linalg.norm(v)


class ScriptedDetector:
    def __init__(self, per_frame): self.per_frame = per_frame   # {frame_idx: [Detection]}
    def detect(self, image): return self.per_frame[image]       # here image == frame idx
    def warmup(self, image): pass


def _det(box, crop_vec):
    return Detection(box=box, score=0.9, label="x", crop=np.asarray(crop_vec, np.float32))


def test_loop_learns_then_recognizes(tmp_path):
    box, bottle = (10, 10, 50, 50), [1, 0, 0, 0]
    # present 0-2 (confirms, NOVEL), gone 3-7 (track retires), returns 8-10 (new track, KNOWN)
    script = {0: [_det(box, bottle)], 1: [_det(box, bottle)], 2: [_det(box, bottle)],
              3: [], 4: [], 5: [], 6: [], 7: [],
              8: [_det(box, bottle)], 9: [_det(box, bottle)], 10: [_det(box, bottle)]}
    store = MemoryStore(dim=4, local_path=str(tmp_path / "mem"))
    loop = PerceptionLoop(ScriptedDetector(script),
                          Tracker(n=3, m=5, max_age=3),
                          Recognizer(FakeEmbedder(), store, threshold=0.8))

    events = []
    for i in range(11):
        events += loop.step(i, i)

    assert len(events) == 2
    assert events[0].result.status == "novel"            # first appearance
    assert events[1].result.status == "known"            # came back
    assert events[1].result.score > 0.9
    assert events[1].result.prototype.observation_count == 2
    store.close()