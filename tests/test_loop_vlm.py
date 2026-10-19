import numpy as np
from perception.detector import Detection
from perception.track import Tracker
from perception.recognizer import Recognizer
from perception.loop import PerceptionLoop
from perception.vlm_worker import VLMWorker
from memory.store import MemoryStore


class FakeEmbedder:
    def embed_image(self, v):
        v = np.asarray(v, np.float32); return v / np.linalg.norm(v)


class _D:
    candidate_labels = ["bottle"]; category = "container"
    color = "green"; material = "plastic"; features = "has a cap"


class FakeDescriber:
    def describe(self, image, known_objects=None, location=""):
        return _D()


class ScriptedDetector:
    def __init__(self, s): self.s = s
    def detect(self, image): return self.s[image]
    def warmup(self, image): pass


def _det(box, vec):
    return Detection(box=box, score=0.9, label="x", crop=np.asarray(vec, np.float32))


def test_loop_enqueues_and_names(tmp_path):
    box, bottle = (10, 10, 50, 50), [1, 0, 0, 0]
    script = {0: [_det(box, bottle)], 1: [_det(box, bottle)], 2: [_det(box, bottle)]}
    store = MemoryStore(dim=4, local_path=str(tmp_path / "mem"))
    worker = VLMWorker(FakeDescriber(), store)
    loop = PerceptionLoop(ScriptedDetector(script), Tracker(n=3, m=5),
                          Recognizer(FakeEmbedder(), store, threshold=0.8), vlm_worker=worker)

    events = []
    for i in range(3):
        events += loop.step(i, i)
    worker.join(); worker.stop()

    assert len(events) == 1 and events[0].result.status == "novel"
    proto = store.get(events[0].result.object_id)
    assert proto.status == "named" and proto.canonical_label == "bottle"
    store.close()