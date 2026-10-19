import numpy as np
from memory.store import MemoryStore
from perception.vlm_worker import VLMWorker


class _FakeDesc:
    candidate_labels = ["mug"]
    category = "container"
    color = "white"
    material = "ceramic"
    features = "has a handle"


class FakeDescriber:
    def describe(self, image, known_objects=None, location=""):
        return _FakeDesc()


def test_worker_names_prototype(tmp_path):
    store = MemoryStore(dim=4, local_path=str(tmp_path / "mem"))
    oid = store.add(np.array([1, 0, 0, 0], np.float32), status="provisional")

    w = VLMWorker(FakeDescriber(), store)
    w.enqueue(oid, np.zeros((4, 4, 3), np.uint8))
    w.join()                                  # wait for the describe to finish
    w.stop()

    proto = store.get(oid)
    assert proto.status == "named"
    assert proto.canonical_label == "mug"
    assert proto.candidate_labels == ["mug"]
    store.close()