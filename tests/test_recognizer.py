import numpy as np
import pytest
from memory.store import MemoryStore
from perception.recognizer import Recognizer


class FakeEmbedder:
    """Returns the 'crop' as a unit vector -- lets tests control embeddings
    without downloading CLIP weights. In tests we pass vectors as the 'crop'."""
    def embed_image(self, v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture
def rec(tmp_path):
    store = MemoryStore(dim=4, local_path=str(tmp_path / "mem"))
    return Recognizer(FakeEmbedder(), store, threshold=0.80)


def test_first_object_is_novel(rec):
    r = rec.recognize(np.array([1, 0, 0, 0], np.float32))
    assert r.status == "novel"
    assert r.score == 0.0                       # memory was empty
    assert r.prototype.status == "provisional"
    assert r.label is None


def test_same_object_recognized(rec):
    rec.recognize(np.array([1, 0, 0, 0], np.float32))         # learn it
    r = rec.recognize(np.array([1, 0, 0, 0], np.float32))     # see it again
    assert r.status == "known"
    assert r.score == pytest.approx(1.0, abs=1e-5)
    assert r.prototype.observation_count == 2                 # bumped


def test_different_object_is_novel(rec):
    rec.recognize(np.array([1, 0, 0, 0], np.float32))
    r = rec.recognize(np.array([0, 1, 0, 0], np.float32))     # orthogonal -> sim 0
    assert r.status == "novel"
    assert r.score == pytest.approx(0.0, abs=1e-5)


def test_near_object_recognized(rec):
    rec.recognize(np.array([1, 0, 0, 0], np.float32))
    r = rec.recognize(np.array([0.95, 0.05, 0, 0], np.float32))  # sim ~0.998 >= 0.8
    assert r.status == "known"

def test_borderline_match_adds_view(rec):
    rec.recognize(np.array([1, 0, 0, 0], np.float32))              # learn it
    oid = rec.store.match(np.array([1, 0, 0, 0], np.float32))[0][0]
    assert rec.store.view_count(oid) == 1
    # sim == 0.85: clearly known, but borderline -> a new view is stored
    r = rec.recognize(np.array([0.85, 0.5268, 0, 0], np.float32))
    assert r.status == "known"
    assert r.object_id == oid
    assert rec.store.view_count(oid) == 2

def test_strong_match_adds_no_view(rec):
    rec.recognize(np.array([1, 0, 0, 0], np.float32))
    oid = rec.store.match(np.array([1, 0, 0, 0], np.float32))[0][0]
    r = rec.recognize(np.array([0.99, 0.01, 0, 0], np.float32))    # sim ~1.0 >= ceiling
    assert r.status == "known"
    assert rec.store.view_count(oid) == 1                          # nothing added
    assert r.prototype.observation_count == 2                      # count still bumped