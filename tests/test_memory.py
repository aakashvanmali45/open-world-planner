import numpy as np
from memory.store import MemoryStore
import pytest

def rand_vec(seed, dim = 512):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)

def test_inmemory_roundtrip(tmp_path):
    store = MemoryStore(local_path = str(tmp_path / "local"))
    v =  rand_vec(1)
    oid = store.add(v, candidate_labels = ["mug"], canonical_label = "mug", status = "named")
    hits = store.match(v, top_k = 1)
    assert hits, "expected a match"
    hit_id, score, payload = hits[0]
    assert hit_id == oid
    assert score > 0.99
    assert payload.canonical_label == "mug"
    store.close()

def test_match_discriminates(tmp_path):
    store = MemoryStore(local_path=str(tmp_path / "local"))
    store.add(rand_vec(1), canonical_label="A")
    id_b = store.add(rand_vec(2), canonical_label="B")
    assert store.match(rand_vec(2), top_k=1)[0][0] == id_b
    store.close()

def test_local_persistence(tmp_path):
    path = str(tmp_path / "local")
    v = rand_vec(3)
    store = MemoryStore(local_path=path)
    oid = store.add(v, canonical_label="valve")
    store.close()
    store2 = MemoryStore(local_path=path)        # reopen: index rebuilt from disk
    hits = store2.match(v, top_k=1)
    assert hits and hits[0][0] == oid
    assert store2.get(oid).canonical_label == "valve"
    store2.close()

def _shared_up(url = "http://localhost:6333"):
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url=url).get_collections()
        return True
    except Exception:
        return False

@pytest.mark.skipif(not _shared_up(), reason = "shared Qdrant (docker) not running")
def test_sync_roundtrip(tmp_path):
    """Robot A learns an object, pushes to shared; a fresh client B pulls it
    and its in-memory index contains it. This is the Phase 0 acceptance gate."""

    import uuid
    from memory.store import MemoryStore

    URL = "http://localhost:6333"

    v = rand_vec(42)

    a = MemoryStore(local_path = str(tmp_path/"a"), shared_url = URL, robot_id = "A")
    oid = a.add(v, canonical_label="hydrant", status = "named")
    pushed, _ = a.sync()
    assert pushed >= 1
    a.close()

    b = MemoryStore(local_path= str(tmp_path/"b"), shared_url = URL, robot_id="B")
    assert b.match(v) == []
    _, pulled = b.sync()
    assert pulled >= 1
    hits = b.match(v, top_k = 1)
    assert hits and hits[0][2].canonical_label == "hydrant"
    b.close()