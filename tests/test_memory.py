import numpy as np
from memory.store import MemoryStore

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