from __future__ import annotations
import time, uuid, threading
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

DIM_DEFAULT = 512

@dataclass
class Prototype:
    """The payload stored per object. Vectors are kept separate from this metadata."""
    object_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "provisional"
    candidate_labels: list = field(default_factory=list)
    canonical_label: Optional[str] = None
    category: Optional[str] = None
    attributes: dict = field(default_factory=dict)
    world_xyz: Optional[list] = None
    observation_count: int = 1
    robot_id: str = "robot_0"
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class MemoryStore:
    """
    Three-tier memory. Tiers 1 and 2 are built here; tier 3 (shared sync) is
    stubbed until Docker is available.
      - Tier 1  in-memory NumPy index  -> match() ALWAYS hits this. No network.
      - Tier 2  embedded Qdrant (local dir, no Docker) -> survives restart.
      - Tier 3  shared Qdrant server    -> background sync target (deferred).

    Multi-view: the Tier-1 index is FLAT -- one row per stored VIEW. Several rows
    may share one object_id (several embeddings of the same object from different
    angles). match() is nearest-of-ANY-view, de-duplicated to distinct objects.
    """

    def __init__(self, dim: int = DIM_DEFAULT, local_path: str = "./memory_data/local",
                 robot_id: str = "robot_0", shared_url: Optional[str] = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.dim = dim
        self.robot_id = robot_id
        self._lock = threading.Lock()

        # Tier 1: flat in-memory index. Row i belongs to object _view_oids[i],
        # and is persisted as Qdrant point _view_pids[i].
        self._vecs = np.zeros((0, dim), dtype=np.float32)   # (M, dim), unit rows
        self._view_oids: list[str] = []                     # object_id per row
        self._view_pids: list[str] = []                     # qdrant point id per row
        self._payloads: dict[str, Prototype] = {}           # one Prototype per object_id

        # Tier 2: embedded Qdrant
        self._local = QdrantClient(path=local_path)
        self._collection = "prototypes"
        if not self._local.collection_exists(self._collection):
            self._local.create_collection(
                self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._load_from_local()

        # Tier 3: shared server (deferred)
        self._shared = QdrantClient(url=shared_url) if shared_url else None

    # ---------- helpers ----------
    @staticmethod
    def _norm(v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _rows_for(self, object_id: str) -> list[int]:
        return [i for i, o in enumerate(self._view_oids) if o == object_id]

    def _append_row(self, object_id: str, vector, pid: Optional[str] = None) -> str:
        pid = pid or str(uuid.uuid4())
        self._vecs = np.vstack([self._vecs, self._norm(vector)[None, :]])
        self._view_oids.append(object_id)
        self._view_pids.append(pid)
        return pid

    def _drop_row(self, idx: int):
        self._vecs = np.delete(self._vecs, idx, axis=0)
        del self._view_oids[idx]
        del self._view_pids[idx]

    # ---------- tier-2 persistence ----------
    def _upsert_point(self, pid: str, vector, proto: Prototype):
        from qdrant_client.models import PointStruct
        self._local.upsert(self._collection, points=[PointStruct(
            id=pid, vector=self._norm(vector).tolist(), payload=asdict(proto))])

    def _persist_object(self, object_id: str):
        """Rewrite every point of this object with the current payload (so a label
        backfilled by the VLM lands on all of an object's view-points, not just one)."""
        proto = self._payloads[object_id]
        for i in self._rows_for(object_id):
            self._upsert_point(self._view_pids[i], self._vecs[i], proto)

    def _delete_point(self, pid: str):
        from qdrant_client.models import PointIdsList
        self._local.delete(self._collection, points_selector=PointIdsList(points=[pid]))

    def _load_from_local(self):
        points, _ = self._local.scroll(self._collection, limit=100000, with_vectors=True)
        for p in points:
            oid = p.payload["object_id"]
            if oid not in self._payloads:
                self._payloads[oid] = Prototype(**p.payload)
            self._append_row(oid, np.asarray(p.vector, dtype=np.float32), pid=p.id)

    def _ensure_shared_collection(self):
        from qdrant_client.models import Distance, VectorParams
        if not self._shared.collection_exists(self._collection):
            self._shared.create_collection(
                self._collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def _evict_redundant(self, rows: list[int]):
        """Drop the view most similar to its siblings (least diverse) so a capped
        set keeps its spread of angles rather than filling up with near-duplicates."""
        if len(rows) < 2:
            victim_idx = rows[0]
        else:
            sub = self._vecs[rows]            # (k, dim), unit rows
            sims = sub @ sub.T                # (k, k)
            np.fill_diagonal(sims, np.nan)
            mean_sim = np.nanmean(sims, axis=1)
            victim_idx = rows[int(np.nanargmax(mean_sim))]
        pid = self._view_pids[victim_idx]
        self._drop_row(victim_idx)
        self._delete_point(pid)

    # ---------- public API ----------
    def add(self, vector, **kwargs) -> str:
        """Create a new object from its first view. Returns the new object_id."""
        proto = Prototype(robot_id=self.robot_id, **kwargs)
        with self._lock:
            self._payloads[proto.object_id] = proto
            pid = self._append_row(proto.object_id, vector)
            self._upsert_point(pid, vector, proto)
        return proto.object_id

    def add_view(self, object_id: str, vector, max_views: int = 8) -> bool:
        """Append another view embedding to an existing object (multi-view prototype).
        Caps the set at max_views, evicting the most redundant existing view first."""
        with self._lock:
            proto = self._payloads.get(object_id)
            if proto is None:
                return False
            rows = self._rows_for(object_id)
            if len(rows) >= max_views:
                self._evict_redundant(rows)
            pid = self._append_row(object_id, vector)
            proto.observation_count += 1
            proto.last_seen = time.time()
            self._upsert_point(pid, vector, proto)
        return True

    def match(self, vector, top_k: int = 1):
        """Nearest-of-ANY-view, de-duplicated to distinct objects.
        Returns [(object_id, best_score, payload)] with up to top_k objects."""
        with self._lock:
            if self._vecs.shape[0] == 0:
                return []
            sims = self._vecs @ self._norm(vector)
            order = np.argsort(-sims)
            out, seen = [], set()
            for i in order:
                oid = self._view_oids[i]
                if oid in seen:
                    continue
                seen.add(oid)
                out.append((oid, float(sims[i]), self._payloads[oid]))
                if len(out) >= top_k:
                    break
            return out

    def update(self, object_id, vector=None, **fields) -> bool:
        """Merge fields into an existing prototype (used to backfill the VLM label:
        provisional -> named). If a vector is given it replaces the object's
        most-recently-added view. Rewrites all of the object's points to tier-2."""
        with self._lock:
            proto = self._payloads.get(object_id)
            if proto is None:
                return False
            for k, v in fields.items():
                setattr(proto, k, v)
            proto.last_seen = time.time()
            if vector is not None:
                rows = self._rows_for(object_id)
                if rows:
                    self._vecs[rows[-1]] = self._norm(vector)
            self._persist_object(object_id)
        return True

    def get(self, object_id: str) -> Optional[Prototype]:
        return self._payloads.get(object_id)

    def view_count(self, object_id: str) -> int:
        """How many view embeddings are stored for this object."""
        return len(self._rows_for(object_id))

    def sync(self):
        """Background-safe push/pull with the shared server. NEVER on the match() path.
        Pushes every view-point; pulls view-points this robot doesn't have yet.
        Returns (pushed, pulled)."""
        if self._shared is None:
            raise RuntimeError("no shared_url configured, needs a shared server")
        from qdrant_client.models import PointStruct

        self._ensure_shared_collection()

        with self._lock:
            local_pids = set(self._view_pids)
            points = [PointStruct(
                id=self._view_pids[i],
                vector=self._vecs[i].tolist(),
                payload=asdict(self._payloads[self._view_oids[i]]),
            ) for i in range(len(self._view_pids))]
            if points:
                self._shared.upsert(self._collection, points=points)
            pushed = len(points)

        remote, _ = self._shared.scroll(self._collection, limit=100000, with_vectors=True)
        pulled = 0
        for p in remote:
            if p.id in local_pids:
                continue
            oid = p.payload["object_id"]
            vec = np.asarray(p.vector, dtype=np.float32)
            with self._lock:
                if oid not in self._payloads:
                    self._payloads[oid] = Prototype(**p.payload)
                self._append_row(oid, vec, pid=p.id)
                self._upsert_point(p.id, vec, self._payloads[oid])
            pulled += 1

        return pushed, pulled

    def close(self):
        self._local.close()