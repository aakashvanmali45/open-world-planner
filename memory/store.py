from __future__ import annotations
import time, uuid, threading
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

DIM_DEFAULT = 512

@dataclass
class Prototype:
    """The payload stored per object. Vector is kept separate from this metadata."""
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
    last_seen:float = field(default_factory=time.time)

class MemoryStore:
    """
    Three-tier memory. Tiers 1 and 2 are built here; tier 3 (shared sync) is
    stubbed until Docker is available.
      - Tier 1  in-memory NumPy index  -> match() ALWAYS hits this. No network.
      - Tier 2  embedded Qdrant (local dir, no Docker) -> survives restart.
      - Tier 3  shared Qdrant server    -> background sync target (deferred).
    """

    def __init__(self, dim: int = DIM_DEFAULT, local_path: str = "./memory_data/local",
                 robot_id: str = "robot_0", shared_url: Optional[str] = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.dim = dim
        self.robot_id = robot_id
        self._lock = threading.Lock()

        # Tier 1: in-memory index

        self._vecs = np.zeros((0, dim), dtype = np.float32)
        self._ids: list[str] = []
        self._payloads: dict[str, Prototype] = {}

        # Tier 2: embedded Qdrant

        self._local = QdrantClient(path = local_path)
        self._collection = "prototypes"
        if not self._local.collection_exists(self._collection):
            self._local.create_collection(
                self._collection,
                vectors_config = VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._load_from_local()

        #Tier 3: Shared server
        self._shared = QdrantClient(url=shared_url) if shared_url else None

    @staticmethod
    def _norm(v) -> np.ndarray:
        v = np.asarray(v, dtype = np.float32).reshape(-1)
        n = np.linalg.norm(v)
        return v/n if n>0 else v

    def _index_add(self, vector, proto: Prototype):
        self._vecs = np.vstack([self._vecs, self._norm(vector)[None, :]])
        self._ids.append(proto.object_id)
        self._payloads[proto.object_id] = proto

    def _upsert_local(self, vector, proto: Prototype):
        from qdrant_client.models import PointStruct
        self._local.upsert(self._collection, points = [PointStruct(
            id = proto.object_id, vector = self._norm(vector).tolist(), payload = asdict(proto))])

    def _load_from_local(self):
        points, _ = self._local.scroll(self._collection, limit = 100000, with_vectors = True)

        for p in points:
            self._index_add(np.asarray(p.vector, dtype = np.float32), Prototype(**p.payload))

    def _ensure_shared_collection(self):
        """Create the collection on the shared server if it isn't there yet."""
        from qdrant_client.models import Distance, VectorParams
        if not self._shared.collection_exists(self._collection):
            self._shared.create_collection(
                self._collection,
                vectors_config = VectorParams(size=self.dim, distance = Distance.COSINE)
            )

    ##### Public API #####

    def add(self, vector, **kwargs) -> str: 
        """Append a new prototype. Writes in-memory + local; returns its object_id."""
        proto = Prototype(robot_id = self.robot_id, **kwargs)
        with self._lock:
            self._index_add(vector, proto)
            self._upsert_local(vector, proto)
        return proto.object_id

    def match(self, vector, top_k:int=1):
        """Nearest-neighbour against the in-memory index. Returns [(id, score, payload)]."""

        with self._lock:
            if not self._ids:
                return []

            sims = self._vecs @ self._norm(vector)
            order = np.argsort(-sims)[:top_k]
            return [(self._ids[i], float(sims[i]), self._payloads[self._ids[i]]) for i in order ]

    def get(self, object_id: str) -> Optional[Prototype]:
        return self._payloads.get(object_id)

    def sync(self):
        """Background-safe push/pull with the shared server. NEVER on the match() path.
        Push: upsert all local prototypes to the shared server.
        Pull: fetch shared prototypes this robot doesn't have, add them to
              local persistence + the in-memory index.
        Returns (pushed, pulled) counts."""

        if self._shared is None:
            raise RuntimeError("no shared_url configured, needs a shared server")
        from qdrant_client.models import PointStruct

        self._ensure_shared_collection()

        with self._lock:
            local_ids = set(self._ids)

            # Pushing everything we know
            points = []
            for oid in self._ids:
                idx = self._ids.index(oid)
                vec = self._vecs[idx].tolist()
                points.append(PointStruct(
                    id=oid, vector=vec, payload=asdict(self._payloads[oid])
                ))

            if points:
                self._shared.upsert(self._collection, points=points)
            pushed = len(points)

        remote, _ = self._shared.scroll(
            self._collection, limit = 100000, with_vectors = True)

        pulled = 0
        for p in remote:
            if p.id in local_ids:
                continue

            proto = Prototype(**p.payload)
            vec = np.asarray(p.vector, dtype = np.float32)
            with self._lock:
                self._index_add(vec, proto)
                self._upsert_local(vec, proto)
            pulled += 1

        return pushed, pulled
    def close(self):
        self._local.close()