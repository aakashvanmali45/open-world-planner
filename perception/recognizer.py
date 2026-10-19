from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

from memory.store import MemoryStore, Prototype

@dataclass
class MatchResult:
    status: str          # "known" or "novel"
    object_id: str       # prototype id (matched / newly created)
    score: float         # top cosine similarity
    label: Optional[str]
    prototype: Prototype

class Recognizer:
    def __init__(self, embedder, store: MemoryStore, threshold: float = 0.80,
                 multiview: bool = True, view_add_ceiling: float = 0.92,
                 max_views: int = 8, masker=None):
        self.embedder = embedder
        self.store = store
        self.threshold = threshold
        self.multiview = multiview            # off-switch for multi-view growth
        self.view_add_ceiling = view_add_ceiling
        self.max_views = max_views
        self.masker = masker                  # pass a Masker to enable; None = off

    def recognize(self, crop) -> MatchResult:
        if self.masker is not None:
            crop = self.masker.apply(crop)              # object pixels only, bg neutralized
        vec = self.embedder.embed_image(crop)
        hits = self.store.match(vec, top_k=1)

        if hits and hits[0][1] >= self.threshold:
            object_id, score, proto = hits[0]
            # Borderline re-match = a genuinely new angle -> remember this view.
            if self.multiview and score < self.view_add_ceiling:
                self.store.add_view(object_id, vec, max_views=self.max_views)
            else:
                proto.observation_count += 1
                proto.last_seen = time.time()
            return MatchResult("known", object_id, score, proto.canonical_label, proto)

        score = hits[0][1] if hits else 0.0
        object_id = self.store.add(vec, status="provisional")
        return MatchResult("novel", object_id, score, None, self.store.get(object_id))