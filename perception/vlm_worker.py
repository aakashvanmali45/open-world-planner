"""
perception/vlm_worker.py -- run describe-once in a background thread so the
perception loop never blocks on the ~tens-of-seconds VLM call.

enqueue(object_id, crop, known_objects) -> the worker describes the crop and
writes the label back onto the prototype via store.update(status='named').
The caller enqueues ONE job per object (guarded in the loop, next build).
Because the VLM runs in Ollama (a separate process), this thread mostly waits
on I/O -- the GIL is free, so the main loop keeps running.
"""
from __future__ import annotations
import queue
import threading


class VLMWorker:
    def __init__(self, describer, store):
        self.describer = describer
        self.store = store
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, object_id: str, crop, known_objects=None) -> None:
        self._q.put((object_id, crop, known_objects or []))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                object_id, crop, known = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                d = self.describer.describe(crop, known_objects=known)
                label = d.candidate_labels[0] if d.candidate_labels else None
                self.store.update(
                    object_id,
                    status="named",
                    candidate_labels=d.candidate_labels,
                    canonical_label=label,
                    category=d.category,
                    attributes={"color": d.color, "material": d.material, "features": d.features},
                )
            except Exception:
                self.store.update(object_id, status="name_failed")
            finally:
                self._q.task_done()

    def join(self) -> None:
        """Block until every queued describe job is done (used by the offline render)."""
        self._q.join()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)