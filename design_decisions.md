# Design Decisions

A running log of the architectural choices behind **Open World Mission Learner**, with the reasoning for each. Newest phase at the bottom. The plan is a scaffold, not a contract — decisions here are revisited when evidence says so.

---

## Core architecture (locked)

These predate the build and shape everything downstream.

### Frozen CLIP — memory *is* the classifier
CLIP is never trained on the live path. Learning an object = appending an embedding to the memory store. No gradients on the mission loop. This is what makes **"no retraining" literally true online**. Identity is decided by cosine distance to stored prototypes, never a trained softmax.

### Two separate scores, two separate jobs
- **Objectness** (from the detector) filters junk boxes — answers *"is there a thing here?"* only. Never decides identity.
- **Embedding distance to memory** (from CLIP) decides *known vs novel*.

These are deliberately different signals. The detector's class label is a *hint* stored in `candidate_labels`, never the identity.

### Known vs novel = threshold on CLIP distance, confirmed over k frames
- `s = max cosine similarity` of a crop's embedding against the in-memory index.
- `s ≥ T` → **known** (assign stored label, bump `observation_count`).
- `s < T` → **novelty candidate**.
- A candidate must read novel across **≥ k frames** before it is trusted. Cross-frame consistency — not a single-frame verdict, not a VLM yes/no — is the real false-positive killer.
- The detector does **not** get a vote on known/novel. It only supplies the box.
- `T` starts as a fixed number; calibration to an open-set score is Phase 1 (where the OOD/MVTec background becomes the contribution).

### VLM runs once, gated
A confirmed novelty gets **one** context-aware VLM call returning a candidate-label list + attributes. Identity lives in the embedding; the label is metadata. The VLM never runs on a known object. Latency is kept off the recognition path.

### ADAM-style recognise-path
Re-sightings resolve by frequency-vote over neighbours' candidate lists → cross-modal rerank (`argmax cos(image, text)`). No new VLM call, no network.

### Two-brain split: planner ≠ perception
The planner is a separate ≥7B LLM. Small VLMs can't tool-call reliably, so reasoning is split off from perception.

---

## Phase 0 — Foundations, shared memory & test harness

### Environment
- **Python 3.11** — the sweet spot for ROS 2 Jazzy, `ultralytics`, and `open_clip`. (LangChain v1 later confirmed this: it dropped Python 3.9; 3.11 is clear.)
- **`uv`** for env + dependency management; pinned via `uv.lock`.
- **Windows for Phases 0–2** (pure Python: perception on video + agent). Switch to **Linux for Phase 3+** (ROS 2 + Gazebo). GitHub is the handoff mechanism; `.gitattributes` with `* text=auto eol=lf` keeps line endings sane across the round-trip.
- `ros_ws/` stays empty on Windows (built natively on Linux in Phase 3).

### Memory backends
- **In-memory index** = NumPy cosine over normalised vectors. Tiny N through Phase 2; FAISS deferrable behind the same interface (and FAISS-CPU wheels are finicky on Windows). `match()` always hits this — no network.
- **Local + shared** = Qdrant. Identical API for both, so **sync is not a rebuild**. Local tier uses **embedded Qdrant** (`QdrantClient(path=...)`) — *no Docker needed*, survives restarts offline.
- **Shared tier** = Qdrant server (Docker) — the background sync target. Deferred until Docker is set up.
- Embedding **dimension is config-driven, default 512**; encoder chosen late.
- Rule: `match()` is local-only; `sync()` runs on a background thread, never on the hot path.

### Encoder
- **CLIP ViT-B/32** (`open_clip`, `openai` weights), 512-dim, **frozen** (`eval()`, no grad).
- Embeddings **L2-normalised** so dot product = cosine, matching the MemoryStore convention.
- SigLIP + the CLIP-vs-SigLIP benchmark deferred to Phase 1 (Kaggle).
- Encoder is interface-hidden; swapping it doesn't touch callers.

### Detector
- **YOLO-World** (`yolov8s-world.pt`, ultralytics), open-vocabulary via `set_classes`.
- Answers objectness/location only, never identity.
- Low default `conf` (chosen below) to catch unknowns; junk filtered downstream.
- **Colour convention:** the model is fed **BGR** (ultralytics' convention); crops are returned **RGB** (CLIP's convention). Mixing these up silently degrades detection — learned the hard way.
- **Chosen conf threshold: 0.10
- **CPU detect latency: 938 ms/frame**  → caps keyframe rate at ~`1000 / latency` fps. This is *why* we process keyframes, not every frame.

### VLM (describe-once)
- **gemma3:4b** via Ollama.
- **Chosen over moondream**, which produced *degenerate repetition* (`"white mug", "white mug", ...`) and unparseable JSON on the same crop — the "higher hallucination rate" the benchmarks warned about. gemma3 returned clean, correct, parseable JSON.
- Called with `format="json"` + `temperature=0.0` for deterministic, parseable output.
- **Latency ~36 s cold on CPU.** Acceptable *only because* the VLM is gated to one call per novel object.
- **This number justifies the whole architecture:** CLIP match is milliseconds, the VLM is ~36 s — a ~10,000× gap. That gap *is* why the cheap always-on hot path and the expensive gated path are split, and why a provisional prototype is written *before* the VLM call so re-sightings match instantly while the slow call happens off the critical path.

### Planner LLM
- **llama3.1:8b** via Ollama, using LangChain v1 **`create_agent`** (the `langgraph.prebuilt.create_react_agent` prebuilt was deprecated in LangGraph v1).
- Tool-calling verified on CPU: a ReAct loop correctly called a tool, used its result, and answered.
- **Note:** small models may emit *duplicate tool calls* (llama3.1 called the dummy tool twice). Phase 2 tools with side effects (`add_to_memory`, `navigate`) must therefore be **idempotent** or guarded — which is already why memory dedups by embedding similarity rather than trusting call counts.
- Planner is model-swappable; a hosted API is an option if CPU latency bites in Phase 2.

### Weights & caching
- Model weights (CLIP ~350 MB, YOLO ~50 MB) are **git-ignored** — they live in the local cache and re-fetch on first run on any machine. Keeps the repo small and makes the Linux migration clean (no large binaries dragged across).

---

## Still open at end of Phase 0
- **Docker** → shared Qdrant tier + `sync()` implementation + the sync-round-trip acceptance test (the one remaining red item on the Phase 0 gate; memory round-trip and CLIP sanity are green).
- **Kaggle batch workflow** check — the "ground-truth factory" for Phase 1 gold labels and the CLIP-vs-SigLIP benchmark.
- **Novelty threshold `T`** — fixed for now; calibrated open-set score is Phase 1.
