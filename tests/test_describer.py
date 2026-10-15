import os
import pytest
from perception.describer import Describer

# These tests need Ollama running with gemma3:4b pulled. Skip cleanly if absent,
# so CI (which has no Ollama) stays green.
def _ollama_ready(model="gemma3:4b"):
    try:
        import ollama
        tags = ollama.Client().list()
        names = [m.get("model", "") for m in tags.get("models", [])]
        return any(model in n for n in names)
    except Exception:
        return False

pytestmark = pytest.mark.skipif(not _ollama_ready(), reason="Ollama/gemma3:4b not available")

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "sample.jpeg")

@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="no sample.jpeg")
def test_describe_returns_parsed_json():
    from perception.detector import Detector
    det = Detector(); det.set_vocabulary(["bottle", "mug", "cup", "book"])
    crop = det.detect(SAMPLE)[0].crop
    d = Describer("gemma3:4b").describe(crop, known_objects=["table"])
    assert d.candidate_labels, f"expected labels, got raw: {d.raw[:150]!r}"
    assert isinstance(d.candidate_labels, list)
    assert d.latency_ms > 0