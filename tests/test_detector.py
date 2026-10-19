import os
import numpy as np
import pytest
from perception.detector import Detector, Detection

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "sample.jpeg")


def test_bad_backend_raises():
    # Rejected before any model loads, so this needs no weights.
    with pytest.raises(ValueError):
        Detector(backend="not-a-real-backend")


def test_set_vocabulary_rejected_on_yoloe():
    # yoloe-pf is prompt-free; set_vocabulary must refuse. No predict() => no weights.
    det = Detector(backend="yoloe")
    with pytest.raises(RuntimeError):
        det.set_vocabulary(["bottle", "mug"])


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="no sample.jpeg exists")
def test_detects_expected_objects_yoloworld():
    det = Detector(backend="yoloworld")
    det.set_vocabulary(["bottle", "mug", "cup", "book"])
    dets = det.detect(SAMPLE)
    found = {d.label for d in dets}
    assert len(dets) >= 2, f"expected at least 2 objects, found {len(dets)}: {found}"
    for d in dets:
        assert d.crop.ndim == 3 and d.crop.shape[2] == 3 and d.crop.size > 0


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="no sample.jpeg exists")
def test_yoloe_detects_and_crops():
    det = Detector(backend="yoloe")
    dets = det.detect(SAMPLE)
    assert dets, "yoloe prompt-free should find at least one object"
    for d in dets:
        assert isinstance(d, Detection)
        assert d.crop.ndim == 3 and d.crop.shape[2] == 3 and d.crop.size > 0


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="no sample.jpeg exists")
def test_crop_feeds_clip(clip_embedder):
    det = Detector(backend="yoloworld")
    det.set_vocabulary(["bottle", "mug", "cup", "book"])
    dets = det.detect(SAMPLE)
    assert dets, "needs at least one detection"
    v = clip_embedder.embed_image(dets[0].crop)
    assert v.shape == (512,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4
