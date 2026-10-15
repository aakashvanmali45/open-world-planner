import os
import numpy as np
import pytest
from perception.detector import Detector

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "sample.jpeg")

@pytest.mark.skipif(not os.path.exists(SAMPLE), reason = "no sample.jpeg exists")
def test_detects_expected_objects():
    det = Detector()
    det.set_vocabulary(["bottle", "mug", "cup", "book"])
    dets = det.detect(SAMPLE)
    found = {d.label for d in dets}
    assert len(dets) >= 2, f"expected atleast 2 objects, found {len(dets)}: {found}"

    for d in dets:
        assert d.crop.ndim == 3 and d.crop.shape[2] == 3 and d.crop.size > 0


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason = "no sample.jpeg exists")
def test_crop_feeds_clip(clip_embedder):
    det = Detector()
    det.set_vocabulary(["bottle", "mug", "cup", "book"])
    dets = det.detect(SAMPLE)
    assert dets, "needs atleast one detection"
    v = clip_embedder.embed_image(dets[0].crop)
    assert v.shape == (512,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4