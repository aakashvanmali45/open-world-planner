import numpy as np
from perception.detector import Detection
from perception.track import Tracker, iou


def _det(box, score=0.9):
    return Detection(box=box, score=score, label="x", crop=np.zeros((4, 4, 3), np.uint8))


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-6


def test_confirms_after_n_of_m():
    trk = Tracker(iou_thresh=0.3, n=3, m=5, max_age=3)
    box = (10, 10, 50, 50)
    confirmed = []
    for f in range(3):
        confirmed = trk.update([_det(box)], f)
    assert len(confirmed) == 1
    assert confirmed[0].best_crop is not None


def test_transient_flicker_never_confirms():
    trk = Tracker(iou_thresh=0.3, n=3, m=5, max_age=3)
    trk.update([_det((10, 10, 50, 50))], 0)
    for f in range(1, 6):
        out = trk.update([], f)
    assert out == []


def test_two_objects_two_tracks():
    trk = Tracker(iou_thresh=0.3, n=1, m=5)
    out = trk.update([_det((0, 0, 20, 20)), _det((100, 100, 140, 140))], 0)
    assert len({t.id for t in out}) == 2