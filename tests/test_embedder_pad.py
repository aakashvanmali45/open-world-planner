import numpy as np
from perception.embedder import _pad_to_square


def test_pad_makes_square_and_centers():
    img = np.full((10, 4, 3), 50, np.uint8)        # tall, non-square
    out = _pad_to_square(img, (255, 255, 255))
    assert out.shape == (10, 10, 3)                 # now square
    x0 = (10 - 4) // 2
    assert (out[:, x0:x0 + 4] == 50).all()          # original centered, intact
    assert (out[:, :x0] == 255).all()               # sides padded