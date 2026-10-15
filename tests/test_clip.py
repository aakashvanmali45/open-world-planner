import numpy as np
from PIL import Image, ImageDraw
from perception.embedder import Embedder

def _circle(color, offset = 0):
    img = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(img).ellipse([60+offset, 60+offset, 160+offset, 160+offset], fill = color)
    return img

def _square(color):
    img = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(img).rectangle([60, 60, 160, 160], fill = color)
    return img

def test_clip_same_scores_higher(clip_embedder):
    a1 = clip_embedder.embed_image(_circle("red"))
    a2 = clip_embedder.embed_image(_circle("red", offset = 20))
    b = clip_embedder.embed_image(_square("blue"))

    same, diff = float(a1 @ a2), float(a1 @ b)

    assert same > diff, f"Same= {same:.3f} should exceed diff= {diff:.3f}"

def test_embeddings_are_normalized(clip_embedder):
    v = clip_embedder.embed_image(_circle("green"))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4