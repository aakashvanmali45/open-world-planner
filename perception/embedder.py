from __future__ import annotations
import numpy as np
import torch
from PIL import Image

def _pad_to_square(img:np.ndarray, pad_color) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w: 
        return img

    s = max(h, w)
    canvas = np.full((s, s, 3), pad_color, dtype=img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0: x0 + w] = img
    return canvas

class Embedder:
    """Frozen CLIP/SigLIP encoder. Returns L2-normalized vectors
    so that a dot product IS cosine similarity (matches MemoryStore's convention)."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", 
                 device: str = "cpu", pad_to_square: bool = True, pad_color = (114, 114, 114)):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained = pretrained, device = device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.dim = self.model.visual.output_dim
        self.pad_to_square = pad_to_square
        self.pad_color = pad_color

    @torch.no_grad()
    def embed_image(self, image) -> np.ndarray:
        """Accepts PIL.Image or an HxWx3 RGB ndarray (OpenCV Crop)"""

        if not isinstance(image, np.ndarray):
            image = np.asarray(image.convert("RGB"))

        if self.pad_to_square:
            image = _pad_to_square(image, self.pad_color)

        image = Image.fromarray(image)

        x = self.preprocess(image).unsqueeze(0).to(self.device)
        f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim = True)

        return f.cpu().numpy().reshape(-1).astype(np.float32)

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Needed for cross modal reranking"""

        toks = self.tokenizer([text]).to(self.device)
        f = self.model.encode_text(toks)
        f = f / f.norm(dim=-1, keepdim = True)

        return f.cpu().numpy().reshape(-1).astype(np.float32)
