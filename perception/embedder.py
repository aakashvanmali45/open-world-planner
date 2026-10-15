from __future__ import annotations
import numpy as np
import torch
from PIL import Image

class Embedder:
    """Frozen CLIP/SigLIP encoder. Returns L2-normalized vectors
    so that a dot product IS cosine similarity (matches MemoryStore's convention)."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained = pretrained, device = device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.dim = self.model.visual.output_dim

    @torch.no_grad()
    def embed_image(self, image) -> np.ndarray:
        """Accepts PIL.Image or an HxWx3 RGB ndarray (OpenCV Crop)"""

        if isinstance(image, np.ndarray):
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
