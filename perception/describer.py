from __future__ import annotations
import io, json, base64, time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image

@dataclass
class Description:
    candidate_labels: list = field(default_factory=list)
    category: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    features: Optional[str] = None
    raw: str = ""
    latency_ms: float = 0.0

PROMPT = """You are labelling a single object a robot has just found.
Return ONLY a compact JSON object, no prose, with these keys:
    candidate_labels: a list of your top 3-5 guesses for what this object is (strings)
    category: a one word high level category
    color: the dominant color
    material: a most likely material
    features: a short phrase of distinctive visual features
Known objects already in view: {known_objects}
{location}
Describe the object in the image"""

class Describer: 
    """The VLM describes the Novel object"""

    def __init__(self, model: str = "gemma3:4B", host: Optional[str] = None):
        import ollama
        self.model = model
        self.client = ollama.Client(host = host) if host else ollama.Client()

    @staticmethod
    def _to_png_bytes(image) -> bytes:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def describe(self, image, known_objects: Optional[list] = None, 
                 location:str="") -> Description:

        prompt = PROMPT.format(
            known_objects = ", ".join(known_objects) if known_objects else "none",
            location = f"Object location: {location}" if location else "",
        )

        img_bytes = self._to_png_bytes(image)

        t = time.perf_counter()
        resp = self.client.generate(
            model = self.model, prompt = prompt,
            images = [base64.b64encode(img_bytes).decode()],
            format = "json",
            options={"temperature":0.0})

        dt = (time.perf_counter() - t) * 1000.0

        raw = resp.get("response", "")
        d = Description(raw = raw, latency_ms = dt)

        try: 
            parsed = json.loads(raw)
            d.candidate_labels = parsed.get("candidate_labels", []) or []
            d.category = parsed.get("category")
            d.color = parsed.get("color")
            d.material = parsed.get("material")
            d.features = parsed.get("features")

        except json.JSONDecodeError:
            pass

        return d
