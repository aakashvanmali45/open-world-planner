import sys, cv2
from detector import Detector
from describer import Describer

path = sys.argv[1] if len(sys.argv) > 1 else "tests/data/sample.jpeg"

det = Detector()
det.set_vocabulary(["bottle", "mug", "cup", "book"])
dets = det.detect(path)
crop = dets[0].crop
print(f"Describing a '{dets[0].label}' crop ({crop.shape[1]}x{crop.shape[0]} px)\n")

for model in ["gemma3:4b", "moondream"]:
    try:
        d = Describer(model).describe(crop, known_objects = ["table"])
        print(f"=== {model} :: {d.latency_ms:.0f} ms ===")
        print(f"   labels   : {d.candidate_labels}")
        print(f"   category:  {d.category}color: {d.color} | material: {d.material}")
        print(f"   features:  {d.features}")

        if not d.candidate_labels:
            print(f" (JSON parse failed; raw head: {d.raw[:120]!r})")

        print()
    except Exception as e:
        print(f"=== {model} failed: {e}")