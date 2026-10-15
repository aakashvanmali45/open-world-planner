import sys, time, cv2
from detector import Detector

path = sys.argv[1] if len(sys.argv) > 1 else "tests/data/sample.jpeg"

det = Detector(conf=0.1)                      # very low, just to SEE everything
det.set_vocabulary(["bottle", "mug", "cup", "book"])

t = time.perf_counter()
dets = det.detect(path)
print(f"detect: {(time.perf_counter()-t)*1000:.0f} ms, {len(dets)} objects")
for d in dets:
    print(f"  {d.label:8s} conf={d.score:.3f} box={d.box}")

bgr = cv2.imread(path)
for d in dets:
    x1, y1, x2, y2 = d.box
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 180, 0), 2)
    cv2.putText(bgr, f"{d.label} {d.score:.2f}", (x1, max(y1-6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
cv2.imwrite("detect_out.jpg", bgr)
print("wrote detect_out.jpg")