import cv2
import numpy as np
import requests
from paddleocr import PaddleOCR

def draw_boxes(img, polys, texts, scores=None):
    """Draw detected boxes and text labels."""
    for i, poly in enumerate(polys):
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = pts[0][0]
        label = texts[i]
        if scores is not None:
            label = f"{label}:{scores[i]:.2f}"
        cv2.putText(img, label, (x, max(y - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return img

def run_ocr_and_show(img):
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang="en"
    )

    results = ocr.predict(img)
    res = results[0]

    # Try multiple possible field names for polygons
    polys = getattr(res, "rec_polys", None) or getattr(res, "dt_polys", None) or []
    texts = getattr(res, "rec_texts", [])
    scores = getattr(res, "rec_scores", [])

    print(f"Detected {len(texts)} text lines")
    for t, s in zip(texts, scores):
        print(f"{t!r}: {s:.3f}")

    vis = img.copy()
    if len(polys) > 0:
        vis = draw_boxes(vis, polys, texts, scores)

    cv2.imshow("PaddleOCR v3.2 Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/english.png"
    arr = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to load image from internet.")
    run_ocr_and_show(img)
"""
Tested with:
  pip install paddlepaddle==3.2.0 paddleocr==3.2.0 opencv-python requests
"""
import cv2, numpy as np, requests
from paddleocr import PaddleOCR

def draw(img, polys, txts, scores):
    for i, p in enumerate(polys):
        pts = np.array(p, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,0), 2)
        x, y = pts[0][0]
        cv2.putText(img, f"{txts[i]}:{scores[i]:.2f}", (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return img

# 1) download example
url = "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/english.png"
img = cv2.imdecode(np.frombuffer(requests.get(url).content, np.uint8), cv2.IMREAD_COLOR)

# 2) create engine  (***doc-preprocessor OFF***)
ocr = PaddleOCR(
        lang="en",
        use_doc_preprocessor=False,          # <-- key line
        use_textline_orientation=True,       # replaces deprecated use_angle_cls
        text_det_thresh=0.1,
        text_det_box_thresh=0.2)

# 3) run
res = ocr.predict(img)[0]                   # first (and only) page

polys  = getattr(res, "rec_polys", [])      # v3.x
texts  = res.rec_texts
scores = res.rec_scores

print(f"Detected {len(texts)} lines")
for t, s in zip(texts, scores):
    print(f"{t:20s}  {s:.3f}")

cv2.imshow("PaddleOCR v3.2 Result", draw(img.copy(), polys, texts, scores))
cv2.waitKey(0)
cv2.destroyAllWindows()
