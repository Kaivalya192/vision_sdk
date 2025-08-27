


# ================================
# FILE: dexsdk/camera/webcam.py
# ================================
from typing import Optional
import cv2, numpy as np


def open_capture(index: int = 0):
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def grab_frame(cap) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return np.ascontiguousarray(frame)

