
# ==========================
# FILE: dexsdk/utils.py (add)
# ==========================
# If you already have a utils.py, append this helper.
import cv2
import numpy as np

def rotate90(img: np.ndarray, k: int = 1) -> np.ndarray:
    k %= 4
    if k == 0:
        return img
    if k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)