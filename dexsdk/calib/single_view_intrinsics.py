from __future__ import annotations

"""Single-view intrinsic calibration utilities."""

import numpy as np
import cv2
from typing import Any, Dict, List, Tuple
from aprilgrid import Detector

# ---- Board definition (edit to match your print) ----
ROWS, COLS = 6, 5
TAG_CM, GAP_CM = 3.0, 0.6
TAG_MM = TAG_CM * 10.0
GAP_MM = GAP_CM * 10.0
PITCH_MM = TAG_MM + GAP_MM
DICT = "t36h11"


def id_to_rc(tag_id: int) -> Tuple[int, int]:
    return tag_id // COLS, tag_id % COLS


def tag_corners_object_mm(tag_id: int) -> np.ndarray:
    r, c = id_to_rc(tag_id)
    cx = c * PITCH_MM
    cy = r * PITCH_MM
    h = TAG_MM * 0.5
    return np.array(
        [
            [cx - h, cy - h, 0.0],
            [cx + h, cy - h, 0.0],
            [cx + h, cy + h, 0.0],
            [cx - h, cy + h, 0.0],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 3)


def detect(gray: np.ndarray):
    dets = Detector(DICT).detect(gray)
    out = []
    for d in dets:
        out.append({"id": int(d.tag_id), "corners_px": np.array(d.corners, dtype=np.float32)})
    return out


def build_points(dets: List[Dict[str, Any]]):
    obj, img = [], []
    for d in dets:
        tid = d["id"]
        if 0 <= tid < ROWS * COLS:
            obj.append(tag_corners_object_mm(tid))
            img.append(d["corners_px"].astype(np.float32))
    if not obj:
        return None, None
    return np.vstack(obj), np.vstack(img)


def estimate_intrinsics(gray: np.ndarray) -> Dict[str, Any]:
    """Estimate camera intrinsics from a single AprilGrid view."""

    h, w = gray.shape[:2]

    dets = detect(gray)
    if not dets:
        raise RuntimeError("No AprilTags detected.")

    obj, img = build_points(dets)
    if obj is None:
        raise RuntimeError("No in-grid tag IDs detected.")

    # ---- Constrained single-image calibration ----
    f0 = 1.2 * max(w, h)
    K0 = np.array([[f0, 0, w / 2.0], [0, f0, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
    dist0 = np.zeros((8, 1), dtype=np.float64)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_SKEW
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_FIX_K6
        | cv2.CALIB_FIX_ASPECT_RATIO
    )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj],
        [img],
        (w, h),
        K0,
        dist0,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if not (0.2 * max(w, h) < fx < 10 * max(w, h)):
        print("[WARN] fx seems off:", fx)
    if not (0.2 * max(w, h) < fy < 10 * max(w, h)):
        print("[WARN] fy seems off:", fy)
    if abs(cx - w / 2.0) > 3 or abs(cy - h / 2.0) > 3:
        print("[WARN] principal point drifted; constraints may not have been applied as expected.")

    proj, _ = cv2.projectPoints(obj, rvecs[0], tvecs[0], K, dist)
    rmse = float(cv2.norm(img, proj, cv2.NORM_L2) / max(len(obj), 1)) ** 0.5

    return {
        "model": "pinhole",
        "K": K.tolist(),
        "dist": dist.tolist(),
        "image_size": {"w": w, "h": h},
        "rms_reprojection_error_px": float(ret),
        "rmse_px": rmse,
    }

