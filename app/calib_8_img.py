#!/usr/bin/env python3
"""
calib_8_img.py — constrained single-image intrinsics from an AprilGrid.

Strategy (pinhole):
  • Detect all tag corners with aprilgrid.Detector("t36h11")
  • Build object points on a Z=0 plane (ROWS×COLS, TAG_MM, GAP_MM)
  • Solve cv2.calibrateCamera with strong constraints:
        - FIX_PRINCIPAL_POINT at image center
        - FIX_SKEW, FIX_ASPECT_RATIO (square pixels)
        - ZERO_TANGENT_DIST
        - estimate f, k1, k2 (k3..k6 fixed)
  • (Optional) visualize undistortion

This is a minimal calibration; multi-view is still superior.

Usage:
  python calib_8_img.py --image img.jpg --save-k K.json
"""

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2
from aprilgrid import Detector

# ---- Board definition (edit to match your print) ----
ROWS, COLS = 6, 5
TAG_CM, GAP_CM = 3.0, 0.6
TAG_MM = TAG_CM*10.0
GAP_MM = GAP_CM*10.0
PITCH_MM = TAG_MM + GAP_MM
DICT = "t36h11"

def to_json_safe(x):
    if isinstance(x, (np.floating, np.float32, np.float64)): return float(x)
    if isinstance(x, (np.integer,  np.int32,   np.int64)):   return int(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return x

def id_to_rc(tag_id: int)->Tuple[int,int]:
    return tag_id // COLS, tag_id % COLS

def tag_corners_object_mm(tag_id: int) -> np.ndarray:
    r, c = id_to_rc(tag_id)
    cx = c * PITCH_MM
    cy = r * PITCH_MM
    h  = TAG_MM * 0.5
    return np.array([
        [cx-h, cy-h, 0.0],
        [cx+h, cy-h, 0.0],
        [cx+h, cy+h, 0.0],
        [cx-h, cy+h, 0.0],
    ], dtype=np.float32).reshape(-1,1,3)

def detect(gray: np.ndarray):
    dets = Detector(DICT).detect(gray)
    out = []
    for d in dets:
        out.append({"id": int(d.tag_id), "corners_px": np.array(d.corners, dtype=np.float32)})
    return out

def build_points(dets: List[Dict[str,Any]]):
    obj, img = [], []
    for d in dets:
        tid = d["id"]
        if 0 <= tid < ROWS*COLS:
            obj.append(tag_corners_object_mm(tid))
            img.append(d["corners_px"].astype(np.float32))
    if not obj: 
        return None, None
    return np.vstack(obj), np.vstack(img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--save-k", default="", help="Path to save K/dist JSON (recommended).")
    ap.add_argument("--show", action="store_true", help="Show undistorted overlay.")
    args = ap.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None: sys.exit(f"[ERROR] Could not read {args.image}")
    h, w = gray.shape[:2]

    dets = detect(gray)
    if not dets: sys.exit("[ERROR] No AprilTags detected.")

    obj, img = build_points(dets)
    if obj is None: sys.exit("[ERROR] No in-grid tag IDs detected.")

    # ---- Constrained single-image calibration ----
    # Intrinsic guess
    f0 = 1.2*max(w, h)  # rough, stable starting point
    K0 = np.array([[f0, 0, w/2.0],
                   [0,  f0, h/2.0],
                   [0,   0,   1.0]], dtype=np.float64)
    dist0 = np.zeros((8,1), dtype=np.float64)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS    |
        cv2.CALIB_FIX_PRINCIPAL_POINT    |  # keep at image center
        cv2.CALIB_FIX_SKEW               |
        cv2.CALIB_ZERO_TANGENT_DIST      |
        cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
        cv2.CALIB_FIX_ASPECT_RATIO          # fx/fy ratio from K0 (1.0)
    )

    # Calibrate from ONE view (many points)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj], [img], (w,h), K0, dist0, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    )

    # Simple sanity checks
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    if not (0.2*max(w,h) < fx < 10*max(w,h)): print("[WARN] fx seems off:", fx)
    if not (0.2*max(w,h) < fy < 10*max(w,h)): print("[WARN] fy seems off:", fy)
    if abs(cx - w/2.0) > 3 or abs(cy - h/2.0) > 3:
        print("[WARN] principal point drifted; constraints may not have been applied as expected.")

    # Reproject RMSE
    proj, _ = cv2.projectPoints(obj, rvecs[0], tvecs[0], K, dist)
    rmse = float(cv2.norm(img, proj, cv2.NORM_L2) / max(len(obj),1))**0.5

    out = {"model":"pinhole","K":K.tolist(),"dist":dist.tolist(),
           "image_size":{"w":w,"h":h},"rms_reprojection_error_px":float(ret),
           "rmse_px": rmse}
    print("\n=== SINGLE-VIEW INTRINSICS ===")
    print(json.dumps(out, indent=2))

    if args.save_k:
        Path(args.save_k).write_text(json.dumps(out, indent=2))
        print(f"[OK] Saved to {args.save_k}")

    if args.show:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0.0)
        und = cv2.undistort(gray, K, dist, None, newK)
        # draw detections on undistorted for a quick look
        det2 = detect(und)
        vis = cv2.cvtColor(und, cv2.COLOR_GRAY2BGR)
        for d in det2:
            c = d["corners_px"].astype(np.int32).reshape(-1,2)
            cv2.polylines(vis, [c], True, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Undistorted (single-view intrinsics)", vis)
        print("[INFO] press any key to close")
        cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()