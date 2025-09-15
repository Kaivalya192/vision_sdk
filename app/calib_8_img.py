#!/usr/bin/env python3
"""Constrained single-image intrinsic calibration."""

import argparse
import json
import sys

import cv2
import numpy as np

from dexsdk.calib import estimate_intrinsics, detect, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--save-k", default="", help="Path to save K/dist JSON (recommended).")
    ap.add_argument("--show", action="store_true", help="Show undistorted overlay.")
    args = ap.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        sys.exit(f"[ERROR] Could not read {args.image}")

    out = estimate_intrinsics(gray)

    print("\n=== SINGLE-VIEW INTRINSICS ===")
    print(json.dumps(out, indent=2))

    if args.save_k:
        save_json(args.save_k, out)
        print(f"[OK] Saved to {args.save_k}")

    if args.show:
        K = np.array(out["K"], dtype=np.float64)
        dist = np.array(out["dist"], dtype=np.float64)
        h, w = out["image_size"]["h"], out["image_size"]["w"]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
        und = cv2.undistort(gray, K, dist, None, newK)
        det2 = detect(und)
        vis = cv2.cvtColor(und, cv2.COLOR_GRAY2BGR)
        for d in det2:
            c = d["corners_px"].astype(int).reshape(-1, 2)
            cv2.polylines(vis, [c], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Undistorted (single-view intrinsics)", vis)
        print("[INFO] press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

