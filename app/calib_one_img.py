#!/usr/bin/env python3
"""Estimate plane scale from a single AprilGrid image."""

import argparse
import json
import sys

import cv2

from dexsdk.calib import compute_plane_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--k-json", default="", help="Intrinsics JSON from single_view_intrinsics.py")
    ap.add_argument("--target-width", type=int, default=0, help="Downscale for detection speed.")
    ap.add_argument("--pair-weight", type=float, default=0.5, help="Blend neighbor vs edge scale (0..1).")
    ap.add_argument("--no-bias", action="store_true", help="Disable final median-edge bias correction.")
    args = ap.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        sys.exit(f"[ERROR] Could not read {args.image}")

    res = compute_plane_scale(
        gray,
        args.k_json or None,
        {
            "target_width": args.target_width,
            "pair_weight": args.pair_weight,
            "no_bias": args.no_bias,
        },
    )

    summary = res["summary"]
    print("\n=== PX↔MM SUMMARY ===")
    print(json.dumps(summary, indent=2))

    gray = res["gray"]
    px_per_mm_x, px_per_mm_y = res["px_per_mm"]
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for d in res["detections"]:
        c = d["corners_px"].astype(int).reshape(-1, 2)
        tid = int(d["id"])
        cv2.polylines(vis, [c], True, (0, 255, 0), 2, cv2.LINE_AA)
        cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
        cv2.putText(vis, f"{tid}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"px/mm X={px_per_mm_x:.4f}  Y={px_per_mm_y:.4f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 220, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Detections (undistorted if K given)", vis)
    print("[INFO] press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

