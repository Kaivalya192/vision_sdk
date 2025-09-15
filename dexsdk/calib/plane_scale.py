from __future__ import annotations

"""Plane scale estimation utilities."""

import numpy as np
import cv2
from typing import Any, Dict, List, Tuple
from aprilgrid import Detector

from .store import load_json

# ---- Board definition ----
ROWS, COLS = 6, 5
TAG_CM, GAP_CM = 3.0, 0.6
TAG_MM = TAG_CM * 10.0
GAP_MM = GAP_CM * 10.0
PITCH_MM = TAG_MM + GAP_MM
DICT = "t36h11"

# Bias-correct final scale so median tag edge == TAG_MM
BIAS_ENFORCE = True
BIAS_BLEND = 1.0  # 1=force exact, 0.5=halfway, 0=off


def resize_to_width(img: np.ndarray, width: int):
    if not width or width <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    s = float(width) / w
    return cv2.resize(img, (width, int(round(h * s))), interpolation=cv2.INTER_AREA), s


def detect(gray: np.ndarray, target_width: int = 0):
    work, s = (gray, 1.0) if not target_width or target_width <= 0 else resize_to_width(gray, target_width)
    dets = Detector(DICT).detect(work)
    out = []
    for d in dets:
        out.append({
            "id": int(d.tag_id),
            "corners_px": (np.array(d.corners, dtype=np.float32) / s).astype(np.float32),
        })
    return out


def edge_sizes_px(corners_4x1x2: np.ndarray) -> Tuple[float, float]:
    p = corners_4x1x2.reshape(-1, 2)
    tl, tr, br, bl = p[0], p[1], p[2], p[3]
    w = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    h = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    return float(w), float(h)


def robust_median_trim(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    if arr.size < 5:
        return med
    mad = float(np.median(np.abs(arr - med)))
    if mad == 0:
        return med
    keep = arr[np.abs(arr - med) <= 3.0 * 1.4826 * mad]
    return float(np.median(keep)) if keep.size else med


def _stats(v: List[float]) -> Dict[str, Any]:
    if not v:
        return {"count": 0}
    a = np.asarray(v, dtype=float)
    return {
        "count": int(a.size),
        "min": float(a.min()),
        "q25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "q75": float(np.percentile(a, 75)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


def compute_plane_scale(gray: np.ndarray, K_json: str | None, cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Estimate plane scale from a single AprilGrid image.

    Parameters
    ----------
    gray : np.ndarray
        Input grayscale image.
    K_json : str | None
        Optional path to intrinsics JSON for undistortion.
    cfg : Dict[str, Any] | None
        Optional configuration dict with keys:
        ``target_width`` (int), ``pair_weight`` (float), ``no_bias`` (bool).

    Returns
    -------
    Dict[str, Any]
        Contains ``summary`` (public JSON), ``detections`` (list), ``gray`` and
        ``px_per_mm`` tuple for visualization.
    """

    if cfg is None:
        cfg = {}
    target_width = int(cfg.get("target_width", 0))
    pair_weight = float(cfg.get("pair_weight", 0.5))
    no_bias = bool(cfg.get("no_bias", False))

    h, w = gray.shape[:2]

    # Optional undistort
    if K_json:
        data = load_json(K_json)
        if data is None:
            raise FileNotFoundError(K_json)
        model = data.get("model", "pinhole")
        K = np.array(data["K"], dtype=np.float64)
        dist = np.array(data["dist"], dtype=np.float64)
        if model == "fisheye":
            mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
                K, dist, np.eye(3), K, (w, h), cv2.CV_32FC1
            )
            gray = cv2.remap(gray, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        else:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
            gray = cv2.undistort(gray, K, dist, None, newK)

    dets = detect(gray, target_width)
    if not dets:
        raise RuntimeError("No AprilTags detected.")

    # Build centers/widths
    centers, widths = {}, {}
    ids = sorted(int(d["id"]) for d in dets)
    for d in dets:
        tid = int(d["id"])
        c = d["corners_px"].astype(np.float32)
        w_px, h_px = edge_sizes_px(c)
        centers[tid] = (
            float(c.reshape(-1, 2)[:, 0].mean()),
            float(c.reshape(-1, 2)[:, 1].mean()),
        )
        widths[tid] = (w_px, h_px)

    # Edge-based px/cm
    edge_x = [widths[i][0] / (TAG_MM / 10.0) for i in ids]
    edge_y = [widths[i][1] / (TAG_MM / 10.0) for i in ids]

    # Neighbor px/cm from center pitch
    start_id = min(ids)
    dset = set(ids)
    nbr_x, nbr_y = [], []
    for r in range(ROWS):
        for c in range(COLS - 1):
            i1 = start_id + r * COLS + c
            i2 = i1 + 1
            if i1 in dset and i2 in dset:
                dx = abs(centers[i2][0] - centers[i1][0])
                nbr_x.append(dx / (PITCH_MM / 10.0))
    for r in range(ROWS - 1):
        for c in range(COLS):
            i1 = start_id + r * COLS + c
            i2 = i1 + COLS
            if i1 in dset and i2 in dset:
                dy = abs(centers[i2][1] - centers[i1][1])
                nbr_y.append(dy / (PITCH_MM / 10.0))

    ex = robust_median_trim(edge_x)
    ey = robust_median_trim(edge_y)
    nx = robust_median_trim(nbr_x) if nbr_x else np.nan
    ny = robust_median_trim(nbr_y) if nbr_y else np.nan

    w_pair = np.clip(pair_weight, 0.0, 1.0)
    px_per_cm_x = (1 - w_pair) * ex + w_pair * (nx if not np.isnan(nx) else ex)
    px_per_cm_y = (1 - w_pair) * ey + w_pair * (ny if not np.isnan(ny) else ey)

    px_per_mm_x, px_per_mm_y = px_per_cm_x / 10.0, px_per_cm_y / 10.0

    # Optional bias: make median tag edge exactly TAG_MM
    if (not no_bias) and BIAS_ENFORCE and BIAS_BLEND > 0:
        w_mm = [widths[i][0] / px_per_mm_x for i in ids]
        h_mm = [widths[i][1] / px_per_mm_y for i in ids]
        med_w, med_h = float(np.median(w_mm)), float(np.median(h_mm))
        kx = (TAG_MM / med_w) if med_w > 1e-9 else 1.0
        ky = (TAG_MM / med_h) if med_h > 1e-9 else 1.0
        px_per_mm_x *= kx ** BIAS_BLEND
        px_per_mm_y *= ky ** BIAS_BLEND

    # Per-tag edges & gaps
    tag_w_mm = [widths[i][0] / px_per_mm_x for i in ids]
    tag_h_mm = [widths[i][1] / px_per_mm_y for i in ids]

    horiz_center_mm, vert_center_mm = [], []
    horiz_gap_mm, vert_gap_mm = [], []
    for r in range(ROWS):
        for c in range(COLS - 1):
            i1 = start_id + r * COLS + c
            i2 = i1 + 1
            if i1 in dset and i2 in dset:
                dx_mm = abs(centers[i2][0] - centers[i1][0]) / px_per_mm_x
                w1 = widths[i1][0] / px_per_mm_x
                w2 = widths[i2][0] / px_per_mm_x
                gap = dx_mm - 0.5 * w1 - 0.5 * w2
                horiz_center_mm.append(dx_mm)
                horiz_gap_mm.append(gap)
    for r in range(ROWS - 1):
        for c in range(COLS):
            i1 = start_id + r * COLS + c
            i2 = i1 + COLS
            if i1 in dset and i2 in dset:
                dy_mm = abs(centers[i2][1] - centers[i1][1]) / px_per_mm_y
                h1 = widths[i1][1] / px_per_mm_y
                h2 = widths[i2][1] / px_per_mm_y
                gap = dy_mm - 0.5 * h1 - 0.5 * h2
                vert_center_mm.append(dy_mm)
                vert_gap_mm.append(gap)

    summary = {
        "plane_scale": {
            "px_per_mm_x": float(px_per_mm_x),
            "px_per_mm_y": float(px_per_mm_y),
            "mm_per_px_x": float(1.0 / px_per_mm_x),
            "mm_per_px_y": float(1.0 / px_per_mm_y),
        },
        "length_summary": {
            "edge_x_mm_stats": _stats(tag_w_mm),
            "edge_y_mm_stats": _stats(tag_h_mm),
            "expected_tag_mm": TAG_MM,
        },
        "gap_summary": {
            "horiz_center_pitch_stats": _stats(horiz_center_mm),
            "vert_center_pitch_stats": _stats(vert_center_mm),
            "horiz_edge_to_edge_gap_stats": _stats(horiz_gap_mm),
            "vert_edge_to_edge_gap_stats": _stats(vert_gap_mm),
            "expected": {"gap_mm": GAP_MM, "pitch_mm": PITCH_MM},
        },
        "detected_ids": ids,
    }

    return {
        "summary": summary,
        "detections": dets,
        "gray": gray,
        "px_per_mm": (px_per_mm_x, px_per_mm_y),
    }

