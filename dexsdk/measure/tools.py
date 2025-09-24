# d exsdk/measure/tools.py
from __future__ import annotations
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2

from .schema import CaliperGuide, AngleBetween, CircleParams, Measurement, Packet, ROI
from ._subpix import subpix_edge_1d  # already in your repo (kept)
from .geometry import fit_line_total_least_squares, line_angle_deg, fit_circle_taubin  # keep or add if present


def _clip_roi(img: np.ndarray, roi: ROI) -> np.ndarray:
    H, W = img.shape[:2]
    x = max(0, min(roi.x, W))
    y = max(0, min(roi.y, H))
    w = max(0, min(roi.w, W - x))
    h = max(0, min(roi.h, H - y))
    return img[y:y+h, x:x+w].copy()


def _sample_along(normalized_t: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    # linear interpolate along guide
    return (1.0 - normalized_t)[:, None] * p0[None, :] + normalized_t[:, None] * p1[None, :]


def _extract_caliper_points(gray: np.ndarray, guide: CaliperGuide) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pts (N,2): subpixel edge points (x,y) in ROI coordinates
      good_mask (N,): boolean
    """
    h, w = gray.shape[:2]

    p0 = np.array(guide.p0, dtype=np.float32)
    p1 = np.array(guide.p1, dtype=np.float32)
    v = p1 - p0
    L = np.linalg.norm(v) + 1e-6
    t_scan = np.linspace(0.05, 0.95, guide.n_scans).astype(np.float32)
    centers = _sample_along(t_scan, p0, p1)

    # unit normal (perpendicular) to guide
    n = np.array([-v[1], v[0]], dtype=np.float32) / L
    half = max(1, int(guide.band_px // 2))

    pts = []
    ok = []
    for c in centers:
        a = c - n * half
        b = c + n * half
        # sample intensity profile between a..b
        ts = np.linspace(0.0, 1.0, guide.samples_per_scan).astype(np.float32)
        ray = _sample_along(ts, a, b)
        # bilinear sample
        xs = np.clip(ray[:, 0], 0, w - 1)
        ys = np.clip(ray[:, 1], 0, h - 1)
        prof = cv2.remap(gray, xs.reshape(-1, 1), ys.reshape(-1, 1),
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        prof = prof.flatten().astype(np.float32)

        # subpixel edge along 1D profile
        edge_pos, edge_score, edge_sign = subpix_edge_1d(prof, polarity=guide.polarity)
        if edge_pos is None or edge_score < guide.min_contrast:
            ok.append(False); pts.append([np.nan, np.nan]); continue

        # map back to XY
        pt = a + (b - a) * float(edge_pos) / (guide.samples_per_scan - 1)
        pts.append([float(pt[0]), float(pt[1])]); ok.append(True)

    pts = np.array(pts, dtype=np.float32)
    ok = np.array(ok, dtype=bool)
    return pts, ok


def tool_line_caliper(gray_roi: np.ndarray, guide: CaliperGuide) -> Tuple[Measurement, Dict[str, Any]]:
    pts, mask = _extract_caliper_points(gray_roi, guide)
    good = pts[mask]
    if len(good) < 5:
        return Measurement(id="line_caliper", kind="line_angle_deg", value=float("nan"), sigma=float("inf")), {"pts": pts, "ok": mask}

    # TLS line fit: ax + by + c = 0  (a^2 + b^2 = 1)
    line = fit_line_total_least_squares(good)
    ang = line_angle_deg(line)  # 0..180
    # residuals: distance to line
    a, b, c = line
    d = np.abs(a * good[:, 0] + b * good[:, 1] + c)
    sigma = float(np.std(d)) if len(d) > 1 else 0.0
    return Measurement(id="line", kind="angle_deg", value=float(ang), sigma=sigma), {"pts": pts, "ok": mask, "line": line}


def tool_edge_pair_width(gray_roi: np.ndarray, gA: CaliperGuide, gB: CaliperGuide) -> Tuple[Measurement, Dict[str, Any]]:
    # extract points for each guide, then fit a line to each and compute shortest distance between them
    mA, dbgA = tool_line_caliper(gray_roi, gA)
    mB, dbgB = tool_line_caliper(gray_roi, gB)
    lineA = dbgA.get("line"); lineB = dbgB.get("line")

    if lineA is None or lineB is None or np.any(np.isnan(lineA)) or np.any(np.isnan(lineB)):
        return Measurement(id="width", kind="px", value=float("nan"), sigma=float("inf")), {"A": dbgA, "B": dbgB}

    # distance between (approximately parallel) lines: |c2 - c1| / sqrt(a^2 + b^2)
    a1, b1, c1 = lineA; a2, b2, c2 = lineB
    denom = np.sqrt(a1*a1 + b1*b1) + 1e-9
    width = abs(c2 - c1) / denom
    # rough sigma from residuals
    sig = 0.5 * (mA.sigma + mB.sigma)
    return Measurement(id="width", kind="px", value=float(width), sigma=float(sig)), {"A": dbgA, "B": dbgB}


def tool_angle_between(gray_roi: np.ndarray, ab: AngleBetween) -> Tuple[Measurement, Dict[str, Any]]:
    m1, d1 = tool_line_caliper(gray_roi, ab.g1)
    m2, d2 = tool_line_caliper(gray_roi, ab.g2)
    if np.isnan(m1.value) or np.isnan(m2.value):
        return Measurement(id="angle", kind="deg", value=float("nan"), sigma=float("inf")), {"g1": d1, "g2": d2}

    # angle between two (a,b) normals (or simply difference of angles)
    ang = abs(m1.value - m2.value)
    if ang > 90.0:
        ang = 180.0 - ang
    sig = 0.5 * (m1.sigma + m2.sigma)
    return Measurement(id="angle", kind="deg", value=float(ang), sigma=float(sig)), {"g1": d1, "g2": d2}


def tool_circle_diameter(gray_roi: np.ndarray, prm: CircleParams) -> Tuple[Measurement, Dict[str, Any]]:
    h, w = gray_roi.shape[:2]
    cx = float(np.clip(prm.cx, 0, w-1))
    cy = float(np.clip(prm.cy, 0, h-1))
    rays = []
    radii = np.linspace(prm.r_min, prm.r_max, prm.samples_per_ray).astype(np.float32)

    pts = []
    oks = []
    for k in range(prm.n_rays):
        ang = 2.0*np.pi * (k / prm.n_rays)
        dx, dy = np.cos(ang), np.sin(ang)
        xs = np.clip(cx + radii * dx, 0, w - 1)
        ys = np.clip(cy + radii * dy, 0, h - 1)
        prof = cv2.remap(gray_roi, xs.reshape(-1, 1), ys.reshape(-1, 1),
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).flatten().astype(np.float32)
        pos, score, sign = subpix_edge_1d(prof, polarity=prm.polarity)
        if pos is None or score < prm.min_contrast:
            oks.append(False); pts.append([np.nan, np.nan]); continue
        r = float(radii[int(round(pos))])
        pts.append([cx + r*dx, cy + r*dy]); oks.append(True)

    pts = np.asarray(pts, dtype=np.float32)
    mask = np.asarray(oks, dtype=bool)
    good = pts[mask]
    if len(good) < 8:
        return Measurement(id="diameter", kind="px", value=float("nan"), sigma=float("inf")), {"pts": pts, "ok": mask}

    (xc, yc, R), rmse = fit_circle_taubin(good)
    return Measurement(id="diameter", kind="px", value=float(2.0*R), sigma=float(rmse)), {"pts": pts, "ok": mask, "circle": (xc, yc, R)}
