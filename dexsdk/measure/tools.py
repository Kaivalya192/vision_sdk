# d exsdk/measure/tools.py
from __future__ import annotations
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2
import math

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
        xs = np.clip(ray[:, 0], 0, w - 1).astype(np.float32)
        ys = np.clip(ray[:, 1], 0, h - 1).astype(np.float32)
        prof = cv2.remap(
            gray,
            xs.reshape(-1, 1),
            ys.reshape(-1, 1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

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
        xs = np.clip(cx + radii * dx, 0, w - 1).astype(np.float32)
        ys = np.clip(cy + radii * dy, 0, h - 1).astype(np.float32)
        prof = cv2.remap(
            gray_roi,
            xs.reshape(-1, 1),
            ys.reshape(-1, 1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).flatten().astype(np.float32)
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


def _warp_strip(gray: np.ndarray, roi, debug: Dict[str, Any]):
    """
    Extract an upright strip of shape (thickness, length) centered at roi.(cx,cy),
    aligned so that the strip's X axis follows the ROI line direction.
    Returns (strip, M, Minv) where:
      - strip: float32 image [thickness x length]
      - M: 2x3 affine that maps strip coords -> image coords
      - Minv: inverse for back-projection image -> strip
    """
    cx, cy = roi.cx, roi.cy
    L, T = float(roi.length), float(roi.thickness)
    a = math.radians(roi.angle_deg)

    # Strip frame basis: x axis = line direction, y axis = normal
    ux = np.array([ math.cos(a), math.sin(a) ], np.float32)
    uy = np.array([-math.sin(a), math.cos(a) ], np.float32)

    # Strip corners in image coords (centered around (cx,cy))
    # We build an affine that maps strip (0..L, 0..T) -> image
    strip_center = np.array([cx, cy], np.float32)
    origin_img = strip_center - 0.5*L*ux - 0.5*T*uy
    x_axis_img = origin_img + ux * L
    y_axis_img = origin_img + uy * T

    src = np.float32([[0,0],[L,0],[0,T]])
    dst = np.float32([origin_img, x_axis_img, y_axis_img])

    M = cv2.getAffineTransform(src, dst)
    Minv = cv2.invertAffineTransform(M)

    strip = cv2.warpAffine(gray, Minv, (int(round(L)), int(round(T))),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    debug["strip_frame"] = {"origin": origin_img, "x_end": x_axis_img, "y_end": y_axis_img}
    return strip, M, Minv

def _edges_from_strip(strip: np.ndarray, params: Dict[str, Any]):
    """
    Scan along the strip normal → build 1D profile by averaging across thickness.
    Find edge locations (subpixel) along length axis.
    """
    # Average over thickness dimension → 1D profile of length L
    profile = strip.mean(axis=0)  # shape [L]
    # Optional prefilter
    ksize = int(params.get("smooth", 3))
    if ksize >= 3 and (ksize % 2 == 1):
        profile = cv2.GaussianBlur(profile.reshape(1,-1), (ksize,1), 0).ravel()

    # Polarity: "any"|"rise"|"fall"
    polarity = params.get("polarity", "any")
    grad = np.gradient(profile)
    if polarity == "rise":
        grad_raw = grad
    elif polarity == "fall":
        grad_raw = -grad
    else:
        grad_raw = np.abs(grad)

    # Peak picking with min contrast
    min_contrast = float(params.get("min_contrast", 5.0))
    # naive non-maximum suppression
    candidates = []
    for i in range(1, len(grad_raw)-1):
        if grad_raw[i] > grad_raw[i-1] and grad_raw[i] > grad_raw[i+1] and abs(grad[i]) >= min_contrast:
            # refine subpixel around i using parabola fit on grad (not abs)
            idx_f = subpix_edge_1d(profile) if params.get("use_global_peak") else _parabola_refine(grad, i)
            strength = float(abs(grad[int(round(idx_f))])) if 0 <= int(round(idx_f)) < len(grad) else float(abs(grad[i]))
            candidates.append((idx_f, strength))

    # Sort by strength and keep top N
    max_edges = int(params.get("max_edges", 2))
    candidates.sort(key=lambda t: t[1], reverse=True)
    picks = candidates[:max_edges]

    return profile, grad, picks

def _parabola_refine(y: np.ndarray, i: int) -> float:
    # classic 3-point quadratic vertex refinement
    if i <= 0 or i >= len(y)-1: 
        return float(i)
    y1,y2,y3 = y[i-1], y[i], y[i+1]
    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        return float(i)
    offset = 0.5*(y1 - y3)/denom
    return float(i + offset)

def tool_line_caliper(gray: np.ndarray, roi, params: Dict[str, Any]):
    """
    Returns:
      dict with keys:
       - edges_img: list[(x,y,strength)]   # in image coords
       - profile: np.ndarray               # 1D profile
       - grad: np.ndarray
       - strip_box: ((x0,y0),(x1,y1),(x2,y2)) for drawing
       - ok: bool
       - msg: str
    """
    debug = {}
    strip, M, Minv = _warp_strip(gray, roi, debug)
    profile, grad, picks = _edges_from_strip(strip, params)

    # Back-project edges to image coords
    edges_img = []
    for idx_f, strength in picks:
        pt_strip = np.array([[idx_f, strip.shape[0]*0.5, 1.0]], np.float32).T  # center across thickness
        x = M[0,0]*pt_strip[0]+M[0,1]*pt_strip[1]+M[0,2]
        y = M[1,0]*pt_strip[0]+M[1,1]*pt_strip[1]+M[1,2]
        edges_img.append((float(x), float(y), strength))

    # Basic GO/NO-GO (example: need >= 1 edge)
    need_edges = int(params.get("need_edges", 1))
    ok = len(edges_img) >= need_edges
    msg = f"found {len(edges_img)} edges; need ≥ {need_edges}"

    return {
        "edges_img": edges_img,
        "profile": profile,
        "grad": grad,
        "strip_box": (debug["strip_frame"]["origin"],
                      debug["strip_frame"]["x_end"],
                      debug["strip_frame"]["y_end"]),
        "ok": ok,
        "msg": msg
    }
