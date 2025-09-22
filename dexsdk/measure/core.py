# dexsdk/measure/core.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import cv2
import numpy as np

# NOTE: We rely on tools you’ll add in dexsdk/measure/tools.py
from .tools import (
    caliper_line,
    distance_p2p as _distance_p2p,
    distance_point_to_line as _distance_p2l,
    angle_between_lines as _angle_l2l,
)

# ------------------------------
# Helpers
# ------------------------------

def _to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _crop_roi(img, roi):
    x, y, w, h = [int(v) for v in roi]
    H, W = img.shape[:2]
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w))
    h = max(1, min(H - y, h))
    return img[y : y + h, x : x + w].copy(), (x, y)

def _px_to_mm(val_px, mm_per_px_xy: Tuple[float, float]):
    # For scalar distances, use isotropic avg if pixels are near square.
    sx, sy = mm_per_px_xy
    s = 0.5 * (sx + sy)
    return float(val_px) * float(s)

def _draw_text(img, txt, org, color=(0,255,255), scale=0.6, thick=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def _draw_line_param(img, p, d, color=(0, 255, 0), length=1000, thickness=2):
    p = np.asarray(p, np.float32)
    d = np.asarray(d, np.float32)
    a = (p - d * length).astype(int)
    b = (p + d * length).astype(int)
    cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)

def _judgement(value_mm: float, ok_range_mm: Optional[Tuple[float, float]]):
    if not ok_range_mm:
        return None, None
    lo, hi = ok_range_mm
    ok = (value_mm >= float(lo)) and (value_mm <= float(hi))
    return ok, (float(lo), float(hi))

# ------------------------------
# Public entry
# ------------------------------

def run_job(frame_bgr: np.ndarray, job: Dict[str, Any], mm_per_px_xy: Tuple[float, float], units_label: str = "mm"):
    """
    job = {
      "tool": "line_caliper" | "distance_p2p" | "distance_p2l" | "angle_l2l" | "point_pick",
      "roi": [x,y,w,h],
      "params": {...}
    }
    Returns: (packet, overlay_bgr)
    packet = {
      "measures": [ {id, kind, value, units, pass?, sigma?} ],
      "units": units_label
    }
    """
    tool = str(job.get("tool", "")).lower().strip()
    roi  = job.get("roi", None)
    params = dict(job.get("params", {}) or {})
    if roi is None or not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("run_job: ROI [x,y,w,h] required")

    overlay = frame_bgr.copy()
    roi_img, (rx, ry) = _crop_roi(frame_bgr, roi)
    roi_gray = _to_gray(roi_img)

    measures = []

    if tool == "line_caliper" or tool == "line_fit":
        # Expected params (with sensible defaults)
        p0 = params.get("p0", [10, roi_img.shape[0] * 0.5])
        p1 = params.get("p1", [roi_img.shape[1] - 10, roi_img.shape[0] * 0.5])
        band_px = int(params.get("band_px", 24))
        n_scans = int(params.get("n_scans", 32))
        samples_per_scan = int(params.get("samples_per_scan", 64))
        polarity = str(params.get("polarity", "any")).lower()
        min_contrast = float(params.get("min_contrast", 8.0))
        ok_range_mm = params.get("ok_range_mm", None)  # for width later if you use two edges

        res = caliper_line(
            roi_gray, p0, p1,
            band_px=band_px, n_scans=n_scans, samples_per_scan=samples_per_scan,
            polarity=polarity, min_contrast=min_contrast
        )
        # Draw ROI rectangle
        cv2.rectangle(overlay, (rx, ry), (rx + roi_img.shape[1], ry + roi_img.shape[0]), (100, 100, 100), 1, cv2.LINE_AA)

        if not res.get("ok", False):
            _draw_text(overlay, f"Caliper: {res.get('msg','fail')}", (rx + 6, ry + 22), (0, 0, 255))
            packet = {"measures": [], "units": units_label}
            return packet, overlay

        # Draw sub-pixel edge points
        pts = res["points"]
        for q in pts:
            qx, qy = int(round(q[0])) + rx, int(round(q[1])) + ry
            cv2.circle(overlay, (qx, qy), 2, (0, 255, 255), -1, cv2.LINE_AA)

        # Draw best-fit line
        line = res["line"]
        p_global = (np.array(line["point"], np.float32) + np.array([rx, ry], np.float32)).tolist()
        _draw_line_param(overlay, p_global, line["dir"], color=(0, 255, 0), thickness=2)

        # Report line (no scalar value unless user asks angle/offset/width)
        measures.append({
            "id": "L1",
            "kind": "line_fit",
            "value": 0.0,  # placeholder
            "units": units_label,
            "pass": None,
            "sigma": None
        })

        packet = {"measures": measures, "units": units_label, "line": line}
        return packet, overlay

    elif tool == "distance_p2p":
        p1 = params.get("p1", [roi_img.shape[1]*0.25, roi_img.shape[0]*0.5])
        p2 = params.get("p2", [roi_img.shape[1]*0.75, roi_img.shape[0]*0.5])
        ok_range_mm = params.get("ok_range_mm", None)

        # draw points in overlay
        P1 = (int(round(p1[0])) + rx, int(round(p1[1])) + ry)
        P2 = (int(round(p2[0])) + rx, int(round(p2[1])) + ry)
        cv2.circle(overlay, P1, 4, (255, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, P2, 4, (255, 200, 0), -1, cv2.LINE_AA)
        cv2.line(overlay, P1, P2, (255, 200, 0), 2, cv2.LINE_AA)

        dist_px = _distance_p2p(p1, p2)
        dist_mm = _px_to_mm(dist_px, mm_per_px_xy)
        ok, rng = _judgement(dist_mm, ok_range_mm)
        col = (0, 200, 0) if (ok is None or ok) else (0, 0, 255)
        _draw_text(overlay, f"{dist_mm:.3f} {units_label}", (min(P1[0], P2[0]) + 6, min(P1[1], P2[1]) - 8), col)

        measures.append({
            "id": "D_P2P",
            "kind": "distance_p2p",
            "value": float(dist_mm),
            "units": units_label,
            "pass": ok,
            "sigma": None
        })
        return {"measures": measures, "units": units_label}, overlay

    elif tool == "distance_p2l":
        # For P→L you must provide a prior fitted line or a guide (p0,p1) to fit inside ROI
        pt = params.get("pt", [roi_img.shape[1]*0.5, roi_img.shape[0]*0.5])
        ok_range_mm = params.get("ok_range_mm", None)

        if "line" in params:
            line = params["line"]  # {"point":[x,y], "dir":[dx,dy], "normal":[nx,ny]} IN ROI coordinates
        else:
            p0 = params.get("p0", [10, roi_img.shape[0] * 0.5])
            p1 = params.get("p1", [roi_img.shape[1] - 10, roi_img.shape[0] * 0.5])
            cal = caliper_line(_to_gray(roi_img), p0, p1, band_px=int(params.get("band_px", 24)))
            if not cal.get("ok", False):
                packet = {"measures": [], "units": units_label}
                _draw_text(overlay, "P→L: line fit failed", (rx + 6, ry + 22), (0, 0, 255))
                return packet, overlay
            line = cal["line"]

        # overlay
        P = (int(round(pt[0])) + rx, int(round(pt[1])) + ry)
        cv2.circle(overlay, P, 4, (0, 200, 255), -1, cv2.LINE_AA)
        _draw_line_param(overlay, (np.array(line["point"]) + [rx, ry]).tolist(), line["dir"], (0, 255, 0), 2)

        d_px = _distance_p2l(pt, line)
        d_mm = _px_to_mm(d_px, mm_per_px_xy)
        ok, rng = _judgement(d_mm, ok_range_mm)
        col = (0, 200, 0) if (ok is None or ok) else (0, 0, 255)
        _draw_text(overlay, f"{d_mm:.3f} {units_label}", (P[0] + 6, P[1] - 8), col)

        measures.append({
            "id": "D_P2L",
            "kind": "distance_p2l",
            "value": float(d_mm),
            "units": units_label,
            "pass": ok,
            "sigma": None
        })
        return {"measures": measures, "units": units_label}, overlay

    elif tool == "angle_l2l":
        # Accept two fitted lines in params, or fit two with guides
        def _get_line(tag: str):
            if f"{tag}_line" in params:
                return params[f"{tag}_line"]
            p0 = params.get(f"{tag}_p0")
            p1 = params.get(f"{tag}_p1")
            if p0 is None or p1 is None:
                return None
            cal = caliper_line(roi_gray, p0, p1, band_px=int(params.get("band_px", 24)))
            return cal["line"] if cal.get("ok", False) else None

        L1 = _get_line("a")
        L2 = _get_line("b")
        if L1 is None or L2 is None:
            packet = {"measures": [], "units": units_label}
            _draw_text(overlay, "Angle: need two lines", (rx + 6, ry + 22), (0, 0, 255))
            return packet, overlay

        # overlay
        _draw_line_param(overlay, (np.array(L1["point"]) + [rx, ry]).tolist(), L1["dir"], (0, 255, 0), 2)
        _draw_line_param(overlay, (np.array(L2["point"]) + [rx, ry]).tolist(), L2["dir"], (0, 200, 255), 2)

        ang_deg = _angle_l2l(L1, L2)
        _draw_text(overlay, f"{ang_deg:.3f} deg", (rx + 6, ry + 22), (0, 255, 255))

        measures.append({
            "id": "ANG_L2L",
            "kind": "angle_l2l",
            "value": float(ang_deg),
            "units": "deg",
            "pass": None,
            "sigma": None
        })
        return {"measures": measures, "units": units_label}, overlay

    elif tool == "point_pick":
        # Robust point pick with adaptive window and empty-checks
        hint = params.get("hint_xy", [roi_img.shape[1]*0.5, roi_img.shape[0]*0.5])
        hx, hy = int(round(hint[0])), int(round(hint[1]))
        # Use requested radius but clamp to ROI size
        req_r = int(params.get("win_radius", 8))
        max_r = max(1, min(roi_img.shape[1] // 2 - 1, roi_img.shape[0] // 2 - 1))
        r = max(1, min(req_r, max_r))

        # If ROI is extremely small, fall back to center point display and exit
        if roi_img.shape[0] < 3 or roi_img.shape[1] < 3 or max_r < 1:
            P = (int(round(hx)) + rx, int(round(hy)) + ry)
            cv2.circle(overlay, P, 3, (0, 255, 255), -1, cv2.LINE_AA)
            _draw_text(overlay, "point_pick: ROI too small", (rx + 6, ry + 22), (0, 0, 255))
            return {"measures": [], "units": units_label}, overlay

        x1, y1 = max(0, hx - r), max(0, hy - r)
        x2, y2 = min(roi_img.shape[1], hx + r + 1), min(roi_img.shape[0], hy + r + 1)

        # Guard: ensure non-empty window
        if x2 <= x1 or y2 <= y1:
            # Try shrinking radius once
            r = max(1, min(r - 1, max_r))
            x1, y1 = max(0, hx - r), max(0, hy - r)
            x2, y2 = min(roi_img.shape[1], hx + r + 1), min(roi_img.shape[0], hy + r + 1)

        if x2 <= x1 or y2 <= y1:
            P = (int(round(hx)) + rx, int(round(hy)) + ry)
            cv2.circle(overlay, P, 3, (0, 255, 255), -1, cv2.LINE_AA)
            _draw_text(overlay, "point_pick: invalid window", (rx + 6, ry + 22), (0, 0, 255))
            return {"measures": [], "units": units_label}, overlay

        win = roi_gray[y1:y2, x1:x2]
        if win.size == 0:
            P = (int(round(hx)) + rx, int(round(hy)) + ry)
            cv2.circle(overlay, P, 3, (0, 255, 255), -1, cv2.LINE_AA)
            _draw_text(overlay, "point_pick: empty window", (rx + 6, ry + 22), (0, 0, 255))
            return {"measures": [], "units": units_label}, overlay

        gx = cv2.Sobel(win, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        m_sum = float(mag.sum())
        if m_sum <= 1e-9:
            P = (int(round(hx)) + rx, int(round(hy)) + ry)
            cv2.circle(overlay, P, 3, (0, 255, 255), -1, cv2.LINE_AA)
            _draw_text(overlay, "point_pick: low gradient", (rx + 6, ry + 22), (0, 0, 255))
            return {"measures": [], "units": units_label}, overlay

        ys, xs = np.mgrid[y1:y2, x1:x2]
        cx = (float((mag * xs).sum()) / m_sum)
        cy = (float((mag * ys).sum()) / m_sum)
        P = (int(round(cx)) + rx, int(round(cy)) + ry)
        cv2.circle(overlay, P, 4, (0, 255, 255), -1, cv2.LINE_AA)

        measures.append({
            "id": "P_SUBPIX",
            "kind": "point_pick",
            "value": 0.0,
            "units": units_label,
            "pass": None,
            "sigma": None
        })
        return {"measures": measures, "units": units_label, "point": [cx, cy]}, overlay

    else:
        # Unknown tool
        packet = {"measures": [], "units": units_label}
        _draw_text(overlay, f"Unknown tool: {tool}", (10, 24), (0, 0, 255))
        return packet, overlay
