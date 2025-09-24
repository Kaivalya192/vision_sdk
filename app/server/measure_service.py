# app/server/measure_service.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

from dexsdk.measure.tools import (
    tool_line_caliper,
    tool_edge_pair_width,
    tool_angle_between,
    tool_circle_diameter,
)
from dexsdk.measure.schema import CaliperGuide, AngleBetween, CircleParams


def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _clip_roi_wh(frame: np.ndarray, roi: List[int]) -> Tuple[List[int], np.ndarray]:
    H, W = frame.shape[:2]
    if roi is None:
        return [0, 0, W, H], frame.copy()
    x, y, w, h = map(int, roi)
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    x = max(0, min(x, W))
    y = max(0, min(y, H))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    return [x, y, w, h], frame[y:y + h, x:x + w].copy()


def _draw_caliper_debug(overlay: np.ndarray, roi_xywh: List[int], dbg: Dict[str, Any], color_line=(0, 255, 255)):
    x0, y0, w, h = roi_xywh
    pts = dbg.get("pts")
    ok = dbg.get("ok")
    line = dbg.get("line")

    if isinstance(pts, np.ndarray) and isinstance(ok, np.ndarray):
        for i, p in enumerate(pts):
            if not ok[i] or not np.isfinite(p).all():
                continue
            cx = int(round(x0 + p[0]))
            cy = int(round(y0 + p[1]))
            cv2.circle(overlay, (cx, cy), 2, (0, 180, 255), -1)

    if line is not None and np.all(np.isfinite(line)):
        a, b, c = line  # ax + by + c = 0
        if abs(b) > 1e-9:
            xa = x0
            ya = int(round(-(a * xa + c) / b))
            xb = x0 + w
            yb = int(round(-(a * xb + c) / b))
            cv2.line(overlay, (xa, ya), (xb, yb), color_line, 2, cv2.LINE_AA)
        else:  # vertical
            xv = int(round(-c / a))
            cv2.line(overlay, (xv, y0), (xv, y0 + h), color_line, 2, cv2.LINE_AA)


def _bgr_overlay(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        base = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        base = frame.copy()
    return base


class MeasureService:
    """
    Minimal measurement service wrapper that exposes run_job(frame_bgr, job_dict)
    and returns (packet_dict, overlay_bgr_or_None).
    """
    def __init__(self, get_mm_scale=None) -> None:
        # server may pass _get_mm_scale() -> {"px_per_mm_x":..., "px_per_mm_y":...}
        self._get_mm_scale = get_mm_scale
        self.anchor = AnchorHelper()  # compatibility shim

    def run_job(self, frame_bgr: np.ndarray, job: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        tool = (job.get("tool") or "").lower()
        params = job.get("params", {}) or {}
        roi_in = job.get("roi") or [0, 0, frame_bgr.shape[1], frame_bgr.shape[0]]

        roi_xywh = self.anchor.apply(roi_in, detection=None)
        roi_xywh, roi_view_bgr = _clip_roi_wh(frame_bgr, roi_xywh)
        roi_gray = _to_gray(roi_view_bgr)

        if tool == "line_caliper":
            meas, dbg = self._run_line_caliper(roi_gray, params)
            overlay = _bgr_overlay(frame_bgr)
            _draw_caliper_debug(overlay, roi_xywh, dbg, color_line=(0, 255, 0))
            packet = self._packet_from_measure(meas, default_units="px")
            return packet, overlay

        elif tool == "edge_pair_width":
            meas, dbg = self._run_edge_pair_width(roi_gray, params)
            overlay = _bgr_overlay(frame_bgr)
            if isinstance(dbg, dict):
                if "A" in dbg:
                    _draw_caliper_debug(overlay, roi_xywh, dbg["A"], color_line=(255, 200, 0))
                if "B" in dbg:
                    _draw_caliper_debug(overlay, roi_xywh, dbg["B"], color_line=(0, 255, 255))
            packet = self._packet_from_measure(meas, default_units="px")
            return packet, overlay

        elif tool == "angle_between":
            meas, dbg = self._run_angle_between(roi_gray, params)
            overlay = _bgr_overlay(frame_bgr)
            if isinstance(dbg, dict):
                if "g1" in dbg:
                    _draw_caliper_debug(overlay, roi_xywh, dbg["g1"], color_line=(0, 220, 255))
                if "g2" in dbg:
                    _draw_caliper_debug(overlay, roi_xywh, dbg["g2"], color_line=(255, 0, 180))
            packet = self._packet_from_measure(meas, default_units="deg")
            return packet, overlay

        elif tool == "circle_diameter":
            meas, dbg = self._run_circle_diameter(roi_gray, params)
            overlay = _bgr_overlay(frame_bgr)
            x0, y0, w, h = roi_xywh
            pts = dbg.get("pts")
            ok = dbg.get("ok")
            circ = dbg.get("circle")
            if isinstance(pts, np.ndarray) and isinstance(ok, np.ndarray):
                for i, p in enumerate(pts):
                    if not ok[i] or not np.isfinite(p).all():
                        continue
                    cx = int(round(x0 + p[0])); cy = int(round(y0 + p[1]))
                    cv2.circle(overlay, (cx, cy), 2, (200, 255, 0), -1)
            if circ is not None and np.all(np.isfinite(circ)):
                xc, yc, R = circ
                cv2.circle(overlay, (int(round(x0 + xc)), int(round(y0 + yc))), int(round(R)), (0, 255, 0), 2, cv2.LINE_AA)
            packet = self._packet_from_measure(meas, default_units="px")
            return packet, overlay

        else:
            packet = {
                "measures": [{
                    "id": str(tool or "unknown"),
                    "kind": "unknown",
                    "value": float("nan"),
                    "sigma": float("inf"),
                    "pass": None
                }],
                "units": "px"
            }
            return packet, None

    # ---- tool runners ----
    def _run_line_caliper(self, gray_roi: np.ndarray, params: Dict[str, Any]):
        g = _parse_caliper(params, gray_roi.shape)
        return tool_line_caliper(gray_roi, g)

    def _run_edge_pair_width(self, gray_roi: np.ndarray, params: Dict[str, Any]):
        gA = _parse_caliper(params.get("gA", {}), gray_roi.shape)
        gB = _parse_caliper(params.get("gB", {}), gray_roi.shape)
        return tool_edge_pair_width(gray_roi, gA, gB)

    def _run_angle_between(self, gray_roi: np.ndarray, params: Dict[str, Any]):
        g1 = _parse_caliper(params.get("g1", {}), gray_roi.shape)
        g2 = _parse_caliper(params.get("g2", {}), gray_roi.shape)
        ab = AngleBetween(g1=g1, g2=g2)
        return tool_angle_between(gray_roi, ab)

    def _run_circle_diameter(self, gray_roi: np.ndarray, params: Dict[str, Any]):
        cp = _parse_circle_params(params, gray_roi.shape)
        return tool_circle_diameter(gray_roi, cp)

    # ---- packet / units helpers ----
    def _packet_from_measure(self, m, default_units: str = "px") -> Dict[str, Any]:
        val = float(getattr(m, "value", float("nan")))
        sig = float(getattr(m, "sigma", float("nan")))
        units = default_units

        # Optional px->mm conversion if scale is available and default is px
        if default_units == "px" and callable(self._get_mm_scale):
            try:
                ps = self._get_mm_scale() or {}
                pxmmx = float(ps.get("px_per_mm_x", 0.0))
                pxmmy = float(ps.get("px_per_mm_y", 0.0))
                if pxmmx > 0 and pxmmy > 0:
                    px_per_mm = 0.5 * (pxmmx + pxmmy)
                    val /= px_per_mm
                    sig /= px_per_mm
                    units = "mm"
            except Exception:
                pass

        return {
            "measures": [{
                "id": getattr(m, "id", "m"),
                "kind": getattr(m, "kind", ""),
                "value": val,
                "sigma": sig,
                "pass": getattr(m, "pass_", None) if hasattr(m, "pass_") else getattr(m, "pass", None),
            }],
            "units": units,
        }


# ---------- parsing helpers ----------

def _parse_caliper(p: Dict[str, Any], roi_shape: Tuple[int, int]) -> CaliperGuide:
    h, w = roi_shape[:2]
    p0 = p.get("p0", [10.0, h * 0.5])
    p1 = p.get("p1", [max(10.0, w - 10.0), h * 0.5])
    band_px = int(p.get("band_px", 24))
    n_scans = int(p.get("n_scans", 32))
    samples = int(p.get("samples_per_scan", 64))
    polarity = str(p.get("polarity", "any"))
    min_con = float(p.get("min_contrast", 8.0))
    return CaliperGuide(
        p0=(float(p0[0]), float(p0[1])),
        p1=(float(p1[0]), float(p1[1])),
        band_px=band_px,
        n_scans=n_scans,
        samples_per_scan=samples,
        polarity=polarity,
        min_contrast=min_con
    )


def _parse_circle_params(p: Dict[str, Any], roi_shape: Tuple[int, int]) -> CircleParams:
    h, w = roi_shape[:2]
    cx = float(p.get("cx", w * 0.5))
    cy = float(p.get("cy", h * 0.5))
    rmin = float(p.get("r_min", min(w, h) * 0.15))
    rmax = float(p.get("r_max", min(w, h) * 0.45))
    n_rays = int(p.get("n_rays", 48))
    spp = int(p.get("samples_per_ray", 48))
    polarity = str(p.get("polarity", "any"))
    min_con = float(p.get("min_contrast", 6.0))
    return CircleParams(
        cx=cx, cy=cy,
        r_min=rmin, r_max=rmax,
        n_rays=n_rays, samples_per_ray=spp,
        polarity=polarity, min_contrast=min_con
    )


# --- AnchorHelper shim (compatibility) ---------------------------------------
class AnchorHelper:
    """Compatibility shim. Tracks anchor source + enabled flag and
    exposes an apply() that currently returns the ROI unchanged."""
    def __init__(self) -> None:
        self._enabled: bool = False
        self._source: str | None = None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def set_source(self, name: str | None) -> None:
        self._source = name if isinstance(name, str) and name.strip() else None

    def get_source(self) -> str | None:
        return self._source

    def is_enabled(self) -> bool:
        return self._enabled

    def apply(self, roi: List[int] | Tuple[int, int, int, int],
              detection: Dict[str, Any] | None = None) -> List[int]:
        if roi is None:
            return [0, 0, 0, 0]
        x, y, w, h = map(int, roi)
        if w < 0: x, w = x + w, -w
        if h < 0: y, h = y + h, -h
        return [x, y, max(0, w), max(0, h)]
