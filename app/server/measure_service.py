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
from dexsdk.measure.schema import LineROI, CaliperGuide, AngleBetween, CircleParams

def _normalize_measure_output(res):
    """
    Accepts results from dexsdk tools in any of these shapes and normalizes to (measure, debug):
      - measure
      - (measure, debug)
      - (measure, debug, extra...)
      - object with attribute .debug
    """
    # Tuple / list returns
    if isinstance(res, (tuple, list)):
        if len(res) == 0:
            return None, {}
        if len(res) == 1:
            m = res[0]
            dbg = getattr(m, "debug", {}) if m is not None else {}
            return m, dbg
        # len >= 2
        m = res[0]
        dbg = res[1] if res[1] is not None else {}
        return m, dbg
    # Single object return
    m = res
    dbg = getattr(res, "debug", {}) if res is not None else {}
    return m, dbg

def _call_tool_optional_params(fn, *args, **kwargs):
    """
    Call dexsdk tool functions that may or may not require a trailing
    'params' argument. Falls back to passing an empty dict, and also
    tries keyword 'params' if required.
    """
    try:
        return fn(*args, **kwargs)
    except TypeError as e:
        if "params" in str(e):
            try:
                return fn(*args, *(), {**kwargs.get("params", {})})
            except TypeError:
                return fn(*args, **{**kwargs, "params": {}})
        raise


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

    def run_job(self, frame_bgr: np.ndarray, job: Dict[str, Any], anchor: Optional[bool] = None) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        tool = (job.get("tool") or "").lower()
        params = job.get("params", {}) or {}
        roi_in = job.get("roi") or [0, 0, frame_bgr.shape[1], frame_bgr.shape[0]]

        # per-call anchor flag (default False if not provided)
        anchor_flag = bool(anchor)
        # transform ROI by current detection if fixture is enabled
        roi_xywh = self.anchor.apply(roi_in, use_anchor=anchor_flag)

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
        # Prefer LineROI API
        lr = _parse_line_roi(params, gray_roi.shape)
        try:
            res = tool_line_caliper(gray_roi, lr, {})
        except TypeError:
            # Older SDK: CaliperGuide-only
            g = _parse_caliper(params, gray_roi.shape)
            res = _call_tool_optional_params(tool_line_caliper, gray_roi, g)
        return _normalize_measure_output(res)

    def _run_edge_pair_width(self, gray_roi: np.ndarray, params: Dict[str, Any]):
        # Prefer two LineROI arguments
        pA = params.get("g1", params.get("gA", {})) or {}
        pB = params.get("g2", params.get("gB", {})) or {}
        lrA = _parse_line_roi(pA, gray_roi.shape)
        lrB = _parse_line_roi(pB, gray_roi.shape)
        try:
            res = tool_edge_pair_width(gray_roi, lrA, lrB, {})
        except TypeError:
            # Older SDK: two CaliperGuide + optional params
            gA = _parse_caliper(pA, gray_roi.shape)
            gB = _parse_caliper(pB, gray_roi.shape)
            res = _call_tool_optional_params(tool_edge_pair_width, gray_roi, gA, gB)
        return _normalize_measure_output(res)

    def _run_angle_between(self, gray_roi: np.ndarray, params: Dict[str, Any]):
        # Prefer two LineROI arguments
        p1 = params.get("g1", {}) or {}
        p2 = params.get("g2", {}) or {}
        lr1 = _parse_line_roi(p1, gray_roi.shape)
        lr2 = _parse_line_roi(p2, gray_roi.shape)
        try:
            res = tool_angle_between(gray_roi, lr1, lr2, {})
        except TypeError:
            # Older SDK: AngleBetween dataclass
            g1 = _parse_caliper(p1, gray_roi.shape)
            g2 = _parse_caliper(p2, gray_roi.shape)
            ab = AngleBetween(g1=g1, g2=g2)
            res = _call_tool_optional_params(tool_angle_between, gray_roi, ab)
        return _normalize_measure_output(res)

    def _run_circle_diameter(self, gray_roi: np.ndarray, params: Dict[str, Any]):
        cp = _parse_circle_params(params, gray_roi.shape)
        res = _call_tool_optional_params(tool_circle_diameter, gray_roi, cp)
        return _normalize_measure_output(res)

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

def _parse_line_roi(p: Dict[str, Any], roi_shape: Tuple[int, int]) -> LineROI:
    """
    Convert old-style caliper params (p0, p1, band_px/ thickness) into LineROI.
    - center = midpoint(p0, p1)
    - length = |p1 - p0|
    - angle = atan2(p1 - p0)
    - thickness ≈ band_px   (reasonable mapping)
    """
    h, w = roi_shape[:2]
    p0 = p.get("p0", [10.0, h * 0.5])
    p1 = p.get("p1", [max(10.0, w - 10.0), h * 0.5])
    band_px = float(p.get("band_px", 24.0))
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    dx, dy = (x1 - x0), (y1 - y0)
    length = float(max(20.0, (dx * dx + dy * dy) ** 0.5))
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    thickness = max(5.0, float(band_px))

    return LineROI(cx=cx, cy=cy, angle_deg=angle_deg, length=length, thickness=thickness)

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

# --- AnchorHelper (real fixture) --------------------------------------------
def _rot2d(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]], dtype=np.float32)

class AnchorHelper:
    """
    Cognex-style fixture. When enabled and a source name is set:
      - the first time apply() is called with a ROI and a current detection pose,
        we capture that as the reference (ref_pose, ref_roi).
      - for subsequent frames, we rotate+translate ref_roi by the delta pose
        (current_pose relative to ref_pose) and return the axis-aligned bbox.

    Poses: (x, y, theta_deg) in image pixels / degrees.
    """
    def __init__(self) -> None:
        self._enabled: bool = False
        self._source: Optional[str] = None
        self._curr_pose: Optional[Tuple[float, float, float]] = None
        self._ref_pose: Optional[Tuple[float, float, float]] = None
        self._ref_roi: Optional[List[int]] = None

    # configuration
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def set_source(self, name: Optional[str]) -> None:
        self._source = name.strip() if isinstance(name, str) else None

    def get_source(self) -> Optional[str]:
        return self._source

    def is_enabled(self) -> bool:
        return self._enabled

    def reset_ref(self) -> None:
        self._ref_pose = None
        self._ref_roi = None

    # detection feed
    def update_detections(self, objects: List[Dict[str, Any]]) -> None:
        """Pick the best (highest score) detection for the configured source."""
        self._curr_pose = None
        if not self._source:
            return
        for obj in (objects or []):
            if obj.get("name") != self._source:
                continue
            dets = obj.get("detections") or []
            if not dets:
                continue
            det = max(dets, key=lambda d: float(d.get("score", 0.0)))
            pose = det.get("pose") or {}
            x = pose.get("x", det.get("x"))
            y = pose.get("y", det.get("y"))
            th = pose.get("theta_deg", pose.get("theta", 0.0))
            if x is None or y is None:
                c = det.get("center")
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    x, y = float(c[0]), float(c[1])
            if x is None or y is None:
                continue
            self._curr_pose = (float(x), float(y), float(th or 0.0))
            break
        # nothing else to do here

    # core
    def apply(self, roi: List[int] | Tuple[int, int, int, int], use_anchor: bool) -> List[int]:
        """Return transformed ROI [x,y,w,h]. If anchoring is disabled/unavailable, pass-through."""
        if roi is None:
            return [0, 0, 0, 0]
        x, y, w, h = map(int, roi)
        if w < 0: x, w = x + w, -w
        if h < 0: y, h = y + h, -h
        roi_n = [x, y, max(0, w), max(0, h)]

        if not (use_anchor and self._enabled and self._source and self._curr_pose):
            # no anchoring -> pass-through
            return roi_n

        if self._ref_pose is None or self._ref_roi != roi_n:
            # capture reference on first use (or when ROI changed)
            self._ref_pose = self._curr_pose
            self._ref_roi = roi_n
            return roi_n

        xr, yr, tr = self._ref_pose
        xc, yc, tc = self._curr_pose
        dtheta = tc - tr
        R = _rot2d(dtheta)

        # rotate & translate the 4 corners relative to ref pose center
        x0, y0, w0, h0 = self._ref_roi
        corners = np.array([[x0, y0],
                            [x0 + w0, y0],
                            [x0 + w0, y0 + h0],
                            [x0, y0 + h0]], dtype=np.float32)
        rel = corners - np.array([[xr, yr]], dtype=np.float32)  # N×2
        warped = (rel @ R.T) + np.array([[xc, yc]], dtype=np.float32)

        mn = np.min(warped, axis=0)
        mx = np.max(warped, axis=0)
        nx, ny = int(round(mn[0])), int(round(mn[1]))
        nw, nh = int(round(mx[0] - mn[0])), int(round(mx[1] - mn[1]))
        return [nx, ny, max(1, nw), max(1, nh)]

def measure_edge_pair_width(gray, roi, px_to_mm: float, params):
    res = tool_line_caliper(gray, roi, {**params, "max_edges": 2, "need_edges": 2})
    if len(res["edges_img"]) < 2:
        return {"ok": False, "value": None, "msg": res["msg"]}
    (x1,y1,_),(x2,y2,_) = sorted(res["edges_img"], key=lambda t: t[0])  # along strip x
    width_px = np.hypot(x2-x1, y2-y1)
    return {"ok": True, "value": width_px*px_to_mm, "debug": res}

def measure_angle_between_two_lines(line_roi_a, line_roi_b):
    da = (line_roi_a.angle_deg % 180.0)
    db = (line_roi_b.angle_deg % 180.0)
    d = abs(da - db)
    if d > 90: d = 180 - d
    return {"ok": True, "value_deg": d}

def go_no_go(value, lo, hi):
    return (lo <= value <= hi)
