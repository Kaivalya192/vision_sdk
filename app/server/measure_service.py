#!/usr/bin/env python3
"""ROI measurement service wrapper.

This module exposes a single function `run_measure_on_frame` that:
- resolves scale from job or calibration info,
- normalizes and optionally anchors ROI polygons,
- dispatches to measurement tools in `dexsdk.measure.tools`,
- renders an overlay and returns a MeasuresPacket dict and overlay image.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from dexsdk.measure.tools import (
    edge_feature_from_roi,
    distance_between_edges_mm,
    angle_between_edges_deg,
    circle_radius_in_roi,
)

try:
    from dexsdk.calib import store as calib_store  # optional
except Exception:  # pragma: no cover
    calib_store = None  # type: ignore


DEFAULT_MM_PER_PX: Tuple[float, float] = (0.05, 0.05)


def _scale_from_calib_info(info: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    # Accept either plane dict directly or nested under "plane_scale"
    plane = info.get("plane_scale", info)
    if not isinstance(plane, dict):
        return None
    mmx = plane.get("mm_per_px_x")
    mmy = plane.get("mm_per_px_y")
    if mmx and mmy and float(mmx) > 0 and float(mmy) > 0:
        return float(mmx), float(mmy)
    pxx = plane.get("px_per_mm_x")
    pxy = plane.get("px_per_mm_y")
    if pxx and pxy and float(pxx) > 0 and float(pxy) > 0:
        return 1.0 / float(pxx), 1.0 / float(pxy)
    scalar = plane.get("mm_per_px")
    if isinstance(scalar, (int, float)) and scalar > 0:
        return float(scalar), float(scalar)
    return None


def _normalize_mm_per_px(val: Any) -> Tuple[float, float]:
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
        sx = float(val[0])
        sy = float(val[1])
        if sx > 0 and sy > 0:
            return sx, sy
    try:
        f = float(val)
        if f > 0:
            return f, f
    except Exception:
        pass
    raise ValueError("mm_per_px must be > 0 (scalar or length-2)")


def _resolve_mm_per_px(job: Dict[str, Any], calib_info: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    if job.get("mm_per_px") is not None:
        return _normalize_mm_per_px(job["mm_per_px"])
    if isinstance(calib_info, dict):
        s = _scale_from_calib_info(calib_info)
        if s:
            return s
    # best-effort: attempt to load via calib store if it offers a function
    if calib_store is not None:
        for name in ("load_json", "get_cached", "get_latest"):
            fn = getattr(calib_store, name, None)
            try:
                if callable(fn):
                    data = fn("plane_scale.json") if name == "load_json" else fn()
                    if isinstance(data, dict):
                        s = _scale_from_calib_info(data)
                        if s:
                            return s
            except Exception:
                continue
    return DEFAULT_MM_PER_PX


def _rect_to_quad(rect: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(rect[i]) for i in range(4)]
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def _as_quad(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[:2] == (1, 4) and arr.shape[2] == 2:
        arr = arr.reshape(4, 2)
    if arr.shape == (4, 2):
        return arr
    if arr.size == 4:
        return _rect_to_quad(arr.ravel().tolist())
    raise ValueError(f"{name} must be a 4x2 polygon or legacy [x,y,w,h]")


def _extract_roi(job: Dict[str, Any], key_poly: str, legacy_keys: Sequence[str]) -> Optional[np.ndarray]:
    if job.get(key_poly) is not None:
        return _as_quad(job[key_poly], key_poly)
    for k in legacy_keys:
        if job.get(k) is not None:
            return _as_quad(job[k], k)
    return None


def _apply_anchor(roi_xy: np.ndarray, job: Dict[str, Any], detections: Dict[str, Any]) -> np.ndarray:
    if not job.get("anchor"):
        return roi_xy
    anchor_obj = job.get("anchor_object") or job.get("anchorId")
    det = None
    if isinstance(detections, dict) and anchor_obj is not None:
        det = detections.get(str(anchor_obj))
    if not isinstance(det, dict):
        return roi_xy
    H = det.get("H_template_to_frame") or det.get("H")
    if H is None:
        return roi_xy
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    pts = roi_xy.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, H).reshape(4, 2)
    return warped.astype(np.float32)


def _draw_roi(img: np.ndarray, poly: np.ndarray, color: tuple[int, int, int]) -> None:
    pts = poly.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts], True, color, 1, lineType=cv2.LINE_AA)


def _draw_edge(img: np.ndarray, edge, color: tuple[int, int, int]) -> None:
    p = np.array(edge.point_xy, dtype=np.float32)
    d = np.array(edge.dir_xy, dtype=np.float32)
    h, w = img.shape[:2]
    t = np.array([-2000.0, 2000.0], dtype=np.float32)
    pts = (p[None, :] + t[:, None] * d[None, :]).clip([0, 0], [w - 1, h - 1]).astype(np.int32)
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, 2, lineType=cv2.LINE_AA)


def _draw_circle(img: np.ndarray, circle, color: tuple[int, int, int]) -> None:
    c = (int(round(circle.center_xy[0])), int(round(circle.center_xy[1])))
    cv2.circle(img, c, int(round(circle.radius_px)), color, 2, lineType=cv2.LINE_AA)
    cv2.circle(img, c, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)


def _encode_overlay(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("ascii")


def run_measure_on_frame(
    frame_bgr: np.ndarray,
    job: Dict[str, Any],
    last_detections: Optional[Dict[str, Any]],
    calib_info: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Returns (packet: MeasuresPacket, overlay_bgr)."""

    job_dict = dict(job or {})
    tool = str(job_dict.get("tool") or job_dict.get("kind") or "edge").lower()
    if tool not in {"edge", "distance", "angle", "circle_radius"}:
        raise ValueError(f"Unsupported measurement tool: {tool}")

    # ROIs
    roi_a = _extract_roi(job_dict, "roiA_poly", ("roiA", "roi"))
    if roi_a is None:
        raise ValueError("roiA_poly is required")
    roi_b = None
    if tool in {"distance", "angle"}:
        roi_b = _extract_roi(job_dict, "roiB_poly", ("roiB",))
        if roi_b is None:
            raise ValueError("roiB_poly is required for distance/angle tools")

    # Scale
    mm_per_px = _resolve_mm_per_px(job_dict, calib_info)

    # Anchor transform if present
    detections = last_detections or {}
    roi_a = _apply_anchor(roi_a, job_dict, detections)
    if roi_b is not None:
        roi_b = _apply_anchor(roi_b, job_dict, detections)

    roi_a = roi_a.astype(np.float32)
    roi_b = roi_b.astype(np.float32) if roi_b is not None else None

    # Overlay base and ROI drawing
    overlay = frame_bgr.copy()
    _draw_roi(overlay, roi_a, (0, 200, 255))
    if roi_b is not None:
        _draw_roi(overlay, roi_b, (80, 255, 80))

    measure_id = str(job_dict.get("id") or "m1")
    entry: Dict[str, Any]
    packet_units: Optional[str]

    if tool == "edge":
        edge = edge_feature_from_roi(frame_bgr, roi_a)
        _draw_edge(overlay, edge, (0, 0, 255))
        entry = {"id": measure_id, "kind": tool, "value": float(edge.theta_deg), "units": "deg"}
        packet_units = "mm"
    elif tool == "distance":
        ea = edge_feature_from_roi(frame_bgr, roi_a)
        eb = edge_feature_from_roi(frame_bgr, roi_b)  # type: ignore[arg-type]
        _draw_edge(overlay, ea, (0, 0, 255))
        _draw_edge(overlay, eb, (255, 0, 0))
        dist_mm = float(distance_between_edges_mm(ea, eb, mm_per_px))
        entry = {"id": measure_id, "kind": tool, "value": dist_mm, "units": "mm"}
        packet_units = "mm"
    elif tool == "angle":
        ea = edge_feature_from_roi(frame_bgr, roi_a)
        eb = edge_feature_from_roi(frame_bgr, roi_b)  # type: ignore[arg-type]
        _draw_edge(overlay, ea, (0, 0, 255))
        _draw_edge(overlay, eb, (255, 0, 0))
        ang = float(angle_between_edges_deg(ea, eb))
        entry = {"id": measure_id, "kind": tool, "value": ang, "units": "deg"}
        packet_units = None
    else:  # circle_radius
        circle = circle_radius_in_roi(frame_bgr, roi_a, mm_per_px)
        if circle is not None:
            _draw_circle(overlay, circle, (255, 128, 0))
            entry = {"id": measure_id, "kind": tool, "value": float(circle.radius_mm), "units": "mm"}
        else:
            entry = {"id": measure_id, "kind": tool, "value": float("nan"), "units": "mm"}
        packet_units = "mm"

    packet: Dict[str, Any] = {
        "measures": [entry],
        "overlay_jpeg_b64": _encode_overlay(overlay),
    }
    if packet_units == "mm":
        packet["units"] = "mm"

    return packet, overlay
