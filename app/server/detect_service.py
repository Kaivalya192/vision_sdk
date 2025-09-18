from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from dexsdk.detect import MultiTemplateMatcher, SIFTMatcher


LAST_DETS: Dict[str, Dict[str, Any]] = {}
"""Cache last detection per object for downstream anchoring.

Structure:
    LAST_DETS[name] = {
        "ts": int,  # epoch milliseconds
        "H_template_to_frame": np.ndarray (3x3) or None,
        "center": Tuple[float, float],
        "theta_deg": float,
        "score": float,
        "inliers": int,
    }
"""


def _homography_from_detection(template_size: Optional[List[float]], det: Dict[str, Any]) -> Optional[np.ndarray]:
    if not template_size or len(template_size) < 2:
        return None
    quad = det.get("quad")
    if quad is None:
        return None
    try:
        dst = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    except Exception:
        return None
    if dst.shape != (4, 2):
        return None
    w, h = float(template_size[0]), float(template_size[1])
    if not np.isfinite([w, h]).all() or w <= 0 or h <= 0:
        return None
    src = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    return H


def _extract_center(det: Dict[str, Any]) -> Tuple[float, float]:
    pose = det.get("pose") or {}
    x = pose.get("x")
    y = pose.get("y")
    if x is None or y is None:
        ctr = det.get("center")
        if isinstance(ctr, (list, tuple)) and len(ctr) >= 2:
            x, y = ctr[0], ctr[1]
    if x is None or y is None:
        origin = pose.get("origin_xy")
        if isinstance(origin, (list, tuple)) and len(origin) >= 2:
            x, y = origin[0], origin[1]
    try:
        fx = float(x)
    except (TypeError, ValueError):
        fx = 0.0
    try:
        fy = float(y)
    except (TypeError, ValueError):
        fy = 0.0
    return fx, fy


def _update_last_detections(objects_report: List[Dict[str, Any]]) -> None:
    if not objects_report:
        return
    now_ms = int(time.time() * 1000)
    for obj in objects_report:
        name = obj.get("name")
        detections = obj.get("detections") or []
        if not name or not detections:
            continue
        best = max(detections, key=lambda d: (float(d.get("score", 0.0)), int(d.get("inliers", 0))))
        H = _homography_from_detection(obj.get("template_size"), best)
        pose = best.get("pose") or {}
        LAST_DETS[name] = {
            "ts": now_ms,
            "H_template_to_frame": H,
            "center": _extract_center(best),
            "theta_deg": float(pose.get("theta_deg", pose.get("theta", 0.0))),
            "score": float(best.get("score", 0.0)),
            "inliers": int(best.get("inliers", 0)),
        }


def get_last_detection(name: str) -> Optional[Dict[str, Any]]:
    return LAST_DETS.get(name)


def map_poly_via_last(name: str, poly_xy: np.ndarray) -> Optional[np.ndarray]:
    det = get_last_detection(name)
    if not det:
        return None
    H = det.get("H_template_to_frame")
    if H is None:
        return None
    H_arr = np.asarray(H, dtype=np.float32)
    if H_arr.size != 9:
        return None
    H_arr = H_arr.reshape(3, 3)
    try:
        mapped = cv2.perspectiveTransform(poly_xy.reshape(-1, 1, 2).astype(np.float32), H_arr)
    except cv2.error:
        return None
    return mapped.reshape(-1, 2)


class DetectService:
    def __init__(self, max_slots: int = 5, min_center_dist_px: int = 40):
        base_matcher = SIFTMatcher()
        matcher_defaults = base_matcher.params.copy()
        matcher_defaults.pop("max_instances", None)
        self.multi = MultiTemplateMatcher(
            max_slots=max_slots,
            min_center_dist_px=min_center_dist_px,
            matcher_defaults=matcher_defaults,
            angle_tolerance_deg=180.0,
        )
        self.multi.set_min_center_dist(min_center_dist_px)
        self._params = {
            "min_score": matcher_defaults.get("min_score", 0.25),
            "min_inliers": matcher_defaults.get("min_inliers", 4),
            "ransac_thr_px": matcher_defaults.get("ransac_thr_px", 4.0),
            "lowe_ratio": matcher_defaults.get("lowe_ratio", 0.90),
            "max_matches": matcher_defaults.get("max_matches", 150),
            "min_center_dist_px": int(min_center_dist_px),
            "angle_tolerance_deg": 180.0,
        }

    # Slot control wrappers (no logic changes)
    def add_or_replace(self, index: int, name: str, roi_bgr: np.ndarray, *, max_instances: int = 3) -> None:
        self.multi.add_or_replace(index, name, roi_bgr, max_instances=max_instances)

    def add_or_replace_polygon(
        self, index: int, name: str, roi_bgr: np.ndarray, roi_mask: np.ndarray, *, max_instances: int = 3
    ) -> None:
        self.multi.add_or_replace_polygon(index, name, roi_bgr, roi_mask, max_instances=max_instances)

    def clear(self, index: int) -> None:
        self.multi.clear(index)

    def set_enabled(self, index: int, enabled: bool) -> None:
        self.multi.set_enabled(index, enabled)

    def set_max_instances(self, index: int, k: int) -> None:
        self.multi.set_max_instances(index, k)

    def update_params(self, params: Dict[str, float]) -> Dict[str, float]:
        if not isinstance(params, dict):
            return {}

        matcher_updates: Dict[str, float] = {}
        applied: Dict[str, float] = {}

        def _float(key):
            val = params.get(key)
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        def _int(key):
            val = params.get(key)
            try:
                return int(val)
            except (TypeError, ValueError):
                return None

        for key in ("min_score", "ransac_thr_px", "lowe_ratio"):
            val = _float(key)
            if val is not None:
                matcher_updates[key] = val

        max_matches = _int("max_matches")
        if max_matches is not None:
            matcher_updates["max_matches"] = max_matches

        min_inliers = _int("min_inliers")
        if min_inliers is not None:
            matcher_updates["min_inliers"] = max(1, min_inliers)

        if matcher_updates:
            applied_defaults = self.multi.set_matcher_defaults(**matcher_updates)
            for key, value in applied_defaults.items():
                self._params[key] = value
                applied[key] = value

        center_dist = _int("min_center_dist_px")
        if center_dist is not None:
            self.multi.set_min_center_dist(center_dist)
            self._params["min_center_dist_px"] = center_dist
            applied["min_center_dist_px"] = center_dist

        angle_tol = _float("angle_tolerance_deg")
        if angle_tol is not None:
            self.multi.set_angle_tolerance(angle_tol)
            self._params["angle_tolerance_deg"] = angle_tol
            applied["angle_tolerance_deg"] = angle_tol

        return applied

    def get_params(self) -> Dict[str, float]:
        return dict(self._params)

    def compute_all(self, frame_bgr: np.ndarray, *, draw: bool = True):
        overlay, objects_report, total_instances = self.multi.compute_all(frame_bgr, draw=draw)
        _update_last_detections(objects_report)
        return overlay, objects_report, total_instances
