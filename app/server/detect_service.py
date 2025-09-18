from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from dexsdk.detect import MultiTemplateMatcher, SIFTMatcher


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
        return self.multi.compute_all(frame_bgr, draw=draw)

