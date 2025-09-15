from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from dexsdk.detect import MultiTemplateMatcher


class DetectService:
    def __init__(self, max_slots: int = 5, min_center_dist_px: int = 40):
        self.multi = MultiTemplateMatcher(max_slots=max_slots, min_center_dist_px=min_center_dist_px)

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

    def compute_all(self, frame_bgr: np.ndarray, *, draw: bool = True):
        return self.multi.compute_all(frame_bgr, draw=draw)

