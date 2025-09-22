# app/server/measure_service.py
from __future__ import annotations
from typing import Callable, Dict, Any, Tuple
import cv2
import numpy as np

from dexsdk.measure.core import run_job as _run_job

class MeasureService:
    """
    Thin wrapper around the measurement core.
    Accepts a scale getter that returns ((mm_per_px_x, mm_per_px_y), units_str)
    """
    def __init__(self, scale_getter: Callable[[], Tuple[Tuple[float, float], str]]):
        self._scale_getter = scale_getter

    def run_job(self, frame_bgr: np.ndarray, job: Dict[str, Any]):
        (mm_per_px_x, mm_per_px_y), units = self._scale_getter()
        packet, overlay = _run_job(frame_bgr, job, (mm_per_px_x, mm_per_px_y), units_label=units)
        return packet, overlay


# Optional: anchoring helper (you already import AnchorHelper from here)
import numpy as np
import math

class AnchorHelper:
    """
    Maintains an origin pose and current pose (x, y, theta_deg) to warp a base ROI as the object moves.
    """
    def __init__(self):
        self.origin_pose = None
        self.current_pose = None
        self.base_rect = None  # (x,y,w,h) learned at set_origin

    def reset_origin(self):
        self.origin_pose = None
        self.current_pose = None
        self.base_rect = None

    def set_origin(self, pose: Dict[str, float], base_rect):
        self.origin_pose = dict(pose)
        self.current_pose = dict(pose)
        self.base_rect = tuple(base_rect)

    def update(self, pose: Dict[str, float]):
        self.current_pose = dict(pose)

    def warp_rect(self, rect, W, H):
        """
        Given base rect (at origin pose), return new rect under current pose.
        For small rotations, approximate by rotating rect center and preserving size.
        """
        if self.origin_pose is None or self.current_pose is None or self.base_rect is None:
            return rect
        bx, by, bw, bh = self.base_rect
        ox, oy, oth = self.origin_pose["x"], self.origin_pose["y"], math.radians(self.origin_pose["theta_deg"])
        cx, cy, cth = self.current_pose["x"], self.current_pose["y"], math.radians(self.current_pose["theta_deg"])

        # Translate rect center by delta pose (rotate delta around origin center)
        base_cx, base_cy = bx + bw * 0.5, by + bh * 0.5
        # Move vector from origin pose to current pose
        dx, dy = (cx - ox), (cy - oy)
        # Rotate displacement by current rotation minus origin rotation
        dth = cth - oth
        rot = np.array([[math.cos(dth), -math.sin(dth)],
                        [math.sin(dth),  math.cos(dth)]], dtype=np.float32)
        new_c = np.dot(rot, np.array([base_cx, base_cy], np.float32)) + np.array([dx, dy], np.float32)

        nx = int(max(0, min(W - 1, new_c[0] - bw * 0.5)))
        ny = int(max(0, min(H - 1, new_c[1] - bh * 0.5)))
        nw, nh = int(bw), int(bh)
        if nx + nw > W: nw = W - nx
        if ny + nh > H: nh = H - ny
        return (nx, ny, nw, nh)
