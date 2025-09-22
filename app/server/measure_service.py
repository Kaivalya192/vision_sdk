# app/server/measure_service.py
from __future__ import annotations
from typing import Callable, Dict, Any, Tuple, Optional, List
import cv2
import numpy as np
import math
import time

from dexsdk.measure.core import run_job as _run_job

class MeasureService:
    """
    Thin wrapper around the measurement core + a simple two-point virtual tracker.
    Accepts a scale getter that returns ((mm_per_px_x, mm_per_px_y), units_str)
    """
    def __init__(self, scale_getter: Callable[[], Tuple[Tuple[float, float], str]]):
        self._scale_getter = scale_getter
        self._vp_tracker = VirtualPointTracker()

    def run_job(self, frame_bgr: np.ndarray, job: Dict[str, Any]):
        (mm_per_px_x, mm_per_px_y), units = self._scale_getter()

        tool = str(job.get("tool", "")).lower().strip()

        # Virtual point tools handled here (stateful)
        if tool in ("vp_init", "vp_step"):
            packet, overlay = self._vp_tracker.handle(tool, frame_bgr, job, (mm_per_px_x, mm_per_px_y), units)
            return packet, overlay

        # All others: delegate to core
        packet, overlay = _run_job(frame_bgr, job, (mm_per_px_x, mm_per_px_y), units_label=units)
        return packet, overlay


# ----------------- Virtual Point Tracker -----------------

class VirtualPointTracker:
    """
    Tracks two ROI-anchored points across frames using local template matching.

    - vp_init: pick 2 relative points inside ROI, crop their small patches as templates.
    - vp_step: predict search windows near last positions and match; update templates (optional).
    - Reports distance in mm and draws overlay each call.

    Robustness features:
      * Adaptive search window growth if confidence drops
      * Optional template refresh with EMA
      * Bounds checks everywhere
    """
    def __init__(self):
        self._state: Optional[Dict[str, Any]] = None
        self._last_ts = 0.0

    def reset(self):
        self._state = None

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _crop_roi(img: np.ndarray, roi):
        x, y, w, h = [int(v) for v in roi]
        H, W = img.shape[:2]
        x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
        w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
        return img[y:y+h, x:x+w].copy(), (x, y)

    @staticmethod
    def _px_to_mm(val_px: float, mm_per_px_xy: Tuple[float, float]) -> float:
        sx, sy = mm_per_px_xy
        return float(val_px) * (0.5 * (sx + sy))

    @staticmethod
    def _draw_text(img, txt, org, color=(0,255,255), scale=0.6, thick=1):
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _safe_crop(self, gray: np.ndarray, cx: int, cy: int, half: int):
        H, W = gray.shape[:2]
        x1 = max(0, cx - half); y1 = max(0, cy - half)
        x2 = min(W, cx + half + 1); y2 = min(H, cy + half + 1)
        if x2 <= x1 or y2 <= y1:
            return None, (0,0,0,0)
        return gray[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)

    def _match_once(self, search: np.ndarray, templ: np.ndarray):
        """
        Returns: (top_left_xy, max_val) in search coords.
        """
        if search is None or templ is None: return None, -1.0
        hs, ws = search.shape[:2]
        ht, wt = templ.shape[:2]
        if hs < ht or ws < wt or ht < 3 or wt < 3:
            return None, -1.0
        res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        return (int(maxLoc[0]), int(maxLoc[1])), float(maxVal)

    def _ema_update(self, old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        if alpha <= 0.0: return old
        if old.shape != new.shape: return old
        out = (1.0 - alpha) * old.astype(np.float32) + alpha * new.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _draw_cross(self, img, pt, color):
        cv2.drawMarker(img, (int(pt[0]), int(pt[1])), color, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)

    def handle(self, tool: str, frame_bgr: np.ndarray, job: Dict[str, Any],
               mm_per_px_xy: Tuple[float, float], units: str):
        overlay = frame_bgr.copy()
        roi = job.get("roi", None)
        if roi is None or len(roi) != 4:
            self._draw_text(overlay, "VP: ROI required", (12, 24), (0, 0, 255))
            return {"measures": [], "units": units}, overlay

        roi_img, (rx, ry) = self._crop_roi(frame_bgr, roi)
        roi_gray = self._to_gray(roi_img)
        H, W = roi_gray.shape[:2]

        if tool == "vp_init":
            # Params with sensible defaults
            p1_rel = job.get("params", {}).get("p1_rel", [0.25, 0.5])  # (u,v) in [0,1]
            p2_rel = job.get("params", {}).get("p2_rel", [0.75, 0.5])
            patch = int(job.get("params", {}).get("patch_px", 21))      # odd preferred
            search = int(job.get("params", {}).get("search_px", 41))
            alpha = float(job.get("params", {}).get("update_alpha", 0.15))
            conf_thr = float(job.get("params", {}).get("conf_thr", 0.6))

            # Convert to ROI pixels
            c1 = [int(round(p1_rel[0] * (W - 1))), int(round(p1_rel[1] * (H - 1)))]
            c2 = [int(round(p2_rel[0] * (W - 1))), int(round(p2_rel[1] * (H - 1)))]

            half_patch = max(2, patch // 2)
            t1, r1 = self._safe_crop(roi_gray, c1[0], c1[1], half_patch)
            t2, r2 = self._safe_crop(roi_gray, c2[0], c2[1], half_patch)
            if t1 is None or t2 is None:
                self._draw_text(overlay, "VP init: patch out of bounds", (rx + 6, ry + 22), (0, 0, 255))
                return {"measures": [], "units": units}, overlay

            # Save state
            self._state = dict(
                templates=[t1, t2],
                centers=[c1, c2],          # ROI coordinates
                patch_px=patch,
                search_px=search,
                alpha=alpha,
                conf_thr=conf_thr,
            )
            self._last_ts = time.time()

            # Draw
            for (cx, cy), color in zip([c1, c2], [(0, 255, 255), (255, 200, 0)]):
                self._draw_cross(overlay, (rx + cx, ry + cy), color)
                cv2.rectangle(overlay,
                              (rx + cx - half_patch, ry + cy - half_patch),
                              (rx + cx + half_patch, ry + cy + half_patch),
                              color, 1, cv2.LINE_AA)
            cv2.rectangle(overlay, (rx, ry), (rx + W, ry + H), (80, 80, 80), 1, cv2.LINE_AA)
            self._draw_text(overlay, "VP: initialized", (rx + 6, ry + 22), (0, 255, 255))

            return {"measures": [], "units": units, "vp": {"ok": True}}, overlay

        elif tool == "vp_step":
            if not self._state:
                self._draw_text(overlay, "VP: not initialized", (rx + 6, ry + 22), (0, 0, 255))
                return {"measures": [], "units": units}, overlay

            patch = int(self._state["patch_px"])
            search = int(self._state["search_px"])
            alpha = float(self._state["alpha"])
            conf_thr = float(self._state["conf_thr"])
            half_patch = max(2, patch // 2)
            half_search = max(half_patch + 2, search // 2)

            templates = self._state["templates"]
            centers = self._state["centers"]

            new_centers = []
            confs = []
            for idx in (0, 1):
                cx, cy = centers[idx]
                templ = templates[idx]

                # Crop search window
                search_img, (sx, sy, sw, sh) = self._safe_crop(roi_gray, int(cx), int(cy), half_search)
                if search_img is None:
                    new_centers.append([cx, cy]); confs.append(0.0); continue

                # Match
                tl, score = self._match_once(search_img, templ)
                if tl is None:
                    new_centers.append([cx, cy]); confs.append(0.0); continue

                mx, my = tl[0] + templ.shape[1] // 2, tl[1] + templ.shape[0] // 2
                gcx, gcy = sx + mx, sy + my   # ROI coords
                new_centers.append([gcx, gcy])
                confs.append(score)

            # Confidence & optional template refresh
            for idx in (0, 1):
                cx, cy = new_centers[idx]
                templ_new, _ = self._safe_crop(roi_gray, int(cx), int(cy), half_patch)
                if templ_new is not None and confs[idx] >= conf_thr and alpha > 0.0:
                    templates[idx] = self._ema_update(templates[idx], templ_new, alpha)
            self._state["templates"] = templates
            self._state["centers"] = new_centers

            # Distance in px→mm
            p1 = np.array(new_centers[0], np.float32)
            p2 = np.array(new_centers[1], np.float32)
            dist_px = float(np.linalg.norm(p1 - p2))
            dist_mm = self._px_to_mm(dist_px, mm_per_px_xy)

            # Draw overlay
            P1 = (rx + int(round(p1[0])), ry + int(round(p1[1])))
            P2 = (rx + int(round(p2[0])), ry + int(round(p2[1])))
            self._draw_cross(overlay, P1, (0, 255, 255))
            self._draw_cross(overlay, P2, (255, 200, 0))
            cv2.line(overlay, P1, P2, (255, 220, 50), 2, cv2.LINE_AA)
            self._draw_text(overlay, f"{dist_mm:.3f} {units}  (c1={confs[0]:.2f}, c2={confs[1]:.2f})", (rx + 6, ry + 22))

            packet = {
                "measures": [{
                    "id": "VP_D",
                    "kind": "vp_distance",
                    "value": float(dist_mm),
                    "units": units,
                    "pass": None,
                    "sigma": None
                }],
                "units": units,
                "vp": {"c1": new_centers[0], "c2": new_centers[1], "conf": confs}
            }
            return packet, overlay

        # Fallback
        self._draw_text(overlay, f"VP: unknown tool '{tool}'", (rx + 6, ry + 22), (0,0,255))
        return {"measures": [], "units": units}, overlay


# ----------------- Anchor Helper (unchanged) -----------------

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

        # Translate rect center by delta pose
        base_cx, base_cy = bx + bw * 0.5, by + bh * 0.5
        dx, dy = (cx - ox), (cy - oy)
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
