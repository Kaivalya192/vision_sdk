# ======================================
# FILE: dexsdk/detection_multi.py
# ======================================
"""Manage up to 5 templates and aggregate detections for publishing/overlay.
- Supports rectangular and polygon-masked templates.
- Simple duplicate suppression via minimum center distance per object.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import cv2, numpy as np

from .detection import SIFTMatcher

PALETTE = [
    (50, 200, 50),    # green
    (50, 180, 255),   # orange
    (255, 160, 50),   # blue-ish
    (220, 60, 220),   # purple
    (60, 220, 220),   # yellow-ish
]

@dataclass
class TemplateSlot:
    name: str
    matcher: SIFTMatcher
    color: Tuple[int, int, int]  # BGR
    enabled: bool = True

class MultiTemplateMatcher:
    """
    Holds up to `max_slots` SIFTMatcher templates.
    Each slot may return multiple detections (instances).
    Simple duplicate suppression ensures instances are spatially separated.
    """
    def __init__(self, max_slots: int = 5, min_center_dist_px: int = 40):
        self.max_slots = max(1, min(5, max_slots))
        self.min_center_dist_px = max(0, int(min_center_dist_px))
        self.slots: List[TemplateSlot] = []

    # ---- configuration helpers ----
    def set_min_center_dist(self, px: int):
        """Set minimum allowed distance (in processed-frame pixels) between
        centers of multiple instances of the SAME object."""
        self.min_center_dist_px = max(0, int(px))

    # ---- slot management ----
    def _ensure_index(self, index: int):
        """Internal: ensure list has placeholders up to 'index'."""
        while len(self.slots) <= index:
            self.slots.append(
                TemplateSlot(
                    name=f"Obj{len(self.slots)+1}",
                    matcher=SIFTMatcher(),
                    color=PALETTE[len(self.slots) % len(PALETTE)],
                    enabled=False,
                )
            )

    def add_or_replace(self, index: int, name: str, roi_bgr: np.ndarray, *, max_instances: int = 3):
        """Create/replace slot at `index` with a new rectangular template ROI."""
        index = int(index)
        if index < 0 or index >= self.max_slots:
            return
        self._ensure_index(index)
        m = SIFTMatcher(max_instances=max_instances)
        m.set_template(roi_bgr)
        self.slots[index] = TemplateSlot(name=name, matcher=m, color=PALETTE[index % len(PALETTE)], enabled=True)

    def add_or_replace_polygon(self, index: int, name: str, roi_bgr: np.ndarray, roi_mask: np.ndarray, *, max_instances: int = 3):
        """Create/replace slot at `index` with a polygon-masked template.
        roi_mask: uint8 (0/255), same size as roi_bgr. Non-zero = keep.
        """
        index = int(index)
        if index < 0 or index >= self.max_slots:
            return
        self._ensure_index(index)
        m = SIFTMatcher(max_instances=max_instances)
        m.set_template_polygon(roi_bgr, roi_mask)
        self.slots[index] = TemplateSlot(name=name, matcher=m, color=PALETTE[index % len(PALETTE)], enabled=True)

    def clear(self, index: int):
        """Disable and clear template for a slot."""
        if 0 <= index < len(self.slots):
            self.slots[index].matcher.clear_template()
            self.slots[index].enabled = False

    def set_enabled(self, index: int, enabled: bool):
        if 0 <= index < len(self.slots):
            self.slots[index].enabled = bool(enabled)

    def set_max_instances(self, index: int, k: int):
        if 0 <= index < len(self.slots):
            self.slots[index].matcher.update_params(max_instances=int(k))

    # ---- main compute ----
    def compute_all(self, frame_bgr: np.ndarray, *, draw: bool = True):
        """
        Run detection for all enabled slots on the given frame.
        Returns:
            overlay_bgr, objects_report(list), total_instances(int)
        """
        overlay = frame_bgr.copy()
        objects_report: List[Dict] = []
        total_instances = 0

        for i, slot in enumerate(self.slots[: self.max_slots]):
            if not slot.enabled:
                continue
            # Skip if no template set
            if slot.matcher.tpl_des is None or slot.matcher.tpl_kp is None:
                continue

            # Compute without drawing (we'll draw with per-slot color)
            _, _, dbg = slot.matcher.compute(frame_bgr, draw=False)
            raw_dets = dbg.get("poses", []) if isinstance(dbg, dict) else []

            # ---- simple duplicate suppression (per object) ----
            kept: List[Dict] = []
            for det in raw_dets:
                # center preferred; fallback to pose (tx,ty)
                c = det.get("center")
                if c is None:
                    c = [float(det.get("x", 0.0)), float(det.get("y", 0.0))]
                cx, cy = float(c[0]), float(c[1])

                too_close = False
                for kd in kept:
                    kc = kd.get("center")
                    if kc is None:
                        kc = [float(kd.get("x", 0.0)), float(kd.get("y", 0.0))]
                    kx, ky = float(kc[0]), float(kc[1])
                    if np.hypot(cx - kx, cy - ky) < self.min_center_dist_px:
                        too_close = True
                        break
                if not too_close:
                    kept.append(det)

            # Build report + draw from kept only
            inst_list: List[Dict] = []
            for j, det in enumerate(kept):
                # ... inside: for j, det in enumerate(kept):
                ctr = det.get("center")
                if ctr is not None:
                    px, py = float(ctr[0]), float(ctr[1])
                else:
                    # fallback to origin translation if center missing
                    px, py = float(det.get("x", 0.0)), float(det.get("y", 0.0))

                inst = {
                    "instance_id": int(j),
                    "score": float(det.get("score", 0.0)),
                    "inliers": int(det.get("ninliers", 0)),
                    "pose": {
                        # REPORT CENTER here (stable w.r.t rotation)
                        "x": px,
                        "y": py,
                        "theta_deg": float(det.get("theta", 0.0)),
                        "x_scale": float(det.get("x_scale", 1.0)),
                        "y_scale": float(det.get("y_scale", 1.0)),
                        # optional: keep raw origin translation for debugging
                        "origin_xy": [float(det.get("x", 0.0)), float(det.get("y", 0.0))],
                    },
                    "center": det.get("center", None),
                    "quad": det.get("quad", None),
                    "color": {
                        "bhattacharyya": float(det.get("color_bhat", -1.0)),
                        "correlation": float(det.get("color_corr", -1.0)),
                        "deltaE": float(det.get("color_deltaE", -1.0)),
                    },
                }

                inst_list.append(inst)

                # Draw per-slot color overlay
                if draw and det.get("quad") is not None:
                    quad = np.array(det["quad"], dtype=np.int32)
                    cv2.polylines(overlay, [quad], True, slot.color, 2)
                    ctr = det.get("center")
                    if ctr is not None:
                        cv2.circle(overlay, (int(ctr[0]), int(ctr[1])), 5, slot.color, -1)
                        cv2.putText(
                            overlay,
                            f"{slot.name}#{j}",
                            (int(ctr[0]) + 6, int(ctr[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            slot.color,
                            1,
                            cv2.LINE_AA,
                        )

            if inst_list:
                total_instances += len(inst_list)
                obj_report = {
                    "object_id": int(i),
                    "name": slot.name,
                    "template_size": list(slot.matcher.tpl_size) if slot.matcher.tpl_size else None,
                    "detections": inst_list,
                }
                objects_report.append(obj_report)

        return overlay, objects_report, total_instances
