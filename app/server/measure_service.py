from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from dexsdk.measure.core import MeasureContext
from dexsdk.measure.geometry import DistanceP2P, LineFit, PointPick
from dexsdk.measure.schema import make_measure, make_result


class AnchorHelper:
    def __init__(self):
        self.prev_pose: Optional[Dict[str, float]] = None
        self.last_pose: Optional[Dict[str, float]] = None

    def set_reference(self, pose: Dict[str, float]) -> None:
        # pose must contain x, y, theta_deg
        self.prev_pose = self.last_pose
        self.last_pose = {"x": float(pose.get("x", 0.0)), "y": float(pose.get("y", 0.0)), "theta_deg": float(pose.get("theta_deg", 0.0))}

    def warp_rect(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if not self.prev_pose or not self.last_pose:
            return rect
        x, y, w, h = rect
        cx = x + w * 0.5
        cy = y + h * 0.5
        # delta pose (center + rotation only)
        dx = self.last_pose["x"] - self.prev_pose["x"]
        dy = self.last_pose["y"] - self.prev_pose["y"]
        dth = math.radians(self.last_pose["theta_deg"] - self.prev_pose["theta_deg"])
        # rotate rectangle corners around center by dth, then translate by (dx,dy)
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        rot = np.array([[math.cos(dth), -math.sin(dth)], [math.sin(dth), math.cos(dth)]], dtype=np.float32)
        shifted = corners - np.array([[cx, cy]], dtype=np.float32)
        rotated = shifted @ rot.T + np.array([[cx + dx, cy + dy]], dtype=np.float32)
        xmin = float(np.min(rotated[:, 0])); xmax = float(np.max(rotated[:, 0]))
        ymin = float(np.min(rotated[:, 1])); ymax = float(np.max(rotated[:, 1]))
        nx = int(round(xmin)); ny = int(round(ymin))
        nw = max(1, int(round(xmax - xmin)))
        nh = max(1, int(round(ymax - ymin)))
        return nx, ny, nw, nh


class MeasureService:
    def __init__(self, get_mm_scale_fn):
        """get_mm_scale_fn -> (sx, sy), units("mm"|"px")"""
        self.get_mm_scale_fn = get_mm_scale_fn

    def _ctx(self) -> MeasureContext:
        (sx, sy), units = self.get_mm_scale_fn()
        return MeasureContext(scale=(float(sx), float(sy)), units=str(units))

    def _draw_cross(self, img: np.ndarray, pt: Tuple[float, float], color=(0, 255, 255)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2, line_type=cv2.LINE_AA)

    def _draw_line_seg(self, img: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float], color=(0, 255, 0)):
        cv2.line(img, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), color, 2, cv2.LINE_AA)

    def run_job(self, image_bgr: np.ndarray, job: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
        tool = str(job.get("tool", "")).lower()
        params = dict(job.get("params", {}))
        roi = job.get("roi")
        if roi is not None:
            x, y, w, h = [int(v) for v in roi]
        else:
            x, y, w, h = 0, 0, int(image_bgr.shape[1]), int(image_bgr.shape[0])

        ctx = self._ctx()
        overlay = image_bgr.copy()
        measures = []
        prim = {"points": [], "lines": []}

        if tool == "point_pick":
            hint = params.get("hint_xy")
            res = PointPick().run(ctx, image_bgr, roi=(x, y, w, h), hint_xy=tuple(hint) if hint else None)
            px = res.get("point_px", [x + w * 0.5, y + h * 0.5])
            self._draw_cross(overlay, (px[0], px[1]))
            measures.append(
                make_measure(id="point_pick", kind="point_pick", value=0.0, passed=True, meta={"point_px": px, "point_mm": res.get("point_mm")})
            )
            prim["points"].append(px)

        elif tool == "line_fit":
            res = LineFit().run(ctx, image_bgr, roi=(x, y, w, h))
            if res:
                a, b, c = res["line"]
                p1, p2 = res["endpoints_px"][0], res["endpoints_px"][1]
                self._draw_line_seg(overlay, p1, p2, color=(0, 255, 0))
                txt = f"{res['angle_deg']:.1f} deg"
                cv2.putText(overlay, txt, (int(p1[0]), int(p1[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                measures.append(
                    make_measure(
                        id="line_fit", kind="angle_deg", value=float(res["angle_deg"]), sigma=None, passed=True, meta={"line_abc": [a, b, c], "span_mm": res["span_mm"]}
                    )
                )
                prim["lines"].append([p1, p2])

        elif tool == "distance_p2p":
            p1 = tuple(params.get("p1", [x + w * 0.25, y + h * 0.5]))
            p2 = tuple(params.get("p2", [x + w * 0.75, y + h * 0.5]))
            res = DistanceP2P().run(ctx, image_bgr, roi=(x, y, w, h), p1=p1, p2=p2)
            self._draw_line_seg(overlay, p1, p2, color=(0, 200, 255))
            d = float(res.get("distance_mm" if ctx.units == "mm" else "distance_px", 0.0))
            lbl = f"{d:.3f} {ctx.units}"
            mpt = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
            cv2.putText(overlay, lbl, (int(mpt[0]), int(mpt[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            measures.append(
                make_measure(id="distance_p2p", kind="distance_p2p", value=d, sigma=None, passed=True, meta={"p1_px": list(p1), "p2_px": list(p2)})
            )
            prim["points"].extend([list(p1), list(p2)])
            prim["lines"].append([list(p1), list(p2)])

        elif tool == "distance_p2l":
            # params: pt [x,y], line [a,b,c]
            pt = params.get("pt") or [x + w * 0.5, y + h * 0.5]
            line = params.get("line")  # try ROI fit if absent
            if line is None:
                lres = LineFit().run(ctx, image_bgr, roi=(x, y, w, h))
                line = lres.get("line") if lres else [0.0, 1.0, -float(y + h * 0.5)]
            a, b, c = [float(v) for v in line]
            px, py = float(pt[0]), float(pt[1])
            denom = math.hypot(a, b) or 1.0
            d_px = abs(a * px + b * py + c) / denom
            d_val = ctx.px_len_to_mm(d_px, axis="avg") if ctx.units == "mm" else d_px
            # perpendicular foot
            # line normal: n = (a,b), point projection onto line
            # foot = p - ((a*px+b*py+c)/(a^2+b^2)) * (a,b)
            t = (a * px + b * py + c) / (a * a + b * b)
            qx, qy = px - a * t, py - b * t
            self._draw_line_seg(overlay, (px, py), (qx, qy), color=(255, 200, 0))
            self._draw_cross(overlay, (px, py), color=(255, 200, 0))
            lbl = f"{d_val:.3f} {ctx.units}"
            cv2.putText(overlay, lbl, (int(qx), int(qy) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
            measures.append(
                make_measure(id="distance_p2l", kind="distance_p2l", value=float(d_val), sigma=None, passed=True, meta={"pt_px": [px, py], "line": [a, b, c]})
            )
            prim["points"].append([px, py])
            prim["lines"].append([[px, py], [qx, qy]])

        else:
            measures.append(make_measure(id="unknown", kind=tool or "unknown", value=0.0, passed=False, meta={"error": "unsupported tool"}))

        packet = make_result(units=ctx.units, measures=measures, primitives=prim)
        return packet, overlay

