from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .core import MeasureContext, ResultPacket, Tool
from .schema import make_measure, make_result
from ._subpix import refine_corner_subpix, strongest_harris_corner


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _roi_bounds(img: np.ndarray, roi: Optional[Tuple[int, int, int, int]]):
    h, w = img.shape[:2]
    if roi is None:
        return 0, 0, w, h
    x, y, rw, rh = roi
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(w, x0 + int(rw))
    y1 = min(h, y0 + int(rh))
    return x0, y0, x1 - x0, y1 - y0


class PointPick(Tool):
    """Pick a sub-pixel point near a hint using cornerSubPix with Harris fallback."""

    def run(
        self,
        ctx: MeasureContext,
        image_bgr: Any,
        roi: Optional[Tuple[int, int, int, int]] = None,
        hint_xy: Optional[Tuple[float, float]] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        img = np.asarray(image_bgr)
        gray = _ensure_gray(img)

        x0, y0, rw, rh = _roi_bounds(gray, roi)
        hint: Tuple[float, float]
        if hint_xy is not None:
            hint = hint_xy
        else:
            hint = strongest_harris_corner(gray, roi=(x0, y0, rw, rh))

        try:
            x_ref, y_ref = refine_corner_subpix(gray, hint, win_half=7)
        except Exception:
            # Fallback to Harris maximum if subpix fails
            x_ref, y_ref = strongest_harris_corner(gray, roi=(x0, y0, rw, rh))

        # Convert to mm using average scale
        px_val = 0.0
        mm_val = 0.0
        dx_mm, dy_mm = ctx.px_vec_to_mm(x_ref, y_ref)

        return {
            "point_px": [float(x_ref), float(y_ref)],
            "point_mm": [float(dx_mm), float(dy_mm)],
            "roi": [x0, y0, rw, rh],
        }


def _line_from_point_dir(pt: Tuple[float, float], v: Tuple[float, float]):
    x0, y0 = pt
    vx, vy = v
    # Normal vector
    nx, ny = float(vy), float(-vx)
    norm = math.hypot(nx, ny) or 1.0
    nx /= norm
    ny /= norm
    c = -(nx * x0 + ny * y0)
    return nx, ny, c


def _clip_line_to_rect(
    a: float, b: float, c: float, rect: Tuple[int, int, int, int]
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    # Rectangle in image coordinates
    x, y, w, h = rect
    x_min, y_min = x, y
    x_max, y_max = x + w, y + h

    # Intersections with rectangle borders
    points: List[Tuple[float, float]] = []

    def add_if_on_segment(px: float, py: float):
        if x_min - 1e-6 <= px <= x_max + 1e-6 and y_min - 1e-6 <= py <= y_max + 1e-6:
            points.append((px, py))

    # y = y_min
    if abs(b) > 1e-12:
        x_i = -(b * y_min + c) / (a if abs(a) > 1e-12 else 1e-12)
        add_if_on_segment(x_i, y_min)
    # y = y_max
    if abs(b) > 1e-12:
        x_i = -(b * y_max + c) / (a if abs(a) > 1e-12 else 1e-12)
        add_if_on_segment(x_i, y_max)
    # x = x_min
    if abs(a) > 1e-12:
        y_i = -(a * x_min + c) / (b if abs(b) > 1e-12 else 1e-12)
        add_if_on_segment(x_min, y_i)
    # x = x_max
    if abs(a) > 1e-12:
        y_i = -(a * x_max + c) / (b if abs(b) > 1e-12 else 1e-12)
        add_if_on_segment(x_max, y_i)

    # Deduplicate close points
    uniq: List[Tuple[float, float]] = []
    for p in points:
        if not any(math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-6 for q in uniq):
            uniq.append(p)

    if len(uniq) < 2:
        return None
    # Pick the two farthest points
    best = (uniq[0], uniq[1])
    best_d = -1.0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            d = math.hypot(uniq[i][0] - uniq[j][0], uniq[i][1] - uniq[j][1])
            if d > best_d:
                best_d = d
                best = (uniq[i], uniq[j])
    return best


class LineFit(Tool):
    """Fit a dominant line in an ROI using edges and total least squares.

    Pipeline (robust and dependency-light):
    - Canny edges
    - Collect edge points; if insufficient, return empty result
    - Fit line with cv2.fitLine (L2)
    - Convert to ax+by+c=0, angle, clip within ROI
    """

    def run(
        self,
        ctx: MeasureContext,
        image_bgr: Any,
        roi: Optional[Tuple[int, int, int, int]] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        img = np.asarray(image_bgr)
        gray = _ensure_gray(img)
        x0, y0, rw, rh = _roi_bounds(gray, roi)
        if rw <= 1 or rh <= 1:
            return {}

        patch = gray[y0 : y0 + rh, x0 : x0 + rw]
        edges = cv2.Canny(patch, 50, 150, apertureSize=3, L2gradient=True)
        ys, xs = np.nonzero(edges)
        if xs.size < 50:
            # Not enough edge points to fit a line; return empty
            return {}

        pts = np.column_stack([xs + x0, ys + y0]).astype(np.float32)
        # cv2.fitLine expects Nx1x2 or Nx2; it returns [vx, vy, x0, y0]
        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, px, py = [float(x) for x in line.ravel()]

        a, b, c = _line_from_point_dir((px, py), (vx, vy))
        angle_deg = math.degrees(math.atan2(vy, vx))

        seg = _clip_line_to_rect(a, b, c, (x0, y0, rw, rh))
        if seg is None:
            # Fallback: project ROI corners
            corners = [
                (x0, y0),
                (x0 + rw, y0),
                (x0, y0 + rh),
                (x0 + rw, y0 + rh),
            ]
            # projection onto line
            tvals = []
            for cx, cy in corners:
                t = (cx - px) * vx + (cy - py) * vy
                tvals.append(t)
            tmin, tmax = min(tvals), max(tvals)
            p1 = (px + tmin * vx, py + tmin * vy)
            p2 = (px + tmax * vx, py + tmax * vy)
        else:
            p1, p2 = seg

        span_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        span_mm = ctx.px_len_to_mm(span_px, axis="avg")

        return {
            "line": [float(a), float(b), float(c)],
            "angle_deg": float(angle_deg),
            "endpoints_px": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]],
            "span_mm": float(span_mm),
            "roi": [x0, y0, rw, rh],
        }


class DistanceP2P(Tool):
    """Compute distance between two points in pixels and millimeters."""

    def run(
        self,
        ctx: MeasureContext,
        image_bgr: Any,
        roi: Optional[Tuple[int, int, int, int]] = None,
        p1: Tuple[float, float] = (0.0, 0.0),
        p2: Tuple[float, float] = (1.0, 1.0),
        **kw: Any,
    ) -> Dict[str, Any]:
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        d_px = math.hypot(dx, dy)
        d_mm = ctx.px_len_to_mm(d_px, axis="avg")
        return {
            "p1_px": [float(p1[0]), float(p1[1])],
            "p2_px": [float(p2[0]), float(p2[1])],
            "distance_px": float(d_px),
            "distance_mm": float(d_mm),
        }


def _cli_demo(image_path: str) -> Dict[str, Any]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    ctx = MeasureContext(scale=(1.0, 1.0), units="mm")
    h, w = img.shape[:2]
    roi = (0, 0, w, h)

    # Pick a point near the center as a starting hint
    hint = (w * 0.5, h * 0.5)
    pp = PointPick()
    pick = pp.run(ctx, img, roi=roi, hint_xy=hint)

    lf = LineFit()
    line = lf.run(ctx, img, roi=roi)

    measures: List[Dict[str, Any]] = []
    primitives: Dict[str, Any] = {"points": [], "lines": []}

    if pick:
        primitives["points"].append(pick["point_px"])  # px coordinates

    if line:
        primitives["lines"].append(line["endpoints_px"])  # [[x1,y1],[x2,y2]]
        measures.append(
            make_measure(
                id="line_angle",
                kind="angle_deg",
                value=float(line["angle_deg"]),
                sigma=None,
                passed=None,
                meta={"line_abc": line["line"], "span_mm": line["span_mm"]},
            )
        )

    result = make_result(units=ctx.units, measures=measures, primitives=primitives)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m dexsdk.measure.geometry <image_path>")
        sys.exit(2)
    res = _cli_demo(sys.argv[1])
    print(json.dumps(res, indent=2))

