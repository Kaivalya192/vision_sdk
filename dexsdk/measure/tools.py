#!/usr/bin/env python3
"""Cognex-style measurement helpers constrained to ROI polygons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import cv2
import numpy as np


# --- geometry helpers -----------------------------------------------------


def _order_quad_tl_tr_br_bl(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    sums = poly.sum(axis=1)
    diffs = poly[:, 0] - poly[:, 1]
    tl = poly[np.argmin(sums)]
    br = poly[np.argmax(sums)]
    tr = poly[np.argmin(diffs)]
    bl = poly[np.argmax(diffs)]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    area = 0.5 * np.cross(ordered[1] - ordered[0], ordered[3] - ordered[0])
    if area < 0:
        ordered = np.array([tl, bl, br, tr], dtype=np.float32)
    return ordered


def _quad_size_wh(poly: np.ndarray) -> Tuple[int, int]:
    ordered = _order_quad_tl_tr_br_bl(poly)
    w = 0.5 * (np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3]))
    h = 0.5 * (np.linalg.norm(ordered[2] - ordered[1]) + np.linalg.norm(ordered[3] - ordered[0]))
    return max(2, int(round(w))), max(2, int(round(h)))


def _warp_to_rect(img_bgr: np.ndarray, quad_xy: np.ndarray, out_wh: Optional[Tuple[int, int]] = None):
    quad = _order_quad_tl_tr_br_bl(quad_xy)
    if out_wh is None:
        w, h = _quad_size_wh(quad)
    else:
        w, h = out_wh
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    h_img_to_rect = cv2.getPerspectiveTransform(quad, dst)
    h_rect_to_img = cv2.getPerspectiveTransform(dst, quad)
    rect = cv2.warpPerspective(img_bgr, h_img_to_rect, (w, h), flags=cv2.INTER_LINEAR)
    return rect, h_img_to_rect, h_rect_to_img


def _to_iso_scale(mm_per_px: float | Tuple[float, float]) -> Tuple[float, float, float]:
    if isinstance(mm_per_px, (tuple, list)) and len(mm_per_px) == 2:
        sx = float(mm_per_px[0])
        sy = float(mm_per_px[1])
    else:
        sx = sy = float(mm_per_px)
    return sx, sy, 0.5 * (sx + sy)


# --- edge extraction ------------------------------------------------------


@dataclass
class EdgeFeature:
    point_xy: Tuple[float, float]
    dir_xy: Tuple[float, float]
    theta_deg: float
    n_samples: int
    inlier_rmse_px: float


def _subpix_peak_1d(signal: np.ndarray, index: int) -> float:
    idx = int(np.clip(index, 1, signal.size - 2))
    y0 = float(signal[idx - 1])
    y1 = float(signal[idx])
    y2 = float(signal[idx + 1])
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-9:
        return float(idx)
    offset = 0.5 * (y0 - y2) / denom
    return float(idx + offset)


def _fit_line(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    direction = np.array([float(vx), float(vy)], dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1e-12
    anchor = np.array([float(x0), float(y0)], dtype=np.float64)
    diff = points_xy - anchor
    distances = np.abs(diff[:, 0] * direction[1] - diff[:, 1] * direction[0])
    rmse = float(np.sqrt(np.mean(distances ** 2))) if distances.size else 0.0
    return anchor, direction, rmse


def edge_feature_from_roi(
    img_bgr: np.ndarray,
    roi_poly_xy: np.ndarray,
    scan_step: int = 2,
    smooth_kernel: int = 5,
) -> EdgeFeature:
    roi_poly_xy = np.asarray(roi_poly_xy, dtype=np.float32).reshape(4, 2)
    patch, _, h_rect_to_img = _warp_to_rect(img_bgr, roi_poly_xy)
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    height, width = gray.shape[:2]
    scan_rows = width >= height
    ksize = max(1, int(smooth_kernel) | 1)
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag_x = np.abs(sobel_x)
    mag_y = np.abs(sobel_y)
    samples_rect = []
    step = max(1, int(scan_step))
    if scan_rows:
        for y in range(1, height - 1, step):
            column = int(np.argmax(mag_x[y, :]))
            if 1 <= column < width - 1:
                col_refined = _subpix_peak_1d(mag_x[y, :], column)
                samples_rect.append([col_refined, float(y)])
    else:
        for x in range(1, width - 1, step):
            row = int(np.argmax(mag_y[:, x]))
            if 1 <= row < height - 1:
                row_refined = _subpix_peak_1d(mag_y[:, x], row)
                samples_rect.append([float(x), row_refined])
    if len(samples_rect) < 6:
        quad = _order_quad_tl_tr_br_bl(roi_poly_xy)
        vec_a = quad[1] - quad[0]
        vec_b = quad[2] - quad[1]
        if np.linalg.norm(vec_a) >= np.linalg.norm(vec_b):
            point = (quad[0] + quad[1]) * 0.5
            direction = vec_a
        else:
            point = (quad[1] + quad[2]) * 0.5
            direction = vec_b
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        return EdgeFeature(
            point_xy=(float(point[0]), float(point[1])),
            dir_xy=(float(direction[0]), float(direction[1])),
            theta_deg=angle,
            n_samples=len(samples_rect),
            inlier_rmse_px=0.0,
        )
    rect_pts = np.array(samples_rect, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(rect_pts, h_rect_to_img).reshape(-1, 2)
    anchor, direction, rmse = _fit_line(img_pts)
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return EdgeFeature(
        point_xy=(float(anchor[0]), float(anchor[1])),
        dir_xy=(float(direction[0]), float(direction[1])),
        theta_deg=angle,
        n_samples=len(samples_rect),
        inlier_rmse_px=rmse,
    )


# --- edge relationships ---------------------------------------------------


def angle_between_edges_deg(edge_a: EdgeFeature, edge_b: EdgeFeature) -> float:
    a = np.array(edge_a.dir_xy, dtype=np.float64)
    b = np.array(edge_b.dir_xy, dtype=np.float64)
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12
    cross = a[0] * b[1] - a[1] * b[0]
    dot = float(np.dot(a, b))
    angle = float(np.degrees(np.arctan2(abs(cross), dot)))
    return angle


def distance_between_edges_mm(
    edge_a: EdgeFeature,
    edge_b: EdgeFeature,
    mm_per_px: float | Tuple[float, float],
) -> float:
    _, _, s_iso = _to_iso_scale(mm_per_px)
    pa = np.array(edge_a.point_xy, dtype=np.float64)
    pb = np.array(edge_b.point_xy, dtype=np.float64)
    da = np.array(edge_a.dir_xy, dtype=np.float64)
    db = np.array(edge_b.dir_xy, dtype=np.float64)
    da /= np.linalg.norm(da) + 1e-12
    db /= np.linalg.norm(db) + 1e-12
    ang = angle_between_edges_deg(edge_a, edge_b)
    if ang < 15.0:
        diff = pb - pa
        dist_px = abs(diff[0] * da[1] - diff[1] * da[0])
        return float(dist_px * s_iso)
    bisector = da + db
    if np.linalg.norm(bisector) < 1e-12:
        normal = np.array([-da[1], da[0]])
    else:
        bisector /= np.linalg.norm(bisector)
        normal = np.array([-bisector[1], bisector[0]])
    dist_px = abs(np.dot(pb - pa, normal))
    return float(dist_px * s_iso)


# --- circle extraction ----------------------------------------------------


@dataclass
class CircleFeature:
    center_xy: Tuple[float, float]
    radius_px: float
    radius_mm: float
    n_inliers: int


def _fit_circle_kasa(points_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = points_xy.astype(np.float64)
    a_mat = np.hstack((2 * pts, np.ones((pts.shape[0], 1))))
    b_vec = np.sum(pts ** 2, axis=1, keepdims=True)
    sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    cx, cy, c = sol.ravel()
    center = np.array([cx, cy], dtype=np.float64)
    radius = float(np.sqrt(max(1e-9, c + cx * cx + cy * cy)))
    return center, radius


def circle_radius_in_roi(
    img_bgr: np.ndarray,
    roi_poly_xy: np.ndarray,
    mm_per_px: float | Tuple[float, float],
    canny_lower: int = 50,
    canny_upper: int = 150,
    min_edge_pts: int = 30,
) -> Optional[CircleFeature]:
    _, _, s_iso = _to_iso_scale(mm_per_px)
    patch, _, h_rect_to_img = _warp_to_rect(img_bgr, roi_poly_xy)
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    edges = cv2.Canny(gray, canny_lower, canny_upper, apertureSize=3, L2gradient=True)
    ys, xs = np.nonzero(edges)
    if xs.size < min_edge_pts:
        return None
    if xs.size > 4000:
        idx = np.random.choice(xs.size, 4000, replace=False)
        xs = xs[idx]
        ys = ys[idx]
    rect_pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    rect_pts = rect_pts.reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(rect_pts, h_rect_to_img).reshape(-1, 2)
    center, radius_px = _fit_circle_kasa(img_pts)
    distances = np.linalg.norm(img_pts - center[None, :], axis=1)
    inliers = int(np.sum(np.abs(distances - radius_px) < 2.0))
    return CircleFeature(
        center_xy=(float(center[0]), float(center[1])),
        radius_px=float(radius_px),
        radius_mm=float(radius_px * s_iso),
        n_inliers=inliers,
    )


# --- public wrappers ------------------------------------------------------


def measure_edge(img_bgr: np.ndarray, roi_poly_xy: np.ndarray) -> Dict[str, Any]:
    edge = edge_feature_from_roi(img_bgr, roi_poly_xy)
    return {
        "type": "edge",
        "point_xy": [edge.point_xy[0], edge.point_xy[1]],
        "dir_xy": [edge.dir_xy[0], edge.dir_xy[1]],
        "theta_deg": float(edge.theta_deg),
        "n_samples": int(edge.n_samples),
        "inlier_rmse_px": float(edge.inlier_rmse_px),
    }


def measure_distance(
    img_bgr: np.ndarray,
    roi_a_xy: np.ndarray,
    roi_b_xy: np.ndarray,
    mm_per_px: float | Tuple[float, float],
) -> Dict[str, Any]:
    edge_a = edge_feature_from_roi(img_bgr, roi_a_xy)
    edge_b = edge_feature_from_roi(img_bgr, roi_b_xy)
    dist_mm = distance_between_edges_mm(edge_a, edge_b, mm_per_px)
    return {
        "type": "distance",
        "mm": float(dist_mm),
        "edgeA": {"theta_deg": float(edge_a.theta_deg), "rmse_px": float(edge_a.inlier_rmse_px)},
        "edgeB": {"theta_deg": float(edge_b.theta_deg), "rmse_px": float(edge_b.inlier_rmse_px)},
    }


def measure_angle(
    img_bgr: np.ndarray,
    roi_a_xy: np.ndarray,
    roi_b_xy: np.ndarray,
) -> Dict[str, Any]:
    edge_a = edge_feature_from_roi(img_bgr, roi_a_xy)
    edge_b = edge_feature_from_roi(img_bgr, roi_b_xy)
    angle = angle_between_edges_deg(edge_a, edge_b)
    return {
        "type": "angle",
        "deg": float(angle),
        "edgeA": {"theta_deg": float(edge_a.theta_deg), "rmse_px": float(edge_a.inlier_rmse_px)},
        "edgeB": {"theta_deg": float(edge_b.theta_deg), "rmse_px": float(edge_b.inlier_rmse_px)},
    }


def measure_circle_radius(
    img_bgr: np.ndarray,
    roi_poly_xy: np.ndarray,
    mm_per_px: float | Tuple[float, float],
) -> Dict[str, Any]:
    circle = circle_radius_in_roi(img_bgr, roi_poly_xy, mm_per_px)
    if circle is None:
        return {
            "type": "circle_radius",
            "ok": False,
            "reason": "insufficient_edge_points",
        }
    return {
        "type": "circle_radius",
        "ok": True,
        "center_xy": [circle.center_xy[0], circle.center_xy[1]],
        "radius_px": float(circle.radius_px),
        "radius_mm": float(circle.radius_mm),
        "n_inliers": int(circle.n_inliers),
    }
