# d exsdk/measure/core.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import cv2

from .schema import Job, ROI, Packet, Measurement, CaliperGuide, AngleBetween, CircleParams
from .tools import tool_line_caliper, tool_edge_pair_width, tool_angle_between, tool_circle_diameter


def _roi_to_slice(roi: ROI) -> Tuple[slice, slice]:
    return slice(roi.y, roi.y + roi.h), slice(roi.x, roi.x + roi.w)


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _draw_overlay(base: np.ndarray, roi: ROI, dbg: Dict[str, Any], tool: str) -> np.ndarray:
    out = base.copy()
    x, y, w, h = roi.x, roi.y, roi.w, roi.h
    cv2.rectangle(out, (x, y), (x + w, y + h), (32, 180, 255), 1, cv2.LINE_AA)

    def draw_pts(pts, color):
        for p in pts:
            if np.isnan(p[0]) or np.isnan(p[1]): continue
            cv2.circle(out, (int(x + p[0]), int(y + p[1])), 2, color, -1, cv2.LINE_AA)

    if tool in ("line_caliper", "edge_pair_width", "angle_between"):
        if "pts" in dbg:  # single
            draw_pts(dbg["pts"], (255, 220, 0))
        if "A" in dbg:
            if "pts" in dbg["A"]:
                draw_pts(dbg["A"]["pts"], (255, 0, 0))
        if "B" in dbg:
            if "pts" in dbg["B"]:
                draw_pts(dbg["B"]["pts"], (0, 255, 255))
        if "line" in dbg:
            a, b, c = dbg["line"]
            # draw within ROI bounds
            xs = np.array([0, w-1], dtype=np.float32)
            ys = (-a*xs - c) / (b + 1e-9)
            for i in range(2):
                ys[i] = np.clip(ys[i], 0, h-1)
            p0 = (int(x + xs[0]), int(y + ys[0])); p1 = (int(x + xs[1]), int(y + ys[1]))
            cv2.line(out, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
        if "A" in dbg and "line" in dbg["A"]:
            a,b,c = dbg["A"]["line"]
            xs = np.array([0, w-1], dtype=np.float32)
            ys = (-a*xs - c) / (b + 1e-9)
            p0 = (int(x + xs[0]), int(y + np.clip(ys[0], 0, h-1)))
            p1 = (int(x + xs[1]), int(y + np.clip(ys[1], 0, h-1)))
            cv2.line(out, p0, p1, (255, 0, 0), 1, cv2.LINE_AA)
        if "B" in dbg and "line" in dbg["B"]:
            a,b,c = dbg["B"]["line"]
            xs = np.array([0, w-1], dtype=np.float32)
            ys = (-a*xs - c) / (b + 1e-9)
            p0 = (int(x + xs[0]), int(y + np.clip(ys[0], 0, h-1)))
            p1 = (int(x + xs[1]), int(y + np.clip(ys[1], 0, h-1)))
            cv2.line(out, p0, p1, (0, 255, 255), 1, cv2.LINE_AA)

    if tool == "circle_diameter" and "circle" in dbg:
        xc, yc, R = dbg["circle"]
        cv2.circle(out, (int(x + xc), int(y + yc)), int(R), (0, 255, 0), 1, cv2.LINE_AA)

    return out


def run_job(frame_bgr: np.ndarray, job: Job, units: str = "px") -> Packet:
    roi = job.roi
    r = _roi_to_slice(roi)
    roi_img = frame_bgr[r[0], r[1]]
    gray = _to_gray(roi_img)

    tool = job.tool
    params = job.params or {}

    meas: Measurement
    dbg: Dict[str, Any] = {}

    if tool == "line_caliper":
        g = CaliperGuide(**params)
        meas, dbg = tool_line_caliper(gray, g)

    elif tool == "edge_pair_width":
        gA = CaliperGuide(**params["g1"])
        gB = CaliperGuide(**params["g2"])
        meas, dbg = tool_edge_pair_width(gray, gA, gB)

    elif tool == "angle_between":
        ab = AngleBetween(
            g1=CaliperGuide(**params["g1"]),
            g2=CaliperGuide(**params["g2"]),
        )
        meas, dbg = tool_angle_between(gray, ab)

    elif tool == "circle_diameter":
        cp = CircleParams(**params)
        meas, dbg = tool_circle_diameter(gray, cp)

    else:
        meas = Measurement(id="unknown", kind=tool, value=float("nan"), sigma=float("inf"))

    overlay = _draw_overlay(frame_bgr, roi, dbg, tool)
    pkt = Packet(units=units, measures=[meas], overlay=overlay)
    return pkt
