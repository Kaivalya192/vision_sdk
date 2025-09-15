from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def refine_corner_subpix(
    gray: np.ndarray,
    hint_xy: Tuple[float, float],
    win_half: int = 7,
    zero_zone: int = -1,
) -> Tuple[float, float]:
    """Refine a corner location to sub-pixel precision using cornerSubPix.

    Parameters
    - gray: 8-bit or float32 single-channel image
    - hint_xy: initial (x, y) estimate in pixels
    - win_half: half-size of the search window (7 -> 15x15 window)
    - zero_zone: zeroZone for cornerSubPix (use -1 for disabled)
    """

    if gray.ndim != 2:
        raise ValueError("gray must be a single-channel image")

    h, w = gray.shape[:2]
    x0 = float(np.clip(hint_xy[0], 0, w - 1))
    y0 = float(np.clip(hint_xy[1], 0, h - 1))

    # cornerSubPix expects float32
    if gray.dtype != np.float32:
        g = gray.astype(np.float32)
    else:
        g = gray

    pts = np.array([[[x0, y0]]], dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-2,
    )
    win_size = (int(win_half), int(win_half))
    zero = (int(zero_zone), int(zero_zone))
    refined = cv2.cornerSubPix(g, pts, win_size, zero, criteria)
    x, y = refined[0, 0].tolist()
    return float(x), float(y)


def strongest_harris_corner(
    gray: np.ndarray,
    ksize: int = 3,
    k: float = 0.04,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[float, float]:
    """Find the strongest Harris response point (x, y).

    If roi is provided as (x, y, w, h), search is confined to that region.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a single-channel image")

    h, w = gray.shape[:2]
    x0, y0, ww, hh = (0, 0, w, h) if roi is None else roi
    x1, y1 = x0 + ww, y0 + hh
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)

    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return float(w // 2), float(h // 2)

    if patch.dtype != np.float32:
        g = patch.astype(np.float32)
    else:
        g = patch

    R = cv2.cornerHarris(g, blockSize=2, ksize=ksize, k=k)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(R)
    px = float(x0 + max_loc[0])
    py = float(y0 + max_loc[1])
    return px, py

