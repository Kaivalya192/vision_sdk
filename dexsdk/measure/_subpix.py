# d exsdk/measure/_subpix.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def _parabolic_subpixel(y_m1: float, y_0: float, y_p1: float) -> float:
    """
    3-point parabola vertex offset in samples relative to center point (i).
    Returns delta in [-1, 1]. If denominator is near-zero, returns 0.0.
    """
    denom = (y_m1 - 2.0 * y_0 + y_p1)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y_m1 - y_p1) / denom


def subpix_edge_1d(profile: np.ndarray, polarity: str = "any") -> Tuple[Optional[float], float, int]:
    """
    Sub-pixel edge locator along a 1D intensity profile.

    Args:
        profile: 1D array of intensities (0..255 or float). Shape (N,)
        polarity: 'rising', 'falling', or 'any'

    Returns:
        (pos, score, sign)
        pos   : float index in [0, N-1] (None if failed)
        score : edge strength (abs first-derivative at the peak)
        sign  : +1 for rising, -1 for falling (0 if failed)

    Notes:
        - Uses a simple centered derivative and 3-point parabolic refinement.
        - 'score' is in the same units as the derivative; you can compare it
          to your contrast thresholds (e.g. 8.0, 12.0, …).
    """
    if profile is None:
        return None, 0.0, 0
    prof = np.asarray(profile, dtype=np.float32).flatten()
    N = prof.size
    if N < 3:
        return None, 0.0, 0

    # centered derivative ([-1, 0, +1] / 2) with clamped borders
    # use np.gradient which does similar and handles edges
    d = np.gradient(prof)
    if polarity == "rising":
        i = int(np.argmax(d))
        sign = +1
        peak_val = float(d[i])
        if peak_val <= 0:
            return None, 0.0, 0
        y_m1, y_0, y_p1 = d[max(i - 1, 0)], d[i], d[min(i + 1, N - 1)]
        delta = _parabolic_subpixel(y_m1, y_0, y_p1)
        pos = float(i + np.clip(delta, -1.0, 1.0))
        score = float(peak_val)
        return pos, score, sign

    elif polarity == "falling":
        i = int(np.argmin(d))
        sign = -1
        peak_val = float(d[i])
        if peak_val >= 0:
            return None, 0.0, 0
        y_m1, y_0, y_p1 = d[max(i - 1, 0)], d[i], d[min(i + 1, N - 1)]
        delta = _parabolic_subpixel(y_m1, y_0, y_p1)
        pos = float(i + np.clip(delta, -1.0, 1.0))
        score = float(-peak_val)  # strength is magnitude
        return pos, score, sign

    else:  # 'any'
        i_pos = int(np.argmax(d))
        i_neg = int(np.argmin(d))
        v_pos = float(d[i_pos])
        v_neg = float(-d[i_neg])  # magnitude
        if v_pos <= 0 and v_neg <= 0:
            return None, 0.0, 0
        if v_pos >= v_neg:
            y_m1, y_0, y_p1 = d[max(i_pos - 1, 0)], d[i_pos], d[min(i_pos + 1, N - 1)]
            delta = _parabolic_subpixel(y_m1, y_0, y_p1)
            pos = float(i_pos + np.clip(delta, -1.0, 1.0))
            return pos, float(v_pos), +1
        else:
            y_m1, y_0, y_p1 = d[max(i_neg - 1, 0)], d[i_neg], d[min(i_neg + 1, N - 1)]
            delta = _parabolic_subpixel(y_m1, y_0, y_p1)
            pos = float(i_neg + np.clip(delta, -1.0, 1.0))
            return pos, float(v_neg), -1
