# d exsdk/measure/geometry.py
from __future__ import annotations
from typing import Tuple
import numpy as np


def fit_line_total_least_squares(points: np.ndarray) -> np.ndarray:
    """
    TLS fit of a 2D line in implicit form: a*x + b*y + c = 0 with sqrt(a^2+b^2)=1.

    Args:
        points: (N,2) array of (x,y)

    Returns:
        np.array([a, b, c]) normalized so that sqrt(a^2+b^2)=1
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    # subtract centroid
    mean = pts.mean(axis=0)
    X = pts - mean

    # SVD on covariance for direction
    # X = U S Vt; columns of V are principal directions.
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    # direction vector along the largest variance:
    v = Vt[0, :]  # (vx, vy)
    # normal is perpendicular to direction
    n = np.array([+v[1], -v[0]], dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    a, b = n / n_norm
    c = -a * mean[0] - b * mean[1]
    return np.array([a, b, c], dtype=np.float64)


def line_angle_deg(line: np.ndarray) -> float:
    """
    Angle (deg) of the line direction with respect to +X axis in [0, 180).

    Input line is [a, b, c] with sqrt(a^2+b^2)=1. Direction vector is (-b, a).
    """
    a, b, _ = [float(x) for x in line]
    # direction vector
    vx, vy = -b, a
    ang = np.degrees(np.arctan2(vy, vx))
    if ang < 0.0:
        ang += 180.0
    # Ensure [0, 180)
    if ang >= 180.0:
        ang -= 180.0
    return float(ang)


def fit_circle_taubin(points: np.ndarray) -> Tuple[Tuple[float, float, float], float]:
    """
    Taubin (1991) algebraic circle fit.
    Returns ((xc, yc, R), rmse)

    Args:
        points: (N,2) array of (x,y), N >= 3

    Returns:
        center (xc, yc), radius R, and RMSE of radial residuals.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] < 3:
        return (np.nan, np.nan, np.nan), float("inf")

    # Normalize for numerical stability
    mean = P.mean(axis=0)
    X = P - mean
    scale = np.sqrt((X**2).sum(axis=1).mean())
    if scale < 1e-12:
        scale = 1.0
    X /= scale

    x = X[:, 0]
    y = X[:, 1]
    z = x * x + y * y

    Z = np.column_stack([z, x, y, np.ones_like(x)])

    # Build the 4x4 symmetric moment matrix
    M = (Z.T @ Z) / P.shape[0]

    # Constraint matrix (Taubin)
    H = np.array([
        [8, 0, 0, 2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [2, 0, 0, 0]
    ], dtype=np.float64)

    # Solve generalized eigenvalue problem M * a = λ * H * a
    # Pick eigenvector with smallest positive eigenvalue
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(H) @ M)
    except np.linalg.LinAlgError:
        # fallback: standard eigen of symmetric form
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(H) @ M)

    # Filter finite, real eigenvalues
    real_mask = np.isfinite(eigvals) & (np.abs(eigvals.imag) < 1e-10)
    if not np.any(real_mask):
        return (np.nan, np.nan, np.nan), float("inf")
    eigvals = eigvals.real[real_mask]
    eigvecs = eigvecs.real[:, real_mask]

    # Choose the eigenvector associated with the smallest positive eigenvalue
    pos_mask = eigvals > 0
    if np.any(pos_mask):
        idx = int(np.argmin(eigvals[pos_mask]))
        # map idx back to original positions
        take = np.flatnonzero(pos_mask)[idx]
    else:
        # If none positive, just take the smallest absolute
        take = int(np.argmin(np.abs(eigvals)))

    a = eigvecs[:, take]  # [A, B, C, D] in normalized space

    A, B, C, D = [float(v) for v in a]

    # Recover circle parameters in normalized coords:
    # A*z + B*x + C*y + D = 0
    # where z = x^2 + y^2
    # Rewrite to (x - ux)^2 + (y - uy)^2 = R^2
    if abs(A) < 1e-12:
        return (np.nan, np.nan, np.nan), float("inf")

    ux = -B / (2.0 * A)
    uy = -C / (2.0 * A)
    Rn = np.sqrt(max(ux * ux + uy * uy - D / A, 0.0))

    # Denormalize
    xc = ux * scale + mean[0]
    yc = uy * scale + mean[1]
    R = Rn * scale

    # RMSE of radial residuals
    r = np.sqrt((P[:, 0] - xc) ** 2 + (P[:, 1] - yc) ** 2)
    rmse = float(np.sqrt(np.mean((r - R) ** 2)))
    return (float(xc), float(yc), float(R)), rmse
