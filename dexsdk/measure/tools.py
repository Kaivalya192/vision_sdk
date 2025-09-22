# dexsdk/measure/tools.py
import math
import numpy as np
import cv2

def _unit(v):
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)

def _quad_subpixel_peak(y, i):
    # Parabolic interpolation of a 1D peak at integer i using neighbors i-1,i,i+1
    if i <= 0 or i >= len(y)-1: 
        return float(i)
    a = 0.5 * (y[i+1] + y[i-1]) - y[i]
    b = 0.5 * (y[i+1] - y[i-1])
    if abs(a) < 1e-12: 
        return float(i)
    return float(i) - b / (2*a)

def _sample_scanlines(gray, c0, c1, band_px=20, n_scans=24, samples_per_scan=64):
    """
    Sample intensity profiles orthogonal to the line c0->c1.
    Returns sample grid points (x,y) and sampled intensity [n_scans, samples_per_scan].
    """
    h, w = gray.shape[:2]
    c0 = np.asarray(c0, dtype=np.float32)
    c1 = np.asarray(c1, dtype=np.float32)
    t  = _unit(c1 - c0)
    n  = np.array([-t[1], t[0]], dtype=np.float32)  # normal

    # centers of scanlines along the segment
    L = np.linalg.norm(c1 - c0)
    if L < 2:
        raise ValueError("Segment too short")
    us = np.linspace(0, 1, n_scans).astype(np.float32)
    centers = c0[None,:] + (c1 - c0)[None,:] * us[:,None]

    # along-normal coords for each scan
    vs = np.linspace(-band_px/2, band_px/2, samples_per_scan).astype(np.float32)
    # broadcast to grid: [n_scans, samples_per_scan, 2]
    pts = centers[:,None,:] + vs[None,:,None] * n[None,None,:]
    map_x = pts[...,0].astype(np.float32)
    map_y = pts[...,1].astype(np.float32)

    # bilinear sampling (use BORDER_REPLICATE for robustness)
    prof = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                     borderMode=cv2.BORDER_REPLICATE)
    return pts, prof  # pts: [S,N,2], prof: [S,N]

def _find_edge_positions_1d(profile, polarity='any', smooth_ksize=5):
    """
    Given 1D intensity, compute derivative and return peak index (sub-pixel).
    polarity: 'rise', 'fall', or 'any'
    """
    if smooth_ksize >= 3:
        profile = cv2.GaussianBlur(profile[None,None,:], (1, smooth_ksize|1), 0).ravel()
    # Scharr/Sobel derivative along samples
    grad = np.convolve(profile, [ -1, 0, +1 ], mode='same')
    if polarity == 'rise':
        i = int(np.argmax(grad))
        return _quad_subpixel_peak(grad, i)
    elif polarity == 'fall':
        i = int(np.argmin(grad))
        return _quad_subpixel_peak(-grad, i)  # flip so peak logic applies
    else:
        ip = int(np.argmax(grad))
        ineg = int(np.argmin(grad))
        # choose stronger edge by absolute gradient
        return _quad_subpixel_peak(grad, ip) if abs(grad[ip]) >= abs(grad[ineg]) else _quad_subpixel_peak(-grad, ineg)

def caliper_line(gray, p0, p1, band_px=20, n_scans=24, samples_per_scan=64, polarity='any', min_contrast=8.0):
    """
    Return sub-pixel edge points (x,y) along a band orthogonal to p0->p1 and a best-fit line.
    """
    gray = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    pts, prof = _sample_scanlines(gray, np.array(p0, np.float32), np.array(p1, np.float32),
                                  band_px=band_px, n_scans=n_scans, samples_per_scan=samples_per_scan)
    edge_xy = []
    for s in range(prof.shape[0]):
        p = prof[s]
        # simple contrast check
        if (p.max() - p.min()) < min_contrast: 
            continue
        u = _find_edge_positions_1d(p, polarity=polarity, smooth_ksize=5)  # sub-pixel sample index
        # clamp
        u = max(0.0, min(len(p)-1.0, u))
        # interpolate 2D point at fractional index along the scanline
        j = int(np.floor(u))
        a = u - j
        xy = (1.0 - a) * pts[s, j] + a * pts[s, min(j+1, pts.shape[1]-1)]
        edge_xy.append(xy)
    edge_xy = np.array(edge_xy, dtype=np.float32)
    if len(edge_xy) < 5:
        return {"ok": False, "msg": "not enough edges", "points": edge_xy}

    # best-fit line using PCA (robust & fast)
    mean = edge_xy.mean(axis=0)
    U, S, Vt = np.linalg.svd(edge_xy - mean, full_matrices=False)
    dir_vec = Vt[0]  # principal direction
    dir_vec = _unit(dir_vec)
    normal = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32)
    return {
        "ok": True,
        "points": edge_xy,
        "line": {"point": mean.astype(float).tolist(), "dir": dir_vec.astype(float).tolist(), "normal": normal.astype(float).tolist()},
    }

def distance_point_to_line(pt, line):
    p = np.array(pt, np.float32)
    p0 = np.array(line["point"], np.float32)
    n  = np.array(line["normal"], np.float32)
    return float(abs(np.dot(p - p0, n)))

def distance_p2p(p1, p2):
    p1 = np.array(p1, np.float32); p2 = np.array(p2, np.float32)
    return float(np.linalg.norm(p2 - p1))

def angle_between_lines(line1, line2):
    a = _unit(np.array(line1["dir"], np.float32)); b = _unit(np.array(line2["dir"], np.float32))
    ang = math.degrees(math.atan2(a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1]))
    return abs(ang)
