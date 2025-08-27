# ==========================
# FILE: dexsdk/detection.py
# ==========================
"""SIFT matcher with RGB-aware (soft) gating + multi-instance + polygon masks.

New:
- set_template_polygon(roi_bgr, roi_mask) to set a non-rectangular template.
- Masked SIFT detectAndCompute + masked HSV/LAB color gating.
- Warps the template mask and uses it while comparing color for gating.
"""
from typing import Optional, Tuple, Dict, List
import cv2, numpy as np


def _prep_gray(img: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
        g = cv2.GaussianBlur(g, (3, 3), 0)
    return g


def _make_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("Your OpenCV build lacks SIFT. Install 'opencv-contrib-python'.")
    return cv2.SIFT_create()


def _kpdesc(detector, gray: np.ndarray, mask: Optional[np.ndarray] = None):
    # mask: uint8 0/255, same size as gray
    return detector.detectAndCompute(gray, mask)


def _ratio_match_L2(des1, des2, ratio: float, crosscheck_if_empty: bool = True) -> List[cv2.DMatch]:
    if des1 is None or des2 is None:
        return []
    if des1.dtype != des2.dtype or des1.shape[1] != des2.shape[1]:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good: List[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2: continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if not good and crosscheck_if_empty:
        good = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(des1, des2)
    return good


def _hsv_hist_masked(bgr: np.ndarray, mask: Optional[np.ndarray] = None):
    """Return normalized 2D HSV hist (H,S) with low-S/V masked out + valid ratio."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # sat/val validity
    sv_mask = cv2.inRange(hsv, (0, 32, 32), (180, 255, 255))
    if mask is not None:
        # ensure binary
        m = (mask > 0).astype(np.uint8) * 255
        final_mask = cv2.bitwise_and(sv_mask, m)
    else:
        final_mask = sv_mask
    valid = int(cv2.countNonZero(final_mask))
    hist = cv2.calcHist([hsv], [0, 1], final_mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    total = bgr.shape[0] * bgr.shape[1]
    valid_ratio = (valid / max(1, total))
    return hist, valid_ratio


def _lab_mean_deltaE(bgr_a: np.ndarray, bgr_b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean CIE76 ΔE; if mask provided, average over mask>0 region only."""
    lab_a = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    dE = np.linalg.norm(lab_a - lab_b, axis=2)
    if mask is not None:
        m = (mask > 0)
        if m.any():
            return float(dE[m].mean())
    return float(np.mean(dE))


class SIFTMatcher:
    def __init__(
        self,
        *,
        lowe_ratio: float = 0.90,
        max_matches: int = 150,
        min_inliers: int = 4,
        ransac_thr_px: float = 4.0,
        min_score: float = 0.25,
        full_affine_fallback: bool = False,
        use_clahe: bool = True,
        # Color-gate (soft) defaults
        use_color_gate: bool = True,
        color_gate_bhat: float = 0.70,
        color_gate_corr: float = 0.10,
        color_gate_lab: float  = 35.0,
        soft_inlier_boost: int = 3,
        soft_score_boost: float = 0.05,
        max_instances: int = 2,
    ):
        self.sift = _make_sift()
        self.use_clahe = bool(use_clahe)
        self.params = dict(
            lowe_ratio=float(lowe_ratio),
            max_matches=int(max_matches),
            min_inliers=int(min_inliers),
            ransac_thr_px=float(ransac_thr_px),
            min_score=float(min_score),
            full_affine_fallback=bool(full_affine_fallback),
            use_color_gate=bool(use_color_gate),
            color_gate_bhat=float(color_gate_bhat),
            color_gate_corr=float(color_gate_corr),
            color_gate_lab=float(color_gate_lab),
            soft_inlier_boost=int(soft_inlier_boost),
            soft_score_boost=float(soft_score_boost),
            max_instances=int(max_instances),
        )
        # Template cache
        self.template_bgr: Optional[np.ndarray] = None
        self.tpl_mask: Optional[np.ndarray] = None      # uint8 0/255, same size as template
        self.tpl_gray: Optional[np.ndarray] = None
        self.tpl_kp = None
        self.tpl_des = None
        self.tpl_size: Optional[Tuple[int, int]] = None  # (w,h)
        self.tpl_hist: Optional[np.ndarray] = None
        self.tpl_hist_valid_ratio: float = 1.0

    # ---- public API ----
    def set_use_clahe(self, enabled: bool):
        self.use_clahe = bool(enabled)
        if self.template_bgr is not None:
            self._recompute_template()

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.params:
                if isinstance(self.params[k], bool): self.params[k] = bool(v)
                elif isinstance(self.params[k], int): self.params[k] = int(v)
                else: self.params[k] = float(v)

    def set_template(self, roi_bgr: np.ndarray):
        """Rectangular template (no mask)."""
        self.template_bgr = roi_bgr.copy()
        self.tpl_mask = None
        self._recompute_template()

    def set_template_polygon(self, roi_bgr: np.ndarray, roi_mask: np.ndarray):
        """Non-rectangular template with mask (uint8 0/255)."""
        self.template_bgr = roi_bgr.copy()
        m = roi_mask
        if m.dtype != np.uint8: m = m.astype(np.uint8)
        if m.ndim == 3: m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        self.tpl_mask = (m > 0).astype(np.uint8) * 255
        self._recompute_template()

    def clear_template(self):
        self.template_bgr = None
        self.tpl_mask = None
        self.tpl_gray = None
        self.tpl_kp = None
        self.tpl_des = None
        self.tpl_size = None
        self.tpl_hist = None
        self.tpl_hist_valid_ratio = 1.0

    def compute(self, scene_bgr: np.ndarray, *, draw: bool = True):
        """Run matching+pose on a downscaled/processed scene image (BGR).
        Returns (pose_first, overlay, debug) with multi-instance support.
        """
        overlay = scene_bgr.copy()
        pose_first: Optional[dict] = None

        if self.tpl_kp is None or self.tpl_des is None or len(self.tpl_kp) < 4:
            return (None, overlay, {"fail_stage": "no_template", "tpl_kp": 0, "scene_kp": 0, "matches": 0, "inliers": 0, "instances": 0, "poses": []})

        scene_gray = _prep_gray(scene_bgr, use_clahe=self.use_clahe)
        kp2, des2 = _kpdesc(self.sift, scene_gray, None)
        if des2 is None or len(kp2) < 4:
            return (None, overlay, {"fail_stage": "scene_few_kp", "tpl_kp": len(self.tpl_kp), "scene_kp": 0, "matches": 0, "inliers": 0, "instances": 0, "poses": []})

        good = _ratio_match_L2(self.tpl_des, des2, ratio=float(self.params["lowe_ratio"]))
        good = sorted(good, key=lambda m: m.distance)[: int(self.params["max_matches"]) * max(1, int(self.params["max_instances"]))]

        dbg_base = dict(tpl_kp=len(self.tpl_kp), scene_kp=len(kp2), matches=len(good))
        if len(good) < 4:
            db = {**dbg_base, "inliers": 0, "instances": 0, "poses": []}
            db["fail_stage"] = "few_matches"
            return None, overlay, db

        tpl_pts_all = np.float32([self.tpl_kp[m.queryIdx].pt for m in good])
        scn_pts_all = np.float32([kp2[m.trainIdx].pt for m in good])
        active_idx = np.arange(len(good))

        detections: List[dict] = []
        max_inst = max(1, int(self.params["max_instances"]))

        for _ in range(max_inst):
            if active_idx.size < 4: break
            pts1 = tpl_pts_all[active_idx]
            pts2 = scn_pts_all[active_idx]

            M, inliers = cv2.estimateAffinePartial2D(
                pts1, pts2, method=cv2.RANSAC,
                ransacReprojThreshold=float(self.params["ransac_thr_px"]),
                maxIters=4000, confidence=0.995, refineIters=10
            )
            if (M is None or inliers is None) and bool(self.params["full_affine_fallback"]):
                M, inliers = cv2.estimateAffine2D(
                    pts1, pts2, method=cv2.RANSAC,
                    ransacReprojThreshold=float(self.params["ransac_thr_px"]),
                    maxIters=4000, confidence=0.995, refineIters=10
                )
            if M is None or inliers is None: break

            inl_mask = inliers.ravel().astype(bool)
            ninl = int(inl_mask.sum())
            score = float(ninl / max(1, active_idx.size))
            geom_ok = (ninl >= int(self.params["min_inliers"])) and (score >= float(self.params["min_score"]))

            color_ok = True
            color_bhat = color_corr = color_dE = -1.0
            if geom_ok and bool(self.params["use_color_gate"]) and self.tpl_hist is not None and self.tpl_size is not None:
                w, h = self.tpl_size
                warped = cv2.warpAffine(scene_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                warped_mask = None
                if self.tpl_mask is not None:
                    warped_mask = cv2.warpAffine(self.tpl_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                hist2, valid_ratio = _hsv_hist_masked(warped, warped_mask)
                if valid_ratio >= 0.10 and self.tpl_hist_valid_ratio >= 0.10:
                    color_bhat = float(cv2.compareHist(self.tpl_hist, hist2, cv2.HISTCMP_BHATTACHARYYA))
                    color_corr = float(cv2.compareHist(self.tpl_hist, hist2, cv2.HISTCMP_CORREL))
                    color_dE = _lab_mean_deltaE(self.template_bgr, warped, warped_mask)
                    bhat_ok = (color_bhat <= float(self.params["color_gate_bhat"]))
                    corr_ok = (color_corr >= float(self.params["color_gate_corr"]))
                    dE_ok   = (color_dE  <= float(self.params["color_gate_lab"]))
                    color_ok = bhat_ok or corr_ok or dE_ok
                    if not color_ok:
                        if (ninl >= int(self.params["min_inliers"]) + int(self.params["soft_inlier_boost"])) and \
                           (score >= float(self.params["min_score"]) + float(self.params["soft_score_boost"])):
                            color_ok = True

            accept = geom_ok and color_ok
            if accept:
                a, b, tx = M[0]; c, d, ty = M[1]
                if np.isfinite(M).all() and (a*d - b*c) > 1e-6:
                    theta = -np.degrees(np.arctan2(c, a))
                    x_scale = float(np.hypot(a, c))
                    y_scale = float(np.hypot(b, d))
                    wq, hq = self.tpl_size if self.tpl_size is not None else (0, 0)
                    quad, center = None, None
                    if self.tpl_size is not None:
                        corners = np.float32([[0,0],[wq,0],[wq,hq],[0,hq]])
                        quad_m = np.hstack([corners, np.ones((4,1), np.float32)]) @ M.T
                        if np.isfinite(quad_m).all():
                            quad = quad_m.astype(float).tolist()
                            cxy = quad_m.mean(axis=0); center = [float(cxy[0]), float(cxy[1])]
                            if draw:
                                cv2.polylines(overlay, [quad_m.astype(np.int32)], True, (0,255,0), 2)
                                cv2.circle(overlay, (int(center[0]), int(center[1])), 5, (0,255,0), -1)
                    det = dict(
                        x=float(tx), y=float(ty), theta=float(theta),
                        x_scale=x_scale, y_scale=y_scale, score=score, ninliers=ninl,
                        color_bhat=color_bhat, color_corr=color_corr, color_deltaE=color_dE,
                        quad=quad, center=center,
                    )
                    detections.append(det)
                    active_idx = active_idx[~inl_mask]  # peel only on accept
                else:
                    break
            else:
                break

        if not detections:
            db = {**dbg_base, "inliers": 0, "instances": 0, "poses": []}
            db["fail_stage"] = "no_detection"
            return None, overlay, db

        detections_sorted = sorted(detections, key=lambda d: (d["ninliers"], d["score"]), reverse=True)
        pose_first = {k: detections_sorted[0][k] for k in ["x", "y", "theta", "x_scale", "y_scale", "score"]}
        debug_out = {**dbg_base, "inliers": detections_sorted[0]["ninliers"], "instances": len(detections_sorted), "poses": detections_sorted, "fail_stage": "—"}
        return pose_first, overlay, debug_out

    # ---- internals ----
    def _recompute_template(self):
        if self.template_bgr is None:
            self.tpl_mask = None
            self.tpl_gray = None
            self.tpl_kp = None
            self.tpl_des = None
            self.tpl_size = None
            self.tpl_hist = None
            self.tpl_hist_valid_ratio = 1.0
            return
        self.tpl_gray = _prep_gray(self.template_bgr, use_clahe=self.use_clahe)
        # SIFT with optional mask
        self.tpl_kp, self.tpl_des = _kpdesc(self.sift, self.tpl_gray, self.tpl_mask)
        self.tpl_size = (self.template_bgr.shape[1], self.template_bgr.shape[0])
        # color model (masked)
        self.tpl_hist, self.tpl_hist_valid_ratio = _hsv_hist_masked(self.template_bgr, self.tpl_mask)
