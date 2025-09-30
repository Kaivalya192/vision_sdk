# app/measure_ui.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Calibrate + Measure UI (PyQt5)

- Single PiCamera2 pipeline (RGB888 @ 640x480) shared across both tabs
- Tab 1: Calibration (AprilGrid only), follows your working syntax/logic verbatim
- Tab 2: Measurement (distance/angle/circle), px and mm using intrinsics.yaml
- Auto-loads ./intrinsics.yaml for the Measure tab if present
- Saves:
    - Captures:       ./captures
    - Calib overlays: ./calib_vis
    - Measurements:   ./measurements

Author: you + me :)
"""

import os, sys, time, traceback, json, csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# --- Optional fix for Qt plugin resolution on some Pi images (safe to keep) ---
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import cv2

# ---- FIX Qt plugin issue ----
# Force Qt to use system plugins, not OpenCV's bundled ones
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/arm-linux-gnueabihf/qt5/plugins/platforms"
os.environ["QT_QPA_PLATFORM"] = "xcb"  # fallback: try "eglfs" if no desktop/X11

from PyQt5 import QtCore, QtGui, QtWidgets

# Camera
from picamera2 import Picamera2
from libcamera import controls

# ----- AprilGrid detector -----
try:
    from aprilgrid import Detector as AprilDetector
    HAVE_APRILGRID = True
except Exception:
    HAVE_APRILGRID = False


# ======================== Shared camera thread ========================

class CameraThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)   # RGB888

    def __init__(self, picam: Picamera2, parent=None):
        super().__init__(parent)
        self.picam = picam
        self._running = True
        self.sleep_ms = 30

    def run(self):
        while self._running:
            try:
                frame = self.picam.capture_array()  # RGB888 (as per config)
                if frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                    self.frame_ready.emit(frame)
            except Exception:
                pass
            self.msleep(self.sleep_ms)

    def stop(self):
        self._running = False
        self.wait(1000)


# ======================== Calibration data containers ========================

@dataclass
class CalibResult:
    model: str                   # "pinhole" or "fisheye"
    image_size: Tuple[int,int]   # (w, h)
    K: np.ndarray                # 3x3
    dist: np.ndarray             # (14,1) pinhole (could be 5/8/14) or (4,1) fisheye
    rvecs: List[np.ndarray]      # per view (3x1)
    tvecs: List[np.ndarray]      # per view (3x1)
    rms: float
    per_view_errs: List[float]
    used_paths: List[str]        # filenames or synthetic names


# ======================== AprilGrid helpers (verbatim) ======================

def detect_aprilgrid(gray: np.ndarray, family: str) -> List[Dict]:
    """Return list of dicts: {id:int, corners_px:(4,2) float32 in tl,tr,br,bl}"""
    if not HAVE_APRILGRID:
        return []
    dets = AprilDetector(family).detect(gray)
    out = []
    for d in dets:
        out.append({"id": int(d.tag_id),
                    "corners_px": np.array(d.corners, dtype=np.float32).reshape(4,2)})
    return out

def aprilgrid_obj_corners(rows: int, cols: int, tag_mm: float, gap_mm: float,
                          row: int, col: int) -> np.ndarray:
    pitch = tag_mm + gap_mm
    x0 = col * pitch
    y0 = row * pitch
    obj = np.array([
        [x0,         y0,          0.0],   # tl
        [x0+tag_mm,  y0,          0.0],   # tr
        [x0+tag_mm,  y0+tag_mm,   0.0],   # br
        [x0,         y0+tag_mm,   0.0],   # bl
    ], dtype=np.float32)
    return obj

def aprilgrid_build_correspondences(gray: np.ndarray, rows: int, cols: int,
                                    tag_mm: float, gap_mm: float,
                                    family: str = "t36h11"
                                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict]]:
    dets = detect_aprilgrid(gray, family)
    if not dets:
        return None, None, []

    APRIL_BASE_ID = 0  # your board: 0,6,12,... across the first row (column-major)
    pitch = tag_mm + gap_mm

    obj_list, img_list = [], []
    for d in dets:
        tid = int(d["id"])
        idx = tid - APRIL_BASE_ID
        if idx < 0:
            continue

        # Column-major mapping (0,6,12,... left→right for rows=6)
        r = idx % rows        # row index (down)
        c = idx // rows       # col index (right)
        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue

        # --- use tag CENTER, not corners ---
        x0 = c * pitch + 0.5 * tag_mm  # object X (mm)
        y0 = r * pitch + 0.5 * tag_mm  # object Y (mm)
        obj_list.append([x0, y0, 0.0])

        corners = d["corners_px"].astype(np.float32).reshape(-1, 2)
        cx = float(corners[:, 0].mean())
        cy = float(corners[:, 1].mean())
        img_list.append([cx, cy])

    if not obj_list:
        return None, None, dets

    obj = np.asarray(obj_list, dtype=np.float32)   # (N,3)
    img = np.asarray(img_list, dtype=np.float32)   # (N,2)
    return obj, img, dets

def _edge_sizes_px(c4x2: np.ndarray)->Tuple[float,float]:
    p = c4x2.reshape(-1,2)
    tl,tr,br,bl = p[0],p[1],p[2],p[3]
    w = 0.5*(np.linalg.norm(tr-tl)+np.linalg.norm(br-bl))
    h = 0.5*(np.linalg.norm(bl-tl)+np.linalg.norm(br-tr))
    return float(w), float(h)

def _robust_median_trim(vals: List[float])->float:
    a = np.asarray(vals, dtype=float)
    if a.size==0: return float("nan")
    med = float(np.median(a))
    if a.size<5: return med
    mad = float(np.median(np.abs(a-med)))
    if mad==0: return med
    keep = a[np.abs(a-med) <= 3.0*1.4826*mad]
    return float(np.median(keep)) if keep.size else med

def compute_pxmm_from_aprilgrid(gray: np.ndarray, rows: int, cols: int,
                                tag_mm: float, gap_mm: float, family: str,
                                pair_weight: float = 0.5,
                                bias_blend: float = 1.0
                                ) -> Optional[Dict[str, float]]:
    """
    Returns dict with px_per_mm_x, px_per_mm_y, mm_per_px_x, mm_per_px_y.
    Uses tag edge widths and neighbor center pitch (robust medians + blending).
    """
    obj, img, dets = aprilgrid_build_correspondences(gray, rows, cols, tag_mm, gap_mm, family)
    if obj is None or not dets:
        return None
    ids = sorted(int(d["id"]) for d in dets)
    dset = set(ids)
    APRIL_BASE_ID = 0
    PITCH_MM = tag_mm + gap_mm

    # Per-tag centers and (w,h) in px
    centers, widths = {}, {}
    for d in dets:
        tid = int(d["id"])
        c = d["corners_px"].astype(np.float32)
        w_px, h_px = _edge_sizes_px(c)
        centers[tid] = (float(c[:,0].mean()), float(c[:,1].mean()))
        widths[tid]  = (w_px, h_px)

    # Edge-based px/mm
    edge_x = [widths[i][0]/tag_mm for i in ids]  # px per mm (X)
    edge_y = [widths[i][1]/tag_mm for i in ids]
    
    # Neighbor-based px/mm from center pitch (COLUMN-MAJOR)
    nbr_x, nbr_y = [], []
    for r in range(rows):
        for c in range(cols - 1):
            i1 = APRIL_BASE_ID + r + c*rows
            i2 = APRIL_BASE_ID + r + (c+1)*rows
            if i1 in dset and i2 in dset:
                dx = abs(centers[i2][0] - centers[i1][0])
                nbr_x.append(dx / PITCH_MM)
    for r in range(rows - 1):
        for c in range(cols):
            i1 = APRIL_BASE_ID + r + c*rows
            i2 = APRIL_BASE_ID + (r+1) + c*rows
            if i1 in dset and i2 in dset:
                dy = abs(centers[i2][1] - centers[i1][1])
                nbr_y.append(dy / PITCH_MM)
    ex = _robust_median_trim(edge_x)
    ey = _robust_median_trim(edge_y)
    nx = _robust_median_trim(nbr_x) if len(nbr_x)>0 else np.nan
    ny = _robust_median_trim(nbr_y) if len(nbr_y)>0 else np.nan

    w_pair = float(np.clip(pair_weight, 0.0, 1.0))
    px_per_mm_x = (1-w_pair)*ex + w_pair*(nx if not np.isnan(nx) else ex)
    px_per_mm_y = (1-w_pair)*ey + w_pair*(ny if not np.isnan(ny) else ey)

    # Bias-correct so median measured tag edge == tag_mm (like your script)
    if bias_blend > 0:
        w_mm = [widths[i][0]/px_per_mm_x for i in ids]
        h_mm = [widths[i][1]/px_per_mm_y for i in ids]
        med_w, med_h = (float(np.median(w_mm)), float(np.median(h_mm)))
        if med_w > 1e-9:
            px_per_mm_x *= (tag_mm/med_w) ** bias_blend
        if med_h > 1e-9:
            px_per_mm_y *= (tag_mm/med_h) ** bias_blend

    return {
        "px_per_mm_x": float(px_per_mm_x),
        "px_per_mm_y": float(px_per_mm_y),
        "mm_per_px_x": float(1.0/px_per_mm_x) if px_per_mm_x>1e-12 else float("nan"),
        "mm_per_px_y": float(1.0/px_per_mm_y) if px_per_mm_y>1e-12 else float("nan"),
    }


# ======================== Calibration back-end (verbatim) ===================

def calibrate_pinhole(objpoints: List[np.ndarray], imgpoints: List[np.ndarray],
                      imsize: Tuple[int,int]) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_ZERO_TANGENT_DIST
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, imsize, None, None, flags=flags
    )
    return rms, K, dist, rvecs, tvecs

def calibrate_fisheye(objpoints: List[np.ndarray], imgpoints: List[np.ndarray],
                      imsize: Tuple[int,int]) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    obj_fe = [op.reshape(-1,1,3).astype(np.float64) for op in objpoints]
    img_fe = [ip.reshape(-1,1,2).astype(np.float64) for ip in imgpoints]
    K = np.zeros((3,3), dtype=np.float64)
    D = np.zeros((4,1), dtype=np.float64)
    rvecs, tvecs = [], []
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_CHECK_COND |
             cv2.fisheye.CALIB_FIX_SKEW)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_fe, img_fe, imsize, K, D, rvecs, tvecs, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    )
    return rms, K, D, rvecs, tvecs

def per_view_error(obj: np.ndarray, img: np.ndarray,
                   rvec: np.ndarray, tvec: np.ndarray,
                   K: np.ndarray, dist: np.ndarray, model: str) -> float:
    if model == "fisheye":
        proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1,1,3), rvec, tvec, K, dist)
        proj = proj.reshape(-1,2)
    else:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj = proj.reshape(-1,2)
    return float(np.linalg.norm(img - proj, axis=1).mean())


class CalibWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)      # CalibResult
    failed = QtCore.pyqtSignal(str)

    def __init__(self, packs, imsize, model, save_yaml_path):
        super().__init__()
        self.packs = packs            # list of tuples (objpoints Nx3, imgpoints Nx2, path_name)
        self.imsize = imsize
        self.model = model            # 'pinhole' | 'fisheye'
        self.save_yaml_path = save_yaml_path

    def run(self):
        try:
            objpoints = [p[0] for p in self.packs]
            imgpoints = [p[1] for p in self.packs]
            names     = [p[2] for p in self.packs]

            if len(objpoints) < 3:
                raise RuntimeError("Need at least 3 captured views.")

            if self.model == "fisheye":
                rms, K, dist, rvecs, tvecs = calibrate_fisheye(objpoints, imgpoints, self.imsize)
            else:
                rms, K, dist, rvecs, tvecs = calibrate_pinhole(objpoints, imgpoints, self.imsize)

            # Per-view errors
            errs = []
            for (obj, img), rv, tv in zip(zip(objpoints, imgpoints), rvecs, tvecs):
                errs.append(per_view_error(obj, img, rv, tv, K, dist, self.model))

            # Save YAML (baseline; px/mm will be written after in main thread)
            fs = cv2.FileStorage(self.save_yaml_path, cv2.FILE_STORAGE_WRITE)
            fs.write("model", self.model)
            fs.write("image_width", int(self.imsize[0]))
            fs.write("image_height", int(self.imsize[1]))
            fs.write("camera_matrix", K)
            fs.write("distortion_coefficients", dist)
            fs.write("avg_reprojection_error", float(rms))
            fs.write("num_views", int(len(rvecs)))
            fs.write("rvecs", np.array([rv.flatten() for rv in rvecs], dtype=np.float64))
            fs.write("tvecs", np.array([tv.flatten() for tv in tvecs], dtype=np.float64))
            fs.release()

            result = CalibResult(self.model, self.imsize, K, dist, rvecs, tvecs, rms, errs, names)
            self.finished_ok.emit(result)

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ======================== Measurement helpers (from your PoC) =================

def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")

def to_qimage_rgb(rgb: np.ndarray) -> QtGui.QImage:
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)

def px_to_mm(dx_px: float, dy_px: float, mm_per_px: Optional[Tuple[float, float]]) -> float:
    """Anisotropic conversion if separate mm/px for X & Y; returns Euclidean mm length."""
    if not mm_per_px:
        return float("nan")
    mmx, mmy = mm_per_px
    return float(np.sqrt((dx_px * mmx) ** 2 + (dy_px * mmy) ** 2))

class ClickLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(int, int)  # label coords
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(ev.x(), ev.y())
        super().mousePressEvent(ev)

@dataclass
class MeasureItem:
    kind: str                # 'distance' | 'angle' | 'circle'
    pts: List[Tuple[int,int]] # image coords (px) used to define the measurement
    px_value: float          # length (px) or radius (px) or angle (deg)
    mm_value: float          # mm (NaN if scale unknown); for angle, same as px_value
    text: str                # label to draw
    color: Tuple[int,int,int] = (40, 255, 220)
    thickness: int = 2


# ======================== Tab: Calibration ========================

class CalibrateTab(QtWidgets.QWidget):
    def __init__(self, picam: Picamera2, cam_thread: CameraThread, parent=None):
        super().__init__(parent)
        self.picam = picam
        self.cam_thread = cam_thread

        self.live_frame: Optional[np.ndarray] = None
        self.undistort_live = False
        self.undist_map = None               # (map1, map2) or None
        self.calib: Optional[CalibResult] = None
        self.pxmm: Optional[Tuple[float,float]] = None  # (px/mm x, px/mm y)

        # ======= UI =======
        hbox = QtWidgets.QHBoxLayout(self)

        # Left: video + overlays
        left = QtWidgets.QVBoxLayout()
        self.view_label = QtWidgets.QLabel("Live")
        self.view_label.setAlignment(QtCore.Qt.AlignCenter)
        self.view_label.setMinimumSize(640, 480)
        self.view_label.setStyleSheet("background:#111; border:1px solid #333;")
        left.addWidget(self.view_label, stretch=1)

        # Buttons row
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_capture = QtWidgets.QPushButton("Capture Frame")
        self.btn_clear   = QtWidgets.QPushButton("Clear Captures")
        self.btn_calib   = QtWidgets.QPushButton("Calibrate")
        self.chk_undist  = QtWidgets.QCheckBox("Undistort Live")
        btn_row.addWidget(self.btn_capture); btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_calib);   btn_row.addWidget(self.chk_undist)
        left.addLayout(btn_row)

        # Captures list
        self.capt_list = QtWidgets.QListWidget()
        self.capt_list.setMaximumHeight(140)
        left.addWidget(self.capt_list)

        hbox.addLayout(left, stretch=3)

        # Right: controls & debug
        right = QtWidgets.QVBoxLayout()

        # Camera controls
        cam_group = QtWidgets.QGroupBox("Camera Controls")
        cam_form = QtWidgets.QFormLayout(cam_group)
        self.sld_exp  = self._mk_slider(100, 20000, 5000)
        self.sld_gain = self._mk_slider(1, 16, 4)
        self.sld_foc  = self._mk_slider(0, 400, 0)
        cam_form.addRow("Exposure (µs)", self.sld_exp["widget"])
        cam_form.addRow("Analogue Gain",  self.sld_gain["widget"])
        cam_form.addRow("Lens Focus (x100 diopters)", self.sld_foc["widget"])
        right.addWidget(cam_group)

        # Model selection
        patt_group = QtWidgets.QGroupBox("Pattern & Model")
        patt_form = QtWidgets.QFormLayout(patt_group)
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["pinhole", "fisheye"])
        patt_form.addRow("Model",   self.cmb_model)
        right.addWidget(patt_group)

        # AprilGrid params
        self.grp_april = QtWidgets.QGroupBox("AprilGrid Params")
        form_ag = QtWidgets.QFormLayout(self.grp_april)
        self.sp_ag_rows = QtWidgets.QSpinBox(); self.sp_ag_rows.setRange(2, 20); self.sp_ag_rows.setValue(6)
        self.sp_ag_cols = QtWidgets.QSpinBox(); self.sp_ag_cols.setRange(2, 20); self.sp_ag_cols.setValue(5)
        self.dsb_tag_mm = QtWidgets.QDoubleSpinBox(); self.dsb_tag_mm.setRange(1, 1000); self.dsb_tag_mm.setValue(30.0)
        self.dsb_gap_mm = QtWidgets.QDoubleSpinBox(); self.dsb_gap_mm.setRange(0, 1000); self.dsb_gap_mm.setValue(6.0)
        self.le_family  = QtWidgets.QLineEdit("t36h11")
        form_ag.addRow("Rows", self.sp_ag_rows)
        form_ag.addRow("Cols", self.sp_ag_cols)
        form_ag.addRow("Tag size (mm)", self.dsb_tag_mm)
        form_ag.addRow("Gap (mm)", self.dsb_gap_mm)
        form_ag.addRow("Family", self.le_family)
        right.addWidget(self.grp_april)

        # Output paths
        out_group = QtWidgets.QGroupBox("Output")
        out_form = QtWidgets.QFormLayout(out_group)
        self.le_yaml = QtWidgets.QLineEdit(os.path.abspath("intrinsics.yaml"))
        self.btn_browse_yaml = QtWidgets.QPushButton("...")
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.le_yaml, stretch=1); path_row.addWidget(self.btn_browse_yaml)
        out_form.addRow("Save YAML", path_row)
        right.addWidget(out_group)

        # Debug / status
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        right.addWidget(QtWidgets.QLabel("Status / Debug log:"))
        right.addWidget(self.log, stretch=1)

        hbox.addLayout(right, stretch=2)

        # ======= Signals =======
        self.cam_thread.frame_ready.connect(self.on_frame)
        self.sld_exp["slider"].valueChanged.connect(self.on_cam_controls)
        self.sld_gain["slider"].valueChanged.connect(self.on_cam_controls)
        self.sld_foc["slider"].valueChanged.connect(self.on_cam_controls)
        self.btn_capture.clicked.connect(self.on_capture)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_calib.clicked.connect(self.on_calibrate)
        self.chk_undist.toggled.connect(self.on_toggle_undist)
        self.btn_browse_yaml.clicked.connect(self.on_choose_yaml)

        # state
        self.captured_frames: List[np.ndarray] = []
        self.captured_gray:   List[np.ndarray] = []
        self.captured_names:  List[str] = []
        self.load_previous_captures()

        if not HAVE_APRILGRID:
            self.append_log("[ERROR] 'aprilgrid' module not found. Install it and retry.")
        self.append_log("[READY] Configure parameters, capture frames, then Calibrate.")

    # ---------- Helpers ----------
    def _mk_slider(self, minv, maxv, init):
        w = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal); s.setRange(minv, maxv); s.setValue(init)
        v = QtWidgets.QLabel(str(init)); v.setFixedWidth(54)
        lay.addWidget(s, stretch=1); lay.addWidget(v)
        return {"widget": w, "slider": s, "val": v}

    def append_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    def save_detection_vis(self, idx: int, rgb: np.ndarray, pts: np.ndarray, patt: str, dets=None):
        """Save visualization of detected pattern overlay (AprilGrid only)."""
        vis = rgb.copy()
        if dets:
            for d in dets:
                c = d["corners_px"].astype(int)
                cv2.polylines(vis, [c], True, (0,255,0), 2, cv2.LINE_AA)
                cx, cy = c.mean(axis=0).astype(int)
                cv2.putText(vis, str(d["id"]), (cx-10, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)
        outdir = os.path.join(os.path.dirname(self.le_yaml.text()), "calib_vis")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f"cap_{idx:02d}_vis.png")
        cv2.imwrite(outpath, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self.append_log(f"[VIS] Saved {outpath}")

    # ---------- Video ----------
    def on_frame(self, frame_rgb: np.ndarray):
        self.live_frame = frame_rgb
        disp = frame_rgb

        # Undistort live if enabled and we have maps
        if self.undistort_live and self.undist_map and self.calib is not None:
            map1, map2 = self.undist_map
            disp = cv2.remap(frame_rgb, map1, map2, interpolation=cv2.INTER_LINEAR)

        # HUD
        hud = disp.copy()
        hud_line = f"{disp.shape[1]}x{disp.shape[0]}  undist={self.undistort_live}"
        if self.pxmm is not None:
            hud_line += f"  px/mm: X={self.pxmm[0]:.3f} Y={self.pxmm[1]:.3f}"
        cv2.putText(hud, hud_line, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,60), 2, cv2.LINE_AA)

        hud = cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)
        h, w, ch = hud.shape
        qimg = QtGui.QImage(hud.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        self.view_label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.view_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def on_cam_controls(self):
        exposure = int(self.sld_exp["slider"].value())
        gain     = float(self.sld_gain["slider"].value())
        focus    = float(self.sld_foc["slider"].value()/100.0)
        self.sld_exp["val"].setText(str(exposure))
        self.sld_gain["val"].setText(f"{gain:.0f}")
        self.sld_foc["val"].setText(f"{focus:.2f}")
        try:
            self.picam.set_controls({
                "AeEnable": False,
                "ExposureTime": exposure,
                "AnalogueGain": gain,
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": focus,
            })
        except Exception as e:
            self.append_log(f"[WARN] set_controls failed: {e}")
    
    def load_previous_captures(self):
        outdir = os.path.join(os.path.dirname(self.le_yaml.text()), "captures")
        if not os.path.isdir(outdir):
            return
        files = sorted([f for f in os.listdir(outdir) if f.lower().endswith(".png")])
        for f in files:
            path = os.path.join(outdir, f)
            rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            self.captured_frames.append(rgb)
            self.captured_gray.append(gray)
            self.captured_names.append(f)
            self.capt_list.addItem(f)
        if files:
            self.append_log(f"[RESTORE] Loaded {len(files)} previous captures from {outdir}")

    # ---------- Capture ----------
    def on_capture(self):
        if self.live_frame is None:
            self.append_log("[WARN] No frame yet.")
            return
        rgb = self.live_frame.copy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        idx  = len(self.captured_frames)
        name = f"cap_{idx:02d}.png"

        # Save immediately to disk
        outdir = os.path.join(os.path.dirname(self.le_yaml.text()), "captures")
        os.makedirs(outdir, exist_ok=True)
        save_path = os.path.join(outdir, name)
        cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Keep in memory too
        self.captured_frames.append(rgb)
        self.captured_gray.append(gray)
        self.captured_names.append(name)
        self.capt_list.addItem(name)
        self.append_log(f"[CAPTURE] Stored {name} → {save_path}. Total={len(self.captured_frames)}")

    def on_clear(self):
        self.captured_frames.clear()
        self.captured_gray.clear()
        self.captured_names.clear()
        self.capt_list.clear()
        self.append_log("[CLEAR] Cleared captured set.")

    def on_choose_yaml(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save intrinsics YAML",
                                                        self.le_yaml.text(), "YAML (*.yaml)")
        if path:
            self.le_yaml.setText(path)

    # ---------- Calibration ----------
    def on_calibrate(self):
        if not HAVE_APRILGRID:
            self.append_log("[ERROR] AprilGrid detector not available. Install 'aprilgrid'.")
            return
        if len(self.captured_gray) < 3:
            self.append_log("[ERROR] Need at least 3 captures with visible AprilGrid.")
            return

        model = self.cmb_model.currentText()   # 'pinhole' | 'fisheye'
        imsize = (self.captured_gray[0].shape[1], self.captured_gray[0].shape[0])

        rows = int(self.sp_ag_rows.value())
        cols = int(self.sp_ag_cols.value())
        tag_mm = float(self.dsb_tag_mm.value())
        gap_mm = float(self.dsb_gap_mm.value())
        family = self.le_family.text().strip()

        packs = []
        ok_views = 0

        for idx, (gray, rgb, name) in enumerate(zip(self.captured_gray,
                                                    self.captured_frames,
                                                    self.captured_names)):
            obj, img, dets = aprilgrid_build_correspondences(gray, rows, cols, tag_mm, gap_mm, family)
            if obj is None:
                self.append_log(f"[SKIP] No AprilGrid found in {name}")
                continue
            packs.append((obj, img, name))
            self.save_detection_vis(idx, rgb, img, "AprilGrid", dets)
            ok_views += 1

        if ok_views < 3:
            self.append_log(f"[ERROR] Only {ok_views} usable views; need at least 3.")
            return

        yaml_path = self.le_yaml.text().strip()
        self.append_log(f"[INFO] Calibrating ({model}) with {ok_views} views → {yaml_path}")

        # Background worker
        self.btn_calib.setEnabled(False)
        self.work = CalibWorker(packs, imsize, model, yaml_path)
        self.work.finished_ok.connect(self.on_calib_done)
        self.work.failed.connect(self.on_calib_fail)
        self.work.start()

    def on_calib_fail(self, msg: str):
        self.btn_calib.setEnabled(True)
        self.append_log("[ERROR] Calibration failed:\n" + msg)

    def _rewrite_yaml_with_pxmm(self, res: CalibResult, pxmm: Dict[str,float], path: str):
        """Rewrite YAML to include px/mm fields as well."""
        try:
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
            fs.write("model", res.model)
            fs.write("image_width", int(res.image_size[0]))
            fs.write("image_height", int(res.image_size[1]))
            fs.write("camera_matrix", res.K)
            fs.write("distortion_coefficients", res.dist)
            fs.write("avg_reprojection_error", float(res.rms))
            fs.write("num_views", int(len(res.rvecs)))
            fs.write("rvecs", np.array([rv.flatten() for rv in res.rvecs], dtype=np.float64))
            fs.write("tvecs", np.array([tv.flatten() for tv in res.tvecs], dtype=np.float64))
            # extra: px/mm
            fs.write("px_per_mm_x", float(pxmm["px_per_mm_x"]))
            fs.write("px_per_mm_y", float(pxmm["px_per_mm_y"]))
            fs.write("mm_per_px_x", float(pxmm["mm_per_px_x"]))
            fs.write("mm_per_px_y", float(pxmm["mm_per_px_y"]))
            fs.release()
        except Exception as e:
            self.append_log(f"[WARN] Could not write px/mm into YAML: {e}")

    def on_calib_done(self, res: CalibResult):
        self.btn_calib.setEnabled(True)
        self.calib = res
        self.append_log(f"[OK] RMS={res.rms:.4f} px; saved YAML; views={len(res.rvecs)}")
        self.append_log("K=\n" + np.array2string(res.K, precision=3, suppress_small=True))
        self.append_log("dist=\n" + np.array2string(res.dist.ravel(), precision=5, suppress_small=True))

        for i, (name, err) in enumerate(zip(res.used_paths, res.per_view_errs)):
            self.append_log(f"  View {i:02d} {name} : mean reproj err = {err:.4f} px")

        # Build undistort maps for live
        self.build_undistort_maps()
        # Compute px↔mm from first capture (based on AprilGrid)
        try:
            idx = 0  # use the first captured image only
            gray = self.captured_gray[idx]
            rows = int(self.sp_ag_rows.value())
            cols = int(self.sp_ag_cols.value())
            tag_mm = float(self.dsb_tag_mm.value())
            gap_mm = float(self.dsb_gap_mm.value())
            family = self.le_family.text().strip()
            pxmm = compute_pxmm_from_aprilgrid(gray, rows, cols, tag_mm, gap_mm, family)
            if pxmm:
                self.pxmm = (pxmm["px_per_mm_x"], pxmm["px_per_mm_y"])
                self.append_log(f"[SCALE] px/mm: X={pxmm['px_per_mm_x']:.6f}  Y={pxmm['px_per_mm_y']:.6f}")
                self.append_log(f"[SCALE] mm/px: X={pxmm['mm_per_px_x']:.6f}  Y={pxmm['mm_per_px_y']:.6f}")
                self._rewrite_yaml_with_pxmm(res, pxmm, self.le_yaml.text().strip())
            else:
                self.append_log("[SCALE] Could not compute px/mm (no AprilGrid in first capture).")
        except Exception as e:
            self.append_log(f"[SCALE] Failed to compute px/mm: {e}")

        # Show residual overlay for the last captured frame as quick sanity check
        try:
            idx = len(self.captured_frames) - 1
            disp = self.draw_residual_overlay(idx)
            if disp is not None:
                self.show_popup_image("Residuals (last capture)", disp)
        except Exception:
            pass

    # ---------- Undistort live ----------
    def on_toggle_undist(self, checked: bool):
        self.undistort_live = checked
        if checked and not self.undist_map:
            self.build_undistort_maps()

    def build_undistort_maps(self):
        if self.calib is None:
            self.append_log("[WARN] No calibration yet.")
            return
        h = self.live_frame.shape[0] if self.live_frame is not None else self.calib.image_size[1]
        w = self.live_frame.shape[1] if self.live_frame is not None else self.calib.image_size[0]
        K = self.calib.K; D = self.calib.dist

        if self.calib.model == "fisheye":
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w,h), np.eye(3), balance=0.0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), newK, (w,h), cv2.CV_16SC2)
        else:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w,h), alpha=0.0)
            map1, map2 = cv2.initUndistortRectifyMap(
                K, D, None, newK, (w,h), cv2.CV_16SC2)
        self.undist_map = (map1, map2)
        self.append_log("[OK] Undistort maps ready.")

    # ---------- Residual overlay ----------
    def draw_residual_overlay(self, idx: int) -> Optional[np.ndarray]:
        if self.calib is None: return None
        if idx < 0 or idx >= len(self.captured_gray): return None

        gray = self.captured_gray[idx]
        rgb  = self.captured_frames[idx].copy()
        model = self.calib.model
        K, D = self.calib.K, self.calib.dist

        rows = int(self.sp_ag_rows.value())
        cols = int(self.sp_ag_cols.value())
        tag_mm = float(self.dsb_tag_mm.value())
        gap_mm = float(self.dsb_gap_mm.value())
        family = self.le_family.text().strip()
        obj, img, dets = aprilgrid_build_correspondences(gray, rows, cols, tag_mm, gap_mm, family)
        if obj is None: return None
        # SolvePnP with fixed intrinsics to get per-image extrinsics
        flag = cv2.SOLVEPNP_ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=flag)
        if not ok: return None
        if model == "fisheye":
            proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1,1,3), rvec, tvec, K, D)
            proj = proj.reshape(-1,2)
        else:
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
            proj = proj.reshape(-1,2)
        # Draw observed (green) and projected (magenta) + residuals (yellow)
        for (u,v), (x,y) in zip(img, proj):
            cv2.circle(rgb, (int(u),int(v)), 2, (0,255,0), -1, cv2.LINE_AA)
            cv2.circle(rgb, (int(x),int(y)), 2, (255,0,255), -1, cv2.LINE_AA)
            cv2.line(rgb, (int(u),int(v)), (int(x),int(y)), (0,255,255), 1, cv2.LINE_AA)
        return rgb

    def show_popup_image(self, title: str, rgb: np.ndarray):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        v = QtWidgets.QVBoxLayout(dlg)
        lab = QtWidgets.QLabel()
        lab.setAlignment(QtCore.Qt.AlignCenter)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        lab.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(900, 700, QtCore.Qt.KeepAspectRatio,
                                                           QtCore.Qt.SmoothTransformation))
        v.addWidget(lab)
        btn = QtWidgets.QPushButton("Close"); btn.clicked.connect(dlg.accept)
        v.addWidget(btn)
        dlg.resize(920, 740)
        dlg.exec_()


# ======================== Tab: Measurement ========================

class MeasureTab(QtWidgets.QWidget):
    def __init__(self, picam: Picamera2, cam_thread: CameraThread, parent=None):
        super().__init__(parent)
        self.picam = picam
        self.cam_thread = cam_thread

        # State
        self.live_frame: Optional[np.ndarray] = None  # latest RGB
        self.frozen = False
        self.work_frame: Optional[np.ndarray] = None  # frozen frame (RGB)
        self.display_rect = QtCore.QRect(0,0,0,0)     # where pixmap sits inside label
        self.mm_per_px: Optional[Tuple[float, float]] = None  # (mm/px X, mm/px Y)
        self.K = None
        self.D = None
        self.undistort = False
        self.map1 = self.map2 = None

        self.items: List[MeasureItem] = []
        self.current_pts: List[Tuple[int,int]] = []
        self.mode = "distance"  # distance | angle | circle

        # ---------- UI Layout ----------
        hbox = QtWidgets.QHBoxLayout(self)

        # Left: video/annotations
        left = QtWidgets.QVBoxLayout()
        self.view = ClickLabel("Live")
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setMinimumSize(640, 480)
        self.view.setStyleSheet("background:#111; border:1px solid #333;")
        left.addWidget(self.view, stretch=1)

        # Buttons row
        row = QtWidgets.QHBoxLayout()
        self.btn_freeze = QtWidgets.QPushButton("Freeze")
        self.btn_unfreeze = QtWidgets.QPushButton("Unfreeze")
        self.btn_undo_pt = QtWidgets.QPushButton("Undo point")
        self.btn_undo_item = QtWidgets.QPushButton("Undo item")
        self.btn_clear = QtWidgets.QPushButton("Clear all")
        row.addWidget(self.btn_freeze); row.addWidget(self.btn_unfreeze)
        row.addWidget(self.btn_undo_pt); row.addWidget(self.btn_undo_item)
        row.addWidget(self.btn_clear)
        left.addLayout(row)

        # Save
        row2 = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save snapshot + CSV")
        row2.addWidget(self.btn_save)
        left.addLayout(row2)

        hbox.addLayout(left, stretch=3)

        # Right: Controls
        right = QtWidgets.QVBoxLayout()

        # Intrinsics
        boxK = QtWidgets.QGroupBox("Intrinsics / Undistort")
        formK = QtWidgets.QFormLayout(boxK)
        self.le_yaml = QtWidgets.QLineEdit(os.path.abspath("intrinsics.yaml"))
        self.btn_browse_yaml = QtWidgets.QPushButton("...")
        yamlrow = QtWidgets.QHBoxLayout()
        yamlrow.addWidget(self.le_yaml, 1); yamlrow.addWidget(self.btn_browse_yaml)
        self.lbl_scale = QtWidgets.QLabel("px/mm: ?  |  mm/px: ?")
        self.chk_undist = QtWidgets.QCheckBox("Undistort Live/Freeze")
        self.btn_loadK = QtWidgets.QPushButton("Load YAML")
        formK.addRow("intrinsics.yaml", yamlrow)
        formK.addRow(self.btn_loadK)
        formK.addRow(self.chk_undist)
        formK.addRow("Scale", self.lbl_scale)
        right.addWidget(boxK)

        # Camera knobs (basic)
        camg = QtWidgets.QGroupBox("Camera")
        camf = QtWidgets.QFormLayout(camg)
        self.sld_exp = self._mk_slider(100, 20000, 5000)
        self.sld_gain = self._mk_slider(1, 16, 4)
        camf.addRow("Exposure (µs)", self.sld_exp["widget"])
        camf.addRow("Analogue Gain", self.sld_gain["widget"])
        right.addWidget(camg)

        # Measurement
        boxM = QtWidgets.QGroupBox("Measurement")
        formM = QtWidgets.QFormLayout(boxM)
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["Distance (2 pts)", "Angle (3 pts)", "Circle radius (center+point)"])
        self.spin_thick = QtWidgets.QSpinBox(); self.spin_thick.setRange(1, 8); self.spin_thick.setValue(2)
        formM.addRow("Mode", self.cmb_mode)
        formM.addRow("Stroke width", self.spin_thick)
        right.addWidget(boxM)

        # Log
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(1000)
        right.addWidget(QtWidgets.QLabel("Log:"))
        right.addWidget(self.log, stretch=1)

        hbox.addLayout(right, stretch=2)

        # Signals
        self.cam_thread.frame_ready.connect(self.on_frame)
        self.view.clicked.connect(self.on_click)
        self.btn_freeze.clicked.connect(self.on_freeze)
        self.btn_unfreeze.clicked.connect(self.on_unfreeze)
        self.btn_undo_pt.clicked.connect(self.on_undo_point)
        self.btn_undo_item.clicked.connect(self.on_undo_item)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_browse_yaml.clicked.connect(self.on_choose_yaml)
        self.btn_loadK.clicked.connect(self.load_yaml)
        self.chk_undist.toggled.connect(self.on_undist_toggled)
        self.cmb_mode.currentIndexChanged.connect(self.on_mode_changed)

        self.sld_exp["slider"].valueChanged.connect(self.on_cam_controls)
        self.sld_gain["slider"].valueChanged.connect(self.on_cam_controls)

        # Auto-load YAML if present
        if os.path.isfile(self.le_yaml.text().strip()):
            self.load_yaml()

        self.append_log("[READY] Freeze a frame, click to annotate. Load intrinsics for mm units.")

    # ---------- UI helpers ----------
    def _mk_slider(self, minv, maxv, init):
        w = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal); s.setRange(minv, maxv); s.setValue(init)
        v = QtWidgets.QLabel(str(init)); v.setFixedWidth(54)
        lay.addWidget(s, 1); lay.addWidget(v)
        return {"widget": w, "slider": s, "val": v}

    def append_log(self, msg: str):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ---------- Camera + view ----------
    def on_cam_controls(self):
        e = int(self.sld_exp["slider"].value())
        g = float(self.sld_gain["slider"].value())
        self.sld_exp["val"].setText(str(e))
        self.sld_gain["val"].setText(f"{g:.0f}")
        try:
            self.picam.set_controls({"AeEnable": False, "ExposureTime": e, "AnalogueGain": g})
        except Exception as ex:
            self.append_log(f"[WARN] set_controls failed: {ex}")

    def on_frame(self, frm: np.ndarray):
        if not self.frozen:
            if frm is None or frm.ndim != 3: 
                return
            self.live_frame = frm
            show = self.process_for_display(frm)
            self.paint_to_label(show)
        else:
            if self.work_frame is not None:
                show = self.draw_overlays(self.work_frame.copy())
                self.paint_to_label(show)

    def process_for_display(self, rgb: np.ndarray) -> np.ndarray:
        out = rgb
        if self.undistort and self.K is not None and self.D is not None:
            h, w = out.shape[:2]
            if self.map1 is None or self.map1.shape[:2] != (h, w):
                if self.is_fisheye():
                    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D, (w,h), np.eye(3), balance=0.0)
                    self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), newK, (w,h), cv2.CV_16SC2)
                else:
                    newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w,h), alpha=0.0)
                    self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, newK, (w,h), cv2.CV_16SC2)
            out = cv2.remap(out, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        return self.draw_hud(out.copy())

    def draw_hud(self, img: np.ndarray) -> np.ndarray:
        txt = f"{img.shape[1]}x{img.shape[0]}  undist={self.undistort}"
        if self.mm_per_px:
            xmm, ymm = self.mm_per_px
            txt += f"  mm/px: X={xmm:.3f}  Y={ymm:.3f}"
        cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,60), 2, cv2.LINE_AA)
        return img

    def draw_overlays(self, img: np.ndarray) -> np.ndarray:
        # Current unfinished poly
        col_current = (200, 180, 30)
        t = int(self.spin_thick.value())
        for it in self.items:
            self._draw_item(img, it)
        if self.current_pts:
            for p in self.current_pts:
                cv2.circle(img, p, max(2, t), (255, 120, 0), -1, cv2.LINE_AA)
            for i in range(len(self.current_pts)-1):
                cv2.line(img, self.current_pts[i], self.current_pts[i+1], col_current, t, cv2.LINE_AA)
        return img

    def _draw_item(self, img: np.ndarray, it: MeasureItem):
        t = it.thickness
        c = it.color
        if it.kind == "distance":
            p1, p2 = it.pts
            cv2.line(img, p1, p2, c, t, cv2.LINE_AA)
            mid = (int((p1[0]+p2[0])//2), int((p1[1]+p2[1])//2))
            cv2.putText(img, it.text, (mid[0]+6, mid[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        elif it.kind == "angle":
            a,b,cpt = it.pts
            cv2.line(img, b, a, c, t, cv2.LINE_AA)
            cv2.line(img, b, cpt, c, t, cv2.LINE_AA)
            cv2.circle(img, b, 5, c, -1, cv2.LINE_AA)
            cv2.putText(img, it.text, (b[0]+8, b[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        elif it.kind == "circle":
            center, edge = it.pts
            rpx = int(round(np.hypot(edge[0]-center[0], edge[1]-center[1])))
            cv2.circle(img, center, rpx, c, t, cv2.LINE_AA)
            cv2.circle(img, center, 3, c, -1, cv2.LINE_AA)
            cv2.putText(img, it.text, (center[0]+8, center[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)

    def paint_to_label(self, rgb: np.ndarray):
        qimg = to_qimage_rgb(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.view.setPixmap(pix)
        # compute display rect for click mapping
        lw, lh = self.view.width(), self.view.height()
        pw, ph = pix.width(), pix.height()
        ox = (lw - pw)//2
        oy = (lh - ph)//2
        self.display_rect = QtCore.QRect(ox, oy, pw, ph)

    # ---------- Freeze / clicks ----------
    def on_freeze(self):
        if self.live_frame is None:
            self.append_log("[WARN] No frame to freeze yet.")
            return
        self.work_frame = self.process_for_display(self.live_frame.copy())  # include undistort + HUD
        self.frozen = True
        self.append_log("[STATE] Frozen. Click on image to annotate.")

    def on_unfreeze(self):
        self.frozen = False
        self.work_frame = None
        self.current_pts.clear()
        self.append_log("[STATE] Unfrozen. Live view resumed.")

    def label_to_image_xy(self, lx: int, ly: int) -> Optional[Tuple[int,int]]:
        if self.work_frame is None:  # map using live frame shape for consistency
            base = self.live_frame if self.live_frame is not None else None
        else:
            base = self.work_frame
        if base is None: return None
        h, w = base.shape[:2]
        if not self.display_rect.contains(lx, ly):
            return None
        x = (lx - self.display_rect.left()) * (w / self.display_rect.width())
        y = (ly - self.display_rect.top())  * (h / self.display_rect.height())
        return (int(round(x)), int(round(y)))

    def on_click(self, lx: int, ly: int):
        if not self.frozen or (self.work_frame is None):
            return  # only annotate when frozen
        p = self.label_to_image_xy(lx, ly)
        if p is None: return
        self.current_pts.append(p)
        need = 2 if self.mode == "distance" else (3 if self.mode == "angle" else 2)
        if len(self.current_pts) >= need:
            self.commit_measure()

    def on_mode_changed(self, idx: int):
        self.mode = "distance" if idx == 0 else ("angle" if idx == 1 else "circle")
        self.current_pts.clear()
        self.append_log(f"[MODE] {self.mode}")

    def on_undist_toggled(self, checked: bool):
        self.undistort = checked
        self.map1 = self.map2 = None  # rebuild
        self.append_log(f"[OPT] Undistort = {checked}")

    # ---------- Build measurements ----------
    def commit_measure(self):
        thick = int(self.spin_thick.value())
        col = (40, 255, 220)

        if self.mode == "distance":
            p1, p2 = self.current_pts[:2]
            dx, dy = (p2[0]-p1[0], p2[1]-p1[1])
            px_len = float(np.hypot(dx, dy))
            mm_len = px_to_mm(dx, dy, self.mm_per_px)
            txt = f"{px_len:.2f}px"
            if self.mm_per_px: txt += f"  |  {mm_len:.3f} mm"
            it = MeasureItem("distance", [p1, p2], px_len, mm_len, txt, col, thick)
            self.items.append(it)
            self.append_log(f"[ADD] Distance: {txt}")
            self.current_pts = []

        elif self.mode == "angle":
            a, b, c = self.current_pts[:3]
            v1 = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
            v2 = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
            # angle at b between BA and BC
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
            cosang = np.clip(cosang, -1.0, 1.0)
            deg = float(np.degrees(np.arccos(cosang)))
            txt = f"{deg:.2f}°"
            it = MeasureItem("angle", [a,b,c], deg, deg, txt, col, thick)
            self.items.append(it)
            self.append_log(f"[ADD] Angle: {txt}")
            self.current_pts = []

        elif self.mode == "circle":
            ctr, edge = self.current_pts[:2]
            dx, dy = (edge[0]-ctr[0], edge[1]-ctr[1])
            rpx = float(np.hypot(dx, dy))
            rmm = px_to_mm(dx, dy, self.mm_per_px)
            txt = f"r={rpx:.2f}px"
            if self.mm_per_px: txt += f"  |  r={rmm:.3f} mm"
            it = MeasureItem("circle", [ctr, edge], rpx, rmm, txt, col, thick)
            self.items.append(it)
            self.append_log(f"[ADD] Circle radius: {txt}")
            self.current_pts = []

    # ---------- Undo / Clear ----------
    def on_undo_point(self):
        if self.current_pts:
            self.current_pts.pop()
            self.append_log("[UNDO] Last point removed.")

    def on_undo_item(self):
        if self.items:
            it = self.items.pop()
            self.append_log(f"[UNDO] Removed {it.kind}.")

    def on_clear(self):
        self.items.clear()
        self.current_pts.clear()
        self.append_log("[CLEAR] All annotations removed.")

    # ---------- Save snapshot + CSV ----------
    def on_save(self):
        if self.work_frame is None:
            self.append_log("[WARN] Freeze a frame before saving.")
            return
        outdir = os.path.abspath("measurements")
        os.makedirs(outdir, exist_ok=True)
        ts = now_ts()
        # Compose annotated image
        img = self.draw_overlays(self.work_frame.copy())
        png_path = os.path.join(outdir, f"measure_{ts}.png")
        cv2.imwrite(png_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # CSV
        csv_path = os.path.join(outdir, f"measure_{ts}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "kind", "points", "px_value", "mm_value"])
            for it in self.items:
                w.writerow([ts, it.kind, it.pts, f"{it.px_value:.6f}", f"{it.mm_value:.6f}"])
        self.append_log(f"[SAVE] {png_path}\n[CSV]  {csv_path}")

    # ---------- Intrinsics ----------
    def on_choose_yaml(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open intrinsics.yaml", self.le_yaml.text(), "YAML (*.yaml)")
        if path:
            self.le_yaml.setText(path)

    def is_fisheye(self) -> bool:
        # We keep it simple: if dist has 4x1, treat as fisheye
        if self.D is None: return False
        try:
            return (self.D.size == 4)
        except Exception:
            return False

    def load_yaml(self):
        path = self.le_yaml.text().strip()
        if not os.path.isfile(path):
            self.append_log(f"[ERROR] File not found: {path}")
            return
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            self.append_log(f"[ERROR] Could not open YAML: {path}")
            return
        # mandatory
        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("distortion_coefficients").mat()
        # optional scale (your calibration script writes these)
        pxmmx = fs.getNode("px_per_mm_x").real()
        pxmmy = fs.getNode("px_per_mm_y").real()
        mmppx = fs.getNode("mm_per_px_x").real()
        mmppy = fs.getNode("mm_per_px_y").real()
        fs.release()

        if K is None or D is None:
            self.append_log("[ERROR] YAML missing camera_matrix/distortion_coefficients.")
            return
        self.K = K.astype(np.float64)
        self.D = D.astype(np.float64)
        self.map1 = self.map2 = None  # force rebuild

        # Follow your original logic strictly
        mmx = mmy = None
        if not np.isnan(pxmmx) and not np.isnan(pxmmy) and pxmmx > 1e-12 and pxmmy > 1e-12:
            # use px/mm directly (kept as-is per your script)
            mmx, mmy = float(pxmmx), float(pxmmy)
        elif not np.isnan(mmppx) and not np.isnan(mmppy) and mmppx > 1e-12 and mmppy > 1e-12:
            # fallback: invert mm/px
            mmx, mmy = (1.0/float(mmppx), 1.0/float(mmppy))

        self.mm_per_px = (mmx, mmy) if (mmx is not None and mmy is not None) else None

        if self.mm_per_px:
            x, y = self.mm_per_px
            self.lbl_scale.setText(f"px/mm  X={x:.4f}  Y={y:.4f}")
        else:
            self.lbl_scale.setText("px/mm: ?  |  mm/px: ?")

        self.append_log("[OK] Loaded intrinsics.")
        self.append_log("      fisheye=" + str(self.is_fisheye()))


# ======================== Main Window (tabs + shared camera) =================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MV Calibrate + Measure")
        self.resize(1220, 820)

        # ======= Camera setup (shared) =======
        self.picam = Picamera2()
        # 640x480 RGB888 as requested
        cfg = self.picam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(0.15)

        self.cam_thread = CameraThread(self.picam)
        self.cam_thread.start()

        # ======= UI =======
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        self.tab_calib = CalibrateTab(self.picam, self.cam_thread, self)
        self.tab_measure = MeasureTab(self.picam, self.cam_thread, self)

        tabs.addTab(self.tab_calib, "Calibrate")
        tabs.addTab(self.tab_measure, "Measure")

    # ---------- Cleanup ----------
    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self.cam_thread.stop()
        except Exception:
            pass
        try:
            self.picam.stop()
        except Exception:
            pass
        e.accept()


# ======================== main ============================

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
