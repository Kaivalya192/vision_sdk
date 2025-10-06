#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPi Vision Client – Blob UI (Industrial Blob Inspection, GO/NOGO)
- Mirrors your Canny Contour UI’s structure (WS video in, UDP trigger, VGR JSON out).
- Blob pipeline with configurable thresholding, polarity, morphology, and filters.
- GO/NOGO rules (area, circularity, solidity, aspect, holes, intensity) decide publish name.

Requires: PyQt5, numpy, opencv-python
"""

import sys, time, base64, json, socket, threading, inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebSockets

# ───────────────────────── UDP trigger bridge ─────────────────────────

class QtTriggerBridge(QtCore.QObject):
    triggerReceived = QtCore.pyqtSignal()

class QtTriggerListener:
    """
    Background UDP listener. Accepts either b'TRIGGER' or JSON {"cmd":"trigger"}.
    Emits QtTriggerBridge.triggerReceived() in the GUI thread.
    """
    def __init__(self, bridge: QtTriggerBridge, bind_ip="0.0.0.0", port=40002, log=print, enable_broadcast=False):
        self.bridge = bridge
        self._ip = bind_ip
        self._port = int(port)
        self._log = log
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            if enable_broadcast:
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._sock.bind((self._ip, self._port))
            self._sock.settimeout(0.5)
            self._log(f"[UDP] BOUND OK on {self._ip}:{self._port}")
        except Exception as e:
            self._log(f"[UDP][BIND ERROR] {e} (ip={self._ip}, port={self._port})")
            try: self._sock.close()
            except: pass
            self._sock = None

    def start(self):
        if self._sock is None:
            self._log("[UDP] Listener NOT STARTED (socket not bound).")
            return
        self._log(f"[UDP] START thread; listening on {self._ip}:{self._port}")
        self._thr.start()

    def stop(self):
        self._stop.set()
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass

    def _loop(self):
        if self._sock is None: return
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                self._log(f"[UDP][RECV ERROR] {e}")
                continue

            self._log(f"[UDP] FROM {addr[0]}:{addr[1]}  {data!r}")

            msg = data.decode("utf-8", errors="ignore").strip()
            fire = (msg.upper() == "TRIGGER")
            if not fire:
                try:
                    j = json.loads(msg)
                    fire = isinstance(j, dict) and str(j.get("cmd","")).lower() == "trigger"
                except Exception:
                    pass

            if fire:
                try:
                    self._sock.sendto(b'{"status":"armed"}', addr)  # optional ACK
                except Exception:
                    pass
                self.bridge.triggerReceived.emit()

# ───────────────────────── UDP publishers ─────────────────────────

class RobotUDPPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 40001):
        self._host = host
        self._port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def set_target(self, host: str, port: int):
        self._host = host.strip(); self._port = int(port)
    def send_json(self, payload: dict):
        try:
            msg = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._sock.sendto(msg, (self._host, self._port))
        except Exception as e:
            print(f"[PUB] UDP send error: {e}", flush=True)

class VGRResultPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 40003):
        self._host = host
        self._port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def set_target(self, host: str, port: int):
        self._host = host.strip(); self._port = int(port)
    def send_json(self, payload: dict):
        try:
            msg = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._sock.sendto(msg, (self._host, self._port))
        except Exception as e:
            print(f"[VGR-PUB] UDP send error: {e}", flush=True)

# ───────────────────────── VideoLabel (aspect + ROI) ─────────────────────────

class VideoLabel(QtWidgets.QLabel):
    roiSelected = QtCore.pyqtSignal(QtCore.QRect)  # rectangle in display coords
    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_draw_rect = QtCore.QRect()
        self._frame_wh: Tuple[int, int] = (0, 0)
        self._rect_mode = False
        self._origin = QtCore.QPoint()
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self.setMouseTracking(True)
    def setFrameSize(self, wh: Tuple[int, int]): self._frame_wh = (int(wh[0]), int(wh[1]))
    def setPixmapKeepAspect(self, pm: QtGui.QPixmap):
        if pm.isNull():
            super().setPixmap(pm); self._last_draw_rect = QtCore.QRect(); return
        area = self.size()
        scaled = pm.scaled(area, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        x = (area.width() - scaled.width()) // 2
        y = (area.height() - scaled.height()) // 2
        self._last_draw_rect = QtCore.QRect(x, y, scaled.width(), scaled.height())
        canvas = QtGui.QPixmap(area); canvas.fill(QtCore.Qt.black)
        p = QtGui.QPainter(canvas); p.drawPixmap(self._last_draw_rect, scaled); p.end()
        super().setPixmap(canvas)
    def enable_rect_selection(self, ok: bool):
        self._rect_mode = bool(ok)
        if not ok: self._rubber.hide()
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self._rect_mode and ev.button() == QtCore.Qt.LeftButton and self._last_draw_rect.contains(ev.pos()):
            self._origin = ev.pos()
            self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubber.show()
        else:
            super().mousePressEvent(ev)
    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._rect_mode and not self._origin.isNull():
            self._rubber.setGeometry(QtCore.QRect(self._origin, ev.pos()).normalized())
        else:
            super().mouseMoveEvent(ev)
    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self._rect_mode and ev.button() == QtCore.Qt.LeftButton:
            r = self._rubber.geometry().intersected(self._last_draw_rect)
            self._rubber.hide(); self._rect_mode = False
            if r.width() > 5 and r.height() > 5: self.roiSelected.emit(r)
            return
        super().mouseReleaseEvent(ev)

# ───────────────────────── Camera Panel (reuse your local stub if ext missing) ─────────────────────────

try:
    from ui.camera_panel import CameraPanel as _ExtCameraPanel
except Exception:
    _ExtCameraPanel = None

class _LocalCameraPanel(QtWidgets.QWidget):
    paramsChanged      = QtCore.pyqtSignal(dict)
    viewChanged        = QtCore.pyqtSignal(dict)
    afTriggerRequested = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self._debounce = QtCore.QTimer(self, interval=150, singleShot=True, timeout=self._emit_params)
        grid = QtWidgets.QGridLayout(self); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6)
        r=0
        self.chk_ae = QtWidgets.QCheckBox("Auto Exposure"); grid.addWidget(self.chk_ae, r,0,1,2); r+=1
        self.sp_exp = QtWidgets.QSpinBox(); self.sp_exp.setRange(100,33000); self.sp_exp.setValue(10000); self.sp_exp.setSingleStep(100); self.sp_exp.setSuffix(" µs")
        grid.addWidget(QtWidgets.QLabel("Exposure"), r,0); grid.addWidget(self.sp_exp, r,1); r+=1
        self.dsb_gain = QtWidgets.QDoubleSpinBox(); self.dsb_gain.setRange(1.0,16.0); self.dsb_gain.setValue(4.0); self.dsb_gain.setDecimals(2); self.dsb_gain.setSingleStep(0.05)
        grid.addWidget(QtWidgets.QLabel("Analogue Gain"), r,0); grid.addWidget(self.dsb_gain, r,1); r+=1
        self.dsb_fps = QtWidgets.QDoubleSpinBox(); self.dsb_fps.setRange(1.0,120.0); self.dsb_fps.setValue(30.0); self.dsb_fps.setDecimals(1)
        grid.addWidget(QtWidgets.QLabel("Framerate"), r,0); grid.addWidget(self.dsb_fps, r,1); r+=1
        self.cmb_awb = QtWidgets.QComboBox(); self.cmb_awb.addItems(["auto","tungsten","fluorescent","indoor","daylight","cloudy","manual"])
        grid.addWidget(QtWidgets.QLabel("AWB Mode"), r,0); grid.addWidget(self.cmb_awb, r,1); r+=1
        self.dsb_awb_r = QtWidgets.QDoubleSpinBox(); self.dsb_awb_r.setRange(0.1,8.0); self.dsb_awb_r.setValue(2.0); self.dsb_awb_r.setSingleStep(0.05)
        self.dsb_awb_b = QtWidgets.QDoubleSpinBox(); self.dsb_awb_b.setRange(0.1,8.0); self.dsb_awb_b.setValue(2.0); self.dsb_awb_b.setSingleStep(0.05)
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(QtWidgets.QLabel("Gains R/B")); hb.addWidget(self.dsb_awb_r); hb.addWidget(self.dsb_awb_b)
        grid.addLayout(hb, r,1); r+=1
        self.cmb_af = QtWidgets.QComboBox(); self.cmb_af.addItems(["auto","continuous","manual"])
        grid.addWidget(QtWidgets.QLabel("AF Mode"), r,0); grid.addWidget(self.cmb_af, r,1); r+=1
        self.dsb_dioptre = QtWidgets.QDoubleSpinBox(); self.dsb_dioptre.setRange(0.0,10.0); self.dsb_dioptre.setDecimals(2); self.dsb_dioptre.setSingleStep(0.05); self.dsb_dioptre.setValue(0.0)
        self.btn_af_trig = QtWidgets.QPushButton("AF Trigger")
        hf = QtWidgets.QHBoxLayout(); hf.addWidget(self.dsb_dioptre); hf.addWidget(self.btn_af_trig)
        grid.addLayout(hf, r,1); grid.addWidget(QtWidgets.QLabel("Lens (dpt)"), r,0); r+=1
        self.dsb_bri = QtWidgets.QDoubleSpinBox(); self.dsb_bri.setRange(-1.0,1.0); self.dsb_bri.setValue(0.0)
        self.dsb_con = QtWidgets.QDoubleSpinBox(); self.dsb_con.setRange(0.0,2.0); self.dsb_con.setValue(1.0)
        self.dsb_sat = QtWidgets.QDoubleSpinBox(); self.dsb_sat.setRange(0.0,2.0); self.dsb_sat.setValue(1.0)
        self.dsb_sha = QtWidgets.QDoubleSpinBox(); self.dsb_sha.setRange(0.0,2.0); self.dsb_sha.setValue(1.0)
        self.cmb_den = QtWidgets.QComboBox(); self.cmb_den.addItems(["off","fast","high_quality"])
        for lab, w in [("Brightness",self.dsb_bri),("Contrast",self.dsb_con),("Saturation",self.dsb_sat),("Sharpness",self.dsb_sha),("Denoise",self.cmb_den)]:
            grid.addWidget(QtWidgets.QLabel(lab), r,0); grid.addWidget(w, r,1); r+=1
        self.chk_flip_h = QtWidgets.QCheckBox("Flip H"); self.chk_flip_v = QtWidgets.QCheckBox("Flip V"); self.btn_rot = QtWidgets.QPushButton("Rotate 90°")
        hv = QtWidgets.QHBoxLayout(); hv.addWidget(self.chk_flip_h); hv.addWidget(self.chk_flip_v); hv.addWidget(self.btn_rot); hv.addStretch(1)
        grid.addLayout(hv, r,1); grid.addWidget(QtWidgets.QLabel("View"), r,0); r+=1
        self.btn_reset = QtWidgets.QPushButton("Reset tuning"); grid.addWidget(self.btn_reset, r,0,1,2); r+=1
        for w in [self.chk_ae, self.sp_exp, self.dsb_gain, self.dsb_fps, self.cmb_awb, self.dsb_awb_r, self.dsb_awb_b,
                  self.cmb_af, self.dsb_dioptre, self.dsb_bri, self.dsb_con, self.dsb_sat, self.dsb_sha, self.cmb_den]:
            if hasattr(w,'valueChanged'): w.valueChanged.connect(lambda *_: self._debounce.start())
            if hasattr(w,'currentTextChanged'): w.currentTextChanged.connect(lambda *_: self._debounce.start())
            if hasattr(w,'toggled'): w.toggled.connect(lambda *_: self._debounce.start())
        self.chk_flip_h.toggled.connect(lambda *_: self._emit_view())
        self.chk_flip_v.toggled.connect(lambda *_: self._emit_view())
        self.btn_rot.clicked.connect(self._on_rot)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_af_trig.clicked.connect(self.afTriggerRequested.emit)
        self.cmb_awb.currentTextChanged.connect(self._awb_mode_changed)
        self._rot_q = 0
        self._awb_mode_changed(self.cmb_awb.currentText())

    def _emit_params(self):
        p = dict(
            auto_exposure=self.chk_ae.isChecked(),
            exposure_us=int(self.sp_exp.value()),
            gain=float(self.dsb_gain.value()),
            fps=float(self.dsb_fps.value()),
            awb_mode=self.cmb_awb.currentText(),
            awb_rb=[float(self.dsb_awb_r.value()), float(self.dsb_awb_b.value())],
            focus_mode=self.cmb_af.currentText(),
            dioptre=float(self.dsb_dioptre.value()),
            brightness=float(self.dsb_bri.value()),
            contrast=float(self.dsb_con.value()),
            saturation=float(self.dsb_sat.value()),
            sharpness=float(self.dsb_sha.value()),
            denoise=self.cmb_den.currentText(),
        )
        self.paramsChanged.emit(p)
    def _on_rot(self): self._rot_q = (self._rot_q + 1) % 4; self._emit_view()
    def _emit_view(self): self.viewChanged.emit(dict(flip_h=self.chk_flip_h.isChecked(), flip_v=self.chk_flip_v.isChecked(), rot_quadrant=self._rot_q))
    def _awb_mode_changed(self, mode):
        manual = (mode=="manual"); self.dsb_awb_r.setEnabled(manual); self.dsb_awb_b.setEnabled(manual); self._debounce.start()
    def _reset(self):
        self.blockSignals(True)
        self.chk_ae.setChecked(False); self.sp_exp.setValue(6000); self.dsb_gain.setValue(2.0)
        self.dsb_fps.setValue(30.0); self.cmb_awb.setCurrentText("auto")
        self.dsb_awb_r.setValue(2.0); self.dsb_awb_b.setValue(2.0)
        self.cmb_af.setCurrentText("manual"); self.dsb_dioptre.setValue(0.0)
        for w,v in [(self.dsb_bri,0.0),(self.dsb_con,1.0),(self.dsb_sat,1.0),(self.dsb_sha,1.0)]: w.setValue(v)
        self.cmb_den.setCurrentText("fast"); self.chk_flip_h.setChecked(False); self.chk_flip_v.setChecked(False); self._rot_q = 0
        self.blockSignals(False)
        self._emit_params(); self._emit_view()

def make_camera_panel():
    if _ExtCameraPanel is None:
        return _LocalCameraPanel()
    try:
        sig = inspect.signature(_ExtCameraPanel.__init__)
        if "cam" in sig.parameters:
            return _ExtCameraPanel(cam=None)
        else:
            return _ExtCameraPanel()
    except Exception:
        return _LocalCameraPanel()

# ───────────────────────── Main Window ─────────────────────────

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPi Vision Client – Blob Inspection")
        self.resize(1600, 980)

        # websocket
        self.ws = QtWebSockets.QWebSocket()
        self.ws.textMessageReceived.connect(self._ws_txt)
        self.ws.connected.connect(self._ws_ok)
        self.ws.disconnected.connect(self._ws_closed)
        self._frame_counter = 0

        # state
        self.session_id = "-"
        self.last_frame_wh = (0, 0)
        self._fps_acc = 0.0
        self._fps_n = 0
        self._last_ts = time.perf_counter()
        self._settings = QtCore.QSettings('vision_sdk', 'ui_blob')
        self._log_buffer: List[str] = []

        self._last_frame_bgr: Optional[np.ndarray] = None
        self._last_overlay: Optional[QtGui.QImage] = None
        self._overlay_until = 0.0

        self._expect_one_result = False
        self._last_result_time = 0.0

        # central: left controls + video + right panels
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central); root.setContentsMargins(8,8,8,8); root.setSpacing(8)

        # LEFT controls inside a scroll area
        left_panel = QtWidgets.QWidget(); left_v = QtWidgets.QVBoxLayout(left_panel); left_v.setSpacing(10)

        # Connection
        conn_box = QtWidgets.QGroupBox("Connection")
        cf = QtWidgets.QFormLayout(conn_box)
        self.ed_host = QtWidgets.QLineEdit("ws://192.168.1.2:8765")
        self.btn_conn = QtWidgets.QPushButton("Connect")
        self.btn_disc = QtWidgets.QPushButton("Disconnect"); self.btn_disc.setEnabled(False)
        self.ed_robot_ip = QtWidgets.QLineEdit("192.168.1.50")
        self.sp_robot_port = QtWidgets.QSpinBox(); self.sp_robot_port.setRange(1,65535); self.sp_robot_port.setValue(40001)
        self.chk_pub_robot = QtWidgets.QCheckBox("Auto publish to robot"); self.chk_pub_robot.setChecked(True)
        cf.addRow("Server:", self.ed_host)
        cf.addRow("", self.btn_conn)
        cf.addRow("", self.btn_disc)
        cf.addRow("Robot IP:", self.ed_robot_ip)
        cf.addRow("Robot Port:", self.sp_robot_port)
        cf.addRow("", self.chk_pub_robot)
        left_v.addWidget(conn_box)

        # VGR Result target
        self.ed_vgr_ip   = QtWidgets.QLineEdit(self.ed_robot_ip.text())
        self.sp_vgr_port = QtWidgets.QSpinBox(); self.sp_vgr_port.setRange(1,65535); self.sp_vgr_port.setValue(40003)
        cf.addRow("VGR Result IP:", self.ed_vgr_ip)
        cf.addRow("VGR Result Port:", self.sp_vgr_port)

        # UDP publishers
        self.pub_robot = RobotUDPPublisher(self.ed_robot_ip.text(), int(self.sp_robot_port.value()))
        self.ed_robot_ip.textChanged.connect(lambda *_: self.pub_robot.set_target(self.ed_robot_ip.text(), self.sp_robot_port.value()))
        self.sp_robot_port.valueChanged.connect(lambda *_: self.pub_robot.set_target(self.ed_robot_ip.text(), self.sp_robot_port.value()))
        self.pub_vgr = VGRResultPublisher(self.ed_vgr_ip.text(), int(self.sp_vgr_port.value()))
        self.ed_vgr_ip.textChanged.connect(lambda *_: self.pub_vgr.set_target(self.ed_vgr_ip.text(), self.sp_vgr_port.value()))
        self.sp_vgr_port.valueChanged.connect(lambda *_: self.pub_vgr.set_target(self.ed_vgr_ip.text(), self.sp_vgr_port.value()))

        # Mode
        mode_box = QtWidgets.QGroupBox("Mode")
        mv = QtWidgets.QVBoxLayout(mode_box)
        self.rad_train = QtWidgets.QRadioButton("Training")
        self.rad_trig  = QtWidgets.QRadioButton("Trigger")
        self.rad_train.setChecked(True)
        self.btn_trig = QtWidgets.QPushButton("TRIGGER")
        mv.addWidget(self.rad_train); mv.addWidget(self.rad_trig); mv.addWidget(self.btn_trig)
        left_v.addWidget(mode_box)

        # Processing cadence
        proc_box = QtWidgets.QGroupBox("Processing")
        pf = QtWidgets.QFormLayout(proc_box)
        self.cmb_w = QtWidgets.QComboBox()
        [self.cmb_w.addItem(str(w)) for w in [320, 480, 640, 800, 960, 1280]]
        self.cmb_w.setCurrentText("640")
        self.sp_every = QtWidgets.QSpinBox(); self.sp_every.setRange(1, 10); self.sp_every.setValue(1)
        pf.addRow("Proc width", self.cmb_w)
        pf.addRow("Run every Nth", self.sp_every)
        left_v.addWidget(proc_box)

        # Blob pipeline parameters
        blob_box = QtWidgets.QGroupBox("Blob Pipeline")
        bf = QtWidgets.QFormLayout(blob_box)

        # Preprocess
        self.chk_clahe = QtWidgets.QCheckBox("CLAHE"); self.chk_clahe.setChecked(False)
        self.ds_clip = QtWidgets.QDoubleSpinBox(); self.ds_clip.setRange(0.5, 10.0); self.ds_clip.setDecimals(2); self.ds_clip.setValue(2.0)
        self.sp_tiles = QtWidgets.QSpinBox(); self.sp_tiles.setRange(2, 16); self.sp_tiles.setValue(8)
        self.sp_blur = QtWidgets.QSpinBox(); self.sp_blur.setRange(0, 21); self.sp_blur.setSingleStep(2); self.sp_blur.setValue(5)  # 0 = off, odd only
        bf.addRow("CLAHE", self._hbox(self.chk_clahe, QtWidgets.QLabel("clip"), self.ds_clip, QtWidgets.QLabel("tiles"), self.sp_tiles))
        bf.addRow("Gaussian Blur (ksize odd, 0=off)", self.sp_blur)

        # Thresholding
        self.cmb_thr_mode = QtWidgets.QComboBox(); self.cmb_thr_mode.addItems(["Otsu","Fixed","AdaptiveMean","AdaptiveGaussian"])
        self.sp_thr_val = QtWidgets.QSpinBox(); self.sp_thr_val.setRange(0,255); self.sp_thr_val.setValue(128)
        self.cmb_polarity = QtWidgets.QComboBox(); self.cmb_polarity.addItems(["DarkOnLight","LightOnDark"])
        bf.addRow("Threshold mode / value", self._hbox(self.cmb_thr_mode, self.sp_thr_val))
        bf.addRow("Polarity", self.cmb_polarity)

        # Morphology
        self.sp_open = QtWidgets.QSpinBox(); self.sp_open.setRange(0,10); self.sp_open.setValue(1)
        self.sp_close = QtWidgets.QSpinBox(); self.sp_close.setRange(0,10); self.sp_close.setValue(1)
        bf.addRow("Open iters / Close iters", self._hbox(self.sp_open, self.sp_close))

        left_v.addWidget(blob_box)

        # Blob filters + selection
        filt_box = QtWidgets.QGroupBox("Blob Filters & Selection")
        ff = QtWidgets.QFormLayout(filt_box)
        self.sp_min_area = QtWidgets.QSpinBox(); self.sp_min_area.setRange(0, 10_000_000); self.sp_min_area.setValue(300)
        self.sp_max_area = QtWidgets.QSpinBox(); self.sp_max_area.setRange(0, 10_000_000); self.sp_max_area.setValue(0)  # 0 = no limit
        self.ds_aspect_min = QtWidgets.QDoubleSpinBox(); self.ds_aspect_min.setRange(0.0, 100.0); self.ds_aspect_min.setDecimals(2); self.ds_aspect_min.setValue(0.0)
        self.ds_aspect_max = QtWidgets.QDoubleSpinBox(); self.ds_aspect_max.setRange(0.0, 100.0); self.ds_aspect_max.setDecimals(2); self.ds_aspect_max.setValue(100.0)
        self.ds_solidity_min = QtWidgets.QDoubleSpinBox(); self.ds_solidity_min.setRange(0.0, 1.0); self.ds_solidity_min.setDecimals(2); self.ds_solidity_min.setValue(0.0)
        self.ds_circ_min = QtWidgets.QDoubleSpinBox(); self.ds_circ_min.setRange(0.0, 1.0); self.ds_circ_min.setDecimals(3); self.ds_circ_min.setValue(0.0)  # 1.0 = perfect circle
        self.sp_holes_min = QtWidgets.QSpinBox(); self.sp_holes_min.setRange(0, 200); self.sp_holes_min.setValue(0)
        self.sp_holes_max = QtWidgets.QSpinBox(); self.sp_holes_max.setRange(0, 200); self.sp_holes_max.setValue(100)
        self.sp_keepN = QtWidgets.QSpinBox(); self.sp_keepN.setRange(1, 500); self.sp_keepN.setValue(50)
        self.rad_pick_largest = QtWidgets.QRadioButton("Pick Largest")
        self.rad_pick_best = QtWidgets.QRadioButton("Pick Best by Score (circularity * solidity * area)")
        self.rad_pick_largest.setChecked(True)
        ff.addRow("Min / Max area", self._hbox(self.sp_min_area, self.sp_max_area))
        ff.addRow("Aspect min/max", self._hbox(self.ds_aspect_min, self.ds_aspect_max))
        ff.addRow("Solidity ≥", self.ds_solidity_min)
        ff.addRow("Circularity ≥", self.ds_circ_min)
        ff.addRow("Holes min/max", self._hbox(self.sp_holes_min, self.sp_holes_max))
        ff.addRow("Keep top N (by area)", self.sp_keepN)
        ff.addRow(self.rad_pick_largest)
        ff.addRow(self.rad_pick_best)
        left_v.addWidget(filt_box)

        # Optional intensity gate
        int_box = QtWidgets.QGroupBox("Optional Intensity Gate (mean gray inside blob)")
        inf = QtWidgets.QFormLayout(int_box)
        self.chk_int_gate = QtWidgets.QCheckBox("Enable intensity range")
        self.sp_int_min = QtWidgets.QSpinBox(); self.sp_int_min.setRange(0,255); self.sp_int_min.setValue(0)
        self.sp_int_max = QtWidgets.QSpinBox(); self.sp_int_max.setRange(0,255); self.sp_int_max.setValue(255)
        inf.addRow(self.chk_int_gate)
        inf.addRow("Min / Max", self._hbox(self.sp_int_min, self.sp_int_max))
        left_v.addWidget(int_box)
        
        # GO/NOGO by count + holes
        rule_box = QtWidgets.QGroupBox("Decision by Count")
        rv = QtWidgets.QFormLayout(rule_box)

        self.sp_expected_count = QtWidgets.QSpinBox(); self.sp_expected_count.setRange(0, 999); self.sp_expected_count.setValue(1)
        self.sp_expected_holes = QtWidgets.QSpinBox(); self.sp_expected_holes.setRange(0, 999); self.sp_expected_holes.setValue(0)

        self.lbl_count_hint = QtWidgets.QLabel("GO if (detected blobs == expected) AND (holes == expected)")
        self.chk_at_least = QtWidgets.QCheckBox("Use ≥ expected blobs (instead of ==)")
        self.chk_holes_at_least = QtWidgets.QCheckBox("Use ≥ expected holes (instead of ==)")

        rv.addRow("Expected blobs", self.sp_expected_count)
        rv.addRow("Expected holes (primary blob)", self.sp_expected_holes)
        rv.addRow(self.lbl_count_hint)
        rv.addRow(self.chk_at_least)
        rv.addRow(self.chk_holes_at_least)

        left_v.addWidget(rule_box)

        # Draw options
        draw_box = QtWidgets.QGroupBox("Draw Options")
        dv = QtWidgets.QVBoxLayout(draw_box)
        self.chk_draw_cnt = QtWidgets.QCheckBox("Contours"); self.chk_draw_cnt.setChecked(True)
        self.chk_draw_box = QtWidgets.QCheckBox("Boxes"); self.chk_draw_box.setChecked(True)
        self.chk_draw_cent = QtWidgets.QCheckBox("Centroids"); self.chk_draw_cent.setChecked(False)
        self.chk_draw_ids = QtWidgets.QCheckBox("IDs / Metrics"); self.chk_draw_ids.setChecked(True)
        dv.addWidget(self.chk_draw_cnt); dv.addWidget(self.chk_draw_box); dv.addWidget(self.chk_draw_cent); dv.addWidget(self.chk_draw_ids)
        left_v.addWidget(draw_box)

        # Actions
        act_box = QtWidgets.QGroupBox("Actions")
        ah = QtWidgets.QHBoxLayout(act_box)
        self.btn_once = QtWidgets.QPushButton("Run Once")
        self.btn_clear = QtWidgets.QPushButton("Clear Overlay")
        self.btn_once.clicked.connect(self._run_once)
        self.btn_clear.clicked.connect(lambda: self._show_image(self._last_frame_bgr))
        ah.addWidget(self.btn_once); ah.addWidget(self.btn_clear); ah.addStretch(1)
        left_v.addWidget(act_box)

        left_v.addStretch(1)

        # Left scroll
        self.controlsPanel = QtWidgets.QScrollArea(); self.controlsPanel.setWidgetResizable(True)
        self.controlsPanel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controlsPanel.setWidget(left_panel)
        root.addWidget(self.controlsPanel, 0)

        # Center video
        self.video = VideoLabel()
        self.video.setMinimumSize(1024, 576)
        self.video.setStyleSheet("background:#111;")
        root.addWidget(self.video, 1)

        # RIGHT: results + log
        right = QtWidgets.QVBoxLayout()
        grp = QtWidgets.QGroupBox("Blobs (last run)")
        vv = QtWidgets.QVBoxLayout(grp)
        self.tbl = QtWidgets.QTableWidget(0, 12)
        self.tbl.setHorizontalHeaderLabels(["id","area","perim","x","y","w","h","sol","circ","aspect","holes","meanI"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        vv.addWidget(self.tbl)
        right.addWidget(grp, 1)

        self.txt_log = QtWidgets.QTextEdit(); self.txt_log.setReadOnly(True)
        right.addWidget(QtWidgets.QLabel("Log")); right.addWidget(self.txt_log, 1)
        rightw = QtWidgets.QWidget(); rlay = QtWidgets.QVBoxLayout(rightw); rlay.addLayout(right)
        root.addWidget(rightw, 0)

        # Camera control dock
        self.cam_panel = make_camera_panel()
        self.cameraDock = QtWidgets.QDockWidget("Camera Control", self)
        self.cameraDock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.cameraDock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                                    QtWidgets.QDockWidget.DockWidgetMovable |
                                    QtWidgets.QDockWidget.DockWidgetFloatable)
        self.cameraDock.setWidget(self.cam_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.cameraDock)

        # View menu
        view_menu = self.menuBar().addMenu("&View")
        self.controlsPanelAction = QtWidgets.QAction("Controls", self, checkable=True, checked=True)
        self.controlsPanelAction.toggled.connect(self.controlsPanel.setVisible)
        self.controlsPanel.installEventFilter(self)
        self.controlsPanelAction.setChecked(self.controlsPanel.isVisible())
        view_menu.addAction(self.controlsPanelAction)
        view_menu.addAction(self.cameraDock.toggleViewAction())
        self.controlsPanel.setVisible(True)
        self.controlsPanelAction.setChecked(True)

        # status
        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        self.lbl_sess = QtWidgets.QLabel("Disconnected"); self.lbl_fps = QtWidgets.QLabel("FPS: —")
        self.status.addWidget(self.lbl_sess); self.status.addPermanentWidget(self.lbl_fps)

        # signals
        self.btn_conn.clicked.connect(lambda: self.ws.open(QtCore.QUrl(self.ed_host.text().strip())))
        self.btn_disc.clicked.connect(self.ws.close)
        self.rad_train.toggled.connect(lambda: self._send({"type": "set_mode", "mode": "training" if self.rad_train.isChecked() else "trigger"}))
        self.btn_trig.clicked.connect(self._do_trigger)
        self.cmb_w.currentTextChanged.connect(lambda w: self._send({"type": "set_proc_width", "width": int(w)}))
        self.sp_every.valueChanged.connect(lambda n: self._send({"type": "set_publish_every", "n": int(n)}))

        self.cam_panel.paramsChanged.connect(lambda p: self._send({"type": "set_params", "params": p}))
        self.cam_panel.viewChanged.connect(lambda v: self._send({"type": "set_view", **v}))
        if hasattr(self.cam_panel, "afTriggerRequested"):
            self.cam_panel.afTriggerRequested.connect(lambda: self._send({"type": "af_trigger"}))

        self._ping = QtCore.QTimer(interval=10000, timeout=lambda: self._send({"type": "ping"}))
        self._load_settings()

        # UDP trigger listener
        self._udp_bridge = QtTriggerBridge()
        self._udp_bridge.triggerReceived.connect(self._on_udp_trigger)
        self._udp_listener = QtTriggerListener(self._udp_bridge, bind_ip="0.0.0.0", port=40002, log=self._append_log, enable_broadcast=False)
        self._udp_listener.start()

    # ───────────────────────── utilities ─────────────────────────

    @staticmethod
    def _hbox(*widgets):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0,0,0,0)
        for x in widgets: h.addWidget(x)
        h.addStretch(1)
        return w

    def _append_log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {text}"
        self._log_buffer.append(entry)
        if len(self._log_buffer) > 200:
            self._log_buffer = self._log_buffer[-200:]
        self.txt_log.append(entry)

    def _send(self, obj):
        try:
            self.ws.sendTextMessage(json.dumps(obj, separators=(',', ':')))
        except Exception:
            pass

    # ───────────────────────── websocket handlers ─────────────────────────

    def _ws_ok(self):
        self.btn_conn.setEnabled(False)
        self.btn_disc.setEnabled(True)
        self._ping.start()
        self._send({"type": "set_mode", "mode": "training" if self.rad_train.isChecked() else "trigger"})
        self._send({"type": "set_proc_width", "width": int(self.cmb_w.currentText())})
        self._send({"type": "set_publish_every", "n": int(self.sp_every.value())})
        if hasattr(self.cam_panel, "_emit_params"): self.cam_panel._emit_params()
        if hasattr(self.cam_panel, "_emit_view"): self.cam_panel._emit_view()
        self._append_log("Connected")

    def _ws_closed(self):
        self.btn_conn.setEnabled(True)
        self.btn_disc.setEnabled(False)
        self._ping.stop()
        self.lbl_sess.setText("Disconnected")
        self._append_log("Disconnected")

    def _ws_txt(self, txt: str):
        try:
            msg = json.loads(txt)
            t = msg.get("type", "")
        except Exception:
            return

        if t == "hello":
            self.session_id = msg.get("session_id", "-")
            self.lbl_sess.setText(f"Session {self.session_id}")

        elif t == "frame":
            b64 = msg.get("jpeg_b64", "")
            qi = None
            if b64:
                qi, bgr = self._decode_b64_to_qimage_and_bgr(b64)
                if bgr is not None:
                    self._last_frame_bgr = bgr

            w = msg.get("w"); h = msg.get("h")
            if qi is None or w is None or h is None:
                return
            self.last_frame_wh = (int(w), int(h))
            self.video.setFrameSize(self.last_frame_wh)

            # FPS calc
            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - self._last_ts)
            self._last_ts = now
            self._fps_acc += fps; self._fps_n += 1
            if self._fps_n >= 10:
                self.lbl_fps.setText(f"FPS: {self._fps_acc / self._fps_n:.1f}")
                self._fps_acc = 0; self._fps_n = 0

            # show overlay if valid, else raw
            if self._last_overlay and time.time() < self._overlay_until:
                self._set_pixmap(self._last_overlay)
            else:
                self._set_pixmap(qi)

            # Auto processing in Training mode
            self._frame_counter += 1
            if self.rad_train.isChecked():
                if (self._frame_counter % max(1, int(self.sp_every.value()))) == 0:
                    self._process_and_update(publish=True)

        elif t == "ack":
            if not msg.get("ok", True):
                self.status.showMessage(msg.get("error", "error"), 4000)
                self._append_log(f"[ACK-ERR] {msg.get('cmd')} : {msg.get('error')}")

    # ───────────────────────── image helpers ─────────────────────────

    def _decode_b64_to_qimage_and_bgr(self, b64: str) -> Tuple[Optional[QtGui.QImage], Optional[np.ndarray]]:
        try:
            arr = np.frombuffer(base64.b64decode(b64), np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None, None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qi = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
            return qi, bgr
        except Exception:
            return None, None

    def _set_pixmap(self, qi: QtGui.QImage):
        pm = QtGui.QPixmap.fromImage(qi)
        self.video.setPixmapKeepAspect(pm)

    def _show_image(self, bgr: Optional[np.ndarray]):
        if bgr is None: return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        self._set_pixmap(qi)

    # ───────────────────────── actions ─────────────────────────

    def _run_once(self):
        self._process_and_update(publish=True)

    def _do_trigger(self):
        if not self.rad_trig.isChecked():
            self.rad_trig.setChecked(True)  # emits set_mode
        self._send({"type": "trigger"})
        self._append_log("TRIGGER pressed → sending VGR result NOW from current frame")
        if self._last_frame_bgr is not None:
            vis, dets = self._compute_blobs(self._last_frame_bgr)
            self._show_image(vis)
            self.status.showMessage(f"Detected blobs: {len(dets)}   (expected {int(self.sp_expected_count.value())})", 1500)
            self._publish_vgr_result(dets)
            self._last_result_time = time.time()
            self._expect_one_result = False
        else:
            self._append_log("No frame yet → will publish after next frame")
            self._expect_one_result = True

    def _on_udp_trigger(self):
        if not self.rad_trig.isChecked():
            self.rad_trig.setChecked(True)  # emits set_mode
        self._append_log("UDP TRIGGER → sending VGR result NOW from current frame")
        self._send({"type": "trigger"})
        if self._last_frame_bgr is not None:
            vis, dets = self._compute_blobs(self._last_frame_bgr)
            self._show_image(vis)
            self._publish_vgr_result(dets)
            self._last_result_time = time.time()
            self._expect_one_result = False
        else:
            self._append_log("No frame yet → will publish after next frame")
            self._expect_one_result = True

    def _process_and_update(self, publish: bool):
        if self._last_frame_bgr is None:
            self.status.showMessage("No frame yet", 1500)
            return

        vis, dets = self._compute_blobs(self._last_frame_bgr)
        self._show_image(vis)
        self.status.showMessage(
            f"Detected blobs: {len(dets)} (expected {int(self.sp_expected_count.value())}) | "
            f"Holes(primary): {dets[0]['holes_count'] if dets else 0} (expected {int(self.sp_expected_holes.value())})",
            1500
        )

        # fill table
        self.tbl.setRowCount(0)
        for idx, d in enumerate(dets):
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(str(idx)))
            self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(str(int(d["area"]))))
            self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(f'{d["perim"]:.1f}'))
            x,y,w,h = d["bbox"]
            self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(str(int(x))))
            self.tbl.setItem(r, 4, QtWidgets.QTableWidgetItem(str(int(y))))
            self.tbl.setItem(r, 5, QtWidgets.QTableWidgetItem(str(int(w))))
            self.tbl.setItem(r, 6, QtWidgets.QTableWidgetItem(str(int(h))))
            self.tbl.setItem(r, 7, QtWidgets.QTableWidgetItem(f'{d["solidity"]:.2f}'))
            self.tbl.setItem(r, 8, QtWidgets.QTableWidgetItem(f'{d["circularity"]:.3f}'))
            self.tbl.setItem(r, 9, QtWidgets.QTableWidgetItem(f'{d["aspect"]:.3f}'))
            self.tbl.setItem(r,10, QtWidgets.QTableWidgetItem(str(int(d["holes_count"]))))
            self.tbl.setItem(r,11, QtWidgets.QTableWidgetItem(f'{d["mean_intensity"]:.1f}'))

        # In Trigger mode, do one-shot publish and latch result briefly
        if self.rad_trig.isChecked() and self._expect_one_result:
            self._publish_vgr_result(dets)
            self._expect_one_result = False
            self._last_result_time = time.time()

        # In Training mode, publish if desired every run
        if self.rad_train.isChecked() and publish:
            self._publish_vgr_result(dets)

        self._append_log(f"[COUNT] detected={len(dets)} expected={int(self.sp_expected_count.value())}")

    # ───────────────────────── blob pipeline ─────────────────────────

    def _pre_gray(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.chk_clahe.isChecked():
            clip = float(self.ds_clip.value())
            tiles = int(self.sp_tiles.value())
            clahe = cv2.createCLAHE(clipLimit=max(0.1, clip), tileGridSize=(tiles, tiles))
            gray = clahe.apply(gray)
        k = int(self.sp_blur.value())
        if k > 0 and k % 2 == 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        return gray

    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        mode = self.cmb_thr_mode.currentText()
        pol = self.cmb_polarity.currentText()
        invert = (pol == "LightOnDark")  # if features are bright on dark, invert post-threshold
        if mode == "Fixed":
            t = int(self.sp_thr_val.value())
            _, mask = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        elif mode == "AdaptiveMean":
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 21, 5)
        elif mode == "AdaptiveGaussian":
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 21, 5)
        else:  # Otsu
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if invert:
            mask = 255 - mask
        # Morphology cleanup
        it_open = int(self.sp_open.value())
        it_close = int(self.sp_close.value())
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        if it_open > 0: mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=it_open)
        if it_close > 0: mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=it_close)
        # Kill border 1 px to avoid giant border components
        h, w = mask.shape[:2]
        if h > 2 and w > 2:
            mask[0,:] = 0; mask[h-1,:] = 0; mask[:,0] = 0; mask[:,w-1] = 0
        return mask

    def _compute_blobs(self, bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Returns (overlay_bgr, blobs list). Each blob dict contains:
        area, perim, bbox, solidity, circularity, aspect, holes_count, mean_intensity,
        contour, holes[], quad, center[x,y], theta_deg, size_wh[w,h]
        """
        gray = self._pre_gray(bgr)
        mask = self._threshold(gray)

        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or len(cnts) == 0:
            return bgr, []

        hier = hier[0]  # [next, prev, first_child, parent]
        outer_ids = [i for i in range(len(cnts)) if hier[i][3] < 0]

        dets: List[Dict[str, Any]] = []
        eh, ew = mask.shape[:2]
        REJECT_BORDER_TOUCHING = True

        min_area = int(self.sp_min_area.value())
        max_area = int(self.sp_max_area.value())
        asp_min = float(self.ds_aspect_min.value())
        asp_max = float(self.ds_aspect_max.value())
        sol_min = float(self.ds_solidity_min.value())
        circ_min = float(self.ds_circ_min.value())
        holes_min = int(self.sp_holes_min.value())
        holes_max = int(self.sp_holes_max.value())

        for i in outer_ids:
            c = cnts[i]

            # children (holes)
            hole_ids = []
            ch = hier[i][2]
            while ch != -1:
                hole_ids.append(ch)
                ch = hier[ch][0]
            holes = [cnts[k] for k in hole_ids]

            outer_area = float(cv2.contourArea(c))
            holes_area = float(sum(cv2.contourArea(hh) for hh in holes))
            area = max(0.0, outer_area - holes_area)
            perim_outer = float(cv2.arcLength(c, True))
            perim_holes = float(sum(cv2.arcLength(hh, True) for hh in holes))
            perim = perim_outer + perim_holes

            x,y,w,h = cv2.boundingRect(c)
            if REJECT_BORDER_TOUCHING and (x <= 0 or y <= 0 or x + w >= ew-1 or y + h >= eh-1):
                continue

            hull = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / max(1.0, hull_area)
            aspect = w / max(1, h)

            # circularity (robust to scale): 4πA / P^2
            circularity = (4.0 * np.pi * area) / max(1.0, perim*perim)

            # holes count
            holes_count = len(holes)

            # mean intensity inside the filled outer contour (excluding holes)
            mask_poly = np.zeros_like(mask)
            cv2.drawContours(mask_poly, [c], -1, 255, thickness=-1)
            for hh in holes:
                cv2.drawContours(mask_poly, [hh], -1, 0, thickness=-1)
            meanI = float(cv2.mean(gray, mask=mask_poly)[0])

            # UI filters
            if area < min_area: continue
            if max_area > 0 and area > max_area: continue
            if not (asp_min <= aspect <= asp_max): continue
            if solidity < sol_min: continue
            if circularity < circ_min: continue
            if not (holes_min <= holes_count <= holes_max): continue
            if bool(self.chk_int_gate.isChecked()):
                i_min = int(self.sp_int_min.value()); i_max = int(self.sp_int_max.value())
                if not (i_min <= meanI <= i_max): continue
                
            # --- orientation via minAreaRect (snap to {0, 90} and rebuild box) ---
            rect = cv2.minAreaRect(c)
            (cx_r, cy_r), (RW, RH), Rangle = rect   # OpenCV: Rangle in (-90, 0]

            # OpenCV’s angle is defined with the side length ambiguity; make a 0..90 measure
            angle = float(Rangle)
            if RW < RH:
                angle += 90.0  # now 'angle' is 0..90 for the long side

            # If nearly square/round, lock to 0 (to avoid jitter)
            aspect_ratio = min(RW, RH) / max(RW, RH + 1e-9)
            near_square = aspect_ratio > 0.85

            # Final snapped angle: only 0 or 90
            if near_square:
                theta_deg = 0.0
            else:
                theta_deg = 0.0 if abs(angle) < 45.0 else 90.0

            # Rebuild a rectangle with the snapped angle and get its box points
            snapped_rect = ((cx_r, cy_r), (RW, RH), theta_deg if RW >= RH else theta_deg - 90.0)
            box = cv2.boxPoints(snapped_rect)
            box = np.int32(np.round(box))

            # (Optionally keep theta in [-90, 90] for publishing)
            if theta_deg > 90.0:  theta_deg -= 180.0
            if theta_deg < -90.0: theta_deg += 180.0

            dets.append({
                "area": area,
                "perim": perim,
                "bbox": (x, y, w, h),
                "solidity": solidity,
                "circularity": circularity,
                "aspect": aspect,
                "holes_count": holes_count,
                "mean_intensity": meanI,
                "contour": c,
                "holes": holes,
                "quad": box.tolist(),
                "center": [int(x + w/2), int(y + h/2)],
                "theta_deg": theta_deg,
                "size_wh": [int(w), int(h)],
            })

        # rank & keepN
        dets.sort(key=lambda d: d["area"], reverse=True)
        keepN = int(self.sp_keepN.value())
        if len(dets) > keepN: dets = dets[:keepN]

        # draw overlay
        vis = bgr.copy()
        if self.chk_draw_cnt.isChecked():
            for d in dets:
                cv2.drawContours(vis, [d["contour"]], -1, (0,255,255), 2)
                for hh in d.get("holes", []):
                    cv2.drawContours(vis, [hh], -1, (0,255,255), 2)
        for i, d in enumerate(dets):
            x,y,w,h = d["bbox"]
            if self.chk_draw_box.isChecked():
                cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
                if d.get("quad"):
                    q = np.array(d["quad"], dtype=np.int32)
                    cv2.polylines(vis, [q], True, (0,128,255), 2)
            if self.chk_draw_cent.isChecked():
                cx, cy = d["center"]
                cv2.circle(vis, (cx,cy), 4, (255,0,0), -1)
            if self.chk_draw_ids.isChecked():
                lab = f"#{i} A={int(d['area'])} S={d['solidity']:.2f} C={d['circularity']:.3f} H={d['holes_count']}"
                cv2.putText(vis, lab, (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

        try:
            jpeg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()
            qi = QtGui.QImage.fromData(jpeg, "JPG")
            self._last_overlay = qi
            self._overlay_until = time.time() + 1.0
        except Exception:
            pass

        return vis, dets

    # ───────────────────────── GO/NOGO publishing ─────────────────────────

    def _choose_blob(self, dets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not dets:
            return None
        if self.rad_pick_largest.isChecked():
            return dets[0]  # already sorted by area
        # simple score: favor round, solid, big
        return max(dets, key=lambda d: d["circularity"] * d["solidity"] * max(1.0, d["area"]))

    def _passes_rules(self, d: Dict[str, Any]) -> bool:
        # We already applied basic gates in _compute_blobs.
        # Here we only enforce the existence rule (and could add extra business logic if needed).
        if d is None:
            return False
        return True

    def _publish_vgr_result(self, dets: List[Dict[str, Any]]):
        if not self.chk_pub_robot.isChecked():
            return

        # ── rules
        count_detected = len(dets)
        expected_blobs = int(self.sp_expected_count.value())
        expected_holes = int(self.sp_expected_holes.value())

        # Blobs rule
        is_ok_blobs = (count_detected == expected_blobs)
        if self.chk_at_least.isChecked():
            is_ok_blobs = (count_detected >= expected_blobs)

        # Choose primary blob (same policy as before)
        chosen = self._choose_blob(dets) if dets else None

        # Holes rule (by default, use holes in the primary blob)
        holes_measured = chosen["holes_count"] if chosen else 0
        is_ok_holes = (holes_measured == expected_holes)
        if self.chk_holes_at_least.isChecked():
            is_ok_holes = (holes_measured >= expected_holes)

        # If you prefer TOTAL holes across all blobs instead of primary:
        # holes_measured = sum(d["holes_count"] for d in dets)
        # (leave the rest unchanged)

        # Final decision = BOTH rules
        is_ok = (is_ok_blobs and is_ok_holes)
        send_name = "go" if is_ok else "nogo"

        if chosen is None:
            payload = {
                "version":"1.0",
                "sdk":"vision_ui",
                "session": self.session_id,
                "timestamp_ms": int(time.time()*1000),
                "camera": {"proc_width": int(self.last_frame_wh[0]), "proc_height": int(self.last_frame_wh[1])},
                "result": {"objects": [], "counts": {"objects": 0, "detections": 0}}
            }
            self.pub_vgr.send_json(payload)
            self.pub_robot.send_json(payload)
            self._append_log(f"VGR ← {send_name} (blobs={count_detected}/{expected_blobs}, holes={holes_measured}/{expected_holes}) [no blobs]")
            return

        # build one det (unchanged)
        x, y, w, h = chosen["bbox"]
        cx, cy = chosen.get("center", [int(x + w/2), int(y + h/2)])
        quad = chosen.get("quad")
        theta = float(chosen.get("theta_deg", 0.0))
        size_wh = chosen.get("size_wh", [int(w), int(h)])

        score = float(chosen["solidity"])
        score = float(max(0.0, min(1.0, score)))

        det_obj = {
            "instance_id": 0,
            "score": float(score),
            "inliers": int(len(chosen.get("contour", []))),
            "pose": {
                "x": float(cx),
                "y": float(cy),
                "theta_deg": float(theta),
                "x_scale": 1.0,
                "y_scale": 1.0,
                "origin_xy": [float(quad[0][0]), float(quad[0][1])] if quad else [float(x), float(y)]
            },
            "center": [float(cx), float(cy)],
            "quad": quad if quad else None,
        }

        obj = {
            "object_id": 1,
            "name": send_name,
            "template_size": [int(size_wh[0]), int(size_wh[1])],
            "detections": [det_obj]
        }

        payload = {
            "version":"1.0",
            "sdk":"vision_ui",
            "session": self.session_id,
            "timestamp_ms": int(time.time()*1000),
            "camera": {"proc_width": int(self.last_frame_wh[0]), "proc_height": int(self.last_frame_wh[1])},
            "result": {"objects": [obj], "counts": {"objects": 1, "detections": 1}}
        }

        self.pub_vgr.send_json(payload)
        self.pub_robot.send_json(payload)
        self._append_log(
            f"VGR ← {send_name}  blobs={count_detected}/{expected_blobs}  holes={holes_measured}/{expected_holes}  "
            f"px=({cx:.1f},{cy:.1f}) yaw={theta:.2f}° size={size_wh}"
        )

    # ───────────────────────── settings ─────────────────────────

    def eventFilter(self, obj, event):
        if obj is getattr(self, "controlsPanel", None) and event.type() in (QtCore.QEvent.Hide, QtCore.QEvent.Show):
            action = getattr(self, "controlsPanelAction", None)
            if action is not None:
                visible = obj.isVisible()
                if action.isChecked() != visible:
                    action.setChecked(visible)
        return super().eventFilter(obj, event)

    def _load_settings(self):
        S = self._settings
        def to_bool(v):
            if isinstance(v, bool): return v
            if isinstance(v, str):
                vv=v.strip().lower()
                if vv in ("1","true","yes","on"): return True
                if vv in ("0","false","no","off"): return False
            if isinstance(v,(int,float)): return bool(v)
            return None
        def to_int(v):
            try: return int(v)
            except (TypeError,ValueError): return None
        def to_float(v):
            try: return float(v)
            except (TypeError,ValueError): return None
        def to_str(v): return v if isinstance(v,str) else None

        try:
            geom = to_str(S.value('window/geometry'))
            if geom: self.restoreGeometry(QtCore.QByteArray.fromHex(geom.encode()))
        except Exception: pass
        try:
            state = to_str(S.value('window/state'))
            if state: self.restoreState(QtCore.QByteArray.fromHex(state.encode()))
        except Exception: pass

        host = to_str(S.value('connection/host'))
        if host: self.ed_host.setText(host)

        mode = to_str(S.value('mode/current'))
        if mode == 'trigger': self.rad_trig.setChecked(True)
        elif mode == 'training': self.rad_train.setChecked(True)

        self.cmb_w.blockSignals(True); self.sp_every.blockSignals(True)
        try:
            width = to_str(S.value('processing/proc_width'))
            if width and self.cmb_w.findText(width) >= 0:
                self.cmb_w.setCurrentText(width)
            every = to_int(S.value('processing/detect_every'))
            if every is not None:
                self.sp_every.setValue(every)
        finally:
            self.cmb_w.blockSignals(False); self.sp_every.blockSignals(False)

        # Robot + VGR
        robot_ip = to_str(S.value('robot/ip'))
        if robot_ip: self.ed_robot_ip.setText(robot_ip)
        rp = to_int(S.value('robot/port'))
        if rp: self.sp_robot_port.setValue(rp)
        ap = to_bool(S.value('robot/auto_publish'))
        if ap is not None: self.chk_pub_robot.setChecked(ap)
        vgr_ip = to_str(S.value('vgr/ip'))
        if vgr_ip: self.ed_vgr_ip.setText(vgr_ip)
        vp = to_int(S.value('vgr/port'))
        if vp: self.sp_vgr_port.setValue(vp)

        # Blob pipeline
        try:
            self.chk_clahe.setChecked(bool(to_bool(S.value('blob/clahe')) or False))
            v = to_float(S.value('blob/clip')); self.ds_clip.setValue(v if v is not None else 2.0)
            i = to_int(S.value('blob/tiles')); self.sp_tiles.setValue(i if i is not None else 8)
            i = to_int(S.value('blob/blur')); self.sp_blur.setValue(i if i is not None else 5)
            s = to_str(S.value('blob/thr_mode')); 
            if s and self.cmb_thr_mode.findText(s) >= 0: self.cmb_thr_mode.setCurrentText(s)
            i = to_int(S.value('blob/thr_val')); self.sp_thr_val.setValue(i if i is not None else 128)
            s = to_str(S.value('blob/polarity')); 
            if s and self.cmb_polarity.findText(s) >= 0: self.cmb_polarity.setCurrentText(s)
            i = to_int(S.value('blob/open')); self.sp_open.setValue(i if i is not None else 1)
            i = to_int(S.value('blob/close')); self.sp_close.setValue(i if i is not None else 1)
        except Exception:
            pass

        # Filters
        try:
            i = to_int(S.value('filter/min_area')); self.sp_min_area.setValue(i if i is not None else 300)
            i = to_int(S.value('filter/max_area')); self.sp_max_area.setValue(i if i is not None else 0)
            f = to_float(S.value('filter/aspect_min')); self.ds_aspect_min.setValue(f if f is not None else 0.0)
            f = to_float(S.value('filter/aspect_max')); self.ds_aspect_max.setValue(f if f is not None else 100.0)
            f = to_float(S.value('filter/solidity')); self.ds_solidity_min.setValue(f if f is not None else 0.0)
            f = to_float(S.value('filter/circularity')); self.ds_circ_min.setValue(f if f is not None else 0.0)
            i = to_int(S.value('filter/holes_min')); self.sp_holes_min.setValue(i if i is not None else 0)
            i = to_int(S.value('filter/holes_max')); self.sp_holes_max.setValue(i if i is not None else 100)
            i = to_int(S.value('filter/keepN')); self.sp_keepN.setValue(i if i is not None else 50)
        except Exception:
            pass

        # Selection + intensity + decision
        try:
            b = to_bool(S.value('sel/pick_best'))
            if b is True: self.rad_pick_best.setChecked(True)
            else: self.rad_pick_largest.setChecked(True)
            b = to_bool(S.value('intensity/enabled')); self.chk_int_gate.setChecked(b if b is not None else False)
            i = to_int(S.value('intensity/min')); self.sp_int_min.setValue(i if i is not None else 0)
            i = to_int(S.value('intensity/max')); self.sp_int_max.setValue(i if i is not None else 255)
        except Exception:
            pass
        i = to_int(S.value('decision/expected_count')); 
        self.sp_expected_count.setValue(i if i is not None else 1)
        b = to_bool(S.value('decision/at_least'))
        if hasattr(self, 'chk_at_least') and b is not None:
            self.chk_at_least.setChecked(b)
            
        i = to_int(S.value('decision/expected_count')); 
        self.sp_expected_count.setValue(i if i is not None else 1)

        i = to_int(S.value('decision/expected_holes'));
        self.sp_expected_holes.setValue(i if i is not None else 0)

        b = to_bool(S.value('decision/at_least'))
        if b is not None: self.chk_at_least.setChecked(b)

        b = to_bool(S.value('decision/holes_at_least'))
        if b is not None: self.chk_holes_at_least.setChecked(b)


        logs_json = to_str(S.value('logs/history')); self._log_buffer = []
        if logs_json:
            try:
                entries = json.loads(logs_json)
                if isinstance(entries, list):
                    self._log_buffer = [e for e in entries[-200:] if isinstance(e, str)]
            except Exception:
                pass
        if self._log_buffer:
            self.txt_log.clear()
            for entry in self._log_buffer:
                self.txt_log.append(entry)

    def _save_settings(self):
        S = self._settings
        try: S.setValue('window/geometry', bytes(self.saveGeometry().toHex()).decode('ascii'))
        except Exception: pass
        try: S.setValue('window/state', bytes(self.saveState().toHex()).decode('ascii'))
        except Exception: pass

        S.setValue('connection/host', self.ed_host.text())
        S.setValue('mode/current', 'trigger' if self.rad_trig.isChecked() else 'training')
        S.setValue('processing/proc_width', self.cmb_w.currentText())
        S.setValue('processing/detect_every', int(self.sp_every.value()))

        # Blob pipeline
        S.setValue('blob/clahe', bool(self.chk_clahe.isChecked()))
        S.setValue('blob/clip', float(self.ds_clip.value()))
        S.setValue('blob/tiles', int(self.sp_tiles.value()))
        S.setValue('blob/blur', int(self.sp_blur.value()))
        S.setValue('blob/thr_mode', self.cmb_thr_mode.currentText())
        S.setValue('blob/thr_val', int(self.sp_thr_val.value()))
        S.setValue('blob/polarity', self.cmb_polarity.currentText())
        S.setValue('blob/open', int(self.sp_open.value()))
        S.setValue('blob/close', int(self.sp_close.value()))

        # Filters
        S.setValue('filter/min_area', int(self.sp_min_area.value()))
        S.setValue('filter/max_area', int(self.sp_max_area.value()))
        S.setValue('filter/aspect_min', float(self.ds_aspect_min.value()))
        S.setValue('filter/aspect_max', float(self.ds_aspect_max.value()))
        S.setValue('filter/solidity', float(self.ds_solidity_min.value()))
        S.setValue('filter/circularity', float(self.ds_circ_min.value()))
        S.setValue('filter/holes_min', int(self.sp_holes_min.value()))
        S.setValue('filter/holes_max', int(self.sp_holes_max.value()))
        S.setValue('filter/keepN', int(self.sp_keepN.value()))

        # Selection + intensity + decision
        S.setValue('sel/pick_best', bool(self.rad_pick_best.isChecked()))
        S.setValue('intensity/enabled', bool(self.chk_int_gate.isChecked()))
        S.setValue('intensity/min', int(self.sp_int_min.value()))
        S.setValue('intensity/max', int(self.sp_int_max.value()))
        S.setValue('decision/expected_count', int(self.sp_expected_count.value()))
        S.setValue('decision/at_least', bool(self.chk_at_least.isChecked()))
        S.setValue('decision/expected_count', int(self.sp_expected_count.value()))
        S.setValue('decision/expected_holes', int(self.sp_expected_holes.value()))
        S.setValue('decision/at_least', bool(self.chk_at_least.isChecked()))
        S.setValue('decision/holes_at_least', bool(self.chk_holes_at_least.isChecked()))


        # Robot + VGR + logs
        S.setValue('robot/ip', self.ed_robot_ip.text())
        S.setValue('robot/port', int(self.sp_robot_port.value()))
        S.setValue('robot/auto_publish', bool(self.chk_pub_robot.isChecked()))
        S.setValue('vgr/ip', self.ed_vgr_ip.text())
        S.setValue('vgr/port', int(self.sp_vgr_port.value()))
        S.setValue('logs/history', json.dumps(self._log_buffer[-200:]))

        S.sync()

    def closeEvent(self, event):
        try:
            self._save_settings()
            try:
                if hasattr(self, "_udp_listener") and self._udp_listener:
                    self._udp_listener.stop()
            except Exception:
                pass
        finally:
            super().closeEvent(event)

# ───────────────────────── Entrypoint ─────────────────────────

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
