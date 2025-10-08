#!/usr/bin/env python3
"""
RPi Vision Client – Canny Contour UI (Industrial Template Matching)
- Receives frames over WS, applies Canny + contouring (no other binarizers).
- "Teach from ROI" captures template contours into a library (named).
- Runtime matches detections to the selected template via cv2.matchShapes
  (I1/I2/I3 selectable) + area/perimeter tolerances.
- Publishes GO/NOGO + bbox to robot over UDP (best passing match).

Extras:
- CLAHE (optional) + Gaussian blur before Canny.
- Edge dilation (optional) to stabilize contours.
- Filters: area, aspect, solidity, perimeter, keep top-N.
- Template Library: add/remove/rename/select, save/load to JSON.

Requires: PyQt5, numpy, opencv-python
"""

import sys, time, base64, json, socket, threading, inspect, os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebSockets
def _lazy_paddleocr():
    import re
    from paddleocr import PaddleOCR, __version__ as _pocr_ver

    def _parse_ver(s: str):
        m = re.findall(r"\d+", s)
        while len(m) < 3: m.append("0")
        return tuple(int(x) for x in m[:3])

    v = _parse_ver(_pocr_ver)  # e.g., (3,2,0)

    if v >= (3, 0, 0):
        # PaddleOCR 3.x — use new flags; DO NOT pass max_batch_size/use_gpu
        return PaddleOCR(
            device='cpu',
            use_textline_orientation=True,
            lang='en',
            # only set these if you want to override defaults; None means “use built-in”
            text_detection_model_dir=None,
            text_recognition_model_dir=None,
            text_det_limit_side_len=1536,
        )
    else:
        # PaddleOCR 2.x — legacy args
        return PaddleOCR(
            use_angle_cls=True, lang='en',
            det_model_dir=None, rec_model_dir=None,
            use_gpu=False, ir_optim=False, enable_mkldnn=False, cpu_threads=1,
            det_limit_side_len=1536, det_db_score_mode='fast', max_batch_size=4
        )


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
        if self._sock is None:
            return
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

# ───────────────────────── UDP publisher to robot ─────────────────────────

class RobotUDPPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 40001):
        self._host = host
        self._port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def set_target(self, host: str, port: int):
        self._host = host.strip()
        self._port = int(port)

    def send_json(self, payload: dict):
        try:
            msg = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._sock.sendto(msg, (self._host, self._port))
        except Exception as e:
            print(f"[PUB] UDP send error: {e}", flush=True)

# ───────────────────────── VideoLabel (aspect + ROI rectangle) ─────────────────────────

class VideoLabel(QtWidgets.QLabel):
    """Keeps aspect, draws scaled pixmap, and supports rectangular ROI selection."""
    roiSelected = QtCore.pyqtSignal(QtCore.QRect)  # rectangle in display coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_draw_rect = QtCore.QRect()
        self._frame_wh: Tuple[int, int] = (0, 0)
        self._rect_mode = False
        self._origin = QtCore.QPoint()
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self.setMouseTracking(True)

    def setFrameSize(self, wh: Tuple[int, int]):
        self._frame_wh = (int(wh[0]), int(wh[1]))

    def setPixmapKeepAspect(self, pm: QtGui.QPixmap):
        if pm.isNull():
            super().setPixmap(pm)
            self._last_draw_rect = QtCore.QRect()
            return
        area = self.size()
        scaled = pm.scaled(area, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        x = (area.width() - scaled.width()) // 2
        y = (area.height() - scaled.height()) // 2
        self._last_draw_rect = QtCore.QRect(x, y, scaled.width(), scaled.height())

        canvas = QtGui.QPixmap(area)
        canvas.fill(QtCore.Qt.black)
        p = QtGui.QPainter(canvas)
        p.drawPixmap(self._last_draw_rect, scaled)
        p.end()
        super().setPixmap(canvas)

    def enable_rect_selection(self, ok: bool):
        self._rect_mode = bool(ok)
        if not ok:
            self._rubber.hide()

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
            self._rubber.hide()
            self._rect_mode = False
            if r.width() > 5 and r.height() > 5:
                self.roiSelected.emit(r)
            return
        super().mouseReleaseEvent(ev)

# ───────────────────────── Camera Panel (external or local minimal) ─────────────────────────

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
        grid.addWidget(QtWidgets.QLabel("View"), r,0); grid.addLayout(hv, r,1); r+=1
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
        self.setWindowTitle("RPi Vision Client – Canny Contour")
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
        self._settings = QtCore.QSettings('vision_sdk', 'ui_canny_contour')
        self._log_buffer: List[str] = []
        
        # snapshots dir (stable across working dirs)
        self._app_dir = os.path.abspath(os.path.dirname(__file__))
        self._out_dir = os.path.join(self._app_dir, "output")
        os.makedirs(self._out_dir, exist_ok=True)
        # (log later, after self.txt_log exists)

        self._last_frame_bgr: Optional[np.ndarray] = None
        self._last_overlay: Optional[QtGui.QImage] = None
        self._overlay_until = 0.0

        self._expect_one_result = False
        self._last_result_time = 0.0

        # Template library
        # item = {"name": str, "area": float, "perim": float, "contour": ndarray Nx1x2 (int)}
        self._templates: List[Dict[str, Any]] = []
        self._active_index: int = -1

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
        self.ed_vgr_ip   = QtWidgets.QLineEdit(self.ed_robot_ip.text())
        self.sp_vgr_port = QtWidgets.QSpinBox(); self.sp_vgr_port.setRange(1,65535); self.sp_vgr_port.setValue(40003)
        cf.addRow("VGR Result IP:", self.ed_vgr_ip)
        cf.addRow("VGR Result Port:", self.sp_vgr_port)

        # UDP publisher binding
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

        # Canny pipeline parameters
        canny_box = QtWidgets.QGroupBox("Canny Pipeline")
        cf2 = QtWidgets.QFormLayout(canny_box)

        # Preprocess
        self.chk_clahe = QtWidgets.QCheckBox("CLAHE"); self.chk_clahe.setChecked(False)
        self.ds_clip = QtWidgets.QDoubleSpinBox(); self.ds_clip.setRange(0.5, 10.0); self.ds_clip.setDecimals(2); self.ds_clip.setValue(2.0)
        self.sp_tiles = QtWidgets.QSpinBox(); self.sp_tiles.setRange(2, 16); self.sp_tiles.setValue(8)
        self.sp_blur = QtWidgets.QSpinBox(); self.sp_blur.setRange(0, 21); self.sp_blur.setSingleStep(2); self.sp_blur.setValue(5)  # 0 = off, odd only
        cf2.addRow("CLAHE", self._hbox(self.chk_clahe, QtWidgets.QLabel("clip"), self.ds_clip, QtWidgets.QLabel("tiles"), self.sp_tiles))
        cf2.addRow("Gaussian Blur (ksize odd, 0=off)", self.sp_blur)

        # Canny core
        self.sp_canny_lo = QtWidgets.QSpinBox(); self.sp_canny_lo.setRange(0,255); self.sp_canny_lo.setValue(50)
        self.sp_canny_hi = QtWidgets.QSpinBox(); self.sp_canny_hi.setRange(1,255); self.sp_canny_hi.setValue(150)
        self.cmb_canny_ap = QtWidgets.QComboBox(); self.cmb_canny_ap.addItems(["3","5","7"])
        self.chk_l2 = QtWidgets.QCheckBox("L2 gradient"); self.chk_l2.setChecked(True)
        cf2.addRow("Canny lo/hi", self._hbox(self.sp_canny_lo, self.sp_canny_hi))
        cf2.addRow("Aperture", self.cmb_canny_ap)
        cf2.addRow("", self.chk_l2)

        # Post edge
        self.sp_dilate = QtWidgets.QSpinBox(); self.sp_dilate.setRange(0, 10); self.sp_dilate.setValue(1)
        cf2.addRow("Edge dilate (iter)", self.sp_dilate)

        left_v.addWidget(canny_box)

        # Contour filters
        filt_box = QtWidgets.QGroupBox("Contour Filters")
        ff = QtWidgets.QFormLayout(filt_box)
        self.sp_min_area = QtWidgets.QSpinBox(); self.sp_min_area.setRange(0, 1_000_000); self.sp_min_area.setValue(200)
        self.sp_max_area = QtWidgets.QSpinBox(); self.sp_max_area.setRange(0, 10_000_000); self.sp_max_area.setValue(0)  # 0 = no limit
        self.ds_aspect_min = QtWidgets.QDoubleSpinBox(); self.ds_aspect_min.setRange(0.0, 100.0); self.ds_aspect_min.setDecimals(2); self.ds_aspect_min.setValue(0.0)
        self.ds_aspect_max = QtWidgets.QDoubleSpinBox(); self.ds_aspect_max.setRange(0.0, 100.0); self.ds_aspect_max.setDecimals(2); self.ds_aspect_max.setValue(100.0)
        self.ds_solidity_min = QtWidgets.QDoubleSpinBox(); self.ds_solidity_min.setRange(0.0, 1.0); self.ds_solidity_min.setDecimals(2); self.ds_solidity_min.setValue(0.0)
        self.ds_perim_min = QtWidgets.QDoubleSpinBox(); self.ds_perim_min.setRange(0.0, 100000.0); self.ds_perim_min.setDecimals(1); self.ds_perim_min.setValue(0.0)
        self.sp_keepN = QtWidgets.QSpinBox(); self.sp_keepN.setRange(1, 200); self.sp_keepN.setValue(50)
        ff.addRow("Min / Max area", self._hbox(self.sp_min_area, self.sp_max_area))
        ff.addRow("Aspect min/max", self._hbox(self.ds_aspect_min, self.ds_aspect_max))
        ff.addRow("Solidity ≥", self.ds_solidity_min)
        ff.addRow("Perimeter ≥", self.ds_perim_min)
        ff.addRow("Keep top N (by area)", self.sp_keepN)
        left_v.addWidget(filt_box)

        # Template library + matching
        lib_box = QtWidgets.QGroupBox("Template Library & Matching")
        lf = QtWidgets.QFormLayout(lib_box)
        self.lst_tpl = QtWidgets.QListWidget()
        self.btn_teach = QtWidgets.QPushButton("Teach from ROI…")
        self.btn_add_name = QtWidgets.QPushButton("Rename…")
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_save_lib = QtWidgets.QPushButton("Save Library…")
        self.btn_load_lib = QtWidgets.QPushButton("Load Library…")
        libbtns = self._hbox(self.btn_teach, self.btn_add_name, self.btn_delete, self.btn_save_lib, self.btn_load_lib)
        self.cmb_metric = QtWidgets.QComboBox(); self.cmb_metric.addItems(["I1","I2","I3"])
        self.ds_match_thr = QtWidgets.QDoubleSpinBox(); self.ds_match_thr.setRange(0.0, 10.0); self.ds_match_thr.setDecimals(4); self.ds_match_thr.setValue(0.10)
        self.sp_area_tol = QtWidgets.QSpinBox(); self.sp_area_tol.setRange(0, 200); self.sp_area_tol.setValue(25)   # ±%
        self.sp_perim_tol = QtWidgets.QSpinBox(); self.sp_perim_tol.setRange(0, 200); self.sp_perim_tol.setValue(20) # ±%
        self.chk_require_active = QtWidgets.QCheckBox("Require active template for GO"); self.chk_require_active.setChecked(False)

        lf.addRow(self.lst_tpl)
        lf.addRow(libbtns)
        lf.addRow("Match metric / thr", self._hbox(self.cmb_metric, self.ds_match_thr))
        lf.addRow("Area ±% / Perim ±%", self._hbox(self.sp_area_tol, self.sp_perim_tol))
        lf.addRow("", self.chk_require_active)
        left_v.addWidget(lib_box)

        # Draw options
        draw_box = QtWidgets.QGroupBox("Draw Options")
        dv = QtWidgets.QVBoxLayout(draw_box)
        self.chk_draw_cnt = QtWidgets.QCheckBox("Contours"); self.chk_draw_cnt.setChecked(True)
        self.chk_draw_box = QtWidgets.QCheckBox("Boxes"); self.chk_draw_box.setChecked(True)
        self.chk_draw_cent = QtWidgets.QCheckBox("Centroids"); self.chk_draw_cent.setChecked(False)
        self.chk_draw_ids = QtWidgets.QCheckBox("IDs / Scores"); self.chk_draw_ids.setChecked(True)
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
        
        # OCR (GO/NOGO) – minimal controls
        ocr_box = QtWidgets.QGroupBox("OCR (drives GO/NOGO)")
        of = QtWidgets.QFormLayout(ocr_box)
        self.ed_ocr_target = QtWidgets.QLineEdit()
        self.ed_ocr_target.setPlaceholderText("e.g. LOT123, OK, 24-09, ...")
        self.cb_ocr_match = QtWidgets.QComboBox(); self.cb_ocr_match.addItems(["contains","exact","startswith","regex"])
        self.chk_ocr_case = QtWidgets.QCheckBox("Case sensitive")
        self.ds_ocr_minconf = QtWidgets.QDoubleSpinBox(); self.ds_ocr_minconf.setRange(0.0,1.0); self.ds_ocr_minconf.setSingleStep(0.05); self.ds_ocr_minconf.setValue(0.50)
        self.chk_ocr_pre = QtWidgets.QCheckBox("Preprocess (CLAHE + denoise + upscale)"); self.chk_ocr_pre.setChecked(True)
        self.chk_use_ocr = QtWidgets.QCheckBox("Use OCR for GO/NOGO"); self.chk_use_ocr.setChecked(True)
        of.addRow("Target", self.ed_ocr_target)
        of.addRow("Match", self.cb_ocr_match)
        of.addRow(self.chk_ocr_case)
        of.addRow("Min Conf", self.ds_ocr_minconf)
        of.addRow(self.chk_ocr_pre)
        of.addRow(self.chk_use_ocr)
        left_v.addWidget(ocr_box)
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
        grp = QtWidgets.QGroupBox("Contours (last run)")
        vv = QtWidgets.QVBoxLayout(grp)
        self.tbl = QtWidgets.QTableWidget(0, 10)
        self.tbl.setHorizontalHeaderLabels(
            ["id","area","perim","x","y","w","h","solidity","match","ocr"]
        )        
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self._last_ocr_summary = "—"

        vv.addWidget(self.tbl)
        right.addWidget(grp, 1)

        self.txt_log = QtWidgets.QTextEdit(); self.txt_log.setReadOnly(True)
        self._append_log(f"[INIT] snapshot dir = {self._out_dir}")

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

        # teach ROI + library ops
        self.btn_teach.clicked.connect(self._start_teach_roi)
        self.video.roiSelected.connect(self._finish_teach_roi)
        self.lst_tpl.currentRowChanged.connect(self._on_tpl_selected)
        self.btn_add_name.clicked.connect(self._rename_template)
        self.btn_delete.clicked.connect(self._delete_template)
        self.btn_save_lib.clicked.connect(self._save_library)
        self.btn_load_lib.clicked.connect(self._load_library)

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
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        self._set_pixmap(qi)

    # ───────────────────────── teaching via ROI ─────────────────────────

    def _start_teach_roi(self):
        if self._last_frame_bgr is None:
            self.status.showMessage("No frame to teach from", 2000)
            return
        self.status.showMessage("Drag ROI on the image…", 2000)
        self.video.enable_rect_selection(True)

    def _finish_teach_roi(self, rect_disp: QtCore.QRect):
        if self._last_frame_bgr is None: return
        draw = self.video._last_draw_rect
        if draw.isNull() or self.last_frame_wh == (0, 0): return

        fw, fh = self.last_frame_wh
        sx = fw / float(draw.width())
        sy = fh / float(draw.height())
        x0 = int((rect_disp.x() - draw.x()) * sx)
        y0 = int((rect_disp.y() - draw.y()) * sy)
        x1 = int((rect_disp.right() - draw.x()) * sx)
        y1 = int((rect_disp.bottom() - draw.y()) * sy)
        x0 = max(0, min(fw-1, x0)); x1 = max(0, min(fw-1, x1))
        y0 = max(0, min(fh-1, y0)); y1 = max(0, min(fh-1, y1))
        if x1 <= x0 or y1 <= y0:
            self.status.showMessage("ROI too small", 2000); return

        roi = self._last_frame_bgr[y0:y1+1, x0:x1+1].copy()
        vis, dets = self._compute_contours(roi, offset=(x0, y0))

        if not dets:
            self._append_log("Teach: no contour in ROI")
            self.status.showMessage("Teach failed: no contour", 2500)
            return

        # choose largest area detection as template
        best = dets[0]
        # Build a horizontally mirrored version of the taught contour (around its centroid)
        c = best["contour"].astype(np.float32)
        M = cv2.moments(c)
        if M["m00"] > 1e-6:
            cx = M["m10"] / M["m00"]; cy = M["m01"] / M["m00"]
        else:
            # fallback if contour is degenerate
            xs = c[:,:,0]; ys = c[:,:,1]
            cx = float(xs.mean()); cy = float(ys.mean())

        cm = c.copy()
        cm[:,:,0] -= cx; cm[:,:,1] -= cy
        cm[:,:,0] *= -1.0                      # mirror X
        cm[:,:,0] += cx; cm[:,:,1] += cy
        contour_mirror = cm.astype(np.int32)

        name, ok = QtWidgets.QInputDialog.getText(self, "Template name", "Enter name:")
        if not ok or not name: name = f"tpl_{len(self._templates)+1}"

        tpl = dict(
            name=str(name),
            area=float(best["area"]),
            perim=float(best["perim"]),
            contour=best["contour"],           # original
            contour_mirror=contour_mirror      # <- add this
        )

        self._templates.append(tpl)
        self._refresh_tpl_list(select_last=True)
        self._append_log(f"Teach ok: '{name}' area={int(tpl['area'])} perim={tpl['perim']:.1f} bbox={best['bbox']}")

        overlay = self._last_frame_bgr.copy()
        x,y,w,h = best["bbox"]
        cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,255), 2)
        cv2.rectangle(overlay, (x0,y0), (x1,y1), (255,0,0), 2)
        try:
            jpeg = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tobytes()
            qi = QtGui.QImage.fromData(jpeg, "JPG")
            self._last_overlay = qi
            self._overlay_until = time.time() + 1.5
            self._set_pixmap(qi)
        except Exception:
            pass

    def _refresh_tpl_list(self, select_last=False):
        self.lst_tpl.blockSignals(True)
        self.lst_tpl.clear()
        for t in self._templates:
            self.lst_tpl.addItem(f"{t['name']}  (A={int(t['area'])}, P={t['perim']:.1f})")
        if select_last and self._templates:
            self.lst_tpl.setCurrentRow(len(self._templates)-1)
        self.lst_tpl.blockSignals(False)
        self._active_index = self.lst_tpl.currentRow()

    def _on_tpl_selected(self, row: int):
        self._active_index = int(row) if row is not None else -1

    def _rename_template(self):
        i = self._active_index
        if 0 <= i < len(self._templates):
            cur = self._templates[i]["name"]
            name, ok = QtWidgets.QInputDialog.getText(self, "Rename template", "New name:", text=cur)
            if ok and name:
                self._templates[i]["name"] = name
                self._refresh_tpl_list()
                self.lst_tpl.setCurrentRow(i)

    def _delete_template(self):
        i = self._active_index
        if 0 <= i < len(self._templates):
            del self._templates[i]
            self._refresh_tpl_list()
            self._active_index = self.lst_tpl.currentRow()
            
    def _update_ocr_summary(self, decision: str, best: float, res: dict):
        cnt = len(res.get("detections", [])) if isinstance(res, dict) else 0
        ms  = res.get("infer_ms", 0.0) if isinstance(res, dict) else 0.0
        self._last_ocr_summary = f"{decision} {best:.2f} ({cnt},{ms:.0f}ms)"

    def _save_library(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Template Library", "canny_templates.json", "JSON (*.json)")
        if not fn: return
        try:
            out = []
            for t in self._templates:
                c = t["contour"].astype(int).tolist()
                out.append({"name": t["name"], "area": t["area"], "perim": t["perim"], "contour": c})
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            self._append_log(f"Saved {len(out)} templates to {fn}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))

    def _load_library(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Template Library", ".", "JSON (*.json)")
        if not fn: return
        try:
            with open(fn, "r", encoding="utf-8") as f:
                arr = json.load(f)
            self._templates = []
            for t in arr:
                c = np.array(t["contour"], dtype=np.int32)
                c_f = c.astype(np.float32)
                M = cv2.moments(c_f)
                if M["m00"] > 1e-6:
                    cx = M["m10"] / M["m00"]; cy = M["m01"] / M["m00"]
                else:
                    xs = c_f[:,:,0]; ys = c_f[:,:,1]
                    cx = float(xs.mean()); cy = float(ys.mean())
                cm = c_f.copy()
                cm[:,:,0] -= cx; cm[:,:,1] -= cy
                cm[:,:,0] *= -1.0
                cm[:,:,0] += cx; cm[:,:,1] += cy
                c_mirror = cm.astype(np.int32)
                self._templates.append(dict(
                    name=t["name"],
                    area=float(t["area"]),
                    perim=float(t["perim"]),
                    contour=c,
                    contour_mirror=c_mirror
                ))
                # self._templates.append(dict(name=t["name"], area=float(t["area"]), perim=float(t["perim"]), contour=c))
            self._refresh_tpl_list(select_last=True)
            self._append_log(f"Loaded {len(self._templates)} templates from {fn}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            
    # ───────────────────────── OCR helpers (inline, sync) ─────────────────────────
    def _ocr__matches(self, text: str, target: str, mode: str, case: bool) -> bool:
        if not target: return False
        if mode not in ("contains","exact","startswith","regex"): mode = "contains"
        a = text if case else text.lower()
        b = target if case else target.lower()
        if mode == "exact": return a == b
        if mode == "startswith": return a.startswith(b)
        if mode == "regex":
            import re
            flags = 0 if case else re.IGNORECASE
            try: return re.search(target, text, flags) is not None
            except re.error: return False
        return b in a  # contains

    def _ocr__preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        # same spirit as your ocr_worker
        import cv2
        h, w = img_bgr.shape[:2]
        if max(h, w) < 1200:
            s = 1200.0 / max(h, w)
            img_bgr = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = clahe.apply(L)
        lab = cv2.merge([L, A, B])
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=40, sigmaSpace=40)
        return img_bgr
    
    @staticmethod
    def _coerce_blocks(pred):
        """
        Normalize PaddleOCR 3.x predict() outputs to plain dict blocks.
        Handles:
        - objects with attribute `.res` (PaddleX Result)
        - dicts like {'res': {...}}
        - plain dicts already containing polys/texts/scores
        Returns: list of dicts
        """
        seq = pred if isinstance(pred, list) else [pred]
        out = []
        for item in seq:
            blk = item
            # 3.x often returns a PaddleX Result object with `.res`
            if hasattr(blk, "res"):
                blk = blk.res
            # sometimes it’s a dict that nests 'res'
            if isinstance(blk, dict) and "res" in blk and isinstance(blk["res"], dict):
                blk = blk["res"]
            out.append(blk)
        return out

    def _ocr__run(self, bgr: np.ndarray, do_pre: bool, min_conf: float) -> dict:
        """
        Returns: {"detections":[{"text","confidence","bbox"}], "infer_ms": float}
        """
        import time, cv2
        if bgr is None:
            return {"detections": [], "infer_ms": 0.0}

        # if OCR is disabled or target is empty → return immediately (extra safety)
        if (not bool(self.chk_use_ocr.isChecked())) or (self.ed_ocr_target.text().strip() == ""):
            return {"detections": [], "infer_ms": 0.0}

        img = bgr.copy()
        if do_pre:
            img = self._ocr__preprocess(img)

        # modest upscaling only (keeps latency down)
        h0, w0 = img.shape[:2]
        scale = 1.0
        LIM = 896.0  # was 1000+, smaller is faster with very small accuracy cost
        if max(h0, w0) < LIM:
            scale = LIM / max(h0, w0)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        engine = getattr(self, "_paddle_engine", None)
        if engine is None:
            engine = _lazy_paddleocr()
            self._paddle_engine = engine

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        try:
            if hasattr(engine, "predict"):          # PaddleOCR 3.x
                result = engine.predict(input=rgb)
            else:                                   # PaddleOCR 2.x
                result = engine.ocr(rgb, cls=True)
        except Exception as e:
            if any(k in str(e).lower() for k in ("primitive","onednn","fused_conv2d")):
                engine = _lazy_paddleocr()
                self._paddle_engine = engine
                result = engine.predict(input=rgb) if hasattr(engine, "predict") else engine.ocr(rgb, cls=True)
            else:
                raise

        infer_ms = (time.time() - t0) * 1000.0

        dets = []
        inv_s = 1.0 / max(1.0, scale)

        def _emit(poly, text, score):
            if float(score) < float(min_conf):
                return
            box = np.array(poly, dtype=np.float32)
            if box.ndim == 3:
                box = box[:, 0, :]
            if scale != 1.0:
                box[:, 0] *= inv_s
                box[:, 1] *= inv_s
            dets.append({"text": str(text), "confidence": float(score), "bbox": box.tolist()})

        if result:
            if isinstance(result, list) and result and isinstance(result[0], list):  # PaddleOCR 2.x
                for page in result:
                    for item in page or []:
                        poly, (txt, sc) = item
                        _emit(poly, txt, sc)
            else:  # PaddleOCR 3.x shapes
                blocks = self._coerce_blocks(result)
                for blk in blocks:
                    if not isinstance(blk, dict):
                        continue
                    if "results" in blk and isinstance(blk["results"], list):
                        for r in blk["results"]:
                            poly = r.get("polygon") or r.get("poly") or r.get("box") or r.get("bbox")
                            if poly is not None:
                                _emit(poly, r.get("text", ""), r.get("score", 0.0))
                        continue
                    polys  = (blk.get("polys") or blk.get("boxes") or blk.get("det_polys") or blk.get("det_boxes") or blk.get("dt_polys"))
                    texts  = (blk.get("rec_texts") or blk.get("texts") or blk.get("rec_text"))
                    scores = (blk.get("rec_scores") or blk.get("scores") or blk.get("rec_score"))
                    if isinstance(polys, list) and isinstance(texts, list):
                        n = min(len(polys), len(texts), len(scores) if isinstance(scores, list) else len(texts))
                        for i in range(n):
                            _emit(polys[i], texts[i], (scores[i] if isinstance(scores, list) else 1.0))

        return {"detections": dets, "infer_ms": round(infer_ms, 1)}

    def _ocr__decide_go_nogo(self, bgr: np.ndarray) -> tuple[str, float, dict]:
        """
        Returns (name, best_conf, ocr_result)
        name ∈ {"go","nogo"}, best_conf in [0,1] (0 if none), raw result dict.
        """
        if not bool(self.chk_use_ocr.isChecked()):
            return ("nogo", 0.0, {"detections":[], "infer_ms":0.0})
        target = self.ed_ocr_target.text().strip()
        mode   = self.cb_ocr_match.currentText()
        case   = bool(self.chk_ocr_case.isChecked())
        min_c  = float(self.ds_ocr_minconf.value())
        pre    = bool(self.chk_ocr_pre.isChecked())

        res = self._ocr__run(bgr, pre, min_c)
        best = 0.0
        for d in res["detections"]:
            if self._ocr__matches(d.get("text",""), target, mode, case):
                best = max(best, float(d.get("confidence",0.0)))
        return ("go" if best > 0.0 else "nogo", best, res)

    # ───────────────────────── contour pipeline ─────────────────────────

    def _run_once(self):
        self._process_and_update(publish=True)

    def _do_trigger(self):
        if not self.rad_trig.isChecked():
            self.rad_trig.setChecked(True)  # emits set_mode
        self._send({"type": "trigger"})
        self._append_log("TRIGGER pressed → OCR-driven GO/NOGO from current frame")
        self._trigger_publish_once()
        
    def _render_ocr_overlay(self, bgr: np.ndarray, ocr_res: dict) -> np.ndarray:
        img = bgr.copy()
        if not isinstance(ocr_res, dict):
            return img
        for det in ocr_res.get("detections", []):
            poly = np.array(det.get("bbox", []), dtype=np.int32).reshape(-1, 2)
            if poly.shape[0] >= 4:
                cv2.polylines(img, [poly], True, (255, 0, 0), 2, cv2.LINE_AA)
                label = f"{det.get('text','')} ({float(det.get('confidence',0.0)):.2f})"
                x, y = int(poly[0,0]), int(poly[0,1])
                cv2.putText(img, label, (x, max(12, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        return img

    def _save_ocr_snapshot(self, bgr: Optional[np.ndarray], ocr_res: dict, tag: str = "run"):
        if bgr is None:
            self._append_log("[OCR] skip save: no frame")
            return
        try:
            vis = self._render_ocr_overlay(bgr, ocr_res)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(self._out_dir, f"ocr_{tag}_{ts}.jpg")
            ok = cv2.imwrite(fn, vis)
            if ok:
                cnt = len(ocr_res.get("detections", [])) if isinstance(ocr_res, dict) else 0
                self._append_log(f"[OCR] saved {fn}  (detections={cnt})")
            else:
                self._append_log(f"[OCR] save failed (cv2.imwrite returned False): {fn}")
        except Exception as e:
            self._append_log(f"[OCR] save error: {e}")

    def _on_udp_trigger(self):
        if not self.rad_trig.isChecked():
            self.rad_trig.setChecked(True)  # emits set_mode
        self._append_log("UDP TRIGGER → OCR-driven GO/NOGO from current frame")
        self._send({"type": "trigger"})
        self._trigger_publish_once()

    def _trigger_publish_once(self):
        if self._last_frame_bgr is not None:
            vis, dets = self._compute_contours(self._last_frame_bgr)
            name, ocr_conf, ocr_res = self._ocr__decide_go_nogo(self._last_frame_bgr)
            
            self._save_ocr_snapshot(self._last_frame_bgr, ocr_res, tag=f"trigger_{name}")

            self._append_log(f"[OCR] detections={len(ocr_res['detections'])} best_match_conf={ocr_conf:.2f}")

            # draw OCR on top of contour visualization and hold 5s
            self._draw_ocr(vis, ocr_res)
            self._set_overlay(vis, ttl_sec=5.0)

            # update table + publish
            self._update_ocr_summary(name, ocr_conf, ocr_res)
            self._populate_table(dets)
            self._publish_vgr_result(dets, forced_name=name, ocr_score=ocr_conf)

            self._last_result_time = time.time()
            self._expect_one_result = False

        else:
            self._append_log("No frame yet → will publish after next frame")
            self._expect_one_result = True


    def _populate_table(self, dets: List[Dict[str, Any]]):
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
            self.tbl.setItem(r, 8, QtWidgets.QTableWidgetItem(
                "-" if d.get("match") is None else f'{d["match"]:.4f}'))
            # NEW: OCR summary per row (same global OCR result shown in this run)
            self.tbl.setItem(r, 9, QtWidgets.QTableWidgetItem(self._last_ocr_summary))
            
    def _process_and_update(self, publish: bool):
        """Fast path in Training: skip OCR (keeps FPS high). Do OCR only in Trigger."""
        if self._last_frame_bgr is None:
            self.status.showMessage("No frame yet", 1500)
            return

        vis, dets = self._compute_contours(self._last_frame_bgr)

        # ── OCR policy: only when Trigger mode to keep Training fast
        if self.rad_trig.isChecked():
            decision, ocr_conf, ocr_res = self._ocr__decide_go_nogo(self._last_frame_bgr)
            self._save_ocr_snapshot(self._last_frame_bgr, ocr_res, tag="training")
            self._draw_ocr(vis, ocr_res)
        else:
            decision, ocr_conf, ocr_res = ("nogo", 0.0, {"detections": [], "infer_ms": 0.0})

        self._set_overlay(vis, ttl_sec=5.0)

        # table + summary
        self._update_ocr_summary(decision, ocr_conf, ocr_res)
        self._populate_table(dets)

        # publish policy
        if self.rad_trig.isChecked() and self._expect_one_result:
            self._publish_vgr_result(dets, forced_name=decision, ocr_score=ocr_conf)
            self._expect_one_result = False
            self._last_result_time = time.time()

        if self.rad_train.isChecked() and publish:
            # still publish geometry for debugging; name/score reflect the (skipped) OCR
            self._publish_vgr_result(dets, forced_name=decision, ocr_score=ocr_conf)

    def _canny_edges(self, gray: np.ndarray) -> np.ndarray:
        # optional CLAHE
        if self.chk_clahe.isChecked():
            clip = float(self.ds_clip.value())
            tiles = int(self.sp_tiles.value())
            clahe = cv2.createCLAHE(clipLimit=max(0.1, clip), tileGridSize=(tiles, tiles))
            gray = clahe.apply(gray)

        # optional blur
        k = int(self.sp_blur.value())
        if k > 0 and k % 2 == 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        lo = int(self.sp_canny_lo.value())
        hi = int(self.sp_canny_hi.value())
        ap = int(self.cmb_canny_ap.currentText())
        l2 = bool(self.chk_l2.isChecked())
        edges = cv2.Canny(gray, lo, hi, apertureSize=ap, L2gradient=l2)

        # optional dilate to thicken edges (stabilize contours)
        it = int(self.sp_dilate.value())
        if it > 0:
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            edges = cv2.dilate(edges, k3, iterations=it)
        return edges
    
    def _bbox_iou(self, a, b):
        ax, ay, aw, ah = a["bbox"]; bx, by, bw, bh = b["bbox"]
        x1 = max(ax, bx); y1 = max(ay, by)
        x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
        iw = max(0, x2-x1); ih = max(0, y2-y1)
        inter = iw * ih
        union = aw*ah + bw*bh - inter + 1e-6
        return inter / union

    def _suppress_overlaps(self, dets, iou_thr=0.85, center_px=8, prefer="match"):
        """
        Non-maximum suppression on detections. 'prefer' is 'match' or 'area'.
        Keeps the best per overlap cluster and drops the rest.
        """
        if not dets:
            return dets
        if prefer == "match":
            # smaller is better; inf for non-matching
            key = lambda d: d.get("match", float("inf"))
        else:
            key = lambda d: -d["area"]

        dets_sorted = sorted(dets, key=key)
        kept = []
        for d in dets_sorted:
            cx, cy = d["center"]
            drop = False
            for k in kept:
                kcx, kcy = k["center"]
                if (cx-kcx)**2 + (cy-kcy)**2 <= center_px**2:
                    drop = True; break
                if self._bbox_iou(d, k) >= iou_thr:
                    drop = True; break
            if not drop:
                kept.append(d)
        return kept
    def _draw_ocr(self, vis: np.ndarray, ocr_res: dict):
        """Overlay OCR polygons + text onto vis."""
        if not isinstance(ocr_res, dict): 
            return
        for det in ocr_res.get("detections", []):
            poly = np.array(det.get("bbox", []), dtype=np.int32).reshape(-1, 2)
            if poly.shape[0] >= 4:
                cv2.polylines(vis, [poly], True, (255, 0, 0), 2, cv2.LINE_AA)
                label = f"{det.get('text','')} ({float(det.get('confidence',0.0)):.2f})"
                x, y = int(poly[0,0]), int(poly[0,1])
                cv2.putText(vis, label, (x, max(12, y - 6)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    def _set_overlay(self, bgr: np.ndarray, ttl_sec: float = 5.0, jpeg_q: int = 80):
        """Show bgr on screen and hold it as overlay for ttl_sec seconds."""
        try:
            jpeg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])[1].tobytes()
            qi = QtGui.QImage.fromData(jpeg, "JPG")
            self._last_overlay = qi
            self._overlay_until = time.time() + float(ttl_sec)
            self._set_pixmap(qi)
        except Exception:
            pass

    def _compute_contours(self, bgr: np.ndarray, offset: Tuple[int,int]=(0,0)) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Return (overlay_bgr, detections list).
        Each det: {"area":float,"perim":float,"bbox":(x,y,w,h),"solidity":float,"contour":cnt_np_int,"match":float|None}
        If 'offset' is given, bbox/contours are shifted by (ox,oy) (used for ROI teaching).
        Robust hole detection via RETR_CCOMP + border cleanup.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = self._canny_edges(gray)

        # --- kill a 1px frame to prevent the “giant border contour” swallowing everything
        eh, ew = edges.shape[:2]
        if eh > 2 and ew > 2:
            edges[0, :] = 0
            edges[eh-1, :] = 0
            edges[:, 0] = 0
            edges[:, ew-1] = 0

        # --- use CCOMP so we get outer ↔ holes directly
        cnts, hier = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or len(cnts) == 0:
            return (bgr if offset == (0,0) else self._last_frame_bgr), []

        hier = hier[0]  # (N, 4): [next, prev, first_child, parent]
        outer_ids = [i for i in range(len(cnts)) if hier[i][3] < 0]

        ox, oy = offset
        dets: List[Dict[str, Any]] = []
        REJECT_BORDER_TOUCHING = True  # set False if parts genuinely touch image edges

        for i in outer_ids:
            c = cnts[i]

            # gather direct children (holes) by walking siblings
            hole_ids = []
            ch = hier[i][2]  # first_child
            while ch != -1:
                hole_ids.append(ch)
                ch = hier[ch][0]  # next sibling
            holes = [cnts[k] for k in hole_ids]

            # geometry with hole-aware area / perimeter
            outer_area = float(cv2.contourArea(c))
            holes_area = float(sum(cv2.contourArea(hh) for hh in holes))
            area = max(0.0, outer_area - holes_area)

            perim_outer = float(cv2.arcLength(c, True))
            perim_holes = float(sum(cv2.arcLength(hh, True) for hh in holes))
            perim_total = perim_outer + perim_holes
            
            # Oriented bounding box (rotated)
            rect = cv2.minAreaRect(c)
            box  = cv2.boxPoints(rect).astype(np.float32)
            ((cx, cy), (w, h), angle) = rect

            # Apply ROI → full-frame offset (so teaching overlays line up)
            if ox != 0 or oy != 0:
                c       = (c.astype(np.float32) + np.array([[[ox, oy]]], dtype=np.float32)).astype(np.int32)
                holes   = [(hh.astype(np.float32) + np.array([[[ox, oy]]], dtype=np.float32)).astype(np.int32) for hh in holes]
                box[:, 0] += ox
                box[:, 1] += oy
                cx += ox; cy += oy

            # Compute other properties
            area = float(cv2.contourArea(c))
            perim = float(cv2.arcLength(c, True))
            hull = cv2.convexHull(c)
            solidity = area / max(1.0, float(cv2.contourArea(hull)))

            theta_deg = float(angle)
            if w < h:
                theta_deg = theta_deg + 90.0

            # Store detection
            dets.append({
                "area": area,
                "perim": perim,
                "bbox": cv2.boundingRect(box.astype(np.int32)),   # integer axis box from rotated corners (with offset)
                "quad": box.astype(np.int32).tolist(),            # oriented corners (with offset)
                "center": [float(cx), float(cy)],
                "theta_deg": theta_deg,
                "solidity": solidity,
                "contour": c,                                     # ← contour already offset
                "holes": holes,                                   # ← holes already offset
            })

                        


        # --- compute match scores (lower is better) before suppression ---
        tpl_selected = (0 <= self._active_index < len(self._templates))
        if tpl_selected and len(dets) > 0:
            tpl = self._templates[self._active_index]
            area_tol_pct = max(0, int(self.sp_area_tol.value()))
            perim_tol_pct = max(0, int(self.sp_perim_tol.value()))
            method = self.cmb_metric.currentText()
            met = { "I1": cv2.CONTOURS_MATCH_I1, "I2": cv2.CONTOURS_MATCH_I2, "I3": cv2.CONTOURS_MATCH_I3 }.get(method, cv2.CONTOURS_MATCH_I1)

            for d in dets:
                area_ok = True; perim_ok = True
                if tpl["area"] > 1:
                    diffA = abs(d["area"] - tpl["area"]) * 100.0 / tpl["area"]
                    area_ok = diffA <= area_tol_pct
                if tpl["perim"] > 1:
                    diffP = abs(d["perim"] - tpl["perim"]) * 100.0 / tpl["perim"]
                    perim_ok = diffP <= perim_tol_pct
                try:
                    score_n = float(cv2.matchShapes(d["contour"], tpl["contour"], met, 0.0))
                    c_m = tpl.get("contour_mirror")
                    if c_m is not None:
                        score_m = float(cv2.matchShapes(d["contour"], c_m, met, 0.0))
                        score = min(score_n, score_m)
                    else:
                        score = score_n
                except Exception:
                    score = float("inf")
                d["match"] = score if (area_ok and perim_ok) else float("inf")
        else:
            for d in dets:
                d["match"] = None

        # --- non-max suppression (prefer best match if template, else largest area) ---
        dets = self._suppress_overlaps(
            dets,
            iou_thr=0.90,
            center_px=8,
            prefer="match" if tpl_selected else "area"
        )

        # --- filter by match threshold / sort / keepN ---
        if tpl_selected:
            thr = float(self.ds_match_thr.value())
            dets = [d for d in dets if np.isfinite(d["match"]) and d["match"] <= thr]
            dets.sort(key=lambda d: d["match"])  # best first
        else:
            dets.sort(key=lambda d: d["area"], reverse=True)

        keepN = int(self.sp_keepN.value())
        if len(dets) > keepN:
            dets = dets[:keepN]

        # --- draw overlay ---
        vis = bgr.copy() if offset == (0,0) else self._last_frame_bgr.copy()
        if self.chk_draw_cnt.isChecked():
            for d in dets:
                cv2.drawContours(vis, [d["contour"]], -1, (0,255,255), 2)
                for hh in d.get("holes", []):
                    cv2.drawContours(vis, [hh], -1, (0,255,255), 2)
        for i, d in enumerate(dets):
            x,y,w,h = d["bbox"]
            if self.chk_draw_box.isChecked():
                color = (0,255,0)
                if d.get("match") is not None and np.isfinite(d["match"]):
                    color = (0,128,255)
                box = np.int32(d["quad"])
                cv2.polylines(vis, [box], True, color, 2)
                # --- draw orientation label ---
                cx, cy = map(int, d["center"])
                ang = float(d.get("theta_deg", 0.0))
                cv2.putText(vis, f"{ang:.1f}°", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if self.chk_draw_cent.isChecked():
                cx, cy = x + w//2, y + h//2
                cv2.circle(vis, (cx,cy), 4, (255,0,0), -1)
            if self.chk_draw_ids.isChecked():
                label = f"#{i}"
                if d.get("match") is not None and np.isfinite(d["match"]):
                    label += f" m={d['match']:.3f}"
                cv2.putText(vis, label, (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

        # brief persistence for smoother viewing
        try:
            jpeg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()
            qi = QtGui.QImage.fromData(jpeg, "JPG")
            self._last_overlay = qi
            self._overlay_until = time.time() + 1.0
        except Exception:
            pass

        return vis if offset == (0,0) else self._last_frame_bgr, dets
    
    def _publish_vgr_result(self, dets: List[Dict[str, Any]], forced_name: Optional[str] = None, ocr_score: Optional[float] = None):
        """
        Publishes a single object. 'score' is ALWAYS the OCR confidence.
        """
        if not self.chk_pub_robot.isChecked():
            return

        best_det = dets[0] if dets else None

        # If nothing detected, still publish an empty result so downstream is deterministic
        if best_det is None:
            payload = {
                "version": "1.0",
                "sdk": "vision_ui",
                "session": self.session_id,
                "timestamp_ms": int(time.time() * 1000),
                "camera": {"proc_width": int(self.last_frame_wh[0]), "proc_height": int(self.last_frame_wh[1])},
                "result": {"objects": [], "counts": {"objects": 0, "detections": 0}}
            }
            self.pub_vgr.send_json(payload)
            self._append_log("VGR ← empty (no detections)")
            return

        x, y, w, h = best_det["bbox"]
        cx, cy = best_det.get("center", [x + w / 2.0, y + h / 2.0])
        quad = best_det.get("quad")
        theta = float(best_det.get("theta_deg", 0.0))
        size_wh = best_det.get("size_wh", [int(w), int(h)])

        send_name = forced_name if forced_name in ("go", "nogo") else "nogo"

        score = float(ocr_score if (ocr_score is not None) else 0.0)

        _m = best_det.get("match", None)
        if isinstance(_m, (int, float)) and np.isfinite(_m):
            match_score = float(_m)
        else:
            match_score = None

        det_obj = {
            "instance_id": 0,
            "score": score,                 # ← OCR confidence only
            "inliers": int(len(best_det.get("contour", []))),
            "pose": {
                "x": float(cx), "y": float(cy), "theta_deg": float(theta),
                "x_scale": 1.0, "y_scale": 1.0,
                "origin_xy": [float(quad[0][0]), float(quad[0][1])] if quad else [float(x), float(y)]
            },
            "center": [float(cx), float(cy)],
            "quad": quad if quad else None,
            "match": match_score            # ← auxiliary (does not affect downstream decisions)
        }

        obj = {
            "object_id": 1,
            "name": send_name,
            "template_size": [int(size_wh[0]), int(size_wh[1])],
            "detections": [det_obj]
        }

        payload = {
            "version": "1.0",
            "sdk": "vision_ui",
            "session": self.session_id,
            "timestamp_ms": int(time.time() * 1000),
            "camera": {"proc_width": int(self.last_frame_wh[0]), "proc_height": int(self.last_frame_wh[1])},
            "result": {"objects": [obj], "counts": {"objects": 1, "detections": 1}}
        }

        self.pub_vgr.send_json(payload)
        self._append_log(f"VGR ← {send_name}  ocr_conf={score:.2f} px=({cx:.1f},{cy:.1f}) yaw={theta:.2f}° size={size_wh} match={match_score}")

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

        # Robot
        robot_ip = to_str(S.value('robot/ip'))
        if robot_ip: self.ed_robot_ip.setText(robot_ip)
        rp = to_int(S.value('robot/port'))
        if rp: self.sp_robot_port.setValue(rp)
        ap = to_bool(S.value('robot/auto_publish'))
        if ap is not None: self.chk_pub_robot.setChecked(ap)
        # --- add to _load_settings(...) after Robot settings ---
        vgr_ip = to_str(S.value('vgr/ip'))
        if vgr_ip:
            self.ed_vgr_ip.setText(vgr_ip)
        vp = to_int(S.value('vgr/port'))
        if vp:
            self.sp_vgr_port.setValue(vp)

        # Canny / Preproc defaults
        try:
            self.chk_clahe.setChecked(bool(to_bool(S.value('canny/clahe')) or False))
            v = to_float(S.value('canny/clip')); self.ds_clip.setValue(v if v is not None else 2.0)
            i = to_int(S.value('canny/tiles')); self.sp_tiles.setValue(i if i is not None else 8)
            i = to_int(S.value('canny/blur')); self.sp_blur.setValue(i if i is not None else 5)
            i = to_int(S.value('canny/lo')); self.sp_canny_lo.setValue(i if i is not None else 50)
            i = to_int(S.value('canny/hi')); self.sp_canny_hi.setValue(i if i is not None else 150)
            s = to_str(S.value('canny/ap')); 
            if s and self.cmb_canny_ap.findText(s) >= 0: self.cmb_canny_ap.setCurrentText(s)
            self.chk_l2.setChecked(bool(to_bool(S.value('canny/l2')) or True))
            i = to_int(S.value('canny/dilate')); self.sp_dilate.setValue(i if i is not None else 1)
        except Exception:
            pass

        # Filters
        try:
            i = to_int(S.value('filter/min_area')); self.sp_min_area.setValue(i if i is not None else 200)
            i = to_int(S.value('filter/max_area')); self.sp_max_area.setValue(i if i is not None else 0)
            f = to_float(S.value('filter/aspect_min')); self.ds_aspect_min.setValue(f if f is not None else 0.0)
            f = to_float(S.value('filter/aspect_max')); self.ds_aspect_max.setValue(f if f is not None else 100.0)
            f = to_float(S.value('filter/solidity')); self.ds_solidity_min.setValue(f if f is not None else 0.0)
            f = to_float(S.value('filter/perim_min')); self.ds_perim_min.setValue(f if f is not None else 0.0)
            i = to_int(S.value('filter/keepN')); self.sp_keepN.setValue(i if i is not None else 50)
        except Exception:
            pass

        # Matching
        try:
            s = to_str(S.value('match/metric')); 
            if s and self.cmb_metric.findText(s) >= 0: self.cmb_metric.setCurrentText(s)
            f = to_float(S.value('match/thr')); self.ds_match_thr.setValue(f if f is not None else 0.10)
            i = to_int(S.value('match/area_tol')); self.sp_area_tol.setValue(i if i is not None else 25)
            i = to_int(S.value('match/perim_tol')); self.sp_perim_tol.setValue(i if i is not None else 20)
            b = to_bool(S.value('match/require_tpl')); self.chk_require_active.setChecked(b if b is not None else False)
        except Exception:
            pass
        
        # OCR
        try:
            s = to_str(S.value('ocr/target'))
            if s is not None:
                self.ed_ocr_target.setText(s)

            s = to_str(S.value('ocr/match'))
            if s and self.cb_ocr_match.findText(s) >= 0:
                self.cb_ocr_match.setCurrentText(s)

            b = to_bool(S.value('ocr/case'))
            if b is not None:
                self.chk_ocr_case.setChecked(b)

            f = to_float(S.value('ocr/minconf'))
            if f is not None:
                self.ds_ocr_minconf.setValue(f)

            b = to_bool(S.value('ocr/pre'))
            if b is not None:
                self.chk_ocr_pre.setChecked(b)

            b = to_bool(S.value('ocr/use'))
            if b is not None:
                self.chk_use_ocr.setChecked(b)
        except Exception:
            pass

        # Template library (paths not persisted; contours can be huge → user saves to JSON)
        # Logs
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

        # Canny / preproc
        S.setValue('canny/clahe', bool(self.chk_clahe.isChecked()))
        S.setValue('canny/clip', float(self.ds_clip.value()))
        S.setValue('canny/tiles', int(self.sp_tiles.value()))
        S.setValue('canny/blur', int(self.sp_blur.value()))
        S.setValue('canny/lo', int(self.sp_canny_lo.value()))
        S.setValue('canny/hi', int(self.sp_canny_hi.value()))
        S.setValue('canny/ap', self.cmb_canny_ap.currentText())
        S.setValue('canny/l2', bool(self.chk_l2.isChecked()))
        S.setValue('canny/dilate', int(self.sp_dilate.value()))

        # Filters
        S.setValue('filter/min_area', int(self.sp_min_area.value()))
        S.setValue('filter/max_area', int(self.sp_max_area.value()))
        S.setValue('filter/aspect_min', float(self.ds_aspect_min.value()))
        S.setValue('filter/aspect_max', float(self.ds_aspect_max.value()))
        S.setValue('filter/solidity', float(self.ds_solidity_min.value()))
        S.setValue('filter/perim_min', float(self.ds_perim_min.value()))
        S.setValue('filter/keepN', int(self.sp_keepN.value()))

        # Matching
        S.setValue('match/metric', self.cmb_metric.currentText())
        S.setValue('match/thr', float(self.ds_match_thr.value()))
        S.setValue('match/area_tol', int(self.sp_area_tol.value()))
        S.setValue('match/perim_tol', int(self.sp_perim_tol.value()))
        S.setValue('match/require_tpl', bool(self.chk_require_active.isChecked()))

        # Camera dock
        S.setValue('camera/state', json.dumps(getattr(self.cam_panel, "get_state", lambda: {})()))
        
        # OCR
        S.setValue('ocr/target', self.ed_ocr_target.text())
        S.setValue('ocr/match', self.cb_ocr_match.currentText())
        S.setValue('ocr/case', bool(self.chk_ocr_case.isChecked()))
        S.setValue('ocr/minconf', float(self.ds_ocr_minconf.value()))
        S.setValue('ocr/pre', bool(self.chk_ocr_pre.isChecked()))
        S.setValue('ocr/use', bool(self.chk_use_ocr.isChecked()))


        # Robot + logs
        S.setValue('robot/ip', self.ed_robot_ip.text())
        S.setValue('robot/port', int(self.sp_robot_port.value()))
        S.setValue('robot/auto_publish', bool(self.chk_pub_robot.isChecked()))
        S.setValue('logs/history', json.dumps(self._log_buffer[-200:]))
        # --- add to _save_settings(...) after Robot settings ---
        S.setValue('vgr/ip', self.ed_vgr_ip.text())
        S.setValue('vgr/port', int(self.sp_vgr_port.value()))

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
