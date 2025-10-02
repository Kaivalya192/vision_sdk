#!/usr/bin/env python3
"""
RPi Vision Client – Color Sorting UI
Mirrors the detection-only UI structure but drives color rule-based backend.
"""

import sys, time, base64, json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
import socket

from PyQt5 import QtCore, QtGui, QtWidgets, QtWebSockets

import threading
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
            parsed = "text"
            if not fire:
                try:
                    j = json.loads(msg)
                    parsed = "json"
                    fire = isinstance(j, dict) and str(j.get("cmd","")).lower() == "trigger"
                except Exception:
                    parsed = "other"

            self._log(f"[UDP] parsed={parsed} fire={fire}")
            if fire:
                self._log("[UDP] >>> TRIGGER RECEIVED <<<")
                try:
                    self._sock.sendto(b'{"status":"armed"}', addr)  # optional ACK
                except Exception:
                    pass
                self.bridge.triggerReceived.emit()

# ------------------------ Video Label (keeps aspect + ROI + pixel click) ------------------------

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

class VideoLabel(QtWidgets.QLabel):
    roiSelected     = QtCore.pyqtSignal(QtCore.QRect)   # rectangle selection (display coords)
    polygonSelected = QtCore.pyqtSignal(object)         # list[QPoint] (display coords)
    pixelClicked    = QtCore.pyqtSignal(int, int)       # display coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._last_draw_rect = QtCore.QRect()  # where the video is drawn inside the label
        self._frame_wh: Tuple[int, int] = (0, 0)

        # Rect ROI
        self._rect_mode = False
        self._origin = QtCore.QPoint()
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

        # Polygon ROI
        self._poly_mode = False
        self._poly_pts: List[QtCore.QPoint] = []

        # Pick (HSV) mode
        self._pick_mode = False

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

        # draw polygon preview if capturing
        if self._poly_mode and self._poly_pts:
            pen = QtGui.QPen(QtCore.Qt.yellow)
            pen.setWidth(2)
            p.setPen(pen)
            for i in range(1, len(self._poly_pts)):
                p.drawLine(self._poly_pts[i - 1], self._poly_pts[i])
            for q in self._poly_pts:
                p.drawEllipse(q, 3, 3)

        p.end()
        super().setPixmap(canvas)

    def enable_rect_selection(self, ok: bool):
        self._rect_mode = ok
        self._poly_mode = False
        self._poly_pts.clear()
        if not ok:
            self._rubber.hide()

    def enable_polygon_selection(self, ok: bool):
        self._poly_mode = ok
        self._rect_mode = False
        self._poly_pts.clear()
        if not ok:
            self.update()

    def enable_pick(self, ok: bool):
        self._pick_mode = ok
        print(f"[DEBUG] Pick mode set to {ok}", flush=True)
        if ok:
            # disable ROI modes when picking
            self._rect_mode = False
            self._poly_mode = False
            self._poly_pts.clear()
            self._rubber.hide()
            self.update()

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self._pick_mode and ev.button() == QtCore.Qt.LeftButton:
            if self._last_draw_rect.isNull() or not self._last_draw_rect.contains(ev.pos()):
                print("[DEBUG] Click outside video area", flush=True)
            else:
                self.pixelClicked.emit(ev.x(), ev.y())
                print(f"[DEBUG] Pixel clicked at {ev.x()},{ev.y()}", flush=True)

        if self._rect_mode and ev.button() == QtCore.Qt.LeftButton:
            self._origin = ev.pos()
            self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubber.show()
        elif self._poly_mode:
            if ev.button() == QtCore.Qt.LeftButton and self._last_draw_rect.contains(ev.pos()):
                self._poly_pts.append(ev.pos())
                self.update()
            elif ev.button() == QtCore.Qt.RightButton:
                self._finish_poly()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._rect_mode and not self._origin.isNull():
            self._rubber.setGeometry(QtCore.QRect(self._origin, ev.pos()).normalized())
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self._rect_mode and ev.button() == QtCore.Qt.LeftButton:
            r = self._rubber.geometry()
            self._rubber.hide()
            self._rect_mode = False
            if r.width() > 5 and r.height() > 5:
                self.roiSelected.emit(r)
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if self._poly_mode:
            self._finish_poly()
        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        if self._poly_mode and ev.key() == QtCore.Qt.Key_Escape:
            self.enable_polygon_selection(False)
        super().keyPressEvent(ev)

    def _finish_poly(self):
        if len(self._poly_pts) >= 3:
            self.polygonSelected.emit(list(self._poly_pts))
        self._poly_pts.clear()
        self._poly_mode = False
        self.update()


# ------------------------ Camera Panel (same params JSON as detection UI) ------------------------

class CameraPanel(QtWidgets.QWidget):
    paramsChanged      = QtCore.pyqtSignal(dict)
    viewChanged        = QtCore.pyqtSignal(dict)
    afTriggerRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._debounce = QtCore.QTimer(self, interval=150, singleShot=True, timeout=self._emit_params)

        grid = QtWidgets.QGridLayout(self)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        r = 0
        # AE / exposure / gain
        self.chk_ae = QtWidgets.QCheckBox("Auto Exposure"); grid.addWidget(self.chk_ae, r, 0, 1, 2); r += 1
        self.sp_exp = QtWidgets.QSpinBox(); self.sp_exp.setRange(100, 33000); self.sp_exp.setValue(6000); self.sp_exp.setSingleStep(100); self.sp_exp.setSuffix(" µs")
        grid.addWidget(QtWidgets.QLabel("Exposure"), r, 0); grid.addWidget(self.sp_exp, r, 1); r += 1

        self.dsb_gain = QtWidgets.QDoubleSpinBox(); self.dsb_gain.setRange(1.0, 16.0); self.dsb_gain.setValue(2.0); self.dsb_gain.setDecimals(2); self.dsb_gain.setSingleStep(0.05)
        grid.addWidget(QtWidgets.QLabel("Analogue Gain"), r, 0); grid.addWidget(self.dsb_gain, r, 1); r += 1

        self.dsb_fps = QtWidgets.QDoubleSpinBox(); self.dsb_fps.setRange(1.0, 120.0); self.dsb_fps.setValue(30.0); self.dsb_fps.setDecimals(1)
        grid.addWidget(QtWidgets.QLabel("Framerate"), r, 0); grid.addWidget(self.dsb_fps, r, 1); r += 1

        # AWB
        self.cmb_awb = QtWidgets.QComboBox(); self.cmb_awb.addItems(["auto", "tungsten", "fluorescent", "indoor", "daylight", "cloudy", "manual"])
        grid.addWidget(QtWidgets.QLabel("AWB Mode"), r, 0); grid.addWidget(self.cmb_awb, r, 1); r += 1
        self.dsb_awb_r = QtWidgets.QDoubleSpinBox(); self.dsb_awb_r.setRange(0.1, 8.0); self.dsb_awb_r.setValue(2.0); self.dsb_awb_r.setSingleStep(0.05)
        self.dsb_awb_b = QtWidgets.QDoubleSpinBox(); self.dsb_awb_b.setRange(0.1, 8.0); self.dsb_awb_b.setValue(2.0); self.dsb_awb_b.setSingleStep(0.05)
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(QtWidgets.QLabel("Gains R/B")); hb.addWidget(self.dsb_awb_r); hb.addWidget(self.dsb_awb_b)
        grid.addWidget(QtWidgets.QLabel(""), r, 0); grid.addLayout(hb, r, 1); r += 1

        # Focus
        self.cmb_af = QtWidgets.QComboBox(); self.cmb_af.addItems(["auto", "continuous", "manual"])
        grid.addWidget(QtWidgets.QLabel("AF Mode"), r, 0); grid.addWidget(self.cmb_af, r, 1); r += 1
        self.dsb_dioptre = QtWidgets.QDoubleSpinBox(); self.dsb_dioptre.setRange(0.0, 10.0); self.dsb_dioptre.setDecimals(2); self.dsb_dioptre.setSingleStep(0.05); self.dsb_dioptre.setValue(0.0)
        self.btn_af_trig = QtWidgets.QPushButton("AF Trigger")
        hf = QtWidgets.QHBoxLayout(); hf.addWidget(self.dsb_dioptre); hf.addWidget(self.btn_af_trig)
        grid.addWidget(QtWidgets.QLabel("Lens (dpt)"), r, 0); grid.addLayout(hf, r, 1); r += 1

        # Tuning
        self.dsb_bri = QtWidgets.QDoubleSpinBox(); self.dsb_bri.setRange(-1.0, 1.0); self.dsb_bri.setValue(0.0)
        self.dsb_con = QtWidgets.QDoubleSpinBox(); self.dsb_con.setRange(0.0, 2.0); self.dsb_con.setValue(1.0)
        self.dsb_sat = QtWidgets.QDoubleSpinBox(); self.dsb_sat.setRange(0.0, 2.0); self.dsb_sat.setValue(1.0)
        self.dsb_sha = QtWidgets.QDoubleSpinBox(); self.dsb_sha.setRange(0.0, 2.0); self.dsb_sha.setValue(1.0)
        self.cmb_den = QtWidgets.QComboBox(); self.cmb_den.addItems(["off", "fast", "high_quality"])
        for lab, widget in [("Brightness", self.dsb_bri), ("Contrast", self.dsb_con), ("Saturation", self.dsb_sat),
                            ("Sharpness", self.dsb_sha), ("Denoise", self.cmb_den)]:
            grid.addWidget(QtWidgets.QLabel(lab), r, 0); grid.addWidget(widget, r, 1); r += 1

        # View
        self.chk_flip_h = QtWidgets.QCheckBox("Flip H"); self.chk_flip_v = QtWidgets.QCheckBox("Flip V"); self.btn_rot = QtWidgets.QPushButton("Rotate 90°")
        hv = QtWidgets.QHBoxLayout(); hv.addWidget(self.chk_flip_h); hv.addWidget(self.chk_flip_v); hv.addWidget(self.btn_rot); hv.addStretch(1)
        grid.addWidget(QtWidgets.QLabel("View"), r, 0); grid.addLayout(hv, r, 1); r += 1

        self.btn_reset = QtWidgets.QPushButton("Reset tuning"); grid.addWidget(self.btn_reset, r, 0, 1, 2); r += 1

        # wire
        for w in [self.chk_ae, self.sp_exp, self.dsb_gain, self.dsb_fps, self.cmb_awb, self.dsb_awb_r, self.dsb_awb_b,
                  self.cmb_af, self.dsb_dioptre, self.dsb_bri, self.dsb_con, self.dsb_sat, self.dsb_sha, self.cmb_den]:
            if hasattr(w, 'valueChanged'): w.valueChanged.connect(lambda *_: self._debounce.start())
            if hasattr(w, 'currentTextChanged'): w.currentTextChanged.connect(lambda *_: self._debounce.start())
            if hasattr(w, 'toggled'): w.toggled.connect(lambda *_: self._debounce.start())
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

    def _on_rot(self):
        self._rot_q = (self._rot_q + 1) % 4
        self._emit_view()

    def _emit_view(self):
        self.viewChanged.emit(dict(flip_h=self.chk_flip_h.isChecked(),
                                   flip_v=self.chk_flip_v.isChecked(),
                                   rot_quadrant=self._rot_q))

    def _awb_mode_changed(self, mode):
        manual = (mode == "manual")
        self.dsb_awb_r.setEnabled(manual)
        self.dsb_awb_b.setEnabled(manual)
        self._debounce.start()

    def _reset(self):
        self.blockSignals(True)
        self.chk_ae.setChecked(False); self.sp_exp.setValue(6000); self.dsb_gain.setValue(2.0)
        self.dsb_fps.setValue(30.0); self.cmb_awb.setCurrentText("auto")
        self.dsb_awb_r.setValue(2.0); self.dsb_awb_b.setValue(2.0)
        self.cmb_af.setCurrentText("manual"); self.dsb_dioptre.setValue(0.0)
        for w, val in [(self.dsb_bri, 0.0), (self.dsb_con, 1.0), (self.dsb_sat, 1.0), (self.dsb_sha, 1.0)]:
            w.setValue(val)
        self.cmb_den.setCurrentText("fast"); self.chk_flip_h.setChecked(False); self.chk_flip_v.setChecked(False); self._rot_q = 0
        self.blockSignals(False)
        self._emit_params()
        self._emit_view()

    def get_state(self) -> Dict[str, Any]:
        return dict(
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
            flip_h=self.chk_flip_h.isChecked(),
            flip_v=self.chk_flip_v.isChecked(),
            rot_quadrant=int(self._rot_q),
        )

    def apply_state(self, state: Dict[str, Any]):
        if not isinstance(state, dict):
            return

        def as_bool(val):
            if isinstance(val, bool): return val
            if isinstance(val, str): return val.lower() in ("1", "true", "yes", "on")
            if isinstance(val, (int, float)): return bool(val)
            return None

        def as_int(val):
            try: return int(val)
            except (TypeError, ValueError): return None

        def as_float(val):
            try: return float(val)
            except (TypeError, ValueError): return None

        def set_combo_text(combo: QtWidgets.QComboBox, text):
            if isinstance(text, str):
                idx = combo.findText(text)
                if idx >= 0: combo.setCurrentIndex(idx)

        widgets = [self.chk_ae, self.sp_exp, self.dsb_gain, self.dsb_fps, self.cmb_awb, self.dsb_awb_r, self.dsb_awb_b,
                   self.cmb_af, self.dsb_dioptre, self.dsb_bri, self.dsb_con, self.dsb_sat, self.dsb_sha,
                   self.cmb_den, self.chk_flip_h, self.chk_flip_v]
        blocked = []
        for w in widgets:
            try:
                blocked.append((w, w.blockSignals(True)))
            except AttributeError:
                pass
        try:
            val = as_bool(state.get("auto_exposure"));   self.chk_ae.setChecked(val) if val is not None else None
            val = as_int(state.get("exposure_us"));      self.sp_exp.setValue(val) if val is not None else None
            val = as_float(state.get("gain"));           self.dsb_gain.setValue(val) if val is not None else None
            val = as_float(state.get("fps"));            self.dsb_fps.setValue(val) if val is not None else None
            set_combo_text(self.cmb_awb, state.get("awb_mode"))
            rb = state.get("awb_rb")
            if isinstance(rb, (list, tuple)) and len(rb) >= 2:
                val = as_float(rb[0]); self.dsb_awb_r.setValue(val) if val is not None else None
                val = as_float(rb[1]); self.dsb_awb_b.setValue(val) if val is not None else None
            set_combo_text(self.cmb_af, state.get("focus_mode"))
            val = as_float(state.get("dioptre"));        self.dsb_dioptre.setValue(val) if val is not None else None
            val = as_float(state.get("brightness"));     self.dsb_bri.setValue(val) if val is not None else None
            val = as_float(state.get("contrast"));       self.dsb_con.setValue(val) if val is not None else None
            val = as_float(state.get("saturation"));     self.dsb_sat.setValue(val) if val is not None else None
            val = as_float(state.get("sharpness"));      self.dsb_sha.setValue(val) if val is not None else None
            set_combo_text(self.cmb_den, state.get("denoise"))
            val = as_bool(state.get("flip_h"));          self.chk_flip_h.setChecked(val) if val is not None else None
            val = as_bool(state.get("flip_v"));          self.chk_flip_v.setChecked(val) if val is not None else None
            rot = as_int(state.get("rot_quadrant"));     self._rot_q = max(0, min(3, rot)) if rot is not None else self._rot_q
        finally:
            for obj, prev in blocked:
                obj.blockSignals(prev)

        manual = self.cmb_awb.currentText() == "manual"
        self.dsb_awb_r.setEnabled(manual)
        self.dsb_awb_b.setEnabled(manual)


# ------------------------ Main Window (Color Sorting) ------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPi Vision Client – Color Sorting")
        self.resize(1500, 980)

        # websocket
        self.ws = QtWebSockets.QWebSocket()
        self.ws.textMessageReceived.connect(self._ws_txt)
        self.ws.connected.connect(self._ws_ok)
        self.ws.disconnected.connect(self._ws_closed)

        # state
        self.session_id = "-"
        self.last_frame_wh = (0, 0)
        self._fps_acc = 0.0
        self._fps_n = 0
        self._last_ts = time.perf_counter()
        self._settings = QtCore.QSettings('vision_sdk', 'ui_color')
        self._log_buffer: List[str] = []

        # color rules state
        self._classes: List[Dict] = []  # [{'name', 'h_min','h_max','s_min','s_max','v_min','v_max','min_area',...}]
        self._kernel = 5
        self._min_area_global = 100
        self._open_iter = 1
        self._close_iter = 1
        self._pick_enabled = False
        self._last_overlay: Optional[QtGui.QImage] = None
        self._overlay_until = 0.0
        self._expect_one_result = False
        self._last_result_time = 0.0


        # central: left controls + video
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # LEFT controls inside a scroll area
        left_panel = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_panel)
        left_v.setSpacing(10)
        
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

        # ---- ADD: UDP publisher instance + live target binding
        self.pub_robot = RobotUDPPublisher(self.ed_robot_ip.text(), int(self.sp_robot_port.value()))
        self.ed_robot_ip.textChanged.connect(lambda *_: self.pub_robot.set_target(self.ed_robot_ip.text(), self.sp_robot_port.value()))
        self.sp_robot_port.valueChanged.connect(lambda *_: self.pub_robot.set_target(self.ed_robot_ip.text(), self.sp_robot_port.value()))
        
        mode_box = QtWidgets.QGroupBox("Mode")
        mv = QtWidgets.QVBoxLayout(mode_box)
        self.rad_train = QtWidgets.QRadioButton("Training")
        self.rad_trig = QtWidgets.QRadioButton("Trigger")
        self.rad_train.setChecked(True)
        self.btn_trig = QtWidgets.QPushButton("TRIGGER")
        mv.addWidget(self.rad_train)
        mv.addWidget(self.rad_trig)
        mv.addWidget(self.btn_trig)
        left_v.addWidget(mode_box)


        # Processing (reuse proc width + cadence to keep parity with detection)
        proc_box = QtWidgets.QGroupBox("Processing")
        pf = QtWidgets.QFormLayout(proc_box)
        self.cmb_w = QtWidgets.QComboBox()
        [self.cmb_w.addItem(str(w)) for w in [320, 480, 640, 800, 960, 1280]]
        self.cmb_w.setCurrentText("640")
        self.sp_every = QtWidgets.QSpinBox(); self.sp_every.setRange(1, 5); self.sp_every.setValue(1)
        pf.addRow("Proc width", self.cmb_w)
        pf.addRow("Detect every Nth", self.sp_every)
        left_v.addWidget(proc_box)

        # Color rules editor
        color_box = QtWidgets.QGroupBox("Color Rules / HSV")
        grid = QtWidgets.QGridLayout(color_box)

        # HSV sliders + spins
        self._sl = {}
        def _add_row(r, label, lo, hi, init):
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal); s.setRange(lo, hi); s.setValue(init)
            v = QtWidgets.QSpinBox(); v.setRange(lo, hi); v.setValue(init)
            s.valueChanged.connect(v.setValue); v.valueChanged.connect(s.setValue)
            grid.addWidget(s, r, 1); grid.addWidget(v, r, 2)
            self._sl[label] = (s, v)

        _add_row(0, "h_min", 0, 179, 0)
        _add_row(1, "h_max", 0, 179, 179)
        _add_row(2, "s_min", 0, 255, 0)
        _add_row(3, "s_max", 0, 255, 255)
        _add_row(4, "v_min", 0, 255, 0)
        _add_row(5, "v_max", 0, 255, 255)

        grid.addWidget(QtWidgets.QLabel("kernel_size"), 6, 0)
        self.sp_kernel = QtWidgets.QSpinBox(); self.sp_kernel.setRange(1, 31); self.sp_kernel.setValue(5)
        grid.addWidget(self.sp_kernel, 6, 1)

        grid.addWidget(QtWidgets.QLabel("min_area"), 7, 0)
        self.sp_min_area = QtWidgets.QSpinBox(); self.sp_min_area.setRange(1, 1_000_000); self.sp_min_area.setValue(100)
        grid.addWidget(self.sp_min_area, 7, 1)

        grid.addWidget(QtWidgets.QLabel("open iters"), 8, 0)
        self.sp_open = QtWidgets.QSpinBox(); self.sp_open.setRange(0, 8); self.sp_open.setValue(1)
        grid.addWidget(self.sp_open, 8, 1)

        grid.addWidget(QtWidgets.QLabel("close iters"), 9, 0)
        self.sp_close = QtWidgets.QSpinBox(); self.sp_close.setRange(0, 8); self.sp_close.setValue(1)
        grid.addWidget(self.sp_close, 9, 1)

        # pick / class ops
        self.btn_pick = QtWidgets.QPushButton("Pick HSV")
        self.btn_pick.setCheckable(True)
        self.btn_add = QtWidgets.QPushButton("Add Class (using sliders)")
        self.btn_remove = QtWidgets.QPushButton("Remove Selected Class")
        self.btn_send = QtWidgets.QPushButton("Send Rules to Server")
        self.chk_stream = QtWidgets.QCheckBox("Stream results")
        grid.addWidget(self.btn_pick, 10, 0, 1, 3)
        grid.addWidget(self.btn_add, 11, 0, 1, 3)
        grid.addWidget(self.btn_remove, 12, 0, 1, 3)
        grid.addWidget(self.btn_send, 13, 0, 1, 3)
        grid.addWidget(self.chk_stream, 14, 0, 1, 3)

        # class list & presets
        self.lst_classes = QtWidgets.QListWidget()
        grid.addWidget(QtWidgets.QLabel("Classes"), 0, 3)
        grid.addWidget(self.lst_classes, 1, 3, 8, 1)
        self.btn_load = QtWidgets.QPushButton("Load Presets…")
        self.btn_save = QtWidgets.QPushButton("Save Presets…")
        grid.addWidget(self.btn_load, 9, 3)
        grid.addWidget(self.btn_save, 10, 3)

        left_v.addWidget(color_box, 1)

        left_v.addStretch(1)

        self.controlsPanel = QtWidgets.QScrollArea()
        self.controlsPanel.setWidgetResizable(True)
        self.controlsPanel.setWidget(left_panel)
        root.addWidget(self.controlsPanel, 0)

        # Center video
        self.video = VideoLabel()
        self.video.setMinimumSize(1024, 576)
        self.video.setStyleSheet("background:#111;")
        root.addWidget(self.video, 1)
        
        self._last_frame_bgr: Optional[np.ndarray] = None

        # Right: results + logs
        right = QtWidgets.QVBoxLayout()
        grp = QtWidgets.QGroupBox("Color Detections (last packet)")
        vv = QtWidgets.QVBoxLayout(grp)
        self.tbl = QtWidgets.QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["class","area","cx","cy","circularity"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        vv.addWidget(self.tbl)
        right.addWidget(grp, 1)

        self.txt_log = QtWidgets.QTextEdit(); self.txt_log.setReadOnly(True)
        right.addWidget(QtWidgets.QLabel("Log"))
        right.addWidget(self.txt_log, 1)
        rightw = QtWidgets.QWidget(); rlay = QtWidgets.QVBoxLayout(rightw); rlay.addLayout(right)
        root.addWidget(rightw, 0)

        # Camera control dock
        self.cam_panel = CameraPanel()
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
        self.cam_panel.afTriggerRequested.connect(lambda: self._send({"type": "af_trigger"}))

        # color ops
        self.btn_pick.toggled.connect(self._toggle_pick)
        self.video.pixelClicked.connect(self._on_pixel_clicked)
        self.btn_add.clicked.connect(self._add_class_from_sliders)
        self.btn_remove.clicked.connect(self._remove_selected_class)
        self.btn_send.clicked.connect(self._send_rules)
        self.chk_stream.toggled.connect(lambda on: self._send({"type":"color_set_stream","enabled": bool(on)}))
        self.btn_load.clicked.connect(self._load_presets)
        self.btn_save.clicked.connect(self._save_presets)

        self._ping = QtCore.QTimer(interval=10000, timeout=lambda: self._send({"type": "ping"}))
        self._load_settings()
        # ----- UDP trigger listener -----
        self._udp_bridge = QtTriggerBridge()
        self._udp_bridge.triggerReceived.connect(self._on_udp_trigger)
        self._udp_listener = QtTriggerListener(
            self._udp_bridge,
            bind_ip="0.0.0.0",
            port=40002,               # matches your robot->UI trigger port
            log=self._append_log,
            enable_broadcast=False
        )
        self._udp_listener.start()
        
    def _on_udp_trigger(self):
        # Switch to Trigger mode and fire a single detection (same as pressing TRIGGER button)
        if not self.rad_trig.isChecked():
            self.rad_trig.setChecked(True)  # this also sends set_mode via your existing signal
        self._append_log("UDP trigger → requesting one detection…")
        self._expect_one_result = True
        self._send({"type": "trigger"})


    def eventFilter(self, obj, event):
        if obj is getattr(self, "controlsPanel", None) and event.type() in (QtCore.QEvent.Hide, QtCore.QEvent.Show):
            action = getattr(self, "controlsPanelAction", None)
            if action is not None:
                visible = obj.isVisible()
                if action.isChecked() != visible:
                    action.setChecked(visible)
        return super().eventFilter(obj, event)

    # -------------- helpers --------------
    def _do_trigger(self):
        if self.rad_trig.isChecked():
            self._send({"type": "trigger"})
            self._append_log("Trigger pressed → waiting for one detection…")
            self._expect_one_result = True
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
        
    def _base_payload(self) -> dict:
        pw, ph = (int(self.last_frame_wh[0]), int(self.last_frame_wh[1])) if self.last_frame_wh != (0, 0) else (0, 0)
        return {
            "version": "1.0",
            "sdk": "vision_ui",                     # <— was "vision_ui_color"; make it EXACT
            "session": self.session_id,
            "timestamp_ms": int(time.time() * 1000),
            "camera": {"proc_width": pw, "proc_height": ph},
            "result": {"objects": [], "counts": {"objects": 0, "detections": 0}}
        }

    def _publish_color_result_to_robot(self, server_msg: Dict):
        """
        Convert server 'color_results' ({objects: [{class_name, area, centroid, circularity, score?}, ...]})
        into the strict VisionResult envelope expected by vgr_manager: each class -> one object,
        each blob -> one detection with pose, center, quad, template_size, color metrics, counts.
        """
        base = self._base_payload()  # already sdk="vision_ui"
        src_objs = server_msg.get("objects") or []

        # Group detections by class name
        by_class: Dict[str, List[dict]] = {}
        for d in src_objs:
            cname = str(d.get("class_name", "unknown"))
            by_class.setdefault(cname, []).append(d)

        objects_out: List[dict] = []
        total_dets = 0

        for obj_id, (cname, dets_raw) in enumerate(sorted(by_class.items(), key=lambda kv: kv[0].lower())):
            dets_out = []
            for inst_id, d in enumerate(dets_raw):
                cx, cy = (d.get("centroid") or [0, 0])[:2]
                area = float(d.get("area", 0.0))
                circ = float(d.get("circularity", -1.0))
                score = float(d.get("score", 0.5))   # default if not provided
                inliers = int(d.get("inliers", 30))  # harmless default

                # Approximate a square support from area:
                s = max(1.0, area ** 0.5)
                w = h = s
                half = s / 2.0
                # Axis-aligned quad around center (TL, TR, BR, BL) to match your sample ordering
                tl = [cx - half, cy - half]
                tr = [cx + half, cy - half]
                br = [cx + half, cy + half]
                bl = [cx - half, cy + half]
                quad = [tl, tr, br, bl]

                # Minimal pose synthesized from center (no real angle info here)
                pose = {
                    "x": float(cx),
                    "y": float(cy),
                    "theta_deg": 0.0,
                    "x_scale": 1.0,
                    "y_scale": 1.0,
                    "origin_xy": [float(cx), float(cy)]
                }

                dets_out.append({
                    "instance_id": inst_id,
                    "score": score,
                    "inliers": inliers,
                    "pose": pose,
                    "center": [float(cx), float(cy)],
                    "quad": quad,
                    "color": {
                        # Keep semantics compatible with your sample
                        "bhattacharyya": -1.0 if circ < 0 else circ,   # or leave circ separate; sample showed -1.0
                        "correlation": -1.0,
                        "deltaE": -1.0
                    }
                })

            total_dets += len(dets_out)

            # template_size: approximate from square side 's'
            template_size = [int(round(w)), int(round(h))]

            objects_out.append({
                "object_id": obj_id,
                "name": cname,
                "template_size": template_size,
                "detections": dets_out
            })

        base["result"]["objects"] = objects_out
        base["result"]["counts"] = {"objects": len(objects_out), "detections": total_dets}

        if self.chk_pub_robot.isChecked() and total_dets > 0:
            self.pub_robot.send_json(base)
            self._append_log(f"UDP → robot (STRICT): classes={len(objects_out)} dets={total_dets}")

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

    # -------------- WS handlers --------------

    def _ws_ok(self):
        self.btn_conn.setEnabled(False)
        self.btn_disc.setEnabled(True)
        self._ping.start()
        self._send({"type": "set_mode", "mode": "training" if self.rad_train.isChecked() else "trigger"})
        self._send({"type": "set_proc_width", "width": int(self.cmb_w.currentText())})
        self._send({"type": "set_publish_every", "n": int(self.sp_every.value())})
        self.cam_panel._emit_params()
        self.cam_panel._emit_view()
        # also push current color streaming toggle state
        self._send({"type":"color_set_stream","enabled": bool(self.chk_stream.isChecked())})
        # and rules if you want immediate application:
        # self._send_rules()
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
            if b64:
                qi_raw, bgr_for_store = self._decode_b64_to_qimage_and_bgr(b64)
                if bgr_for_store is not None:
                    self._last_frame_bgr = bgr_for_store   # always update!
                qi = qi_raw

            w = msg.get("w"); h = msg.get("h")
            if qi is None or w is None or h is None:
                return

            self.last_frame_wh = (int(w), int(h))
            self.video.setFrameSize(self.last_frame_wh)

            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - self._last_ts)
            self._last_ts = now
            self._fps_acc += fps
            self._fps_n += 1
            if self._fps_n >= 10:
                self.lbl_fps.setText(f"FPS: {self._fps_acc / self._fps_n:.1f}")
                self._fps_acc = 0
                self._fps_n = 0

            # If we have an overlay still valid, keep showing it
            if self._last_overlay and time.time() < self._overlay_until:
                self._set_pixmap(self._last_overlay)
            else:
                self._set_pixmap(qi)


        elif t == "color_results":
            self._update_color_results(msg)
        elif t == "ack":
            if not msg.get("ok", True):
                self.status.showMessage(msg.get("error", "error"), 4000)
                self._append_log(f"[ACK-ERR] {msg.get('cmd')} : {msg.get('error')}")

    # -------------- Color handling --------------

    def _toggle_pick(self, on: bool):
        self._pick_enabled = bool(on)
        self.video.enable_pick(self._pick_enabled)
        self.btn_pick.setText("Pick HSV (active)" if on else "Pick HSV")
        
    def _on_pixel_clicked(self, x_disp: int, y_disp: int):
        print(f"[DEBUG] _on_pixel_clicked called with ({x_disp},{y_disp})", flush=True)
        draw = self.video._last_draw_rect
        if draw.isNull() or self.last_frame_wh == (0, 0):
            print("[DEBUG] No valid frame size yet", flush=True)
            self.btn_pick.setChecked(False)
            return

        if self._last_frame_bgr is None:
            print("[DEBUG] _last_frame_bgr is None!", flush=True)
            self.btn_pick.setChecked(False)
            return

        fw, fh = self.last_frame_wh
        sx = fw / float(draw.width())
        sy = fh / float(draw.height())
        x_img = int((x_disp - draw.x()) * sx)
        y_img = int((y_disp - draw.y()) * sy)
        x_img = max(0, min(fw - 1, x_img))
        y_img = max(0, min(fh - 1, y_img))
        print(f"[DEBUG] mapped to image coords=({x_img},{y_img})", flush=True)

        roi = self._last_frame_bgr[max(0, y_img-4):y_img+5, max(0, x_img-4):x_img+5]
        print(f"[DEBUG] ROI shape={None if roi is None else roi.shape}", flush=True)
        if roi is None or roi.size == 0:
            print("[DEBUG] ROI empty, aborting", flush=True)
            self.btn_pick.setChecked(False)
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h = int(np.median(hsv[..., 0]))
        s = int(np.median(hsv[..., 1]))
        v = int(np.median(hsv[..., 2]))
        print(f"[DEBUG] Median HSV=({h},{s},{v})", flush=True)

        dh, ds, dv = 12, 70, 70
        h_min = max(0, h - dh); h_max = min(179, h + dh)
        s_min = max(0, s - ds); s_max = min(255, s + ds)
        v_min = max(0, v - dv); v_max = min(255, v + dv)

        # Update sliders visibly
        for k, val in [("h_min", h_min), ("h_max", h_max),
                    ("s_min", s_min), ("s_max", s_max),
                    ("v_min", v_min), ("v_max", v_max)]:
            sld, spn = self._sl[k]
            sld.blockSignals(True); spn.blockSignals(True)
            sld.setValue(val); spn.setValue(val)
            sld.blockSignals(False); spn.blockSignals(False)
        
        # no auto class, no auto send — only reflect picked HSV on sliders
        self._append_log(
            f"PICK sliders set → H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}]"
        )


        print(f"[PICK] HSV≈({h},{s},{v}) → "
            f"H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}]",
            flush=True)

        # disable pick mode after click
        self.btn_pick.setChecked(False)

    def _add_class_from_sliders(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Class name", "Enter class name:")
        if not ok or not name:
            return
        c = dict(
            name = str(name),
            h_min = int(self._sl["h_min"][1].value()),
            h_max = int(self._sl["h_max"][1].value()),
            s_min = int(self._sl["s_min"][1].value()),
            s_max = int(self._sl["s_max"][1].value()),
            v_min = int(self._sl["v_min"][1].value()),
            v_max = int(self._sl["v_max"][1].value()),
            min_area = int(self.sp_min_area.value()),
            max_area = 1_000_000,
            aspect_min = 0.0,
            aspect_max = 10.0,
            circularity_min = 0.0
        )
        self._classes.append(c)
        self._refresh_class_list()

    def _remove_selected_class(self):
        idx = self.lst_classes.currentRow()
        if 0 <= idx < len(self._classes):
            del self._classes[idx]
            self._refresh_class_list()

    def _refresh_class_list(self):
        self.lst_classes.clear()
        for c in self._classes:
            self.lst_classes.addItem(c.get("name","(unnamed)"))
        # -------------- presets (load/save) --------------

    def _load_presets(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Color Presets", ".", "JSON (*.json);;All files (*)"
        )
        if not fn:
            return
        try:
            with open(fn, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return

        # expected schema: {"classes":[...], "kernel_size":5, "min_area_global":100, "open_iter":1, "close_iter":1}
        classes = data.get("classes")
        if isinstance(classes, list):
            self._classes = classes
            self._refresh_class_list()
        k = data.get("kernel_size")
        if isinstance(k, int):
            self.sp_kernel.setValue(k)
        mag = data.get("min_area_global")
        if isinstance(mag, int):
            self.sp_min_area.setValue(mag)
        oi = data.get("open_iter")
        if isinstance(oi, int):
            self.sp_open.setValue(oi)
        ci = data.get("close_iter")
        if isinstance(ci, int):
            self.sp_close.setValue(ci)

        self._append_log(f"Loaded presets from {fn} ({len(self._classes)} classes)")

    def _save_presets(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Color Presets", "color_presets.json", "JSON (*.json);;All files (*)"
        )
        if not fn:
            return
        data = dict(
            classes=self._classes,
            kernel_size=int(self.sp_kernel.value()),
            min_area_global=int(self.sp_min_area.value()),
            open_iter=int(self.sp_open.value()),
            close_iter=int(self.sp_close.value()),
        )
        try:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))
            return

        self._append_log(f"Saved presets to {fn} ({len(self._classes)} classes)")

    def _send_rules(self):
        self._kernel = int(self.sp_kernel.value())
        self._min_area_global = int(self.sp_min_area.value())
        self._open_iter = int(self.sp_open.value())
        self._close_iter = int(self.sp_close.value())
        self._send({
            "type":"color_set_rules",
            "classes": self._classes,
            "kernel_size": self._kernel,
            "min_area_global": self._min_area_global,
            "open_iter": self._open_iter,
            "close_iter": self._close_iter
        })
        self._append_log(f"Sent {len(self._classes)} classes (k={self._kernel}, min_area={self._min_area_global})")

    def _update_color_results(self, msg: Dict):
        if self.rad_trig.isChecked():
            if not self._expect_one_result:
                # allow persisting for 1s
                if time.time() - self._last_result_time < 1.0:
                    return  # keep showing old result
                else:
                    # clear after 1 sec
                    self.tbl.setRowCount(0)
                    return
        self._expect_one_result = False
        self._last_result_time = time.time()

        objs = msg.get("objects", []) or []
        self.tbl.setRowCount(0)
        for d in objs:
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(str(d.get("class_name",""))))
            self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(str(int(d.get("area",0)))))
            cx, cy = d.get("centroid",[0,0])
            self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(str(int(cx))))
            self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(str(int(cy))))
            self.tbl.setItem(r, 4, QtWidgets.QTableWidgetItem(f"{float(d.get('circularity',0.0)):.3f}"))

        b64 = msg.get("overlay_jpeg_b64")
        if b64:
            qi, _ = self._decode_b64_to_qimage_and_bgr(b64)  # ← was _decode_b64_to_qimage(...)
            if qi:
                self._last_overlay = qi
                self._overlay_until = time.time() + 1.0
                self._set_pixmap(qi)
                
        # --- ADD: publish to robot ---
        try:
            self._publish_color_result_to_robot(msg)
        except Exception as e:
            self._append_log(f"[PUB] color publish error: {e}")


    # -------------- settings --------------

    def _load_settings(self):
        S = self._settings

        def to_bool(v):
            if isinstance(v, bool): return v
            if isinstance(v, str):
                vv = v.strip().lower()
                if vv in ("1","true","yes","on"): return True
                if vv in ("0","false","no","off"): return False
            if isinstance(v, (int,float)): return bool(v)
            return None

        def to_int(v):
            try: return int(v)
            except (TypeError, ValueError): return None

        def to_float(v):
            try: return float(v)
            except (TypeError, ValueError): return None

        def to_str(v):
            return v if isinstance(v, str) else None

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

        self.cmb_w.blockSignals(True)
        self.sp_every.blockSignals(True)
        try:
            width = to_str(S.value('processing/proc_width'))
            if width and self.cmb_w.findText(width) >= 0:
                self.cmb_w.setCurrentText(width)
            every = to_int(S.value('processing/detect_every'))
            if every is not None:
                self.sp_every.setValue(every)
        finally:
            self.cmb_w.blockSignals(False)
            self.sp_every.blockSignals(False)

        # color sliders / kernel config
        try:
            k = to_int(S.value('color/kernel'));        self.sp_kernel.setValue(k) if k is not None else None
            g = to_int(S.value('color/min_area'));      self.sp_min_area.setValue(g) if g is not None else None
            oi = to_int(S.value('color/open_iter'));    self.sp_open.setValue(oi) if oi is not None else None
            ci = to_int(S.value('color/close_iter'));   self.sp_close.setValue(ci) if ci is not None else None
            stream = to_bool(S.value('color/stream'));  self.chk_stream.setChecked(stream) if stream is not None else None
            classes_json = to_str(S.value('color/classes'))
            if classes_json:
                try:
                    self._classes = json.loads(classes_json)
                    if not isinstance(self._classes, list): self._classes = []
                except Exception:
                    self._classes = []
            self._refresh_class_list()
        except Exception:
            pass

        # camera state
        cam_json = to_str(S.value('camera/state'))
        if cam_json:
            try:
                cam_state = json.loads(cam_json)
                if isinstance(cam_state, dict):
                    self.cam_panel.apply_state(cam_state)
            except Exception:
                pass
        robot_ip = to_str(S.value('robot/ip'))
        if robot_ip: self.ed_robot_ip.setText(robot_ip)
        robot_port = to_int(S.value('robot/port'))
        if robot_port: self.sp_robot_port.setValue(robot_port)
        auto_pub = to_bool(S.value('robot/auto_publish'))
        if auto_pub is not None: self.chk_pub_robot.setChecked(auto_pub)

        # logs
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

        S.setValue('color/kernel', int(self.sp_kernel.value()))
        S.setValue('color/min_area', int(self.sp_min_area.value()))
        S.setValue('color/open_iter', int(self.sp_open.value()))
        S.setValue('color/close_iter', int(self.sp_close.value()))
        S.setValue('color/stream', bool(self.chk_stream.isChecked()))
        S.setValue('color/classes', json.dumps(self._classes))

        S.setValue('camera/state', json.dumps(self.cam_panel.get_state()))
        S.setValue('logs/history', json.dumps(self._log_buffer[-200:]))
        S.setValue('robot/ip', self.ed_robot_ip.text())
        S.setValue('robot/port', int(self.sp_robot_port.value()))
        S.setValue('robot/auto_publish', bool(self.chk_pub_robot.isChecked()))

        S.sync()

    # -------------- close/save --------------

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

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
