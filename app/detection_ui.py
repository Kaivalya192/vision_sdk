#!/usr/bin/env python3
"""
RPi Vision Client – Detection-only UI (pattern matching)
"""

import sys, time, base64, json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets, QtWebSockets


# ------------------------ Video Label (rect/poly capture only) ------------------------

class VideoLabel(QtWidgets.QLabel):
    roiSelected     = QtCore.pyqtSignal(QtCore.QRect)  # rectangle selection (display coords)
    polygonSelected = QtCore.pyqtSignal(object)        # list[QPoint] (display coords)

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

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
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


# ------------------------ Camera Panel (unchanged) ------------------------

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


# ------------------------ Main Window (detection only) ------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPi Vision Client – Detection Only")
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
        self._last_roi: Optional[List[int]] = None
        self._settings = QtCore.QSettings('vision_sdk', 'ui_client')
        self._log_buffer: List[str] = []

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

        # Connection
        conn_box = QtWidgets.QGroupBox("Connection")
        ch = QtWidgets.QHBoxLayout(conn_box)
        self.ed_host = QtWidgets.QLineEdit("ws://192.168.1.2:8765")
        self.btn_conn = QtWidgets.QPushButton("Connect")
        self.btn_disc = QtWidgets.QPushButton("Disconnect"); self.btn_disc.setEnabled(False)
        for w in (self.ed_host, self.btn_conn, self.btn_disc): ch.addWidget(w)
        left_v.addWidget(conn_box)

        # Mode
        mode_box = QtWidgets.QGroupBox("Mode")
        mh = QtWidgets.QHBoxLayout(mode_box)
        self.rad_train = QtWidgets.QRadioButton("Training"); self.rad_trig = QtWidgets.QRadioButton("Trigger")
        self.rad_train.setChecked(True)
        self.btn_trig = QtWidgets.QPushButton("TRIGGER")
        mh.addWidget(self.rad_train); mh.addWidget(self.rad_trig); mh.addStretch(1); mh.addWidget(self.btn_trig)
        left_v.addWidget(mode_box)

        # Processing
        proc_box = QtWidgets.QGroupBox("Processing")
        pf = QtWidgets.QFormLayout(proc_box)
        self.cmb_w = QtWidgets.QComboBox()
        [self.cmb_w.addItem(str(w)) for w in [320, 480, 640, 800, 960, 1280]]
        self.cmb_w.setCurrentText("640")
        self.sp_every = QtWidgets.QSpinBox(); self.sp_every.setRange(1, 5); self.sp_every.setValue(1)
        pf.addRow("Proc width", self.cmb_w)
        pf.addRow("Detect every Nth", self.sp_every)
        left_v.addWidget(proc_box)

        # Templates
        tmpl_box = QtWidgets.QGroupBox("Templates")
        tv = QtWidgets.QVBoxLayout(tmpl_box)
        row = QtWidgets.QHBoxLayout()
        self.cmb_slot = QtWidgets.QComboBox(); [self.cmb_slot.addItem(f"{i+1}") for i in range(5)]
        self.ed_name = QtWidgets.QLineEdit("Obj1")
        self.sp_max = QtWidgets.QSpinBox(); self.sp_max.setRange(1, 10); self.sp_max.setValue(3)
        self.chk_en = QtWidgets.QCheckBox("Enabled"); self.chk_en.setChecked(True)
        for w in (QtWidgets.QLabel("Slot"), self.cmb_slot, QtWidgets.QLabel("Name"), self.ed_name,
                  QtWidgets.QLabel("Max"), self.sp_max, self.chk_en):
            row.addWidget(w)
        tv.addLayout(row)
        left_v.addWidget(tmpl_box)

        # Detect controls (ROI capture)
        det_box = QtWidgets.QGroupBox("Detect")
        dh = QtWidgets.QHBoxLayout(det_box)
        self.btn_rect = QtWidgets.QPushButton("Capture Rect")
        self.btn_poly = QtWidgets.QPushButton("Capture Poly")
        self.btn_clear = QtWidgets.QPushButton("Clear Active")
        for w in (self.btn_rect, self.btn_poly, self.btn_clear): dh.addWidget(w)
        left_v.addWidget(det_box)

        # Detection settings
        det_cfg = QtWidgets.QGroupBox("Detection Settings")
        df = QtWidgets.QFormLayout(det_cfg)
        self.dsb_det_score = QtWidgets.QDoubleSpinBox(); self.dsb_det_score.setRange(0.0, 1.0); self.dsb_det_score.setSingleStep(0.01); self.dsb_det_score.setDecimals(3); self.dsb_det_score.setValue(0.25)
        self.sp_det_inliers = QtWidgets.QSpinBox(); self.sp_det_inliers.setRange(1, 200); self.sp_det_inliers.setValue(4)
        self.dsb_det_angle = QtWidgets.QDoubleSpinBox(); self.dsb_det_angle.setRange(0.0, 180.0); self.dsb_det_angle.setDecimals(1); self.dsb_det_angle.setSingleStep(1.0); self.dsb_det_angle.setValue(180.0)
        self.dsb_det_ransac = QtWidgets.QDoubleSpinBox(); self.dsb_det_ransac.setRange(0.1, 20.0); self.dsb_det_ransac.setDecimals(2); self.dsb_det_ransac.setSingleStep(0.1); self.dsb_det_ransac.setValue(4.0)
        self.dsb_det_ratio = QtWidgets.QDoubleSpinBox(); self.dsb_det_ratio.setRange(0.1, 1.0); self.dsb_det_ratio.setDecimals(2); self.dsb_det_ratio.setSingleStep(0.01); self.dsb_det_ratio.setValue(0.90)
        self.sp_det_matches = QtWidgets.QSpinBox(); self.sp_det_matches.setRange(10, 1000); self.sp_det_matches.setValue(150)
        self.sp_det_center = QtWidgets.QSpinBox(); self.sp_det_center.setRange(0, 500); self.sp_det_center.setValue(40)
        df.addRow("Min score", self.dsb_det_score)
        df.addRow("Min inliers", self.sp_det_inliers)
        df.addRow("Angle tolerance (deg)", self.dsb_det_angle)
        df.addRow("RANSAC thr (px)", self.dsb_det_ransac)
        df.addRow("Lowe ratio", self.dsb_det_ratio)
        df.addRow("Max matches", self.sp_det_matches)
        df.addRow("Min center dist (px)", self.sp_det_center)
        self._det_settings_timer = QtCore.QTimer(self); self._det_settings_timer.setInterval(300); self._det_settings_timer.setSingleShot(True); self._det_settings_timer.timeout.connect(self._emit_detection_settings)
        for w in [self.dsb_det_score, self.sp_det_inliers, self.dsb_det_angle, self.dsb_det_ransac, self.dsb_det_ratio, self.sp_det_matches, self.sp_det_center]:
            w.valueChanged.connect(lambda *_: self._det_settings_timer.start())
        left_v.addWidget(det_cfg)

        # Detections table
        det_tbl = QtWidgets.QGroupBox("Detections")
        dv = QtWidgets.QVBoxLayout(det_tbl)
        self.tbl = QtWidgets.QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(["Obj", "Id", "x", "y", "θ", "score", "inliers", "center"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        dv.addWidget(self.tbl)
        left_v.addWidget(det_tbl, 1)

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

        # Results dock (Log only)
        self.resultsDock = QtWidgets.QDockWidget("Results", self)
        self.resultsDock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.resultsDock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                                     QtWidgets.QDockWidget.DockWidgetMovable |
                                     QtWidgets.QDockWidget.DockWidgetFloatable)
        dockw = QtWidgets.QWidget(); dlay = QtWidgets.QVBoxLayout(dockw)
        self.txt_log = QtWidgets.QTextEdit(); self.txt_log.setReadOnly(True)
        dlay.addWidget(QtWidgets.QLabel("Log")); dlay.addWidget(self.txt_log, 1)
        self.resultsDock.setWidget(dockw)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.resultsDock)

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
        self.controlsPanelAction = QtWidgets.QAction("Detection Controls", self, checkable=True, checked=True)
        self.controlsPanelAction.toggled.connect(self.controlsPanel.setVisible)
        self.controlsPanel.installEventFilter(self)
        self.controlsPanelAction.setChecked(self.controlsPanel.isVisible())
        view_menu.addAction(self.controlsPanelAction)
        view_menu.addAction(self.resultsDock.toggleViewAction())
        view_menu.addAction(self.cameraDock.toggleViewAction())

        # status
        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        self.lbl_sess = QtWidgets.QLabel("Disconnected"); self.lbl_fps = QtWidgets.QLabel("FPS: —")
        self.status.addWidget(self.lbl_sess); self.status.addPermanentWidget(self.lbl_fps)

        # signals
        self.btn_conn.clicked.connect(lambda: self.ws.open(QtCore.QUrl(self.ed_host.text().strip())))
        self.btn_disc.clicked.connect(self.ws.close)
        self.rad_train.toggled.connect(lambda: self._send({"type": "set_mode", "mode": "training" if self.rad_train.isChecked() else "trigger"}))
        self.btn_trig.clicked.connect(lambda: self._send({"type": "trigger"}))
        self.cmb_w.currentTextChanged.connect(lambda w: self._send({"type": "set_proc_width", "width": int(w)}))
        self.sp_every.valueChanged.connect(lambda n: self._send({"type": "set_publish_every", "n": int(n)}))
        self.cmb_slot.currentIndexChanged.connect(self._slot_state)
        self.sp_max.valueChanged.connect(self._slot_state)
        self.chk_en.toggled.connect(self._slot_state)
        self.btn_rect.clicked.connect(lambda: self.video.enable_rect_selection(True))
        self.btn_poly.clicked.connect(lambda: self.video.enable_polygon_selection(True))
        self.btn_clear.clicked.connect(lambda: self._send({"type": "clear_template", "slot": int(self.cmb_slot.currentIndex())}))
        self.cam_panel.paramsChanged.connect(lambda p: self._send({"type": "set_params", "params": p}))
        self.cam_panel.viewChanged.connect(lambda v: self._send({"type": "set_view", **v}))
        self.cam_panel.afTriggerRequested.connect(lambda: self._send({"type": "af_trigger"}))
        self.video.roiSelected.connect(self._rect_roi)
        self.video.polygonSelected.connect(self._poly_roi)

        self._ping = QtCore.QTimer(interval=10000, timeout=lambda: self._send({"type": "ping"}))
        self._load_settings()

    def eventFilter(self, obj, event):
        if obj is getattr(self, "controlsPanel", None) and event.type() in (QtCore.QEvent.Hide, QtCore.QEvent.Show):
            action = getattr(self, "controlsPanelAction", None)
            if action is not None:
                visible = obj.isVisible()
                if action.isChecked() != visible:
                    action.setChecked(visible)
        return super().eventFilter(obj, event)

    # ---- websocket handlers ----

    def _ws_ok(self):
        self.btn_conn.setEnabled(False)
        self.btn_disc.setEnabled(True)
        self._ping.start()

        # mode / processing
        self._send({"type": "set_mode", "mode": "training" if self.rad_train.isChecked() else "trigger"})
        self._send({"type": "set_proc_width", "width": int(self.cmb_w.currentText())})
        self._send({"type": "set_publish_every", "n": int(self.sp_every.value())})

        # initial params/view
        self.cam_panel._emit_params()
        self.cam_panel._emit_view()
        self._emit_detection_settings()

    def _ws_closed(self):
        self.btn_conn.setEnabled(True)
        self.btn_disc.setEnabled(False)
        self._ping.stop()
        self.lbl_sess.setText("Disconnected")

    def _append_log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {text}"
        self._log_buffer.append(entry)
        if len(self._log_buffer) > 50:
            self._log_buffer = self._log_buffer[-50:]
        self.txt_log.append(entry)

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
            self._frame(msg)
        elif t == "detections":
            self._dets(msg)
        elif t == "ack":
            if not msg.get("ok", True):
                self.status.showMessage(msg.get("error", "error"), 4000)
                self._append_log(f"[ACK-ERR] {msg.get('cmd')} : {msg.get('error')}")

    # ---- frame/dets ----
    def _decode(self, b64: str) -> Optional[QtGui.QImage]:
        try:
            arr = np.frombuffer(base64.b64decode(b64), np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            return QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        except Exception:
            return None

    def _frame(self, msg: Dict):
        qi = self._decode(msg.get("jpeg_b64", ""))
        w = msg.get("w")
        h = msg.get("h")
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

        self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qi))

    def _dets(self, msg: Dict):
        objs = msg.get("payload", {}).get("result", {}).get("objects", [])

        # Update table
        rows = sum(len(o.get("detections", [])) for o in objs)
        self.tbl.setRowCount(rows)
        r = 0
        for o in objs:
            name = o.get("name", "?")
            for det in o.get("detections", []):
                pose = det.get("pose", {})
                if not pose:
                    pose = {
                        "x": det.get("x", 0),
                        "y": det.get("y", 0),
                        "theta_deg": det.get("theta_deg", det.get("theta", 0)),
                    }
                x = pose.get("x", 0.0)
                y = pose.get("y", 0.0)
                th = pose.get("theta_deg", 0.0)
                vals = [
                    name,
                    str(det.get("instance_id", 0)),
                    f"{x:.1f}",
                    f"{y:.1f}",
                    f"{th:.1f}",
                    f"{det.get('score', 0.0):.3f}",
                    str(det.get('inliers', 0)),
                    f"({int(det['center'][0])},{int(det['center'][1])})" if det.get('center') else "?",
                ]
                for c, v in enumerate(vals):
                    self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
                r += 1

        # If server provides a detection overlay image, show it briefly (or immediately)
        if "overlay_jpeg_b64" in msg and msg["overlay_jpeg_b64"]:
            qi = self._decode(msg["overlay_jpeg_b64"])
            if qi:
                self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qi))

    # ---- ROI capture mapping ----
    def _disp_to_proc(self, qp: QtCore.QPointF) -> Tuple[float, float]:
        draw = self.video._last_draw_rect
        fw, fh = self.last_frame_wh
        sx, sy = fw / float(draw.width()), fh / float(draw.height())
        return (qp.x() - draw.x()) * sx, (qp.y() - draw.y()) * sy

    def _rect_roi(self, rect: QtCore.QRect):
        draw = self.video._last_draw_rect
        if draw.isNull():
            return
        sel = rect.intersected(draw)
        if sel.isEmpty():
            return
        (x, y) = self._disp_to_proc(sel.topLeft())
        w = sel.width() * self.last_frame_wh[0] / draw.width()
        h = sel.height() * self.last_frame_wh[1] / draw.height()
        if w < 10 or h < 10:
            return
        self._last_roi = [int(x), int(y), int(w), int(h)]
        self._send({
            "type": "add_template_rect",
            "slot": int(self.cmb_slot.currentIndex()),
            "name": self.ed_name.text() or f"Obj{self.cmb_slot.currentIndex() + 1}",
            "rect": self._last_roi,
            "max_instances": int(self.sp_max.value()),
        })

    def _poly_roi(self, qpts: List[QtCore.QPoint]):
        draw = self.video._last_draw_rect
        if draw.isNull():
            return
        pts = [self._disp_to_proc(p) for p in qpts]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) - min(xs) < 10 or max(ys) - min(ys) < 10:
            return
        self._send({
            "type": "add_template_poly",
            "slot": int(self.cmb_slot.currentIndex()),
            "name": self.ed_name.text() or f"Obj{self.cmb_slot.currentIndex() + 1}",
            "points": [[round(x, 1), round(y, 1)] for x, y in pts],
            "max_instances": int(self.sp_max.value()),
        })

    def _emit_detection_settings(self):
        params = dict(
            min_score=float(self.dsb_det_score.value()),
            min_inliers=int(self.sp_det_inliers.value()),
            angle_tolerance_deg=float(self.dsb_det_angle.value()),
            ransac_thr_px=float(self.dsb_det_ransac.value()),
            lowe_ratio=float(self.dsb_det_ratio.value()),
            max_matches=int(self.sp_det_matches.value()),
            min_center_dist_px=int(self.sp_det_center.value()),
        )
        self._send({"type": "set_detection_params", "params": params})

    def _slot_state(self):
        self._send({
            "type": "set_slot_state",
            "slot": int(self.cmb_slot.currentIndex()),
            "enabled": self.chk_en.isChecked(),
            "max_instances": int(self.sp_max.value()),
        })

    # ---- settings ----
    def _load_settings(self):
        settings = self._settings

        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                lower = val.strip().lower()
                if lower in ("1", "true", "yes", "on"):
                    return True
                if lower in ("0", "false", "no", "off"):
                    return False
            if isinstance(val, (int, float)):
                return bool(val)
            return None

        def to_int(val):
            try:
                return int(val)
            except (TypeError, ValueError):
                return None

        def to_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        def to_str(val):
            return val if isinstance(val, str) else None

        try:
            geom = to_str(settings.value('window/geometry'))
            if geom:
                self.restoreGeometry(QtCore.QByteArray.fromHex(geom.encode()))
        except Exception:
            pass

        try:
            state = to_str(settings.value('window/state'))
            if state:
                self.restoreState(QtCore.QByteArray.fromHex(state.encode()))
        except Exception:
            pass

        host = to_str(settings.value('connection/host'))
        if host:
            self.ed_host.setText(host)

        mode = to_str(settings.value('mode/current'))
        if mode == 'trigger':
            self.rad_trig.setChecked(True)
        elif mode == 'training':
            self.rad_train.setChecked(True)

        widgets_to_block = [self.cmb_w, self.sp_every, self.cmb_slot, self.sp_max, self.chk_en]
        blocked = []
        for w in widgets_to_block:
            blocked.append((w, w.blockSignals(True)))
        try:
            width = to_str(settings.value('processing/proc_width'))
            if width and self.cmb_w.findText(width) >= 0:
                self.cmb_w.setCurrentText(width)

            every = to_int(settings.value('processing/detect_every'))
            if every is not None:
                self.sp_every.setValue(every)

            slot = to_int(settings.value('templates/slot_index'))
            if slot is not None and 0 <= slot < self.cmb_slot.count():
                self.cmb_slot.setCurrentIndex(slot)

            name = to_str(settings.value('templates/name'))
            if name is not None:
                self.ed_name.setText(name)

            max_instances = to_int(settings.value('templates/max'))
            if max_instances is not None:
                self.sp_max.setValue(max_instances)

            enabled = to_bool(settings.value('templates/enabled'))
            if enabled is not None:
                self.chk_en.setChecked(enabled)
        finally:
            for w, prev in blocked:
                w.blockSignals(prev)

        det_widgets = [self.dsb_det_score, self.sp_det_inliers, self.dsb_det_angle,
                       self.dsb_det_ransac, self.dsb_det_ratio, self.sp_det_matches, self.sp_det_center]
        for w in det_widgets:
            w.blockSignals(True)
        try:
            val = to_float(settings.value('detection/min_score'));             self.dsb_det_score.setValue(val) if val is not None else None
            val = to_int(settings.value('detection/min_inliers'));            self.sp_det_inliers.setValue(val) if val is not None else None
            val = to_float(settings.value('detection/angle_tol_deg'));        self.dsb_det_angle.setValue(val) if val is not None else None
            val = to_float(settings.value('detection/ransac_thr_px'));        self.dsb_det_ransac.setValue(val) if val is not None else None
            val = to_float(settings.value('detection/lowe_ratio'));           self.dsb_det_ratio.setValue(val) if val is not None else None
            val = to_int(settings.value('detection/max_matches'));            self.sp_det_matches.setValue(val) if val is not None else None
            val = to_int(settings.value('detection/min_center_dist_px'));     self.sp_det_center.setValue(val) if val is not None else None
        finally:
            for w in det_widgets:
                w.blockSignals(False)
        self._det_settings_timer.stop()
        self._emit_detection_settings()

        controls_visible = to_bool(settings.value('panels/controls_visible'))
        if controls_visible is not None:
            self.controlsPanelAction.blockSignals(True)
            self.controlsPanelAction.setChecked(controls_visible)
            self.controlsPanelAction.blockSignals(False)
            self.controlsPanel.setVisible(controls_visible)

        results_visible = to_bool(settings.value('panels/results_visible'))
        if results_visible is not None:
            self.resultsDock.setVisible(results_visible)

        camera_visible = to_bool(settings.value('panels/camera_visible'))
        if camera_visible is not None:
            self.cameraDock.setVisible(camera_visible)

        scroll_val = to_int(settings.value('panels/controls_scroll'))
        if scroll_val is not None:
            self.controlsPanel.verticalScrollBar().setValue(scroll_val)

        cam_json = to_str(settings.value('camera/state'))
        if cam_json:
            try:
                cam_state = json.loads(cam_json)
            except Exception:
                cam_state = None
            if isinstance(cam_state, dict):
                self.cam_panel.apply_state(cam_state)

        logs_json = to_str(settings.value('logs/history'))
        self._log_buffer = []
        if logs_json:
            try:
                entries = json.loads(logs_json)
            except Exception:
                entries = []
            if isinstance(entries, list):
                for entry in entries[-50:]:
                    if isinstance(entry, str):
                        self._log_buffer.append(entry)
        if self._log_buffer:
            self.txt_log.clear()
            for entry in self._log_buffer:
                self.txt_log.append(entry)

    def _save_settings(self):
        settings = self._settings
        try:
            settings.setValue('window/geometry', bytes(self.saveGeometry().toHex()).decode('ascii'))
        except Exception:
            pass
        try:
            settings.setValue('window/state', bytes(self.saveState().toHex()).decode('ascii'))
        except Exception:
            pass
        settings.setValue('connection/host', self.ed_host.text())
        settings.setValue('mode/current', 'trigger' if self.rad_trig.isChecked() else 'training')
        settings.setValue('processing/proc_width', self.cmb_w.currentText())
        settings.setValue('processing/detect_every', int(self.sp_every.value()))
        settings.setValue('templates/slot_index', int(self.cmb_slot.currentIndex()))
        settings.setValue('templates/name', self.ed_name.text())
        settings.setValue('templates/max', int(self.sp_max.value()))
        settings.setValue('templates/enabled', bool(self.chk_en.isChecked()))
        settings.setValue('panels/controls_visible', bool(self.controlsPanel.isVisible()))
        settings.setValue('panels/results_visible', bool(self.resultsDock.isVisible()))
        settings.setValue('panels/camera_visible', bool(self.cameraDock.isVisible()))
        settings.setValue('panels/controls_scroll', int(self.controlsPanel.verticalScrollBar().value()))
        settings.setValue('detection/min_score', float(self.dsb_det_score.value()))
        settings.setValue('detection/min_inliers', int(self.sp_det_inliers.value()))
        settings.setValue('detection/angle_tol_deg', float(self.dsb_det_angle.value()))
        settings.setValue('detection/ransac_thr_px', float(self.dsb_det_ransac.value()))
        settings.setValue('detection/lowe_ratio', float(self.dsb_det_ratio.value()))
        settings.setValue('detection/max_matches', int(self.sp_det_matches.value()))
        settings.setValue('detection/min_center_dist_px', int(self.sp_det_center.value()))
        settings.setValue('camera/state', json.dumps(self.cam_panel.get_state()))
        settings.setValue('logs/history', json.dumps(self._log_buffer[-50:]))
        settings.sync()

    # ---- send ----
    def _send(self, obj):
        try:
            self.ws.sendTextMessage(json.dumps(obj, separators=(',', ':')))
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self._save_settings()
        finally:
            super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
