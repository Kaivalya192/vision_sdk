#!/usr/bin/env python3
"""
RPi Vision Client — tailored for the LED-aware server
----------------------------------------------------

‣ Removed controls: HDR modes, digital zoom.  
‣ Added “manual” AWB mode (enables R/B gain boxes).  
‣ Added AF-Trigger button (sends {"type":"af_trigger"}).

Run:  python vision_client.py   (PyQt5 required)
"""

import sys, time, base64, json
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets, QtWebSockets


# ---------------------------------------------------------------------------#
#                               Helper widgets                               #
# ---------------------------------------------------------------------------#

class VideoLabel(QtWidgets.QLabel):
    roiSelected      = QtCore.pyqtSignal(QtCore.QRect)
    polygonSelected  = QtCore.pyqtSignal(object)            # list[QPoint]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._last_draw_rect = QtCore.QRect()

        # rect ROI
        self._rect_mode = False
        self._origin    = QtCore.QPoint()
        self._rubber    = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

        # polygon ROI
        self._poly_mode = False
        self._poly_pts  = []

    # --- public -------------------------------------------------------------
    def setPixmapKeepAspect(self, pm: QtGui.QPixmap):
        if pm.isNull():
            super().setPixmap(pm); self._last_draw_rect = QtCore.QRect(); return
        area = self.size()
        scaled = pm.scaled(area, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        x = (area.width()  - scaled.width())  // 2
        y = (area.height() - scaled.height()) // 2
        self._last_draw_rect = QtCore.QRect(x, y, scaled.width(), scaled.height())

        canvas = QtGui.QPixmap(area); canvas.fill(QtCore.Qt.black)
        p = QtGui.QPainter(canvas); p.drawPixmap(self._last_draw_rect, scaled)

        if self._poly_mode and self._poly_pts:
            pen = QtGui.QPen(QtCore.Qt.yellow); pen.setWidth(2); p.setPen(pen)
            for i in range(1, len(self._poly_pts)):
                p.drawLine(self._poly_pts[i-1], self._poly_pts[i])
            for q in self._poly_pts:
                p.drawEllipse(q, 3, 3)
        p.end()
        super().setPixmap(canvas)

    # --- mode toggles -------------------------------------------------------
    def enable_rect_selection(self, ok: bool):
        self._rect_mode = ok
        self._poly_mode = False
        self._poly_pts.clear()
        if not ok: self._rubber.hide()

    def enable_polygon_selection(self, ok: bool):
        self._poly_mode = ok
        self._rect_mode = False
        self._poly_pts.clear()
        if not ok: self.update()

    # --- events -------------------------------------------------------------
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self._rect_mode and ev.button() == QtCore.Qt.LeftButton:
            self._origin = ev.pos(); self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize())); self._rubber.show()
        elif self._poly_mode:
            if ev.button() == QtCore.Qt.LeftButton and self._last_draw_rect.contains(ev.pos()):
                self._poly_pts.append(ev.pos()); self.update()
            elif ev.button() == QtCore.Qt.RightButton: self._finish_poly()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._rect_mode and not self._origin.isNull():
            self._rubber.setGeometry(QtCore.QRect(self._origin, ev.pos()).normalized())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self._rect_mode and ev.button() == QtCore.Qt.LeftButton:
            r = self._rubber.geometry(); self._rubber.hide(); self._rect_mode=False
            if r.width()>5 and r.height()>5: self.roiSelected.emit(r)
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):     # finish poly w/ dbl-click
        if self._poly_mode: self._finish_poly()
        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        if self._poly_mode and ev.key()==QtCore.Qt.Key_Escape:
            self.enable_polygon_selection(False)
        super().keyPressEvent(ev)

    def _finish_poly(self):
        if len(self._poly_pts) >= 3:
            self.polygonSelected.emit(list(self._poly_pts))
        self._poly_pts.clear(); self._poly_mode=False; self.update()


class CollapsibleTray(QtWidgets.QWidget):
    def __init__(self, title="Camera Controls", parent=None):
        super().__init__(parent)
        self._body   = QtWidgets.QWidget()
        self._toggle = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(QtCore.Qt.RightArrow)
        self._toggle.clicked.connect(lambda: self.set_collapsed(self._body.isVisible()))
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        top=QtWidgets.QHBoxLayout(); top.addWidget(self._toggle); top.addStretch(1)
        wrap=QtWidgets.QWidget(); v=QtWidgets.QVBoxLayout(wrap); v.setContentsMargins(0,0,0,0); v.addWidget(self._body)
        lay.addLayout(top); lay.addWidget(wrap)
        self.set_collapsed(True)

    def set_body_layout(self, l): self._body.setLayout(l)
    def set_collapsed(self, c):
        self._body.setVisible(not c); self._toggle.setArrowType(QtCore.Qt.DownArrow if not c else QtCore.Qt.RightArrow)


# ---------------------------------------------------------------------------#
#                              Camera control UI                             #
# ---------------------------------------------------------------------------#

class CameraPanel(QtWidgets.QWidget):
    paramsChanged      = QtCore.pyqtSignal(dict)
    viewChanged        = QtCore.pyqtSignal(dict)
    afTriggerRequested = QtCore.pyqtSignal()          # new

    def __init__(self, parent=None):
        super().__init__(parent)
        self._debounce = QtCore.QTimer(self, interval=150, singleShot=True,
                                       timeout=self._emit_params)

        grid = QtWidgets.QGridLayout(self); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6)
        r=0
        # --- AE / exposure / gain ------------------------------------------
        self.chk_ae = QtWidgets.QCheckBox("Auto Exposure"); grid.addWidget(self.chk_ae, r,0,1,2); r+=1
        self.sp_exp = QtWidgets.QSpinBox()
        self.sp_exp.setRange(100, 33000)
        self.sp_exp.setValue(6000)
        self.sp_exp.setSuffix(" µs")
        self.sp_exp.setSingleStep(100)
        grid.addWidget(QtWidgets.QLabel("Exposure"), r,0); grid.addWidget(self.sp_exp,r,1); r+=1

        self.dsb_gain = QtWidgets.QDoubleSpinBox()
        self.dsb_gain.setRange(1.0, 16.0)
        self.dsb_gain.setValue(2.0)
        self.dsb_gain.setSingleStep(0.05)
        self.dsb_gain.setDecimals(2)
        grid.addWidget(QtWidgets.QLabel("Analogue Gain"), r,0); grid.addWidget(self.dsb_gain,r,1); r+=1

        self.dsb_fps  = QtWidgets.QDoubleSpinBox()
        self.dsb_fps.setRange(1.0, 120.0)
        self.dsb_fps.setValue(30.0)
        self.dsb_fps.setDecimals(1)
        grid.addWidget(QtWidgets.QLabel("Framerate"), r,0); grid.addWidget(self.dsb_fps,r,1); r+=1

        # --- AWB ------------------------------------------------------------
        self.cmb_awb = QtWidgets.QComboBox(); self.cmb_awb.addItems(
            ["auto","tungsten","fluorescent","indoor","daylight","cloudy","manual"])
        grid.addWidget(QtWidgets.QLabel("AWB Mode"), r,0); grid.addWidget(self.cmb_awb,r,1); r+=1
        self.dsb_awb_r = QtWidgets.QDoubleSpinBox()
        self.dsb_awb_r.setRange(0.1, 8.0)
        self.dsb_awb_r.setValue(2.0)
        self.dsb_awb_r.setSingleStep(0.05)
        self.dsb_awb_r.setDecimals(2)

        self.dsb_awb_b = QtWidgets.QDoubleSpinBox()
        self.dsb_awb_b.setRange(0.1, 8.0)
        self.dsb_awb_b.setValue(2.0)
        self.dsb_awb_b.setSingleStep(0.05)
        self.dsb_awb_b.setDecimals(2)
        hb=QtWidgets.QHBoxLayout(); hb.addWidget(QtWidgets.QLabel("Gains R/B")); hb.addWidget(self.dsb_awb_r); hb.addWidget(self.dsb_awb_b)
        grid.addWidget(QtWidgets.QLabel(""), r,0); grid.addLayout(hb,r,1); r+=1

        # --- focus ----------------------------------------------------------
        self.cmb_af = QtWidgets.QComboBox(); self.cmb_af.addItems(["auto","continuous","manual"])
        grid.addWidget(QtWidgets.QLabel("AF Mode"), r,0); grid.addWidget(self.cmb_af,r,1); r+=1
        self.dsb_dioptre = QtWidgets.QDoubleSpinBox()
        self.dsb_dioptre.setRange(0.0, 10.0)
        self.dsb_dioptre.setValue(0.0)
        self.dsb_dioptre.setDecimals(2)
        self.dsb_dioptre.setSingleStep(0.05)

        self.btn_af_trig = QtWidgets.QPushButton("AF Trigger")
        hf=QtWidgets.QHBoxLayout(); hf.addWidget(self.dsb_dioptre); hf.addWidget(self.btn_af_trig)
        grid.addWidget(QtWidgets.QLabel("Lens (dpt)"), r,0); grid.addLayout(hf,r,1); r+=1

        # --- image tuning ---------------------------------------------------
        self.dsb_bri = QtWidgets.QDoubleSpinBox()
        self.dsb_bri.setRange(-1.0, 1.0)
        self.dsb_bri.setValue(0.0)
        self.dsb_bri.setDecimals(2)

        self.dsb_con = QtWidgets.QDoubleSpinBox()
        self.dsb_con.setRange(0.0, 2.0)
        self.dsb_con.setValue(1.0)
        self.dsb_con.setDecimals(2)

        self.dsb_sat = QtWidgets.QDoubleSpinBox()
        self.dsb_sat.setRange(0.0, 2.0)
        self.dsb_sat.setValue(1.0)
        self.dsb_sat.setDecimals(2)

        self.dsb_sha = QtWidgets.QDoubleSpinBox()
        self.dsb_sha.setRange(0.0, 2.0)
        self.dsb_sha.setValue(1.0)
        self.dsb_sha.setDecimals(2)

        self.cmb_den = QtWidgets.QComboBox(); self.cmb_den.addItems(["off","fast","high_quality"])
        for lab,widget in [("Brightness",self.dsb_bri),("Contrast",self.dsb_con),
                           ("Saturation",self.dsb_sat),("Sharpness",self.dsb_sha),
                           ("Denoise",self.cmb_den)]:
            grid.addWidget(QtWidgets.QLabel(lab), r,0); grid.addWidget(widget,r,1); r+=1

        # --- view flips -----------------------------------------------------
        self.chk_flip_h = QtWidgets.QCheckBox("Flip H"); self.chk_flip_v=QtWidgets.QCheckBox("Flip V")
        self.btn_rot=QtWidgets.QPushButton("Rotate 90°")
        hv=QtWidgets.QHBoxLayout(); hv.addWidget(self.chk_flip_h); hv.addWidget(self.chk_flip_v); hv.addWidget(self.btn_rot); hv.addStretch()
        grid.addWidget(QtWidgets.QLabel("View"), r,0); grid.addLayout(hv,r,1); r+=1

        # reset --------------------------------------------------------------
        self.btn_reset = QtWidgets.QPushButton("Reset tuning")
        grid.addWidget(self.btn_reset,r,0,1,2); r+=1

        # -------- event wiring ----------------------------------------------
        for w in [self.chk_ae,self.sp_exp,self.dsb_gain,self.dsb_fps,
                  self.cmb_awb,self.dsb_awb_r,self.dsb_awb_b,
                  self.cmb_af,self.dsb_dioptre,
                  self.dsb_bri,self.dsb_con,self.dsb_sat,self.dsb_sha,self.cmb_den]:
            w.installEventFilter(self)
            if hasattr(w,'valueChanged'): w.valueChanged.connect(self._any)
            elif hasattr(w,'currentTextChanged'): w.currentTextChanged.connect(self._any)
            elif hasattr(w,'toggled'): w.toggled.connect(self._any)

        self.chk_flip_h.toggled.connect(self._emit_view)
        self.chk_flip_v.toggled.connect(self._emit_view)
        self.btn_rot.clicked.connect(self._on_rot)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_af_trig.clicked.connect(self.afTriggerRequested.emit)
        self.cmb_awb.currentTextChanged.connect(self._awb_mode_changed)

        self._rot_q=0
        self._awb_mode_changed(self.cmb_awb.currentText())

    # --- helpers -----------------------------------------------------------
    def _any(self,*_): self._debounce.start()
    def _emit_params(self):
        p=dict(auto_exposure=self.chk_ae.isChecked(),
               exposure_us=int(self.sp_exp.value()),
               gain=float(self.dsb_gain.value()),
               fps=float(self.dsb_fps.value()),
               awb_mode=self.cmb_awb.currentText(),
               awb_rb=[float(self.dsb_awb_r.value()),float(self.dsb_awb_b.value())],
               focus_mode=self.cmb_af.currentText(),
               dioptre=float(self.dsb_dioptre.value()),
               brightness=float(self.dsb_bri.value()),
               contrast=float(self.dsb_con.value()),
               saturation=float(self.dsb_sat.value()),
               sharpness=float(self.dsb_sha.value()),
               denoise=self.cmb_den.currentText())
        self.paramsChanged.emit(p)

    def _on_rot(self): self._rot_q=(self._rot_q+1)%4; self._emit_view()
    def _emit_view(self):
        self.viewChanged.emit(dict(flip_h=self.chk_flip_h.isChecked(),
                                   flip_v=self.chk_flip_v.isChecked(),
                                   rot_quadrant=self._rot_q))
    def _awb_mode_changed(self,mode):
        manual = (mode=="manual")
        self.dsb_awb_r.setEnabled(manual); self.dsb_awb_b.setEnabled(manual)
        self._any()

    def _reset(self):
        self.blockSignals(True)
        self.chk_ae.setChecked(False); self.sp_exp.setValue(6000); self.dsb_gain.setValue(2.0)
        self.dsb_fps.setValue(30.0); self.cmb_awb.setCurrentText("auto")
        self.dsb_awb_r.setValue(2.0); self.dsb_awb_b.setValue(2.0)
        self.cmb_af.setCurrentText("manual"); self.dsb_dioptre.setValue(0.0)
        for w,val in [(self.dsb_bri,0.0),(self.dsb_con,1.0),(self.dsb_sat,1.0),(self.dsb_sha,1.0)]: w.setValue(val)
        self.cmb_den.setCurrentText("fast")
        self.chk_flip_h.setChecked(False); self.chk_flip_v.setChecked(False); self._rot_q=0
        self.blockSignals(False); self._emit_params(); self._emit_view()


# ---------------------------------------------------------------------------#
#                                 Main window                                #
# ---------------------------------------------------------------------------#

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPi Vision Client"); self.resize(1480,980)

        self.session_id="-"; self.last_frame_wh=(0,0)
        self._fps_acc=0.0; self._fps_n=0; self._last_ts=time.perf_counter()
        self._overlay_until=0.0; self._last_overlay_img=None

        # websocket ----------------------------------------------------------
        self.ws=QtWebSockets.QWebSocket(); self.ws.textMessageReceived.connect(self._ws_txt)
        self.ws.connected.connect(self._ws_ok); self.ws.disconnected.connect(self._ws_closed)
        self.ws.error.connect(lambda e: self.status.showMessage(self.ws.errorString(),3000))

        # ui scaffold --------------------------------------------------------
        central=QtWidgets.QWidget(); self.setCentralWidget(central); root=QtWidgets.QHBoxLayout(central)
        # -- left stack ------------------------------------------------------
        left=QtWidgets.QVBoxLayout(); topbar=QtWidgets.QHBoxLayout()
        self.ed_host=QtWidgets.QLineEdit("ws://192.168.1.2:8765")
        self.btn_conn=QtWidgets.QPushButton("Connect")
        self.btn_disc=QtWidgets.QPushButton("Disconnect"); self.btn_disc.setEnabled(False)
        self.btn_rect=QtWidgets.QPushButton("Capture Rect")
        self.btn_poly=QtWidgets.QPushButton("Capture Poly")
        self.btn_clear=QtWidgets.QPushButton("Clear Active")
        for w in (self.ed_host,self.btn_conn,self.btn_disc,self.btn_rect,self.btn_poly,self.btn_clear):
            topbar.addWidget(w); topbar.addSpacing(6)
        left.addLayout(topbar)

        self.video=VideoLabel(); self.video.setMinimumSize(1024,576); self.video.setStyleSheet("background:#111;")
        self.cam_panel=CameraPanel()
        tray=CollapsibleTray(); body=QtWidgets.QVBoxLayout(); scr=QtWidgets.QScrollArea()
        scr.setWidgetResizable(True); scr.setFrameShape(QtWidgets.QFrame.NoFrame)
        scr.setWidget(self.cam_panel); body.addWidget(scr); tray.set_body_layout(body)
        vsplit=QtWidgets.QSplitter(QtCore.Qt.Vertical); vsplit.addWidget(self.video); vsplit.addWidget(tray)
        vsplit.setCollapsible(1,True); vsplit.setStretchFactor(0,3)
        left.addWidget(vsplit,1); root.addLayout(left,2)

        # -- right stack -----------------------------------------------------
        right=QtWidgets.QVBoxLayout()
        mode_box=QtWidgets.QGroupBox("Mode"); hb=QtWidgets.QHBoxLayout(mode_box)
        self.rad_train=QtWidgets.QRadioButton("Training"); self.rad_trig=QtWidgets.QRadioButton("Trigger")
        self.rad_train.setChecked(True); self.btn_trig=QtWidgets.QPushButton("TRIGGER")
        for w in (self.rad_train,self.rad_trig): hb.addWidget(w); hb.addStretch(1); hb.addWidget(self.btn_trig)
        right.addWidget(mode_box)

        proc_box=QtWidgets.QGroupBox("Processing"); pf=QtWidgets.QFormLayout(proc_box)
        self.cmb_w=QtWidgets.QComboBox(); [self.cmb_w.addItem(str(w)) for w in [320,480,640,800,960,1280]]
        self.cmb_w.setCurrentText("640"); self.sp_every=QtWidgets.QSpinBox()
        self.sp_every.setRange(1, 5)
        self.sp_every.setValue(1)
        pf.addRow("Proc width",self.cmb_w); pf.addRow("Detect every Nth",self.sp_every)
        right.addWidget(proc_box)

        tmpl_box=QtWidgets.QGroupBox("Templates"); tv=QtWidgets.QVBoxLayout(tmpl_box)
        row=QtWidgets.QHBoxLayout(); self.cmb_slot=QtWidgets.QComboBox(); [self.cmb_slot.addItem(f"{i+1}") for i in range(5)]
        self.ed_name=QtWidgets.QLineEdit("Obj1"); self.sp_max=QtWidgets.QSpinBox()
        self.sp_max.setRange(1, 10)
        self.sp_max.setValue(3)
        self.chk_en=QtWidgets.QCheckBox("Enabled"); self.chk_en.setChecked(True)
        for w in (QtWidgets.QLabel("Slot"),self.cmb_slot,QtWidgets.QLabel("Name"),self.ed_name,
                  QtWidgets.QLabel("Max"),self.sp_max,self.chk_en): row.addWidget(w)
        tv.addLayout(row); right.addWidget(tmpl_box)

        det_box=QtWidgets.QGroupBox("Detections"); dv=QtWidgets.QVBoxLayout(det_box)
        self.tbl=QtWidgets.QTableWidget(0,8)
        self.tbl.setHorizontalHeaderLabels(["Obj","Id","x","y","θ","score","inliers","center"])
        self.tbl.horizontalHeader().setStretchLastSection(True); dv.addWidget(self.tbl)
        right.addWidget(det_box,1); root.addLayout(right,1)

        # status -------------------------------------------------------------
        self.status=QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        self.lbl_sess=QtWidgets.QLabel("Disconnected"); self.lbl_fps=QtWidgets.QLabel("FPS: —")
        self.status.addWidget(self.lbl_sess); self.status.addPermanentWidget(self.lbl_fps)

        # signals ------------------------------------------------------------
        self.btn_conn.clicked.connect(lambda: self.ws.open(QtCore.QUrl(self.ed_host.text().strip())))
        self.btn_disc.clicked.connect(self.ws.close)
        self.btn_rect.clicked.connect(lambda: self.video.enable_rect_selection(True))
        self.btn_poly.clicked.connect(lambda: self.video.enable_polygon_selection(True))
        self.btn_clear.clicked.connect(lambda: self._send({"type":"clear_template","slot":int(self.cmb_slot.currentIndex())}))
        self.rad_train.toggled.connect(lambda: self._send({"type":"set_mode","mode":"training" if self.rad_train.isChecked() else "trigger"}))
        self.btn_trig.clicked.connect(lambda: self._send({"type":"trigger"}))
        self.cmb_w.currentTextChanged.connect(lambda w: self._send({"type":"set_proc_width","width":int(w)}))
        self.sp_every.valueChanged.connect(lambda n: self._send({"type":"set_publish_every","n":int(n)}))
        self.cmb_slot.currentIndexChanged.connect(self._slot_state); self.sp_max.valueChanged.connect(self._slot_state); self.chk_en.toggled.connect(self._slot_state)
        self.video.roiSelected.connect(self._rect_roi); self.video.polygonSelected.connect(self._poly_roi)
        self.cam_panel.paramsChanged.connect(lambda p: self._send({"type":"set_params","params":p}))
        self.cam_panel.viewChanged.connect(lambda v: self._send({"type":"set_view",**v}))
        self.cam_panel.afTriggerRequested.connect(lambda: self._send({"type":"af_trigger"}))

        self._ping=QtCore.QTimer(interval=10000,timeout=lambda: self._send({"type":"ping"}))

    # ---------------- websocket handlers -----------------------------------
    def _ws_ok(self):
        self.btn_conn.setEnabled(False); self.btn_disc.setEnabled(True); self._ping.start()
        self._send({"type":"set_mode","mode":"training" if self.rad_train.isChecked() else "trigger"})
        self._send({"type":"set_proc_width","width":int(self.cmb_w.currentText())})
        self._send({"type":"set_publish_every","n":int(self.sp_every.value())})
        # initial params / view
        self.cam_panel._emit_params(); self.cam_panel._emit_view()

    def _ws_closed(self):
        self.btn_conn.setEnabled(True); self.btn_disc.setEnabled(False); self._ping.stop()
        self.lbl_sess.setText("Disconnected")

    def _ws_txt(self,txt:str):
        try: msg=json.loads(txt); t=msg.get("type","")
        except Exception: return
        if t=="hello": self.session_id=msg.get("session_id","-"); self.lbl_sess.setText(f"Session {self.session_id}")
        elif t=="frame": self._frame(msg)
        elif t=="detections": self._dets(msg)
        elif t=="ack":
            if msg.get("cmd")=="af_trigger" and msg.get("ok") and msg.get("error") is None:
                d=msg.get("dioptre")
                if d is not None: self.status.showMessage(f"AF done – lens {d:.2f} dpt",2000)
            if not msg.get("ok",True):
                self.status.showMessage(f"{msg.get('cmd')}: {msg.get('error')}",4000)

    # ---------------- frame / det handling ---------------------------------
    def _decode(self,b64:str):
        try:
            arr=np.frombuffer(base64.b64decode(b64),np.uint8); bgr=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            if bgr is None: return None
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2]
            return QtGui.QImage(rgb.data,w,h,rgb.strides[0],QtGui.QImage.Format_RGB888).copy()
        except Exception: return None

    def _frame(self,msg:Dict):
        qi=self._decode(msg.get("jpeg_b64","")); w=msg.get("w"); h=msg.get("h")
        if qi is None or w is None or h is None: return
        self.last_frame_wh=(int(w),int(h))
        now=time.perf_counter(); fps=1.0/max(1e-6,now-self._last_ts); self._last_ts=now
        self._fps_acc+=fps; self._fps_n+=1
        if self._fps_n>=10: self.lbl_fps.setText(f"FPS: {self._fps_acc/self._fps_n:.1f}"); self._fps_acc=0; self._fps_n=0
        if time.time()<self._overlay_until and self._last_overlay_img is not None: qi=self._last_overlay_img
        self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qi))

    def _dets(self,msg:Dict):
        objs=msg.get("payload",{}).get("result",{}).get("objects",[]); self._table(objs)
        if "overlay_jpeg_b64" in msg:
            qi=self._decode(msg["overlay_jpeg_b64"]); 
            if qi: self._last_overlay_img=qi; self._overlay_until=time.time()+1.0; self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qi))

    def _table(self, objs: List[Dict]):
        rows = sum(len(o.get("detections", [])) for o in objs)
        self.tbl.setRowCount(rows)
        r = 0
        for o in objs:
            name = o.get("name", "?")
            for det in o.get("detections", []):
                # ------------- flexible pose extraction ------------------
                pose = det.get("pose", {})          # new format
                if not pose:                        # fallback to flat keys
                    pose = {"x": det.get("x", 0),
                            "y": det.get("y", 0),
                            "theta_deg": det.get("theta_deg", det.get("theta", 0))}
                x = pose.get("x", 0.0)
                y = pose.get("y", 0.0)
                th = pose.get("theta_deg", 0.0)
                # ---------------------------------------------------------
                vals = [
                    name,
                    str(det.get("instance_id", 0)),
                    f"{x:.1f}", f"{y:.1f}", f"{th:.1f}",
                    f"{det.get('score', 0.0):.3f}",
                    str(det.get('inliers', 0)),
                    f"({int(det['center'][0])},{int(det['center'][1])})"
                    if det.get('center') else "—",
                ]
                for c, v in enumerate(vals):
                    self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
                r += 1
        if rows == 0:
            self.tbl.setRowCount(0)


    # ---------------- ROI emit ---------------------------------------------
    def _disp_to_proc(self,qp:QtCore.QPointF):
        draw=self.video._last_draw_rect; fw,fh=self.last_frame_wh
        sx,sy=fw/float(draw.width()),fh/float(draw.height())
        return (qp.x()-draw.x())*sx,(qp.y()-draw.y())*sy

    def _rect_roi(self,rect:QtCore.QRect):
        draw = self.video._last_draw_rect
        if draw.isNull():
            return
        sel = rect.intersected(draw)
        if sel.isEmpty():
            return
        (x,y)=self._disp_to_proc(sel.topLeft()); w=sel.width()*self.last_frame_wh[0]/draw.width()
        h=sel.height()*self.last_frame_wh[1]/draw.height()
        if w<10 or h<10: return
        self._send({"type":"add_template_rect","slot":int(self.cmb_slot.currentIndex()),
                    "name":self.ed_name.text() or f"Obj{self.cmb_slot.currentIndex()+1}",
                    "rect":[int(x),int(y),int(w),int(h)],"max_instances":int(self.sp_max.value())})

    def _poly_roi(self,qpts:List[QtCore.QPoint]):
        draw=self.video._last_draw_rect
        if draw.isNull():
            return
        pts=[self._disp_to_proc(p) for p in qpts]
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        if max(xs)-min(xs)<10 or max(ys)-min(ys)<10: return
        self._send({"type":"add_template_poly","slot":int(self.cmb_slot.currentIndex()),
                    "name":self.ed_name.text() or f"Obj{self.cmb_slot.currentIndex()+1}",
                    "points":[[round(x,1),round(y,1)] for x,y in pts],
                    "max_instances":int(self.sp_max.value())})

    # ---------------- misc --------------------------------------------------
    def _slot_state(self):
        self._send({"type":"set_slot_state","slot":int(self.cmb_slot.currentIndex()),
                    "enabled":self.chk_en.isChecked(),"max_instances":int(self.sp_max.value())})

    def _send(self,obj): 
        try: self.ws.sendTextMessage(json.dumps(obj,separators=(',',':')))
        except Exception: pass


# ---------------------------------------------------------------------------#
def main():
    app=QtWidgets.QApplication(sys.argv)
    w=MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
