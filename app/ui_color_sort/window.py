from __future__ import annotations
import json, base64
from typing import List, Dict, Optional
import numpy as np, cv2

from PyQt5 import QtWidgets, QtCore, QtGui, QtWebSockets

# reuse your shared panels
from app.ui.net_panel import NetPanel
from app.ui.camera_panel import CameraPanel
from app.ui.mode_panel import ModePanel
from app.ui.detections_table import DetectionsTable  # we’ll use its table shell for listing

def _b64_to_qimage(b64: str) -> Optional[QtGui.QImage]:
    try:
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None: return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h,w = rgb.shape[:2]
        return QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
    except Exception:
        return None

class ColorSortWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision SDK – Color Sorting")
        self.resize(1500, 980)

        # WS
        self.ws = QtWebSockets.QWebSocket()
        self.ws.textMessageReceived.connect(self._ws_txt)
        self.ws.connected.connect(self._ws_ok)
        self.ws.disconnected.connect(self._ws_closed)

        # state
        self.session_id = "-"
        self._last_overlay: Optional[QtGui.QImage] = None
        self._overlay_until = 0.0
        self._log_buffer: List[str] = []
        self._classes: List[Dict] = []      # same schema as your POC
        self._kernel = 5
        self._min_area_global = 100
        self._open_iter = 1
        self._close_iter = 1

        self._build()

    def _build(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central); root.setContentsMargins(8,8,8,8); root.setSpacing(8)

        # LEFT: stacked controls like your detection UI
        left = QtWidgets.QVBoxLayout()
        # connection
        self.net = NetPanel()
        left.addWidget(self.net)
        # mode (training/trigger)  – identical behavior
        self.mode = ModePanel()
        left.addWidget(self.mode)
        # camera params
        self.cam = CameraPanel()
        left.addWidget(self.cam)

        # color rules editor (very small)
        grp = QtWidgets.QGroupBox("Color Rules")
        f = QtWidgets.QFormLayout(grp)
        self.btn_import = QtWidgets.QPushButton("Import JSON…")
        self.btn_import.clicked.connect(self._import_json)
        self.btn_send = QtWidgets.QPushButton("Send to Server")
        self.btn_send.clicked.connect(self._send_rules)
        self.chk_stream = QtWidgets.QCheckBox("Stream results")
        self.chk_stream.toggled.connect(self._toggle_stream)
        self.sp_k = QtWidgets.QSpinBox(); self.sp_k.setRange(1,31); self.sp_k.setValue(5)
        self.sp_min_area = QtWidgets.QSpinBox(); self.sp_min_area.setRange(1,1_000_000); self.sp_min_area.setValue(100)
        self.sp_open = QtWidgets.QSpinBox(); self.sp_open.setRange(0,8); self.sp_open.setValue(1)
        self.sp_close= QtWidgets.QSpinBox(); self.sp_close.setRange(0,8); self.sp_close.setValue(1)
        f.addRow(self.btn_import, self.btn_send)
        f.addRow("Kernel", self.sp_k)
        f.addRow("Min area", self.sp_min_area)
        f.addRow("Open iters", self.sp_open)
        f.addRow("Close iters", self.sp_close)
        f.addRow(self.chk_stream)
        left.addWidget(grp)

        left.addStretch(1)
        left_wrap = QtWidgets.QScrollArea(); left_wrap.setWidgetResizable(True)
        ww = QtWidgets.QWidget(); wl = QtWidgets.QVBoxLayout(ww); wl.addLayout(left); wl.addStretch(1)
        left_wrap.setWidget(ww)
        root.addWidget(left_wrap, 0)

        # CENTER: video
        self.lbl_video = QtWidgets.QLabel(); self.lbl_video.setMinimumSize(1024, 576)
        self.lbl_video.setStyleSheet("background:#111;")
        root.addWidget(self.lbl_video, 1)

        # RIGHT: results + logs
        right = QtWidgets.QVBoxLayout()
        self.tbl = QtWidgets.QTableWidget(0,5)
        self.tbl.setHorizontalHeaderLabels(["class","area","cx","cy","circularity"])
        right.addWidget(QtWidgets.QLabel("Color Results"))
        right.addWidget(self.tbl, 1)
        self.txt_log = QtWidgets.QTextEdit(); self.txt_log.setReadOnly(True)
        right.addWidget(QtWidgets.QLabel("Log"))
        right.addWidget(self.txt_log, 1)
        right_wrap = QtWidgets.QWidget(); rlay = QtWidgets.QVBoxLayout(right_wrap); rlay.addLayout(right)
        root.addWidget(right_wrap, 0)

        # wire net / cam / mode
        self.net.btnConnect.clicked.connect(lambda: self.ws.open(QtCore.QUrl(self.net.edUrl.text().strip())))
        self.net.btnDisconnect.clicked.connect(self.ws.close)
        self.mode.training.toggled.connect(lambda: self._send({"type":"set_mode","mode": "training" if self.mode.training.isChecked() else "trigger"}))
        self.cam.paramsChanged.connect(lambda p: self._send({"type":"set_params","params":p}))
        self.cam.viewChanged.connect(lambda v: self._send({"type":"set_view", **v}))
        self.mode.btnTrigger.clicked.connect(lambda: self._send({"type":"trigger"}))

    # ---------- helpers ----------
    def _send(self, obj: Dict):
        try: self.ws.sendTextMessage(json.dumps(obj, separators=(',',':')))
        except Exception: pass

    def _log(self, s: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {s}"
        self._log_buffer.append(line)
        self.txt_log.append(line)

    # ---------- WS ----------
    def _ws_ok(self):
        self._send({"type":"set_mode", "mode":"training" if self.mode.training.isChecked() else "trigger"})
        self.cam._emit_params(); self.cam._emit_view()
        self._log("connected")

    def _ws_closed(self):
        self._log("disconnected")

    def _ws_txt(self, txt: str):
        try: msg = json.loads(txt)
        except Exception: return
        t = msg.get("type","")
        if t == "hello":
            self.session_id = msg.get("session_id","-")
            self._log(f"session {self.session_id}")
        elif t == "frame":
            # only draw server overlay for color if present in color_results;
            # otherwise show raw stream
            if self._last_overlay is not None and time.time() < self._overlay_until:
                qi = self._last_overlay
            else:
                b64 = msg.get("jpeg_b64")
                if b64:
                    qi = _b64_to_qimage(b64)
                else:
                    qi = None
            if qi: self._set_pixmap(qi)
        elif t == "color_results":
            self._update_results(msg)
        elif t == "ack":
            if not msg.get("ok", True):
                self._log(f"ACK-ERR {msg.get('cmd')}: {msg.get('error')}")

    def _set_pixmap(self, qi: QtGui.QImage):
        pm = QtGui.QPixmap.fromImage(qi)
        self.lbl_video.setPixmap(pm.scaled(self.lbl_video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _update_results(self, msg: Dict):
        objs = msg.get("objects", []) or []
        self.tbl.setRowCount(0)
        for d in objs:
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            self.tbl.setItem(r,0, QtWidgets.QTableWidgetItem(str(d.get("class_name",""))))
            self.tbl.setItem(r,1, QtWidgets.QTableWidgetItem(str(d.get("area",0))))
            cx,cy = d.get("centroid",[0,0])
            self.tbl.setItem(r,2, QtWidgets.QTableWidgetItem(str(int(cx))))
            self.tbl.setItem(r,3, QtWidgets.QTableWidgetItem(str(int(cy))))
            self.tbl.setItem(r,4, QtWidgets.QTableWidgetItem(f"{float(d.get('circularity',0.0)):.3f}"))
        b64 = msg.get("overlay_jpeg_b64")
        if b64:
            qi = _b64_to_qimage(b64)
            if qi:
                self._last_overlay = qi
                self._overlay_until = time.time() + 1.0
                self._set_pixmap(qi)

    # ---------- UI actions ----------
    def _import_json(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Color Rules", ".", "JSON (*.json);;All files(*)")
        if not fn: return
        try:
            data = json.loads(open(fn,"r").read())
            self._classes = data.get("classes", [])
            self._log(f"Loaded {len(self._classes)} classes")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def _send_rules(self):
        self._kernel = int(self.sp_k.value())
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

    def _toggle_stream(self, on: bool):
        self._send({"type":"color_set_stream","enabled": bool(on)})
        if not on:
            self._last_overlay = None
            self._overlay_until = 0.0
