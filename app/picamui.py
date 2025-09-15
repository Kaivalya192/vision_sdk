# ======================
# FILE: app/picamui.py
# ======================
import sys, time, uuid
from typing import Optional, Dict, List, Tuple
import cv2, numpy as np
import json
import socket
import signal, atexit
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtNetwork

from dexsdk.ui.video_label import VideoLabel
from dexsdk.utils import rotate90
from dexsdk.detect import MultiTemplateMatcher
from dexsdk.net.publisher import UDPPublisher
from dexsdk.camera.picam3 import PiCam3

from app.ui import CollapsibleTray, CameraPanel, ModePanel, NetPanel, DetectionsTable


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, camera_num: int = 0):
        super().__init__()
        self.setWindowTitle("PiCam3 — Vision-Guided Sorting (dexsdk)")
        self.resize(1480, 980)

        # Camera & session
        self.cam = PiCam3(width=640, height=480, fps=30.0, camera_num=camera_num, preview=None)
        self.session_id = str(uuid.uuid4())[:8]
        self.frame_id = 0

        # Publisher
        self.publisher = UDPPublisher()
        self.publish_enabled = False

        # Multi-template & state
        self.multi = MultiTemplateMatcher(max_slots=5, min_center_dist_px=40)
        self.active_slot = 0
        self.mode: str = "training"  # training | trigger
        self._last_overlay: Optional[np.ndarray] = None
        self._overlay_until: float = 0.0

        # ---- Layout scaffold ----
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central); root.setContentsMargins(8,8,8,8); root.setSpacing(8)

        # Left stack: topbar + video + collapsible camera tray
        left = QtWidgets.QVBoxLayout(); left.setSpacing(8)

        topbar = QtWidgets.QHBoxLayout()
        self.btn_capture = QtWidgets.QPushButton("Capture Rect ROI ↘ Active")
        self.btn_capture_poly = QtWidgets.QPushButton("Capture Poly ROI ↘ Active")
        self.btn_clear   = QtWidgets.QPushButton("Clear Active")
        self.btn_rotate  = QtWidgets.QPushButton("Rotate 90° view")
        for w in (self.btn_capture, self.btn_capture_poly, self.btn_clear, self.btn_rotate):
            topbar.addWidget(w); topbar.addSpacing(6)
        topbar.addStretch(1)
        left.addLayout(topbar)

        # Video
        self.video = VideoLabel()
        self.video.setMinimumSize(1024, 576)
        self.video.setStyleSheet("background:#111;")

        # Collapsible camera tray under monitor (SCROLLABLE)
        self.cam_panel = CameraPanel(self.cam)
        self.tray = CollapsibleTray("Camera Controls")
        body = QtWidgets.QVBoxLayout()
        body.setContentsMargins(8, 4, 8, 4)
        self.cam_scroll = QtWidgets.QScrollArea()
        self.cam_scroll.setWidgetResizable(True)
        self.cam_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.cam_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.cam_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.cam_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.cam_scroll.setWidget(self.cam_panel)
        body.addWidget(self.cam_scroll)
        self.tray.set_body_layout(body)

        self.left_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.left_split.addWidget(self.video)
        self.left_split.addWidget(self.tray)
        self.left_split.setCollapsible(0, False)
        self.left_split.setCollapsible(1, True)
        self.left_split.setStretchFactor(0, 3)
        self.left_split.setStretchFactor(1, 1)
        left.addWidget(self.left_split, 1)
        root.addLayout(left, 2)

        # Right: Control stack (Mode, Publisher, Processing, Templates, Detections)
        right = QtWidgets.QVBoxLayout(); right.setSpacing(10)

        # Mode panel
        self.mode_panel = ModePanel()
        right.addWidget(self.mode_panel)
        # Expose expected attributes
        self.rad_training = self.mode_panel.rad_training
        self.rad_trigger = self.mode_panel.rad_trigger
        self.btn_trigger = self.mode_panel.btn_trigger

        # Net panel
        self.net_panel = NetPanel()
        right.addWidget(self.net_panel)
        # Expose expected attributes
        self.ed_ip = self.net_panel.ed_ip
        self.sp_port = self.net_panel.sp_port
        self.chk_publish = self.net_panel.chk_publish
        self.sp_cmd_port = self.net_panel.sp_cmd_port
        self.chk_cmd_guard = self.net_panel.chk_cmd_guard

        # Processing
        proc_box = QtWidgets.QGroupBox("Processing")
        proc_form = QtWidgets.QFormLayout(proc_box)
        self.cmb_proc_w = QtWidgets.QComboBox()
        for w in [320, 480, 640, 800, 960, 1280]:
            self.cmb_proc_w.addItem(str(w))
        self.cmb_proc_w.setCurrentText("640")
        self.sp_every = QtWidgets.QSpinBox(); self.sp_every.setRange(1, 5); self.sp_every.setValue(1)
        self.chk_clahe = QtWidgets.QCheckBox("CLAHE + Blur"); self.chk_clahe.setChecked(True)
        self.cmb_mode = QtWidgets.QComboBox(); self.cmb_mode.addItems(["Color + geometry", "Grayscale (geometry only)"]); self.cmb_mode.setCurrentIndex(0)
        proc_form.addRow("Process width", self.cmb_proc_w)
        proc_form.addRow("Process every Nth frame", self.sp_every)
        proc_form.addRow(self.chk_clahe)
        proc_form.addRow("Detection mode", self.cmb_mode)
        right.addWidget(proc_box)

        # Templates
        tmpl_box = QtWidgets.QGroupBox("Templates (max 5)")
        tmpl_v = QtWidgets.QVBoxLayout(tmpl_box)
        row1 = QtWidgets.QHBoxLayout()
        self.cmb_slot = QtWidgets.QComboBox(); [self.cmb_slot.addItem(f"Slot {i+1}") for i in range(5)]
        self.ed_name = QtWidgets.QLineEdit("Obj1")
        self.sp_maxinst = QtWidgets.QSpinBox(); self.sp_maxinst.setRange(1, 10); self.sp_maxinst.setValue(3)
        self.chk_enable = QtWidgets.QCheckBox("Enabled"); self.chk_enable.setChecked(True)
        row1.addWidget(QtWidgets.QLabel("Active:")); row1.addWidget(self.cmb_slot)
        row1.addWidget(QtWidgets.QLabel("Name:")); row1.addWidget(self.ed_name)
        row1.addWidget(QtWidgets.QLabel("Max Inst:")); row1.addWidget(self.sp_maxinst)
        row1.addWidget(self.chk_enable)
        tmpl_v.addLayout(row1)
        right.addWidget(tmpl_box)

        # Detections table
        tbl_box = QtWidgets.QGroupBox("Detections")
        vtbl = QtWidgets.QVBoxLayout(tbl_box)
        self.tbl = DetectionsTable()
        vtbl.addWidget(self.tbl)
        right.addWidget(tbl_box, 1)

        # Status bar
        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        self.lbl_fps = QtWidgets.QLabel("FPS: —"); self.lbl_info = QtWidgets.QLabel(f"Session: {self.session_id}")
        self.status.addWidget(self.lbl_info); self.status.addPermanentWidget(self.lbl_fps)

        root.addLayout(right, 1)

        # ---- Signals ----
        self.btn_capture.clicked.connect(lambda: self.video.enable_rect_selection(True))
        self.btn_capture_poly.clicked.connect(lambda: self.video.enable_polygon_selection(True))
        self.btn_clear.clicked.connect(self._on_clear_active)
        self.btn_rotate.clicked.connect(self._on_rotate)

        self.video.roiSelected.connect(self._on_rect_selected)
        self.video.polygonSelected.connect(self._on_poly_selected)

        self.cmb_slot.currentIndexChanged.connect(self._on_slot_changed)
        self.chk_publish.toggled.connect(self._on_publish_toggle)
        self.chk_clahe.toggled.connect(self._on_clahe_toggle)
        self.sp_maxinst.valueChanged.connect(self._on_maxinst_changed)
        self.chk_enable.toggled.connect(self._on_enable_toggle)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.rad_training.toggled.connect(self._on_mode_radio)
        self.rad_trigger.toggled.connect(self._on_mode_radio)
        self.btn_trigger.clicked.connect(self._on_trigger_once)

        # ---- Timer/State ----
        self.rot_quadrant = 0
        self.last_frame_rgb: Optional[np.ndarray] = None
        self.frame_count = 0
        self._fps_acc = 0.0; self._fps_n = 0

        # --- UDP command listener (remote trigger) ---
        self.cmd_sock = QtNetwork.QUdpSocket(self)
        self._bind_cmd_socket(int(self.sp_cmd_port.value()))
        self.cmd_sock.readyRead.connect(self._on_cmd_ready)
        self.sp_cmd_port.valueChanged.connect(lambda v: self._bind_cmd_socket(int(v)))
        self._last_remote_trigger_ts = 0.0
        self._trigger_busy = False
        # --- LED trigger client ---
        self._led_host = "127.0.0.1"
        self._led_port = 12345
        self._led_training_on = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(0)
        self._on_mode_radio()

    # ---- UI callbacks ----
    def _on_mode_radio(self):
        self.mode = "training" if self.rad_training.isChecked() else "trigger"
        self.btn_trigger.setEnabled(self.mode == "trigger")
        self.cam_panel.set_enabled_by_mode(training=self.mode == "training")
        self._led_set_training(self.mode == "training")

    def _on_trigger_once(self):
        if self.last_frame_rgb is None or self._trigger_busy:
            return
        self._trigger_busy = True

        def do_detect_then_postoff():
            proc = self._proc_frame_copy()
            overlay, objects_report, _ = self.multi.compute_all(proc, draw=True)
            self._last_overlay = overlay.copy()
            self._overlay_until = time.time() + 1.0
            self._populate_table(objects_report)
            if self.publish_enabled:
                payload = self._build_payload(objects_report, proc.shape[1], proc.shape[0], timings={"detect_ms": 0.0, "draw_ms": 0.0})
                self.publisher.send(payload)
            QtCore.QTimer.singleShot(100, lambda: (self._led_send(False), setattr(self, "_trigger_busy", False)))

        self._led_send(True)
        QtCore.QTimer.singleShot(200, do_detect_then_postoff)

    def _on_mode_changed(self, idx: int):
        use_color = (idx == 0)
        for s in self.multi.slots:
            s.matcher.update_params(use_color_gate=use_color)

    def _on_slot_changed(self, idx: int):
        self.active_slot = int(idx)
        self.ed_name.setText(self.ed_name.text() or f"Obj{self.active_slot+1}")

    def _on_publish_toggle(self, ok: bool):
        self.publish_enabled = bool(ok)
        try:
            self.publisher.configure(self.ed_ip.text().strip(), int(self.sp_port.value()))
        except Exception:
            pass

    def _on_clahe_toggle(self, ok: bool):
        for s in self.multi.slots:
            s.matcher.set_use_clahe(bool(ok))

    def _on_maxinst_changed(self, v: int):
        self.multi.set_max_instances(self.active_slot, int(v))

    def _on_enable_toggle(self, ok: bool):
        self.multi.set_enabled(self.active_slot, bool(ok))

    def _on_clear_active(self):
        self.multi.clear(self.active_slot)

    def _on_rotate(self):
        self.rot_quadrant = (self.rot_quadrant + 1) % 4

    def _on_rect_selected(self, rect: QtCore.QRect):
        if self.last_frame_rgb is None:
            return
        frame = self._proc_frame_copy()
        fh, fw = frame.shape[:2]
        draw = self.video._last_draw_rect
        if draw.isNull():
            return
        sel = rect.intersected(draw)
        if sel.isEmpty():
            return
        sx = fw / float(draw.width()); sy = fh / float(draw.height())
        x = max(0, int((sel.x() - draw.x()) * sx)); y = max(0, int((sel.y() - draw.y()) * sy))
        w = max(1, int(sel.width() * sx));          h = max(1, int(sel.height() * sy))
        x2 = min(fw, x + w); y2 = min(fh, y + h)
        roi = frame[y:y2, x:x2].copy()
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            QtWidgets.QMessageBox.warning(self, "ROI too small", "Please select a larger area.")
            return
        name = self.ed_name.text().strip() or f"Obj{self.active_slot+1}"
        self.multi.add_or_replace(self.active_slot, name, roi_bgr=roi, max_instances=int(self.sp_maxinst.value()))

    def _on_poly_selected(self, qpoints: List[QtCore.QPoint]):
        if self.last_frame_rgb is None or not qpoints:
            return
        frame = self._proc_frame_copy()
        fh, fw = frame.shape[:2]
        draw = self.video._last_draw_rect
        if draw.isNull():
            return
        sx = fw / float(draw.width()); sy = fh / float(draw.height())
        pts_img = []
        for p in qpoints:
            if not draw.contains(p):
                px = min(max(p.x(), draw.left()), draw.right())
                py = min(max(p.y(), draw.top()), draw.bottom())
                p = QtCore.QPoint(px, py)
            xi = (p.x() - draw.x()) * sx
            yi = (p.y() - draw.y()) * sy
            pts_img.append([xi, yi])
        pts_np = np.array(pts_img, dtype=np.float32)

        if len(pts_np) == 4:
            s = pts_np.sum(axis=1)
            diff = np.diff(pts_np, axis=1).ravel()
            tl = pts_np[np.argmin(s)]; br = pts_np[np.argmax(s)]
            tr = pts_np[np.argmin(diff)]; bl = pts_np[np.argmax(diff)]
            src = np.array([tl, tr, br, bl], dtype=np.float32)

            def L2(a, b): return float(np.linalg.norm(a - b))
            widthA  = L2(br, bl)
            widthB  = L2(tr, tl)
            heightA = L2(tr, br)
            heightB = L2(tl, bl)
            W = int(round(max(widthA, widthB)))
            H = int(round(max(heightA, heightB)))

            if W < 10 or H < 10:
                QtWidgets.QMessageBox.warning(self, "ROI too small", "Please select a larger rectangle.")
                return

            dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            rectified = cv2.warpPerspective(frame, M, (W, H))

            name = self.ed_name.text().strip() or f"Obj{self.active_slot+1}"
            self.multi.add_or_replace(self.active_slot, name, roi_bgr=rectified, max_instances=int(self.sp_maxinst.value()))
            return

        x1 = max(0, int(np.floor(np.min(pts_np[:, 0]))))
        y1 = max(0, int(np.floor(np.min(pts_np[:, 1]))))
        x2 = min(fw, int(np.ceil(np.max(pts_np[:, 0]))))
        y2 = min(fh, int(np.ceil(np.max(pts_np[:, 1]))))
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        if w < 10 or h < 10:
            QtWidgets.QMessageBox.warning(self, "ROI too small", "Please select a larger polygon.")
            return

        pts_shift = (pts_np - np.array([[x1, y1]], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_shift], 255)

        roi = frame[y1:y2, x1:x2].copy()
        if roi.shape[:2] != mask.shape[:2]:
            QtWidgets.QMessageBox.warning(self, "Mask mismatch", "Internal error building polygon mask.")
            return

        name = self.ed_name.text().strip() or f"Obj{self.active_slot+1}"
        self.multi.add_or_replace_polygon(self.active_slot, name, roi_bgr=roi, roi_mask=mask, max_instances=int(self.sp_maxinst.value()))

    # ---- Processing helpers ----
    def _grab(self):
        return self.cam.get_frame()

    def _proc_frame_copy(self) -> np.ndarray:
        rgb = self.last_frame_rgb
        if rgb is None:
            return np.zeros((480, 640, 3), np.uint8)
        if self.cam_panel.chk_flip_h.isChecked():
            rgb = cv2.flip(rgb, 1)
        if self.cam_panel.chk_flip_v.isChecked():
            rgb = cv2.flip(rgb, 0)
        if self.rot_quadrant:
            rgb = rotate90(rgb, self.rot_quadrant)
        try:
            target_w = int(self.cmb_proc_w.currentText())
        except Exception:
            target_w = 640
        h, w = rgb.shape[:2]
        if target_w > 0 and w != target_w:
            target_h = max(1, int(h * (target_w / float(w))))
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def _update(self):
        t_all0 = time.perf_counter()
        frame_rgb = self._grab()
        if frame_rgb is None:
            return
        self.last_frame_rgb = frame_rgb
        self.frame_id += 1

        proc_bgr = self._proc_frame_copy()
        if self.mode == "training":
            self.frame_count += 1
            if (self.frame_count % max(1, int(self.sp_every.value()))) == 0:
                overlay, objects_report, _ = self.multi.compute_all(proc_bgr, draw=True)
                disp_bgr = overlay
                self._populate_table(objects_report)
            else:
                disp_bgr = proc_bgr
        else:
            if time.time() < self._overlay_until and self._last_overlay is not None:
                disp_bgr = self._last_overlay
            else:
                disp_bgr = proc_bgr

        disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(disp_rgb.data, disp_rgb.shape[1], disp_rgb.shape[0], disp_rgb.strides[0], QtGui.QImage.Format_RGB888)
        self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qimg))

        dt = time.perf_counter() - t_all0
        fps = 1.0 / max(1e-6, dt)
        self._fps_acc += fps; self._fps_n += 1
        if self._fps_n >= 10:
            self.lbl_fps.setText(f"FPS: {self._fps_acc / self._fps_n:.1f}")
            self._fps_acc = 0.0; self._fps_n = 0

    def _populate_table(self, objects_report: List[Dict]):
        # Delegate to table wrapper to keep API stable
        if hasattr(self.tbl, "populate"):
            self.tbl.populate(objects_report)
            return
        # Fallback (if plain QTableWidget)
        rows = sum(len(o["detections"]) for o in objects_report)
        self.tbl.setRowCount(rows)
        r = 0
        for obj in objects_report:
            name = obj.get("name", "?")
            for det in obj.get("detections", []):
                ctr = det.get("center")
                vals = [
                    name,
                    str(det.get("instance_id", 0)),
                    f"{det['pose']['x']:.1f}", f"{det['pose']['y']:.1f}", f"{det['pose']['theta_deg']:.1f}",
                    f"{det.get('score', 0.0):.3f}", str(det.get('inliers', 0)),
                    f"({int(ctr[0])},{int(ctr[1])})" if ctr else "?",
                ]
                for c, text in enumerate(vals):
                    self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem(text))
                r += 1
        if rows == 0:
            self.tbl.setRowCount(0)

    def _build_payload(self, objects_report: List[Dict], width: int, height: int, timings: Dict[str, float]):
        now_ms = int(time.time() * 1000)
        counts = {"objects": len(objects_report), "detections": int(sum(len(o.get("detections", [])) for o in objects_report))}
        return {
            "version": "1.0",
            "sdk": {"name": "dexsdk", "module": "SIFT", "soft_color_gate": self.cmb_mode.currentIndex() == 0},
            "session": {"session_id": self.session_id, "frame_id": self.frame_id},
            "timestamp_ms": now_ms,
            "camera": {"source_index": 0, "proc_width": width, "proc_height": height, "coordinate_space": "processed", "units": "pixels"},
            "result": {"counts": counts, "objects": objects_report},
            "timing_ms": {**timings, "total_ms": round(sum(timings.values()), 1)}
        }

    def _bind_cmd_socket(self, port: int):
        try:
            self.cmd_sock.close()
        except Exception:
            pass
        ok = self.cmd_sock.bind(QtNetwork.QHostAddress.AnyIPv4, int(port), QtNetwork.QUdpSocket.ShareAddress | QtNetwork.QUdpSocket.ReuseAddressHint)
        if hasattr(self, "status"):
            self.status.showMessage(f"Cmd UDP bind {'ok' if ok else 'FAILED'} on :{port}", 3000)

    def _on_cmd_ready(self):
        now = time.time()
        while self.cmd_sock.hasPendingDatagrams():
            size = self.cmd_sock.pendingDatagramSize()
            data, sender, sender_port = self.cmd_sock.readDatagram(size)
            sender_ip = sender.toString().split('%')[0]
            if self.chk_cmd_guard.isChecked():
                allow_ip = self.ed_ip.text().strip()
                if allow_ip and allow_ip != sender_ip:
                    continue
            text = (data.decode('utf-8', errors='ignore')).strip()
            cmd = text.lower()
            payload = {}
            try:
                payload = json.loads(text)
                cmd = str(payload.get('cmd', cmd)).lower()
            except Exception:
                pass
            if 'trigger' in cmd:
                if (now - self._last_remote_trigger_ts) < 0.25:
                    continue
                self._last_remote_trigger_ts = now
                if self.mode != 'trigger':
                    self.rad_trigger.setChecked(True)
                    self._on_mode_radio()
                pub = payload.get('publish') if isinstance(payload, dict) else None
                if isinstance(pub, dict) and 'port' in pub and ('ip' in pub or 'host' in pub):
                    ip = pub.get('ip') or pub.get('host')
                    prt = int(pub['port'])
                    try:
                        self.ed_ip.setText(str(ip))
                        self.sp_port.setValue(prt)
                        self.publisher.configure(ip, prt)
                        self.publish_enabled = True
                        self.chk_publish.setChecked(True)
                    except Exception:
                        pass
                self._on_trigger_once()
                try:
                    ack = json.dumps({
                        "ok": True,
                        "session": self.session_id,
                        "frame_id": self.frame_id,
                        "mode": self.mode,
                        "ts_ms": int(time.time() * 1000)
                    }).encode('utf-8')
                    self.cmd_sock.writeDatagram(ack, sender, sender_port)
                except Exception:
                    pass
            elif 'ping' in cmd or 'status' in cmd:
                try:
                    st = json.dumps({
                        "ok": True,
                        "mode": self.mode,
                        "session": self.session_id,
                        "frame_id": self.frame_id
                    }).encode('utf-8')
                    self.cmd_sock.writeDatagram(st, sender, sender_port)
                except Exception:
                    pass

    def _led_send(self, on: bool):
        try:
            with socket.create_connection((self._led_host, self._led_port), timeout=0.2) as s:
                s.sendall(b'1' if on else b'0')
                try:
                    s.recv(64)
                except Exception:
                    pass
        except Exception:
            pass

    def _led_set_training(self, training: bool):
        if training and not self._led_training_on:
            self._led_send(True)
            self._led_training_on = True
        elif not training and self._led_training_on:
            self._led_send(False)
            self._led_training_on = False

    def closeEvent(self, event: QtGui.QCloseEvent):
        try: self.timer.stop()
        except Exception: pass
        try: self.publisher.close()
        except Exception: pass
        try: self.cam.stop()
        except Exception: pass
        try: self.cmd_sock.close()
        except Exception: pass
        try: self._led_send(False)
        except Exception: pass
        super().closeEvent(event)


def main():
    cam_num = 0
    if len(sys.argv) >= 2:
        try: cam_num = int(sys.argv[1])
        except ValueError: pass
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(camera_num=cam_num)
    w.show()
    def _graceful_kill(*_):
        try: w._led_send(False)
        except Exception: pass
        QtWidgets.QApplication.quit()

    signal.signal(signal.SIGINT, _graceful_kill)
    signal.signal(signal.SIGTERM, _graceful_kill)
    atexit.register(lambda: w._led_send(False))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

