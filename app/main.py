# ======================
# FILE: app/main.py
# ======================
import sys, time, uuid
from typing import Optional, Dict, List
import cv2, numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from dexsdk.ui.video_label import VideoLabel
from dexsdk.utils import rotate90
from dexsdk.camera.webcam import open_capture, grab_frame
from dexsdk.detection_multi import MultiTemplateMatcher
from dexsdk.net.publisher import UDPPublisher


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cam_index: int = 0):
        super().__init__()
        self.setWindowTitle("Vision-Guided Sorting — dexsdk")
        self.resize(1400, 960)

        # Camera & session
        self.cam_index = cam_index
        self.cap = open_capture(cam_index)
        self.session_id = str(uuid.uuid4())[:8]
        self.frame_id = 0

        # Publisher
        self.publisher = UDPPublisher()
        self.publish_enabled = False

        # Multi-template (max 5 objects)
        self.multi = MultiTemplateMatcher(max_slots=5, min_center_dist_px=40)
        self.active_slot = 0  # which slot ROI capture goes to

        # ---- Layout ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left: Video + topbar
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)

        topbar = QtWidgets.QHBoxLayout()
        self.btn_capture = QtWidgets.QPushButton("Capture Rect ROI → Active")
        self.btn_capture_poly = QtWidgets.QPushButton("Capture Poly ROI → Active")  # NEW
        self.btn_clear   = QtWidgets.QPushButton("Clear Active")
        self.btn_rotate  = QtWidgets.QPushButton("Rotate 90° view")
        self.chk_flip_h  = QtWidgets.QCheckBox("Flip H")
        self.chk_flip_v  = QtWidgets.QCheckBox("Flip V")
        for w in (self.btn_capture, self.btn_capture_poly, self.btn_clear, self.btn_rotate, self.chk_flip_h, self.chk_flip_v):
            topbar.addWidget(w); topbar.addSpacing(6)
        topbar.addStretch(1)
        left.addLayout(topbar)

        self.video = VideoLabel()
        self.video.setMinimumSize(1024, 576)
        self.video.setStyleSheet("background:#111;")
        left.addWidget(self.video, 1)
        root.addLayout(left, 2)

        # Right: Control stack
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        # Publisher card
        net_box = QtWidgets.QGroupBox("Publisher")
        net_form = QtWidgets.QFormLayout(net_box)
        self.ed_ip = QtWidgets.QLineEdit("127.0.0.1")
        self.sp_port = QtWidgets.QSpinBox(); self.sp_port.setRange(1, 65535); self.sp_port.setValue(40001)
        self.chk_publish = QtWidgets.QCheckBox("Publish JSON over UDP")
        net_form.addRow("IP", self.ed_ip)
        net_form.addRow("Port", self.sp_port)
        net_form.addRow(self.chk_publish)
        right.addWidget(net_box)

        # Processing card
        proc_box = QtWidgets.QGroupBox("Processing")
        proc_form = QtWidgets.QFormLayout(proc_box)
        self.cmb_proc_w = QtWidgets.QComboBox()
        for w in [320, 480, 640, 800, 960, 1280]:
            self.cmb_proc_w.addItem(str(w))
        self.cmb_proc_w.setCurrentText("640")

        self.sp_every = QtWidgets.QSpinBox(); self.sp_every.setRange(1, 5); self.sp_every.setValue(1)
        self.chk_clahe = QtWidgets.QCheckBox("CLAHE + Blur"); self.chk_clahe.setChecked(True)

        proc_form.addRow("Process width", self.cmb_proc_w)
        proc_form.addRow("Process every Nth frame", self.sp_every)
        proc_form.addRow(self.chk_clahe)
        right.addWidget(proc_box)

        # Templates manager
        tmpl_box = QtWidgets.QGroupBox("Templates (max 5)")
        tmpl_v = QtWidgets.QVBoxLayout(tmpl_box)
        row1 = QtWidgets.QHBoxLayout()
        self.cmb_slot = QtWidgets.QComboBox()
        [self.cmb_slot.addItem(f"Slot {i+1}") for i in range(5)]
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
        self.tbl = QtWidgets.QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(["Obj", "Inst", "x", "y", "θ(deg)", "score", "inliers", "center"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        vtbl.addWidget(self.tbl)
        right.addWidget(tbl_box, 1)

        # Status bottom
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_fps = QtWidgets.QLabel("FPS: —")
        self.lbl_info = QtWidgets.QLabel(f"Session: {self.session_id}")
        self.status.addWidget(self.lbl_info)
        self.status.addPermanentWidget(self.lbl_fps)

        root.addLayout(right, 1)

        # ---- Signals ----
        self.btn_capture.clicked.connect(lambda: self.video.enable_rect_selection(True))
        self.btn_capture_poly.clicked.connect(lambda: self.video.enable_polygon_selection(True))  # NEW
        self.btn_clear.clicked.connect(self._on_clear_active)
        self.btn_rotate.clicked.connect(self._on_rotate)

        self.video.roiSelected.connect(self._on_rect_selected)
        self.video.polygonSelected.connect(self._on_poly_selected)  # NEW

        self.cmb_slot.currentIndexChanged.connect(self._on_slot_changed)
        self.chk_publish.toggled.connect(self._on_publish_toggle)
        self.chk_clahe.toggled.connect(self._on_clahe_toggle)
        self.sp_maxinst.valueChanged.connect(self._on_maxinst_changed)
        self.chk_enable.toggled.connect(self._on_enable_toggle)

        # ---- Timer/State ----
        self.rot_quadrant = 0
        self.last_frame_bgr: Optional[np.ndarray] = None
        self.frame_count = 0
        self._fps_t0 = time.perf_counter()
        self._fps_acc = 0
        self._fps_n = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(0)

    # ---- UI callbacks ----
    def _on_slot_changed(self, idx: int):
        self.active_slot = int(idx)
        # update defaults in the editor (soft)
        self.ed_name.setText(self.ed_name.text() or f"Obj{self.active_slot+1}")

    def _on_publish_toggle(self, ok: bool):
        self.publish_enabled = bool(ok)
        # reconfigure socket with current values
        try:
            self.publisher.configure(self.ed_ip.text().strip(), int(self.sp_port.value()))
        except Exception:
            pass

    def _on_clahe_toggle(self, ok: bool):
        # apply to all existing matchers
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

    # ---- ROI capture (rect) ----
    def _on_rect_selected(self, rect: QtCore.QRect):
        if self.last_frame_bgr is None:
            return
        frame = self._proc_frame_copy()  # downscaled
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

    # ---- ROI capture (polygon) ----
    def _on_poly_selected(self, qpoints: List[QtCore.QPoint]):
        """qpoints are in label coordinates; map them to processed-frame coords,
        crop to polygon bbox, and build a mask for non-rectangular template."""
        if self.last_frame_bgr is None or not qpoints:
            return

        frame = self._proc_frame_copy()  # downscaled (processed space)
        fh, fw = frame.shape[:2]
        draw = self.video._last_draw_rect
        if draw.isNull():
            return

        # Map label points -> processed-frame space
        sx = fw / float(draw.width()); sy = fh / float(draw.height())
        pts_img = []
        for p in qpoints:
            if not draw.contains(p):
                # clamp into draw rect to be safe
                px = min(max(p.x(), draw.left()), draw.right())
                py = min(max(p.y(), draw.top()), draw.bottom())
                p = QtCore.QPoint(px, py)
            xi = (p.x() - draw.x()) * sx
            yi = (p.y() - draw.y()) * sy
            pts_img.append([xi, yi])

        pts_np = np.array(pts_img, dtype=np.float32)

        # Compute tight bbox (clip to frame)
        x1 = max(0, int(np.floor(np.min(pts_np[:, 0]))))
        y1 = max(0, int(np.floor(np.min(pts_np[:, 1]))))
        x2 = min(fw, int(np.ceil(np.max(pts_np[:, 0]))))
        y2 = min(fh, int(np.ceil(np.max(pts_np[:, 1]))))
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w < 10 or h < 10:
            QtWidgets.QMessageBox.warning(self, "ROI too small", "Please select a larger polygon.")
            return

        # Shift polygon to bbox space and rasterize mask
        pts_shift = (pts_np - np.array([[x1, y1]], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_shift], 255)

        roi = frame[y1:y2, x1:x2].copy()
        # sanity check
        if roi.shape[:2] != mask.shape[:2]:
            QtWidgets.QMessageBox.warning(self, "Mask mismatch", "Internal error building polygon mask.")
            return

        name = self.ed_name.text().strip() or f"Obj{self.active_slot+1}"
        self.multi.add_or_replace_polygon(self.active_slot, name, roi_bgr=roi, roi_mask=mask, max_instances=int(self.sp_maxinst.value()))

    # ---- Processing helpers ----
    def _grab(self):
        return grab_frame(self.cap)

    def _proc_frame_copy(self) -> np.ndarray:
        bgr = self.last_frame_bgr
        if bgr is None:
            return np.zeros((480, 640, 3), np.uint8)
        if self.chk_flip_h.isChecked():
            bgr = cv2.flip(bgr, 1)
        if self.chk_flip_v.isChecked():
            bgr = cv2.flip(bgr, 0)
        if self.rot_quadrant:
            bgr = rotate90(bgr, self.rot_quadrant)
        try:
            target_w = int(self.cmb_proc_w.currentText())
        except Exception:
            target_w = 640
        h, w = bgr.shape[:2]
        if target_w > 0 and w != target_w:
            target_h = max(1, int(h * (target_w / float(w))))
            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return bgr

    def _update(self):
        t_all0 = time.perf_counter()
        # Grab
        frame = self._grab()
        if frame is None:
            return
        self.last_frame_bgr = frame
        self.frame_id += 1

        # Preprocess
        proc = self._proc_frame_copy()

        # Frame skip
        self.frame_count += 1
        if (self.frame_count % max(1, int(self.sp_every.value()))) != 0:
            # just display latest processed frame
            disp_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(disp_rgb.data, disp_rgb.shape[1], disp_rgb.shape[0],
                                disp_rgb.strides[0], QtGui.QImage.Format_RGB888)
            self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qimg))
            return

        # Detect across templates
        t2 = time.perf_counter()
        overlay, objects_report, total_instances = self.multi.compute_all(proc, draw=True)
        t_det = (time.perf_counter() - t2) * 1000.0

        # Display
        t3 = time.perf_counter()
        disp_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(disp_rgb.data, disp_rgb.shape[1], disp_rgb.shape[0],
                            disp_rgb.strides[0], QtGui.QImage.Format_RGB888)
        self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qimg))
        t_draw = (time.perf_counter() - t3) * 1000.0

        # Update table
        self._populate_table(objects_report)

        # Publish JSON (industrial-ish schema)
        if self.publish_enabled:
            payload = self._build_payload(objects_report, proc.shape[1], proc.shape[0], timings={
                "detect_ms": round(t_det, 1),
                "draw_ms": round(t_draw, 1),
            })
            self.publisher.send(payload)

        # FPS
        dt = time.perf_counter() - t_all0
        self._fps_acc += 1.0 / max(1e-6, dt)
        self._fps_n += 1
        if self._fps_n >= 10:
            fps = self._fps_acc / self._fps_n
            self.lbl_fps.setText(f"FPS: {fps:.1f}")
            self._fps_acc = 0.0
            self._fps_n = 0

    def _populate_table(self, objects_report: List[Dict]):
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
                    f"({int(ctr[0])},{int(ctr[1])})" if ctr else "—",
                ]
                for c, text in enumerate(vals):
                    item = QtWidgets.QTableWidgetItem(text)
                    self.tbl.setItem(r, c, item)
                r += 1
        if rows == 0:
            self.tbl.setRowCount(0)

    def _build_payload(self, objects_report: List[Dict], width: int, height: int, timings: Dict[str, float]):
        now_ms = int(time.time() * 1000)
        counts = {
            "objects": len(objects_report),
            "detections": int(sum(len(o.get("detections", [])) for o in objects_report)),
        }
        payload = {
            "version": "1.0",
            "sdk": {"name": "dexsdk", "module": "SIFT", "soft_color_gate": True},
            "session": {"session_id": self.session_id, "frame_id": self.frame_id},
            "timestamp_ms": now_ms,
            "camera": {
                "source_index": self.cam_index,
                "proc_width": width,
                "proc_height": height,
                "coordinate_space": "processed",
                "units": "pixels"
            },
            "result": {
                "counts": counts,
                "objects": objects_report
            },
            "timing_ms": {
                **timings,
                "total_ms": round(sum(timings.values()), 1)
            }
        }
        return payload

    # ---- Cleanup ----
    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            self.publisher.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    cam_index = 0
    if len(sys.argv) >= 2:
        try:
            cam_index = int(sys.argv[1])
        except ValueError:
            pass
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(cam_index=cam_index)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
