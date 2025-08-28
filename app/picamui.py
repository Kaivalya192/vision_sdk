# ======================
# FILE: app/picam3_ui.py
# ======================
import sys, time, uuid
from typing import Optional, Dict, List, Tuple
import cv2, numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from dexsdk.ui.video_label import VideoLabel
from dexsdk.utils import rotate90
from dexsdk.detection_multi import MultiTemplateMatcher
from dexsdk.net.publisher import UDPPublisher
from dexsdk.camera.picam3 import PiCam3


# -------- Small helper: collapsible tray --------
class CollapsibleTray(QtWidgets.QWidget):
    toggled = QtCore.pyqtSignal(bool)

    def __init__(self, title: str = "Camera Controls", parent=None):
        super().__init__(parent)
        self._body = QtWidgets.QWidget()
        self._body.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._toggle = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(QtCore.Qt.RightArrow)
        self._toggle.clicked.connect(self._on_toggle)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._toggle)
        top.addStretch(1)

        self._wrap = QtWidgets.QWidget()
        wrap_l = QtWidgets.QVBoxLayout(self._wrap)
        wrap_l.setContentsMargins(0, 0, 0, 0)
        wrap_l.addWidget(self._body)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self._wrap)

        self.set_collapsed(True)

    def set_body_layout(self, layout: QtWidgets.QLayout):
        self._body.setLayout(layout)

    def set_collapsed(self, collapsed: bool):
        self._wrap.setVisible(not collapsed)
        self._toggle.setChecked(not collapsed)
        self._toggle.setArrowType(QtCore.Qt.DownArrow if not collapsed else QtCore.Qt.RightArrow)
        self.toggled.emit(not collapsed)

    def _on_toggle(self):
        self.set_collapsed(self._wrap.isVisible())


# -------- Camera control panel (enabled only in Training mode) --------
class CameraPanel(QtWidgets.QWidget):
    """Binds UI controls to PiCam3 setter methods. Call set_enabled_by_mode() on mode changes."""
    def __init__(self, cam: PiCam3, parent=None):
        super().__init__(parent)
        self.cam = cam
        self._building = False  # guard to avoid re-entrancy

        # Query ranges (fall back safely if not available)
        info = cam.controls_info()
        def rng(key, default: Tuple[int,int,int]):
            try:
                lo, hi, _def = info.get(key, default)
                return int(lo), int(hi), int(_def if _def is not None else (lo+hi)//2)
            except Exception:
                return default

        exp_lo, exp_hi, _ = rng("ExposureTime", (100, 33000, 5000))
        gain_lo, gain_hi, _ = rng("AnalogueGain", (1, 16, 2))
        fdl_lo, fdl_hi, _ = rng("FrameDurationLimits", (5000, 33333, 33333))

        # ---- Layout ----
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6)

        r = 0
        # AE / Exposure / Gain
        self.chk_ae = QtWidgets.QCheckBox("Auto Exposure")
        self.chk_ae.setChecked(True)
        self.chk_ae.toggled.connect(self._on_ae_toggled)
        grid.addWidget(self.chk_ae, r, 0, 1, 2); r += 1

        self.sp_exposure = QtWidgets.QSpinBox(); self.sp_exposure.setRange(exp_lo, exp_hi); self.sp_exposure.setSuffix(" µs")
        self.sp_exposure.setSingleStep(100); self.sp_exposure.setValue(min(max(6000, exp_lo), exp_hi))
        self.sp_exposure.valueChanged.connect(self._apply_manual_exposure)
        grid.addWidget(QtWidgets.QLabel("Exposure"), r, 0); grid.addWidget(self.sp_exposure, r, 1); r += 1

        self.dsb_gain = QtWidgets.QDoubleSpinBox(); self.dsb_gain.setRange(float(gain_lo), float(gain_hi))
        self.dsb_gain.setDecimals(2); self.dsb_gain.setSingleStep(0.05); self.dsb_gain.setValue(2.0)
        self.dsb_gain.valueChanged.connect(self._apply_manual_exposure)
        grid.addWidget(QtWidgets.QLabel("Analogue Gain"), r, 0); grid.addWidget(self.dsb_gain, r, 1); r += 1

        # FPS (FrameDurationLimits)
        self.dsb_fps = QtWidgets.QDoubleSpinBox(); self.dsb_fps.setRange(1.0, 120.0); self.dsb_fps.setDecimals(1)
        # derive default FPS from FDL (µs)
        default_fps = round(1_000_000.0 / max(1, fdl_hi), 1)
        self.dsb_fps.setValue(30.0 if default_fps < 1.0 else 30.0)
        self.dsb_fps.valueChanged.connect(lambda _: self.cam.set_framerate(self.dsb_fps.value()))
        grid.addWidget(QtWidgets.QLabel("Framerate (FPS)"), r, 0); grid.addWidget(self.dsb_fps, r, 1); r += 1

        # Metering
        self.cmb_meter = QtWidgets.QComboBox(); self.cmb_meter.addItems(["centre", "spot", "matrix"])
        self.cmb_meter.currentTextChanged.connect(self.cam.set_metering)
        grid.addWidget(QtWidgets.QLabel("Metering"), r, 0); grid.addWidget(self.cmb_meter, r, 1); r += 1

        # Flicker avoidance
        self.cmb_flicker = QtWidgets.QComboBox(); self.cmb_flicker.addItems(["off", "auto", "manual"])
        self.sp_flicker_hz = QtWidgets.QSpinBox(); self.sp_flicker_hz.setRange(10, 1000); self.sp_flicker_hz.setValue(50)
        self.cmb_flicker.currentTextChanged.connect(self._apply_flicker)
        self.sp_flicker_hz.valueChanged.connect(self._apply_flicker)
        grid.addWidget(QtWidgets.QLabel("Flicker"), r, 0); grid.addWidget(self.cmb_flicker, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Flicker Hz"), r, 0); grid.addWidget(self.sp_flicker_hz, r, 1); r += 1

        # AWB
        self.cmb_awb = QtWidgets.QComboBox(); self.cmb_awb.addItems(["auto", "tungsten", "fluorescent", "indoor", "daylight", "cloudy"])
        self.cmb_awb.currentTextChanged.connect(self.cam.set_awb_mode)
        grid.addWidget(QtWidgets.QLabel("AWB Mode"), r, 0); grid.addWidget(self.cmb_awb, r, 1); r += 1

        self.dsb_awb_r = QtWidgets.QDoubleSpinBox(); self.dsb_awb_r.setRange(0.1, 8.0); self.dsb_awb_r.setSingleStep(0.05); self.dsb_awb_r.setValue(2.0)
        self.dsb_awb_b = QtWidgets.QDoubleSpinBox(); self.dsb_awb_b.setRange(0.1, 8.0); self.dsb_awb_b.setSingleStep(0.05); self.dsb_awb_b.setValue(2.0)
        wb_row = QtWidgets.QHBoxLayout(); wb_row.addWidget(QtWidgets.QLabel("AWB Gains R/B")); wb_row.addWidget(self.dsb_awb_r); wb_row.addWidget(self.dsb_awb_b)
        self.btn_awb_lock = QtWidgets.QPushButton("Lock gains (manual)")
        self.btn_awb_lock.clicked.connect(lambda: self.cam.set_awb_gains(self.dsb_awb_r.value(), self.dsb_awb_b.value()))
        vbox_awb = QtWidgets.QVBoxLayout(); vbox_awb.addLayout(wb_row); vbox_awb.addWidget(self.btn_awb_lock)
        grid.addWidget(QtWidgets.QLabel(""), r, 0); grid.addLayout(vbox_awb, r, 1); r += 1

        # Focus
        self.cmb_af = QtWidgets.QComboBox(); self.cmb_af.addItems(["auto", "continuous", "manual"])
        self.cmb_af.currentTextChanged.connect(self.cam.set_focus_mode)
        grid.addWidget(QtWidgets.QLabel("AF Mode"), r, 0); grid.addWidget(self.cmb_af, r, 1); r += 1

        self.dsb_dioptre = QtWidgets.QDoubleSpinBox(); self.dsb_dioptre.setRange(0.0, 10.0); self.dsb_dioptre.setDecimals(2); self.dsb_dioptre.setSingleStep(0.05)
        self.dsb_dioptre.setValue(0.0)
        self.dsb_dioptre.valueChanged.connect(lambda v: self.cam.set_lens_position(v))
        self.btn_af_trigger = QtWidgets.QPushButton("AF Trigger")
        self.btn_af_trigger.clicked.connect(self.cam.af_trigger)
        af_row = QtWidgets.QHBoxLayout(); af_row.addWidget(self.dsb_dioptre); af_row.addWidget(self.btn_af_trigger)
        grid.addWidget(QtWidgets.QLabel("Lens (dioptres)"), r, 0); grid.addLayout(af_row, r, 1); r += 1

        # Image tuning
        self.dsb_brightness = QtWidgets.QDoubleSpinBox(); self.dsb_brightness.setRange(-1.0, 1.0); self.dsb_brightness.setSingleStep(0.05); self.dsb_brightness.setValue(0.0)
        self.dsb_contrast   = QtWidgets.QDoubleSpinBox(); self.dsb_contrast.setRange(0.0, 2.0); self.dsb_contrast.setSingleStep(0.05); self.dsb_contrast.setValue(1.0)
        self.dsb_saturation = QtWidgets.QDoubleSpinBox(); self.dsb_saturation.setRange(0.0, 2.0); self.dsb_saturation.setSingleStep(0.05); self.dsb_saturation.setValue(1.0)
        self.dsb_sharpness  = QtWidgets.QDoubleSpinBox(); self.dsb_sharpness.setRange(0.0, 2.0); self.dsb_sharpness.setSingleStep(0.05); self.dsb_sharpness.setValue(1.0)
        self.cmb_denoise = QtWidgets.QComboBox(); self.cmb_denoise.addItems(["off", "fast", "high_quality"])
        for w in (self.dsb_brightness, self.dsb_contrast, self.dsb_saturation, self.dsb_sharpness):
            w.valueChanged.connect(self._apply_tuning)
        self.cmb_denoise.currentTextChanged.connect(self._apply_tuning)
        grid.addWidget(QtWidgets.QLabel("Brightness"), r, 0); grid.addWidget(self.dsb_brightness, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Contrast"),   r, 0); grid.addWidget(self.dsb_contrast,   r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Saturation"), r, 0); grid.addWidget(self.dsb_saturation, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Sharpness"),  r, 0); grid.addWidget(self.dsb_sharpness,  r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Denoise"),    r, 0); grid.addWidget(self.cmb_denoise,    r, 1); r += 1

        # Zoom (ScalerCrop)
        self.dsb_zoom = QtWidgets.QDoubleSpinBox(); self.dsb_zoom.setRange(1.0, 5.0); self.dsb_zoom.setDecimals(2); self.dsb_zoom.setSingleStep(0.1); self.dsb_zoom.setValue(1.0)
        self.dsb_zoom.valueChanged.connect(lambda v: self.cam.set_zoom(v))
        grid.addWidget(QtWidgets.QLabel("Digital Zoom (×)"), r, 0); grid.addWidget(self.dsb_zoom, r, 1); r += 1

        # Display flips (UI level)
        self.chk_flip_h = QtWidgets.QCheckBox("Flip H (display only)")
        self.chk_flip_v = QtWidgets.QCheckBox("Flip V (display only)")
        flip_row = QtWidgets.QHBoxLayout(); flip_row.addWidget(self.chk_flip_h); flip_row.addWidget(self.chk_flip_v); flip_row.addStretch(1)
        grid.addWidget(QtWidgets.QLabel("View flips"), r, 0); grid.addLayout(flip_row, r, 1); r += 1

        # HDR (Pi 5 HdrMode)
        self.cmb_hdr = QtWidgets.QComboBox(); self.cmb_hdr.addItems(["off", "single", "multi", "night", "unmerged"])
        self.cmb_hdr.currentTextChanged.connect(self.cam.set_pi5_hdr_mode)
        grid.addWidget(QtWidgets.QLabel("Pi5 HDR mode"), r, 0); grid.addWidget(self.cmb_hdr, r, 1); r += 1

        # Reset
        self.btn_reset = QtWidgets.QPushButton("Reset tuning")
        self.btn_reset.clicked.connect(self._reset_defaults)
        grid.addWidget(self.btn_reset, r, 0, 1, 2); r += 1

        self.setLayout(grid)
        self._reset_defaults()

    # ---- API for MainWindow ----
    def set_enabled_by_mode(self, training: bool):
        """Enable all live-tune controls only in training mode."""
        for child in self.findChildren((QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QCheckBox, QtWidgets.QPushButton)):
            # Keep view-flip checkboxes always enabled (display-only)
            if isinstance(child, QtWidgets.QCheckBox) and child.text().startswith("Flip"):
                child.setEnabled(True)
                continue
            if child is self.btn_reset:
                child.setEnabled(training)
                continue
            child.setEnabled(training if child is not self.btn_awb_lock else training)
        # Freeze AE state visually (no change to current camera state)
        return

    # ---- Slots ----
    def _on_ae_toggled(self, enabled: bool):
        self.cam.set_auto_exposure(enabled)
        self.sp_exposure.setEnabled(not enabled)
        self.dsb_gain.setEnabled(not enabled)
        if enabled:
            # AE ON: no-op (AE will overwrite exposure/gain)
            pass
        else:
            self._apply_manual_exposure()

    def _apply_manual_exposure(self):
        if self.chk_ae.isChecked():
            return
        self.cam.set_manual_exposure(self.sp_exposure.value(), self.dsb_gain.value())

    def _apply_flicker(self):
        mode = self.cmb_flicker.currentText()
        hz = int(self.sp_flicker_hz.value())
        self.cam.set_flicker_avoidance(mode, hz if mode == "manual" else None)

    def _apply_tuning(self):
        self.cam.set_image_adjustments(
            brightness=self.dsb_brightness.value(),
            contrast=self.dsb_contrast.value(),
            saturation=self.dsb_saturation.value(),
            sharpness=self.dsb_sharpness.value(),
            denoise=self.cmb_denoise.currentText(),
        )

    def _reset_defaults(self):
        self._building = True
        self.chk_ae.setChecked(True); self._on_ae_toggled(True)
        self.cmb_meter.setCurrentText("matrix"); self.cam.set_metering("matrix")
        self.cmb_flicker.setCurrentText("auto"); self._apply_flicker()
        self.cmb_awb.setCurrentText("auto"); self.cam.set_awb_mode("auto")
        self.cmb_af.setCurrentText("continuous"); self.cam.set_focus_mode("continuous")
        self.dsb_dioptre.setValue(0.0)
        self.dsb_brightness.setValue(0.0); self.dsb_contrast.setValue(1.0)
        self.dsb_saturation.setValue(1.0); self.dsb_sharpness.setValue(1.0)
        self.cmb_denoise.setCurrentText("fast"); self._apply_tuning()
        self.dsb_zoom.setValue(1.0)
        self.cmb_hdr.setCurrentText("off"); self.cam.set_pi5_hdr_mode("off")
        self.dsb_fps.setValue(30.0); self.cam.set_framerate(30.0)
        self._building = False


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
        self.btn_capture = QtWidgets.QPushButton("Capture Rect ROI → Active")
        self.btn_capture_poly = QtWidgets.QPushButton("Capture Poly ROI → Active")
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
        left.addWidget(self.video, 1)

        # Collapsible camera tray under monitor
        self.cam_panel = CameraPanel(self.cam)
        self.tray = CollapsibleTray("Camera Controls")
        body = QtWidgets.QVBoxLayout()
        body.setContentsMargins(8, 4, 8, 4)
        body.addWidget(self.cam_panel)
        self.tray.set_body_layout(body)
        left.addWidget(self.tray)

        root.addLayout(left, 2)

        # Right: Control stack (Mode, Publisher, Processing, Templates, Detections)
        right = QtWidgets.QVBoxLayout(); right.setSpacing(10)

        # Mode card
        mode_box = QtWidgets.QGroupBox("Mode")
        mode_h = QtWidgets.QHBoxLayout(mode_box)
        self.rad_training = QtWidgets.QRadioButton("Training"); self.rad_trigger = QtWidgets.QRadioButton("Trigger")
        self.rad_training.setChecked(True)
        self.btn_trigger = QtWidgets.QPushButton("TRIGGER Detect + Publish"); self.btn_trigger.setEnabled(False)
        mode_h.addWidget(self.rad_training); mode_h.addWidget(self.rad_trigger); mode_h.addStretch(1); mode_h.addWidget(self.btn_trigger)
        right.addWidget(mode_box)

        # Publisher
        net_box = QtWidgets.QGroupBox("Publisher")
        net_form = QtWidgets.QFormLayout(net_box)
        self.ed_ip = QtWidgets.QLineEdit("127.0.0.1")
        self.sp_port = QtWidgets.QSpinBox(); self.sp_port.setRange(1, 65535); self.sp_port.setValue(40001)
        self.chk_publish = QtWidgets.QCheckBox("Publish JSON over UDP")
        net_form.addRow("IP", self.ed_ip); net_form.addRow("Port", self.sp_port); net_form.addRow(self.chk_publish)
        right.addWidget(net_box)

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
        self.tbl = QtWidgets.QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(["Obj", "Inst", "x", "y", "θ(deg)", "score", "inliers", "center"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
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

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(0)

    # ---- UI callbacks ----
    def _on_mode_radio(self):
        self.mode = "training" if self.rad_training.isChecked() else "trigger"
        self.btn_trigger.setEnabled(self.mode == "trigger")
        # Freeze camera panel when in trigger mode
        self.cam_panel.set_enabled_by_mode(training=self.mode == "training")
        # In training mode, never send JSON automatically (same rule as before)

    def _on_trigger_once(self):
        if self.last_frame_rgb is None:
            return
        proc = self._proc_frame_copy()
        overlay, objects_report, _ = self.multi.compute_all(proc, draw=True)
        self._last_overlay = overlay.copy()
        self._overlay_until = time.time() + 1.0
        self._populate_table(objects_report)
        if self.publish_enabled:
            payload = self._build_payload(objects_report, proc.shape[1], proc.shape[0], timings={"detect_ms": 0.0, "draw_ms": 0.0})
            self.publisher.send(payload)

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

    # ---- ROI capture (rect) ----
    def _on_rect_selected(self, rect: QtCore.QRect):
        if self.last_frame_rgb is None:
            return
        frame = self._proc_frame_copy()  # downscaled BGR (processed)
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
        if self.last_frame_rgb is None or not qpoints:
            return
        frame = self._proc_frame_copy()  # downscaled BGR (processed)
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
        # PiCam3 returns RGB; keep it for display, convert to BGR for processing
        return self.cam.get_frame()

    def _proc_frame_copy(self) -> np.ndarray:
        rgb = self.last_frame_rgb
        if rgb is None:
            return np.zeros((480, 640, 3), np.uint8)
        # Display-level flips from camera panel
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
        # detection expects BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # optional CLAHE/Blur controlled by matcher (we just toggle flags)
        return bgr

    def _update(self):
        t_all0 = time.perf_counter()
        # Grab
        frame_rgb = self._grab()
        if frame_rgb is None:
            return
        self.last_frame_rgb = frame_rgb
        self.frame_id += 1

        # Prepare display frame (processed path also applies flips/rotation/resize)
        proc_bgr = self._proc_frame_copy()

        # Mode
        if self.mode == "training":
            self.frame_count += 1
            if (self.frame_count % max(1, int(self.sp_every.value()))) == 0:
                t2 = time.perf_counter()
                overlay, objects_report, _ = self.multi.compute_all(proc_bgr, draw=True)
                _ = (time.perf_counter() - t2) * 1000.0
                disp_bgr = overlay
                self._populate_table(objects_report)
            else:
                disp_bgr = proc_bgr
        else:
            if time.time() < self._overlay_until and self._last_overlay is not None:
                disp_bgr = self._last_overlay
            else:
                disp_bgr = proc_bgr

        # Display (BGR->RGB)
        disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(disp_rgb.data, disp_rgb.shape[1], disp_rgb.shape[0],
                            disp_rgb.strides[0], QtGui.QImage.Format_RGB888)
        self.video.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qimg))

        # FPS
        dt = time.perf_counter() - t_all0
        fps = 1.0 / max(1e-6, dt)
        self._fps_acc += fps; self._fps_n += 1
        if self._fps_n >= 10:
            self.lbl_fps.setText(f"FPS: {self._fps_acc / self._fps_n:.1f}")
            self._fps_acc = 0.0; self._fps_n = 0

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

    # ---- Cleanup ----
    def closeEvent(self, event: QtGui.QCloseEvent):
        try: self.timer.stop()
        except Exception: pass
        try: self.publisher.close()
        except Exception: pass
        try: self.cam.stop()
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
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
