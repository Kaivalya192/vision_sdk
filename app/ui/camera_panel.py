from typing import Tuple
from PyQt5 import QtCore, QtWidgets


class CameraPanel(QtWidgets.QWidget):
    """Binds UI controls to PiCam3-like setter methods. Call set_enabled_by_mode() on mode changes."""

    def __init__(self, cam, parent=None):
        super().__init__(parent)
        self.cam = cam
        self._building = False  # guard to avoid re-entrancy

        info = cam.controls_info()

        def rng(key, default: Tuple[int, int, int]):
            try:
                lo, hi, _def = info.get(key, default)
                return int(lo), int(hi), int(_def if _def is not None else (lo + hi) // 2)
            except Exception:
                return default

        exp_lo, exp_hi, _ = rng("ExposureTime", (100, 33000, 5000))
        gain_lo, gain_hi, _ = rng("AnalogueGain", (1, 16, 2))
        _fdl_lo, fdl_hi, _ = rng("FrameDurationLimits", (5000, 33333, 33333))

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        r = 0
        self.chk_ae = QtWidgets.QCheckBox("Auto Exposure")
        self.chk_ae.setChecked(True)
        self.chk_ae.toggled.connect(self._on_ae_toggled)
        grid.addWidget(self.chk_ae, r, 0, 1, 2)
        r += 1

        self.sp_exposure = QtWidgets.QSpinBox()
        self.sp_exposure.setRange(exp_lo, exp_hi)
        self.sp_exposure.setSuffix(" µs")
        self.sp_exposure.setSingleStep(100)
        self.sp_exposure.setValue(min(max(6000, exp_lo), exp_hi))
        self.sp_exposure.valueChanged.connect(self._apply_manual_exposure)
        grid.addWidget(QtWidgets.QLabel("Exposure"), r, 0)
        grid.addWidget(self.sp_exposure, r, 1)
        r += 1

        self.dsb_gain = QtWidgets.QDoubleSpinBox()
        self.dsb_gain.setRange(float(gain_lo), float(gain_hi))
        self.dsb_gain.setDecimals(2)
        self.dsb_gain.setSingleStep(0.05)
        self.dsb_gain.setValue(2.0)
        self.dsb_gain.valueChanged.connect(self._apply_manual_exposure)
        grid.addWidget(QtWidgets.QLabel("Analogue Gain"), r, 0)
        grid.addWidget(self.dsb_gain, r, 1)
        r += 1

        self.dsb_fps = QtWidgets.QDoubleSpinBox()
        self.dsb_fps.setRange(1.0, 120.0)
        self.dsb_fps.setDecimals(1)
        default_fps = round(1_000_000.0 / max(1, fdl_hi), 1)
        self.dsb_fps.setValue(30.0 if default_fps < 1.0 else 30.0)
        self.dsb_fps.valueChanged.connect(lambda _: self.cam.set_framerate(self.dsb_fps.value()))
        grid.addWidget(QtWidgets.QLabel("Framerate (FPS)"), r, 0)
        grid.addWidget(self.dsb_fps, r, 1)
        r += 1

        self.cmb_meter = QtWidgets.QComboBox()
        self.cmb_meter.addItems(["centre", "spot", "matrix"])
        self.cmb_meter.currentTextChanged.connect(self.cam.set_metering)
        grid.addWidget(QtWidgets.QLabel("Metering"), r, 0)
        grid.addWidget(self.cmb_meter, r, 1)
        r += 1

        self.cmb_flicker = QtWidgets.QComboBox()
        self.cmb_flicker.addItems(["off", "auto", "manual"])
        self.sp_flicker_hz = QtWidgets.QSpinBox()
        self.sp_flicker_hz.setRange(10, 1000)
        self.sp_flicker_hz.setValue(50)
        self.cmb_flicker.currentTextChanged.connect(self._apply_flicker)
        self.sp_flicker_hz.valueChanged.connect(self._apply_flicker)
        grid.addWidget(QtWidgets.QLabel("Flicker"), r, 0)
        grid.addWidget(self.cmb_flicker, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Flicker Hz"), r, 0)
        grid.addWidget(self.sp_flicker_hz, r, 1)
        r += 1

        self.cmb_awb = QtWidgets.QComboBox()
        self.cmb_awb.addItems(["auto", "tungsten", "fluorescent", "indoor", "daylight", "cloudy"])
        self.cmb_awb.currentTextChanged.connect(self.cam.set_awb_mode)
        grid.addWidget(QtWidgets.QLabel("AWB Mode"), r, 0)
        grid.addWidget(self.cmb_awb, r, 1)
        r += 1

        self.dsb_awb_r = QtWidgets.QDoubleSpinBox()
        self.dsb_awb_r.setRange(0.1, 8.0)
        self.dsb_awb_r.setSingleStep(0.05)
        self.dsb_awb_r.setValue(2.0)
        self.dsb_awb_b = QtWidgets.QDoubleSpinBox()
        self.dsb_awb_b.setRange(0.1, 8.0)
        self.dsb_awb_b.setSingleStep(0.05)
        self.dsb_awb_b.setValue(2.0)
        wb_row = QtWidgets.QHBoxLayout()
        wb_row.addWidget(QtWidgets.QLabel("AWB Gains R/B"))
        wb_row.addWidget(self.dsb_awb_r)
        wb_row.addWidget(self.dsb_awb_b)
        self.btn_awb_lock = QtWidgets.QPushButton("Lock gains (manual)")
        self.btn_awb_lock.clicked.connect(
            lambda: self.cam.set_awb_gains(self.dsb_awb_r.value(), self.dsb_awb_b.value())
        )
        vbox_awb = QtWidgets.QVBoxLayout()
        vbox_awb.addLayout(wb_row)
        vbox_awb.addWidget(self.btn_awb_lock)
        grid.addWidget(QtWidgets.QLabel(""), r, 0)
        grid.addLayout(vbox_awb, r, 1)
        r += 1

        self.cmb_af = QtWidgets.QComboBox()
        self.cmb_af.addItems(["auto", "continuous", "manual"])
        self.cmb_af.currentTextChanged.connect(self.cam.set_focus_mode)
        grid.addWidget(QtWidgets.QLabel("AF Mode"), r, 0)
        grid.addWidget(self.cmb_af, r, 1)
        r += 1

        self.dsb_dioptre = QtWidgets.QDoubleSpinBox()
        self.dsb_dioptre.setRange(0.0, 10.0)
        self.dsb_dioptre.setDecimals(2)
        self.dsb_dioptre.setSingleStep(0.05)
        self.dsb_dioptre.setValue(0.0)
        self.dsb_dioptre.valueChanged.connect(lambda v: self.cam.set_lens_position(v))
        self.btn_af_trigger = QtWidgets.QPushButton("AF Trigger")
        self.btn_af_trigger.clicked.connect(self.cam.af_trigger)
        af_row = QtWidgets.QHBoxLayout()
        af_row.addWidget(self.dsb_dioptre)
        af_row.addWidget(self.btn_af_trigger)
        grid.addWidget(QtWidgets.QLabel("Lens (dioptres)"), r, 0)
        grid.addLayout(af_row, r, 1)
        r += 1

        self.dsb_brightness = QtWidgets.QDoubleSpinBox()
        self.dsb_brightness.setRange(-1.0, 1.0)
        self.dsb_brightness.setSingleStep(0.05)
        self.dsb_brightness.setValue(0.0)
        self.dsb_contrast = QtWidgets.QDoubleSpinBox()
        self.dsb_contrast.setRange(0.0, 2.0)
        self.dsb_contrast.setSingleStep(0.05)
        self.dsb_contrast.setValue(1.0)
        self.dsb_saturation = QtWidgets.QDoubleSpinBox()
        self.dsb_saturation.setRange(0.0, 2.0)
        self.dsb_saturation.setSingleStep(0.05)
        self.dsb_saturation.setValue(1.0)
        self.dsb_sharpness = QtWidgets.QDoubleSpinBox()
        self.dsb_sharpness.setRange(0.0, 2.0)
        self.dsb_sharpness.setSingleStep(0.05)
        self.dsb_sharpness.setValue(1.0)
        self.cmb_denoise = QtWidgets.QComboBox()
        self.cmb_denoise.addItems(["off", "fast", "high_quality"])
        for w in (
            self.dsb_brightness,
            self.dsb_contrast,
            self.dsb_saturation,
            self.dsb_sharpness,
        ):
            w.valueChanged.connect(self._apply_tuning)
        self.cmb_denoise.currentTextChanged.connect(self._apply_tuning)
        grid.addWidget(QtWidgets.QLabel("Brightness"), r, 0)
        grid.addWidget(self.dsb_brightness, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Contrast"), r, 0)
        grid.addWidget(self.dsb_contrast, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Saturation"), r, 0)
        grid.addWidget(self.dsb_saturation, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Sharpness"), r, 0)
        grid.addWidget(self.dsb_sharpness, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Denoise"), r, 0)
        grid.addWidget(self.cmb_denoise, r, 1)
        r += 1

        self.dsb_zoom = QtWidgets.QDoubleSpinBox()
        self.dsb_zoom.setRange(1.0, 5.0)
        self.dsb_zoom.setDecimals(2)
        self.dsb_zoom.setSingleStep(0.1)
        self.dsb_zoom.setValue(1.0)
        self.dsb_zoom.valueChanged.connect(lambda v: self.cam.set_zoom(v))
        grid.addWidget(QtWidgets.QLabel("Digital Zoom (×)"), r, 0)
        grid.addWidget(self.dsb_zoom, r, 1)
        r += 1

        self.chk_flip_h = QtWidgets.QCheckBox("Flip H (display only)")
        self.chk_flip_v = QtWidgets.QCheckBox("Flip V (display only)")
        flip_row = QtWidgets.QHBoxLayout()
        flip_row.addWidget(self.chk_flip_h)
        flip_row.addWidget(self.chk_flip_v)
        flip_row.addStretch(1)
        grid.addWidget(QtWidgets.QLabel("View flips"), r, 0)
        grid.addLayout(flip_row, r, 1)
        r += 1

        self.cmb_hdr = QtWidgets.QComboBox()
        self.cmb_hdr.addItems(["off", "single", "multi", "night", "unmerged"])
        self.cmb_hdr.currentTextChanged.connect(self.cam.set_pi5_hdr_mode)
        grid.addWidget(QtWidgets.QLabel("Pi5 HDR mode"), r, 0)
        grid.addWidget(self.cmb_hdr, r, 1)
        r += 1

        self.btn_reset = QtWidgets.QPushButton("Reset tuning")
        self.btn_reset.clicked.connect(self._reset_defaults)
        grid.addWidget(self.btn_reset, r, 0, 1, 2)

        self.setLayout(grid)
        self._reset_defaults()

    def set_enabled_by_mode(self, training: bool):
        for child in self.findChildren(
            (
                QtWidgets.QAbstractSpinBox,
                QtWidgets.QComboBox,
                QtWidgets.QCheckBox,
                QtWidgets.QPushButton,
            )
        ):
            if isinstance(child, QtWidgets.QCheckBox) and child.text().startswith("Flip"):
                child.setEnabled(True)
                continue
            if child is self.btn_reset:
                child.setEnabled(training)
                continue
            child.setEnabled(training if child is not self.btn_awb_lock else training)
        return

    def _on_ae_toggled(self, enabled: bool):
        self.cam.set_auto_exposure(enabled)
        self.sp_exposure.setEnabled(not enabled)
        self.dsb_gain.setEnabled(not enabled)
        if enabled:
            pass
        else:
            self._apply_manual_exposure()

    def _apply_manual_exposure(self):
        if self.chk_ae.isChecked():
            return
        self.cam.set_manual_exposure(self.sp_exposure.value(), self.dsb_gain.value())

    def _apply_flicker(self):
        pass

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
        self.chk_ae.setChecked(False)
        self._on_ae_toggled(False)
        self.cmb_meter.setCurrentText("matrix")
        self.cam.set_metering("matrix")
        self.cmb_flicker.setCurrentText("auto")
        self._apply_flicker()
        self.cmb_awb.setCurrentText("auto")
        self.cam.set_awb_mode("auto")
        self.cmb_af.setCurrentText("manual")
        self.cam.set_focus_mode("manual")
        self.dsb_dioptre.setValue(0.0)
        self.dsb_brightness.setValue(0.0)
        self.dsb_contrast.setValue(1.0)
        self.dsb_saturation.setValue(1.0)
        self.dsb_sharpness.setValue(1.0)
        self.cmb_denoise.setCurrentText("fast")
        self._apply_tuning()
        self.dsb_zoom.setValue(1.0)
        self.cmb_hdr.setCurrentText("off")
        self.cam.set_pi5_hdr_mode("off")
        self.dsb_fps.setValue(30.0)
        self.cam.set_framerate(30.0)
        self._building = False

