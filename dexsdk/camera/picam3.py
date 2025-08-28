# ================================
# FILE: dexsdk/camera/picam3.py
# ================================
"""
High-level wrapper for Raspberry Pi Camera Module 3 (IMX708)
using Picamera2 + libcamera controls.

Tested on Raspberry Pi OS (Bookworm) with Pi 5.

Notes:
- For *sensor* HDR (CM3 built-in HDR), enable/disable *before* creating Picamera2
  via set_imx708_sensor_hdr(camera_num, enabled). See the classmethod below.
- For Pi 5 HDR accumulation (TDN), use set_pi5_hdr_mode().
- Query supported controls at runtime: cam.controls_info()
- Query sensor modes: cam.list_sensor_modes()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import time
import numpy as np
import cv2

from picamera2 import Picamera2, Preview # type: ignore
from libcamera import Transform, Rectangle, controls # type: ignore

# Some enums live under controls.draft on certain builds; make safe access helpers.
try:
    from libcamera.controls import draft as controls_draft  # type: ignore
except Exception:  # pragma: no cover - fallback if draft is not present
    controls_draft = None  # type: ignore


@dataclass
class SensorMode:
    size: Tuple[int, int]
    bit_depth: int
    fps: float
    crop_limits: Tuple[int, int, int, int]
    format: str
    unpacked: Optional[str]
    exposure_limits: Tuple[int, int]


class PiCam3:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: Optional[float] = 30.0,
        pixel_format: str = "RGB888",
        camera_num: int = 0,
        vflip: int = 0,
        hflip: int = 0,
        preview: Optional[str] = None,  # "qt", "qtgl", "drm", or None
    ) -> None:
        self.picam = Picamera2(camera_num)
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.camera_num = camera_num

        # Build a preview configuration by default
        config = self.picam.create_preview_configuration(
            main={"size": (width, height), "format": pixel_format},
            transform=Transform(vflip=vflip, hflip=hflip),
        )

        # If fps requested, embed FrameDurationLimits into the configuration controls
        if fps:
            frame_us = int(1_000_000 / fps)
            config["controls"] = {"FrameDurationLimits": (frame_us, frame_us)}

        self.picam.configure(config)

        # Select preview backend if asked
        if preview:
            self._start_preview(preview)

        self.picam.start()
        time.sleep(0.2)

    # -------------------------- Lifecycle --------------------------
    def _start_preview(self, backend: str) -> None:
        backend = backend.lower()
        if backend == "qtgl":
            self.picam.start_preview(Preview.QTGL)
        elif backend == "qt":
            self.picam.start_preview(Preview.QT)
        elif backend == "drm":
            self.picam.start_preview(Preview.DRM)
        else:
            raise ValueError("preview must be one of: 'qt', 'qtgl', 'drm'")

    def stop(self) -> None:
        try:
            self.picam.stop_preview()
        except Exception:
            pass
        self.picam.stop()

    # -------------------------- Queries ---------------------------
    def controls_info(self) -> Dict[str, Tuple[Union[int, float], Union[int, float], Union[int, float]]]:
        """Return dict of supported controls -> (min, max, default)."""
        return dict(self.picam.camera_controls)

    def properties_info(self) -> Dict[str, Union[int, float, str, Tuple[int, ...]]]:
        """Return read-only camera properties (model, PixelArraySize, etc.)."""
        return dict(self.picam.camera_properties)

    def list_sensor_modes(self) -> List[SensorMode]:
        """Return parsed list of available sensor modes for this camera."""
        modes: List[SensorMode] = []
        for m in self.picam.sensor_modes:
            modes.append(
                SensorMode(
                    size=tuple(m.get("size")) if isinstance(m.get("size"), (list, tuple)) else m.get("size"),
                    bit_depth=int(m.get("bit_depth")),
                    fps=float(m.get("fps")),
                    crop_limits=tuple(m.get("crop_limits")),
                    format=str(m.get("format")),
                    unpacked=str(m.get("unpacked")) if m.get("unpacked") else None,
                    exposure_limits=tuple(m.get("exposure_limits")),
                )
            )
        return modes

    # -------------------------- Capture ---------------------------
    def get_frame(self) -> np.ndarray:
        """Capture and return an RGB array from the main stream."""
        frame = self.picam.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def capture_file(self, path: str, size: Optional[Tuple[int, int]] = None) -> None:
        """Switch to still config (optionally at a different size) and save to file."""
        still_cfg = self.picam.create_still_configuration(
            main={"size": size or (self.width, self.height), "format": self.pixel_format}
        )
        # Apply still config only for the capture
        self.picam.switch_mode_and_capture_file(still_cfg, path)

    # -------------------------- Focus -----------------------------
    def set_focus_mode(self, mode: str = "auto") -> None:
        mode_map = {
            "auto": controls.AfModeEnum.Auto,
            "continuous": controls.AfModeEnum.Continuous,
            "manual": controls.AfModeEnum.Manual,
        }
        if mode not in mode_map:
            raise ValueError("mode must be 'auto'|'continuous'|'manual'")
        self.picam.set_controls({"AfMode": mode_map[mode]})

    def set_lens_position(self, dioptres: float) -> None:
        """Set lens position in dioptres (0.0=infinity, larger=closer)."""
        self.picam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(dioptres)})

    def af_trigger(self) -> None:
        """Trigger a single autofocus scan (when in 'Auto' AF mode)."""
        try:
            self.picam.set_controls({"AfMode": controls.AfModeEnum.Auto, "AfTrigger": controls.AfTriggerEnum.Start})
        except Exception:
            # AfTrigger may not be available on all stacks; ignore gracefully.
            pass

    # -------------------------- Exposure / Gain -------------------
    def set_auto_exposure(self, enable: bool = True) -> None:
        self.picam.set_controls({"AeEnable": bool(enable)})

    def set_exposure_mode(self, mode: str = "normal") -> None:
        emap = {
            "normal": controls.AeExposureModeEnum.Normal,
            "short": controls.AeExposureModeEnum.Short,
            "long": controls.AeExposureModeEnum.Long,
        }
        self.picam.set_controls({"AeExposureMode": emap[mode]})

    def set_metering(self, mode: str = "centre") -> None:
        mmap = {
            "centre": controls.AeMeteringModeEnum.CentreWeighted,
            "spot": controls.AeMeteringModeEnum.Spot,
            "matrix": controls.AeMeteringModeEnum.Matrix,
        }
        self.picam.set_controls({"AeMeteringMode": mmap[mode]})

    def set_flicker_avoidance(self, mode: str = "off", period_hz: Optional[int] = None) -> None:
        fmap = {
            "off": controls.AeFlickerModeEnum.FlickerOff,
            "manual": controls.AeFlickerModeEnum.FlickerManual,
            "auto": controls.AeFlickerModeEnum.FlickerAuto,
        }
        updates = {"AeFlickerMode": fmap[mode]}
        if mode == "manual" and period_hz:
            updates["AeFlickerPeriod"] = int(1_000_000 // period_hz)
        self.picam.set_controls(updates)

    def set_manual_exposure(self, exposure_us: int, gain: float) -> None:
        """Disable AE and set exposure/gain explicitly."""
        self.picam.set_controls({
            "AeEnable": False,
            "ExposureTime": int(exposure_us),
            "AnalogueGain": float(gain),
        })

    def set_framerate(self, fps: float) -> None:
        frame_us = int(1_000_000 / fps)
        self.picam.set_controls({"FrameDurationLimits": (frame_us, frame_us)})

    # -------------------------- White Balance ---------------------
    def set_awb_mode(self, mode: str = "auto") -> None:
        wmap = {
            "auto": controls.AwbModeEnum.Auto,
            "tungsten": controls.AwbModeEnum.Tungsten,
            "fluorescent": controls.AwbModeEnum.Fluorescent,
            "indoor": controls.AwbModeEnum.Indoor,
            "daylight": controls.AwbModeEnum.Daylight,
            "cloudy": controls.AwbModeEnum.Cloudy,
        }
        self.picam.set_controls({"AwbEnable": True, "AwbMode": wmap[mode]})

    def set_awb_gains(self, red_gain: float, blue_gain: float) -> None:
        self.picam.set_controls({"AwbEnable": False, "ColourGains": (float(red_gain), float(blue_gain))})

    # -------------------------- Image Tuning ----------------------
    def set_image_adjustments(
        self,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        sharpness: Optional[float] = None,
        denoise: Optional[str] = None,  # off|fast|high_quality
    ) -> None:
        updates: Dict[str, Union[int, float]] = {}
        if brightness is not None:
            updates["Brightness"] = float(brightness)
        if contrast is not None:
            updates["Contrast"] = float(contrast)
        if saturation is not None:
            updates["Saturation"] = float(saturation)
        if sharpness is not None:
            updates["Sharpness"] = float(sharpness)
        if denoise is not None and controls_draft is not None:
            dmap = {
                "off": controls_draft.NoiseReductionModeEnum.Off,
                "fast": controls_draft.NoiseReductionModeEnum.Fast,
                "high_quality": controls_draft.NoiseReductionModeEnum.HighQuality,
            }
            updates["NoiseReductionMode"] = dmap[denoise]
        if updates:
            self.picam.set_controls(updates)

    # -------------------------- Digital Zoom (ScalerCrop) ---------
    def set_zoom(self, zoom_factor: float, center: Optional[Tuple[float, float]] = None) -> None:
        """
        Digital zoom using ScalerCrop. zoom_factor=1.0 means full FOV; 2.0 halves the FOV.
        center: normalized (cx, cy) in [0,1] relative to full sensor, default = (0.5, 0.5).
        """
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be > 0")
        props = self.picam.camera_properties
        full_w, full_h = props.get("PixelArraySize", (self.width, self.height))
        cx, cy = center if center else (0.5, 0.5)
        crop_w = int(full_w / zoom_factor)
        crop_h = int(full_h / zoom_factor)
        x = max(0, min(int(cx * full_w - crop_w / 2), full_w - crop_w))
        y = max(0, min(int(cy * full_h - crop_h / 2), full_h - crop_h))
        self.picam.set_controls({"ScalerCrop": Rectangle(x, y, crop_w, crop_h)})

    # -------------------------- Sensor Mode -----------------------
    def reconfigure_sensor(self, output_size: Tuple[int, int], bit_depth: int = 10) -> None:
        """Stop, reconfigure to a specific sensor mode (size + bit depth), then restart."""
        was_running = True
        try:
            self.picam.stop()
        except Exception:
            was_running = False
        cfg = self.picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": self.pixel_format},
            sensor={"output_size": tuple(output_size), "bit_depth": int(bit_depth)},
        )
        self.picam.configure(cfg)
        if was_running:
            self.picam.start()
            time.sleep(0.2)

    # -------------------------- HDR -------------------------------
    @staticmethod
    def set_imx708_sensor_hdr(camera_num: int = 0, enable: bool = True) -> None:
        """
        Enable/disable Camera Module 3 sensor HDR *before* creating Picamera2.
        Call this when *no* Picamera2 instance for that camera is open.
        """
        try:
            from picamera2.devices.imx708 import IMX708 # type: ignore
        except ImportError as e:
            raise ImportError("IMX708 module not found. Ensure 'picamera2' is installed with IMX708 support.") from e
        with IMX708(camera_num) as cam:
            cam.set_sensor_hdr_mode(bool(enable))

    def set_pi5_hdr_mode(self, mode: str = "off") -> None:
        """Pi 5 HDR accumulation via HdrMode (requires Pi 5)."""
        hmap = {
            "off": controls.HdrModeEnum.Off,
            "single": controls.HdrModeEnum.SingleExposure,
            "multi": controls.HdrModeEnum.MultiExposure,
            "night": controls.HdrModeEnum.Night,
            "unmerged": controls.HdrModeEnum.MultiExposureUnmerged,
        }
        self.picam.set_controls({"HdrMode": hmap[mode]})


# ------------------------------ Quick demo ------------------------------
if __name__ == "__main__":
    cam = PiCam3(width=640, height=480, fps=30, preview=None)
    try:
        print("Model:", cam.properties_info().get("Model"))
        print("Controls available:", list(cam.controls_info().keys())[:10], "...")
        print("Sensor modes (first 3):")
        for m in cam.list_sensor_modes()[:3]:
            print(m)

        # Example tweaks
        cam.set_focus_mode("continuous")
        cam.set_metering("matrix")
        cam.set_awb_mode("daylight")
        cam.set_image_adjustments(brightness=0.0, contrast=1.0)
        cam.set_framerate(30)

        frame = cam.get_frame()
        print("Captured frame:", frame.shape)
    finally:
        cam.stop()
