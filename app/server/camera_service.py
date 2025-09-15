from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import ViewConfig


def _import_camera_class():
    try:
        from dexsdk.camera.picam3 import PiCam3  # type: ignore

        return PiCam3
    except Exception:
        # Fallback: alias Webcam as PiCam3-like
        try:
            from dexsdk.camera.webcam import Webcam  # type: ignore

            class PiCam3Shim(Webcam):  # type: ignore
                def controls_info(self):
                    return {}

                # No-op control methods to preserve API surface
                def set_framerate(self, *_args, **_kw):
                    pass

                def set_metering(self, *_args, **_kw):
                    pass

                def set_auto_exposure(self, *_args, **_kw):
                    pass

                def set_manual_exposure(self, *_args, **_kw):
                    pass

                def set_awb_mode(self, *_args, **_kw):
                    pass

                def set_awb_gains(self, *_args, **_kw):
                    pass

                def set_focus_mode(self, *_args, **_kw):
                    pass

                def set_lens_position(self, *_args, **_kw):
                    pass

                def af_trigger(self, *_args, **_kw):
                    pass

                def set_image_adjustments(self, *_args, **_kw):
                    pass

                def set_pi5_hdr_mode(self, *_args, **_kw):
                    pass

                def set_zoom(self, *_args, **_kw):
                    pass

            return PiCam3Shim
        except Exception as e:  # pragma: no cover
            raise e


PiCam3Class = _import_camera_class()


class CameraService:
    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0):
        self.cam = PiCam3Class(width=width, height=height, fps=fps, preview=None)

    def stop(self):
        try:
            self.cam.stop()
        except Exception:
            pass

    def get_frame(self, view: ViewConfig) -> np.ndarray:
        rgb = self.cam.get_frame()  # RGB
        if view.flip_h:
            rgb = cv2.flip(rgb, 1)
        if view.flip_v:
            rgb = cv2.flip(rgb, 0)
        if view.rot_quadrant:
            from dexsdk.utils import rotate90

            rgb = rotate90(rgb, int(view.rot_quadrant) % 4)
        # down-scale
        if view.proc_width and rgb.shape[1] != view.proc_width:
            h = int(rgb.shape[0] * (view.proc_width / rgb.shape[1]))
            rgb = cv2.resize(rgb, (view.proc_width, h), interpolation=cv2.INTER_AREA)
        return rgb

