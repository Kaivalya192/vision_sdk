# a pp/server/measure_service.py
from __future__ import annotations
import base64
import json
from typing import Dict, Any

import cv2
import numpy as np

from dexsdk.measure.schema import Job, ROI
from dexsdk.measure.core import run_job


class MeasureService:
    def __init__(self, camera_service, calib_store=None):
        self.camera = camera_service
        self.calib_store = calib_store  # optional future: px→mm scaling

    def handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        # msg: {"type":"run_measure","job":{...},"anchor":bool}
        job = msg.get("job", {})
        frame = self.camera.last_frame_bgr()
        if frame is None:
            return {"type": "ack", "ok": False, "cmd": "run_measure", "error": "no frame"}

        try:
            roi_list = job.get("roi", [0,0,frame.shape[1], frame.shape[0]])
            roi = ROI(x=int(roi_list[0]), y=int(roi_list[1]),
                      w=int(roi_list[2]), h=int(roi_list[3]))
            j = Job(tool=job["tool"], roi=roi, params=job.get("params", {}))
        except Exception as e:
            return {"type": "ack", "ok": False, "cmd": "run_measure", "error": f"bad job: {e}"}

        # TODO: if you have plane_scale → convert units="mm"
        units = "px"
        pkt = run_job(frame, j, units=units)

        # overlay jpeg
        ok, enc = cv2.imencode(".jpg", pkt.overlay if pkt.overlay is not None else frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        overlay_b64 = base64.b64encode(enc.tobytes()).decode("ascii") if ok else None

        out = {
            "type": "measures",
            "packet": {
                "units": pkt.units,
                "measures": [dict(id=m.id, kind=m.kind, value=float(m.value),
                                  passed=None if m.passed is None else bool(m.passed),
                                  sigma=float(m.sigma)) for m in pkt.measures],
            }
        }
        if overlay_b64:
            out["overlay_jpeg_b64"] = overlay_b64
        return out
