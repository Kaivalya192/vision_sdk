from __future__ import annotations

import base64
import json
from typing import Dict, Iterable, List

import cv2


class Streamer:
    def __init__(self, jpeg_quality: int = 80, prefer_b64: bool = True):
        self.enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        # For compatibility with existing clients, keep base64 by default
        self.prefer_b64 = bool(prefer_b64)

    def encode_jpeg(self, bgr) -> bytes:
        ok, buf = cv2.imencode(".jpg", bgr, self.enc_param)
        if not ok:
            return b""
        return buf.tobytes()

    async def broadcast_frame(self, clients: Iterable, bgr, width: int, height: int, *, loop):
        jpeg = await loop.run_in_executor(None, lambda: self.encode_jpeg(bgr))
        if not jpeg:
            return
        if self.prefer_b64:
            payload = {
                "type": "frame",
                "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
                "w": width,
                "h": height,
            }
            txt = json.dumps(payload, separators=(",", ":"))
            for ws in list(clients):
                try:
                    await ws.send(txt)
                except Exception:
                    try:
                        clients.discard(ws)
                    except Exception:
                        pass
        else:
            # future: binary frames; keep JSON-only for compatibility
            payload = {
                "type": "frame",
                "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
                "w": width,
                "h": height,
            }
            txt = json.dumps(payload, separators=(",", ":"))
            for ws in list(clients):
                try:
                    await ws.send(txt)
                except Exception:
                    try:
                        clients.discard(ws)
                    except Exception:
                        pass

    async def broadcast_json(self, clients: Iterable, obj: Dict):
        txt = json.dumps(obj, separators=(",", ":"))
        for ws in list(clients):
            try:
                await ws.send(txt)
            except Exception:
                try:
                    clients.discard(ws)
                except Exception:
                    pass

