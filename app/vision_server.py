#!/usr/bin/env python3
"""
RPi Vision Server — with LED support & manual-WB gains

• Training mode → LEDs always ON
• Trigger mode  → LED ON 200 ms pre-flash, capture next frame, LED OFF 100 ms post-flash

Manual WB fix: if params carry {"awb_mode":"manual", "awb_rb":[R,B]} we call
PiCam3.set_awb_gains(R,B) (auto-WB disabled).

Run:
  pip install websockets==12.* opencv-python numpy
  python app/vision_server.py

This server streams JPEG frames and detection results over WebSockets and
accepts JSON commands for camera settings, viewing, and template management.
"""

import asyncio
import base64
import json
import signal
import socket
import time
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from dexsdk.camera.picam3 import PiCam3
from dexsdk.detect import MultiTemplateMatcher
from dexsdk.utils import rotate90


# ---------------------------------------------------------------------------
JPEG_QUAL = 80
DEFAULT_PROC_WIDTH = 640
DEFAULT_PUBLISH_EVERY = 1
MAX_SLOTS = 5
WS_HOST, WS_PORT = "0.0.0.0", 8765

# --- LED parameters --------------------------------------------------------
LED_HOST, LED_PORT = "127.0.0.1", 12345  # TCP NeoPixel micro-server
PRE_MS, POST_MS = 200, 100  # timings for trigger flashes
# ---------------------------------------------------------------------------


class VisionServer:
    # ---------- life-cycle ----------
    def __init__(self):
        self.cam = PiCam3(width=640, height=480, fps=30.0, preview=None)
        self.multi = MultiTemplateMatcher(max_slots=MAX_SLOTS, min_center_dist_px=40)

        # runtime
        self.clients: Set[WebSocketServerProtocol] = set()
        self.mode = "training"  # training | trigger
        self.proc_width = DEFAULT_PROC_WIDTH
        self.publish_every = DEFAULT_PUBLISH_EVERY
        self.every_counter = 0
        self.view = dict(flip_h=False, flip_v=False, rot_quadrant=0)
        self.session_id = hex(int(time.time() * 1000))[-8:]
        self.frame_id = 0
        self.last_overlay_bgr: Optional[np.ndarray] = None
        self.overlay_until = 0.0
        self.trigger_flag = False  # set by UI
        self.req_stop = False
        self.last_proc_bgr: Optional[np.ndarray] = None

        # LED state machine
        self._led_training_on = False
        self._led_state = False  # physical state
        self._phase = "idle"  # idle|pre|capture|post
        self._phase_t0 = 0.0

        # ensure LEDs ON at start (training default)
        self._led_set_training(True)

    # ---------- LED helpers ----------
    def _led_send(self, on: bool):
        if on == self._led_state:
            return
        try:
            with socket.create_connection((LED_HOST, LED_PORT), timeout=0.2) as s:
                s.sendall(b"1" if on else b"0")
                try:
                    s.recv(32)
                except Exception:
                    pass
            self._led_state = on
        except Exception:
            # Silent failure – avoids crashing if LED service unreachable
            self._led_state = on  # assume success to avoid hammering

    def _led_set_training(self, training: bool):
        if training and not self._led_training_on:
            self._led_send(True)
            self._led_training_on = True
        elif not training and self._led_training_on:
            self._led_send(False)
            self._led_training_on = False

    # ---------- public API ----------
    async def start(self):
        async with websockets.serve(
            self._ws_handler, WS_HOST, WS_PORT, ping_interval=None, max_size=16 * 1024 * 1024
        ):
            print(f"[vision-server] ws://{WS_HOST}:{WS_PORT}")
            await self._run_pipeline()

    # ---------- WebSocket ----------
    async def _ws_handler(self, ws: WebSocketServerProtocol):
        self.clients.add(ws)
        await ws.send(json.dumps({"type": "hello", "session_id": self.session_id}))
        print(f"[ws] +{ws.remote_address}  ({len(self.clients)})")
        try:
            async for txt in ws:
                try:
                    msg = json.loads(txt)
                except Exception:
                    continue
                await self._handle_msg(ws, msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            print(f"[ws] -{ws.remote_address}  ({len(self.clients)})")

    async def _handle_msg(self, ws: WebSocketServerProtocol, msg: Dict):
        t = str(msg.get("type", "")).lower()
        if t == "ping":
            await ws.send(json.dumps({"type": "pong", "ts_ms": int(time.time() * 1000)}))
        elif t == "set_mode":
            self.mode = msg.get("mode", "training").lower()
            self.mode = "training" if self.mode not in ("training", "trigger") else self.mode
            self._led_set_training(self.mode == "training")
            await self._ack(ws, "set_mode")
        elif t == "trigger":
            self.trigger_flag = True
            await self._ack(ws, "trigger")
        elif t == "set_proc_width":
            self.proc_width = max(64, int(msg.get("width", DEFAULT_PROC_WIDTH)))
            await self._ack(ws, "set_proc_width")
        elif t == "set_publish_every":
            self.publish_every = max(1, int(msg.get("n", DEFAULT_PUBLISH_EVERY)))
            await self._ack(ws, "set_publish_every")
        elif t == "set_params":
            await self._apply_cam_params(msg.get("params", {}))
            await self._ack(ws, "set_params")
        elif t == "set_view":
            self.view.update({k: msg.get(k, self.view[k]) for k in ("flip_h", "flip_v", "rot_quadrant")})
            self.view["rot_quadrant"] = int(self.view["rot_quadrant"]) % 4
            await self._ack(ws, "set_view")
        elif t == "add_template_rect":
            await self._add_template_rect(ws, msg)
        elif t == "add_template_poly":
            await self._add_template_poly(ws, msg)
        elif t == "clear_template":
            self.multi.clear(int(msg.get("slot", 0)))
            await self._ack(ws, "clear_template")
        elif t == "set_slot_state":
            s = int(msg.get("slot", 0))
            self.multi.set_enabled(s, bool(msg.get("enabled", True)))
            self.multi.set_max_instances(s, int(msg.get("max_instances", 3)))
            await self._ack(ws, "set_slot_state")
        elif t == "af_trigger":
            try:
                self.cam.af_trigger()
                await asyncio.sleep(3.0)
                try:
                    dioptre = float(self.cam.picam.capture_metadata().get("LensPosition", 0.0))
                except Exception:
                    dioptre = None
                await self._ack(ws, "af_trigger", ok=True, extra={"dioptre": dioptre})
            except Exception as e:
                await self._ack(ws, "af_trigger", ok=False, err=str(e))
        else:
            await self._ack(ws, t, ok=False, err="unknown command")

    async def _ack(self, ws, cmd, ok=True, err=None, extra: Optional[Dict] = None):
        msg = {"type": "ack", "cmd": cmd, "ok": ok, "error": err}
        if extra:
            msg.update(extra)
        try:
            await ws.send(json.dumps(msg))
        except Exception:
            pass

    # ---------- camera params (manual-WB fix) ----------
    async def _apply_cam_params(self, p: Dict):
        if not p:
            return
        # FPS
        if "fps" in p:
            try:
                self.cam.set_framerate(float(p["fps"]))
            except Exception:
                pass

        # Exposure
        if not p.get("auto_exposure", True):
            try:
                self.cam.set_manual_exposure(int(p.get("exposure_us", 6000)), float(p.get("gain", 2.0)))
            except Exception:
                pass
        else:
            try:
                self.cam.set_auto_exposure(True)
            except Exception:
                pass

        # AWB / manual gains
        awb_mode = p.get("awb_mode", "auto")
        if awb_mode == "manual" or ("awb_rb" in p and not p.get("auto_exposure", False)):
            rg, bg = p.get("awb_rb", [2.0, 2.0])
            try:
                self.cam.set_awb_gains(float(rg), float(bg))
            except Exception:
                pass
        else:
            try:
                self.cam.set_awb_mode(awb_mode)
            except Exception:
                pass

        # Focus
        if p.get("focus_mode") == "manual":
            try:
                self.cam.set_lens_position(float(p.get("dioptre", 0.0)))
            except Exception:
                pass

        # Image tuning
        try:
            self.cam.set_image_adjustments(
                brightness=p.get("brightness"),
                contrast=p.get("contrast"),
                saturation=p.get("saturation"),
                sharpness=p.get("sharpness"),
                denoise=p.get("denoise"),
            )
        except Exception:
            pass

    # ---------- template helpers (unchanged) ----------
    async def _add_template_rect(self, ws, msg):
        slot = int(msg.get("slot", 0))
        name = str(msg.get("name", f"Obj{slot+1}"))
        try:
            x, y, w, h = [int(v) for v in msg.get("rect", [0, 0, 0, 0])]
            if self.last_proc_bgr is None:
                raise RuntimeError("no frame")
            fh, fw = self.last_proc_bgr.shape[:2]
            x = max(0, min(fw - 1, x))
            y = max(0, min(fh - 1, y))
            w = max(1, min(fw - x, w))
            h = max(1, min(fh - y, h))
            if w < 10 or h < 10:
                raise ValueError("ROI too small")
            roi = self.last_proc_bgr[y : y + h, x : x + w].copy()
            self.multi.add_or_replace(slot, name, roi_bgr=roi)
            await self._ack(ws, "add_template_rect")
        except Exception as e:
            await self._ack(ws, "add_template_rect", ok=False, err=str(e))

    async def _add_template_poly(self, ws, msg):
        slot = int(msg.get("slot", 0))
        name = str(msg.get("name", f"Obj{slot+1}"))
        pts = msg.get("points", [])
        try:
            if self.last_proc_bgr is None:
                raise RuntimeError("no frame")
            arr = np.asarray(pts, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
                raise ValueError("invalid points")
            fh, fw = self.last_proc_bgr.shape[:2]
            x1, y1 = np.floor(arr.min(axis=0)).astype(int)
            x2, y2 = np.ceil(arr.max(axis=0)).astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(fw, x2)
            y2 = min(fh, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                raise ValueError("ROI too small")
            mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
            cv2.fillPoly(mask, [(arr - [x1, y1]).astype(np.int32)], 255)
            roi = self.last_proc_bgr[y1:y2, x1:x2].copy()
            self.multi.add_or_replace_polygon(slot, name, roi_bgr=roi, roi_mask=mask)
            await self._ack(ws, "add_template_poly")
        except Exception as e:
            await self._ack(ws, "add_template_poly", ok=False, err=str(e))

    # ---------- pipeline ----------
    async def _run_pipeline(self):
        loop = asyncio.get_running_loop()
        enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUAL]

        while not self.req_stop:
            t0 = time.perf_counter()

            # ---------- LED trigger state machine ----------
            if self.mode == "trigger":
                if self.trigger_flag and self._phase == "idle":
                    # start pre-flash
                    self._led_send(True)
                    self._phase = "pre"
                    self._phase_t0 = time.time()
                    self.trigger_flag = False

                elif self._phase == "pre" and (time.time() - self._phase_t0) * 1000 >= PRE_MS:
                    self._phase = "capture"  # capture on this iteration

                elif self._phase == "post" and (time.time() - self._phase_t0) * 1000 >= POST_MS:
                    self._led_send(False)
                    self._phase = "idle"

            # keep LEDs steady in training
            if self.mode == "training":
                self._led_set_training(True)
            else:
                self._led_set_training(False)

            # ---------- capture ----------
            rgb = self.cam.get_frame()

            if self.view["flip_h"]:
                rgb = cv2.flip(rgb, 1)
            if self.view["flip_v"]:
                rgb = cv2.flip(rgb, 0)
            if self.view["rot_quadrant"]:
                rgb = rotate90(rgb, int(self.view["rot_quadrant"]))

            # down-scale
            if self.proc_width and rgb.shape[1] != self.proc_width:
                h = int(rgb.shape[0] * (self.proc_width / rgb.shape[1]))
                rgb_small = cv2.resize(rgb, (self.proc_width, h), interpolation=cv2.INTER_AREA)
            else:
                rgb_small = rgb
            bgr_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
            self.last_proc_bgr = bgr_small

            # ---------- detection decision ----------
            detect_now = False
            if self.mode == "training":
                detect_now = (self.every_counter % self.publish_every) == 0
            elif self._phase == "capture":
                detect_now = True
                self._phase = "post"
                self._phase_t0 = time.time()

            overlay_bgr = None
            objs: List[Dict] = []
            if detect_now:
                overlay_bgr, objs, _ = self.multi.compute_all(bgr_small, draw=True)

            self.every_counter += 1
            self.frame_id += 1

            # ---------- send to clients ----------
            if self.clients:
                send_bgr = (
                    overlay_bgr
                    if overlay_bgr is not None
                    else (
                        self.last_overlay_bgr
                        if time.time() < self.overlay_until and self.last_overlay_bgr is not None
                        else bgr_small
                    )
                )
                if overlay_bgr is not None:
                    self.last_overlay_bgr = overlay_bgr
                    self.overlay_until = time.time() + 1.0

                jpeg = await loop.run_in_executor(
                    None, lambda: cv2.imencode(".jpg", send_bgr, enc_param)[1].tobytes()
                )
                jpeg_b64 = base64.b64encode(jpeg).decode("ascii")
                await self._broadcast(
                    {"type": "frame", "jpeg_b64": jpeg_b64, "w": send_bgr.shape[1], "h": send_bgr.shape[0]}
                )
                if detect_now:
                    await self._broadcast(
                        {"type": "detections", "payload": {"result": {"objects": objs}}, "overlay_jpeg_b64": jpeg_b64}
                    )

            # ---------- pace ----------
            await asyncio.sleep(max(0, (1 / 30.0) - (time.perf_counter() - t0)))

    async def _broadcast(self, obj: Dict):
        txt = json.dumps(obj, separators=(",", ":"))
        for ws in list(self.clients):
            try:
                await ws.send(txt)
            except Exception:
                self.clients.discard(ws)

    # ---------- graceful exit ----------
    def stop(self):
        self.req_stop = True
        self._led_send(False)
        try:
            self.cam.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
def main():
    srv = VisionServer()
    loop = asyncio.get_event_loop()

    def _sig(*_):
        srv.stop()
        for t in list(asyncio.all_tasks(loop)):
            t.cancel()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _sig)
        except Exception:
            pass
    try:
        loop.run_until_complete(srv.start())
    except asyncio.CancelledError:
        pass
    finally:
        srv.stop()
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()


if __name__ == "__main__":
    main()

