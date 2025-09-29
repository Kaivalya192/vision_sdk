from __future__ import annotations

import asyncio
import json
import time
import signal
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from .camera_service import CameraService
from .detect_service import DetectService
from .led_service import LEDClient, LEDStateMachine
from .router import CommandRouter
from .streamer import Streamer
from .types import ViewConfig, WSMessage
import base64
from .color_service import ColorService
import base64

JPEG_QUAL = 80
DEFAULT_PROC_WIDTH = 640
DEFAULT_PUBLISH_EVERY = 1
MAX_SLOTS = 5
WS_HOST, WS_PORT = "0.0.0.0", 8765
LED_HOST, LED_PORT = "127.0.0.1", 12345
PRE_MS, POST_MS = 200, 100


class VisionServer:
    def __init__(self):
        self.camera = CameraService(width=640, height=480, fps=30.0)
        self.detect = DetectService(max_slots=MAX_SLOTS, min_center_dist_px=40)
        self.stream = Streamer(jpeg_quality=JPEG_QUAL, prefer_b64=True)
        self.router = CommandRouter()

        self.clients: Set[WebSocketServerProtocol] = set()
        self.mode = "training"  # training | trigger
        self.view = ViewConfig()
        self.publish_every = DEFAULT_PUBLISH_EVERY
        self.every_counter = 0
        self.session_id = hex(int(time.time() * 1000))[-8:]
        self.frame_id = 0
        self.last_overlay_bgr: Optional[np.ndarray] = None
        self.overlay_until = 0.0
        self.last_proc_bgr: Optional[np.ndarray] = None
        self.color = ColorService()
        self.color_stream_enabled = False

        self.led = LEDStateMachine(LEDClient(LED_HOST, LED_PORT), pre_ms=PRE_MS, post_ms=POST_MS)

        self._register_handlers()

    def _register_handlers(self):
        r = self.router

        async def h_ping(ws, msg: WSMessage):
            await ws.send(json.dumps({
                "type": "pong",
                "ts_ms": int(time.time() * 1000),
                **({"req_id": msg["req_id"]} if "req_id" in msg else {})
            }))

        async def h_set_mode(ws, msg: WSMessage):
            self.mode = str(msg.get("mode", "training")).lower()
            self.mode = "training" if self.mode not in ("training", "trigger") else self.mode
            await r.send_acked(ws, msg)

        async def h_trigger(ws, msg: WSMessage):
            self.led.trigger()
            await r.send_acked(ws, msg)

        async def h_set_proc_width(ws, msg: WSMessage):
            w = int(msg.get("width", DEFAULT_PROC_WIDTH))
            self.view.proc_width = max(64, w)
            await r.send_acked(ws, msg)

        async def h_set_publish_every(ws, msg: WSMessage):
            self.publish_every = max(1, int(msg.get("n", DEFAULT_PUBLISH_EVERY)))
            await r.send_acked(ws, msg)

        async def h_set_detection_params(ws, msg: WSMessage):
            params = msg.get("params", {})
            applied = self.detect.update_params(params)
            await r.send_acked(ws, msg, extra={"params": applied})

        async def h_set_params(ws, msg: WSMessage):
            p = msg.get("params", {})
            # fps
            try:
                if "fps" in p:
                    self.camera.cam.set_framerate(float(p["fps"]))
            except Exception:
                pass
            # exposure/gain
            if not p.get("auto_exposure", True):
                try:
                    self.camera.cam.set_manual_exposure(
                        int(p.get("exposure_us", 6000)),
                        float(p.get("gain", 2.0)),
                    )
                except Exception:
                    pass
            else:
                try:
                    self.camera.cam.set_auto_exposure(True)
                except Exception:
                    pass
            # awb
            awb_mode = p.get("awb_mode", "auto")
            if awb_mode == "manual" or ("awb_rb" in p and not p.get("auto_exposure", False)):
                rg, bg = p.get("awb_rb", [2.0, 2.0])
                try:
                    self.camera.cam.set_awb_gains(float(rg), float(bg))
                except Exception:
                    pass
            else:
                try:
                    self.camera.cam.set_awb_mode(awb_mode)
                except Exception:
                    pass
            # focus
            if p.get("focus_mode") == "manual":
                try:
                    self.camera.cam.set_lens_position(float(p.get("dioptre", 0.0)))
                except Exception:
                    pass
            # image adjustments
            try:
                self.camera.cam.set_image_adjustments(
                    brightness=p.get("brightness"),
                    contrast=p.get("contrast"),
                    saturation=p.get("saturation"),
                    sharpness=p.get("sharpness"),
                    denoise=p.get("denoise"),
                )
            except Exception:
                pass
            await r.send_acked(ws, msg)

        async def h_set_view(ws, msg: WSMessage):
            self.view.flip_h = bool(msg.get("flip_h", self.view.flip_h))
            self.view.flip_v = bool(msg.get("flip_v", self.view.flip_v))
            self.view.rot_quadrant = int(msg.get("rot_quadrant", self.view.rot_quadrant)) % 4
            await r.send_acked(ws, msg)

        async def h_add_template_rect(ws, msg: WSMessage):
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
                self.detect.add_or_replace(slot, name, roi_bgr=roi)
                await r.send_acked(ws, msg)
            except Exception as e:
                await r.send_acked(ws, msg, ok=False, error=str(e))

        async def h_add_template_poly(ws, msg: WSMessage):
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
                self.detect.add_or_replace_polygon(slot, name, roi_bgr=roi, roi_mask=mask)
                await r.send_acked(ws, msg)
            except Exception as e:
                await r.send_acked(ws, msg, ok=False, error=str(e))

        async def h_clear_template(ws, msg: WSMessage):
            self.detect.clear(int(msg.get("slot", 0)))
            await r.send_acked(ws, msg)

        async def h_set_slot_state(ws, msg: WSMessage):
            s = int(msg.get("slot", 0))
            self.detect.set_enabled(s, bool(msg.get("enabled", True)))
            self.detect.set_max_instances(s, int(msg.get("max_instances", 3)))
            await r.send_acked(ws, msg)

        async def h_af_trigger(ws, msg: WSMessage):
            ok = True
            err = None
            dioptre = None
            try:
                self.camera.cam.af_trigger()
                await asyncio.sleep(3.0)
                try:
                    dioptre = float(self.camera.cam.picam.capture_metadata().get("LensPosition", 0.0))
                except Exception:
                    dioptre = None
            except Exception as e:
                ok = False
                err = str(e)
            await r.send_acked(ws, msg, ok=ok, error=err, extra={"dioptre": dioptre})
        
        async def h_color_set_rules(ws, msg):
            applied = self.color.set_rules(
                msg.get("classes", []),
                kernel_size       = msg.get("kernel_size"),
                min_area_global   = msg.get("min_area_global"),
                open_iter         = msg.get("open_iter"),
                close_iter        = msg.get("close_iter"),
            )
            await r.send_acked(ws, msg, extra={"applied": applied})

        async def h_color_run_once(ws, msg):
            if self.last_proc_bgr is None:
                await r.send_acked(ws, msg, ok=False, error="no frame")
                return
            vis, dets, _ = self.color.run(self.last_proc_bgr)
            # overlay to b64 for UI
            overlay_b64 = None
            try:
                jpeg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUAL])[1].tobytes()
                overlay_b64 = base64.b64encode(jpeg).decode("ascii")
            except Exception:
                pass
            await self.stream.broadcast_json(self.clients, {
                "type": "color_results",
                "objects": dets,
                "overlay_jpeg_b64": overlay_b64
            })
            await r.send_acked(ws, msg)

        async def h_color_set_stream(ws, msg):
            self.color_stream_enabled = bool(msg.get("enabled", False))
            await r.send_acked(ws, msg, extra={"enabled": self.color_stream_enabled})

        async def h_color_clear_rules(ws, msg):
            self.color.set_rules([])
            await r.send_acked(ws, msg)

        r.register("color_set_rules",  h_color_set_rules)
        r.register("color_run_once",   h_color_run_once)
        r.register("color_set_stream", h_color_set_stream)
        r.register("color_clear_rules",h_color_clear_rules)

        # Register handlers
        r.register("ping", h_ping)
        r.register("set_mode", h_set_mode)
        r.register("trigger", h_trigger)
        r.register("set_proc_width", h_set_proc_width)
        r.register("set_publish_every", h_set_publish_every)
        r.register("set_detection_params", h_set_detection_params)
        r.register("set_params", h_set_params)
        r.register("set_view", h_set_view)
        r.register("add_template_rect", h_add_template_rect)
        r.register("add_template_poly", h_add_template_poly)
        r.register("clear_template", h_clear_template)
        r.register("set_slot_state", h_set_slot_state)
        r.register("af_trigger", h_af_trigger)

    async def start(self):
        async with websockets.serve(
            self._ws_handler, WS_HOST, WS_PORT, ping_interval=None, max_size=16 * 1024 * 1024
        ):
            print(f"[vision-server] ws://{WS_HOST}:{WS_PORT}")
            await self._run_pipeline()

    async def _ws_handler(self, ws: WebSocketServerProtocol):
        self.clients.add(ws)
        await ws.send(json.dumps({"type": "hello", "session_id": self.session_id}))
        try:
            async for txt in ws:
                try:
                    msg = json.loads(txt)
                except Exception:
                    continue
                await self.router.dispatch(ws, msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    async def _run_pipeline(self):
        loop = asyncio.get_running_loop()

        while True:
            t0 = time.perf_counter()

            # LED state; may request capture in trigger mode
            capture_event = self.led.update("training" if self.mode == "training" else "trigger")

            # capture RGB -> BGR (proc)
            rgb = self.camera.get_frame(self.view)
            bgr_small = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.last_proc_bgr = bgr_small

            # detection cadence
            detect_now = False
            if self.mode == "training":
                detect_now = (self.every_counter % self.publish_every) == 0
            elif capture_event:
                detect_now = True
                self.led.notify_captured()

            overlay_bgr = None
            objs: List[Dict] = []
            if detect_now:
                overlay_bgr, objs, _ = self.detect.compute_all(bgr_small, draw=True)

            # broadcast current frame (overlay if available / recent)
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

                await self.stream.broadcast_frame(
                    self.clients, send_bgr, send_bgr.shape[1], send_bgr.shape[0], loop=loop
                )

                if detect_now:
                    # also include a base64 overlay for UIs that want to swap immediately
                    overlay_b64 = None
                    if overlay_bgr is not None:
                        try:
                            jpeg = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUAL])[1].tobytes()
                            overlay_b64 = base64.b64encode(jpeg).decode("ascii")
                        except Exception:
                            overlay_b64 = None

                    await self.stream.broadcast_json(
                        self.clients,
                        {
                            "type": "detections",
                            "payload": {"result": {"objects": objs}},
                            "overlay_jpeg_b64": overlay_b64,
                        },
                    )
                # stream color overlay at camera FPS (lightweight)
                if self.clients and self.color_stream_enabled and self.last_proc_bgr is not None:
                    try:
                        vis, dets, _ = self.color.run(self.last_proc_bgr)
                        overlay_b64 = None
                        try:
                            jpeg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUAL])[1].tobytes()
                            overlay_b64 = base64.b64encode(jpeg).decode("ascii")
                        except Exception:
                            pass
                        await self.stream.broadcast_json(
                            self.clients, {"type": "color_results", "objects": dets, "overlay_jpeg_b64": overlay_b64}
                        )
                    except Exception:
                        # keep loop healthy
                        pass

            self.every_counter += 1
            self.frame_id += 1

            # ~30 fps pacing
            await asyncio.sleep(max(0, (1 / 30.0) - (time.perf_counter() - t0)))


def main():
    srv = VisionServer()
    loop = asyncio.get_event_loop()

    def _sig(*_):
        try:
            for t in list(asyncio.all_tasks(loop)):
                t.cancel()
        except Exception:
            pass

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _sig)
        except Exception:
            pass

    try:
        loop.run_until_complete(srv.start())
    except asyncio.CancelledError:
        pass
