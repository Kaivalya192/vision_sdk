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
from .measure_service import MeasureService, AnchorHelper
from .robot_calib import RobotMap
from .calib_io import ensure_dir, save_k_json, load_k_json
from dexsdk.calib.single_view_intrinsics import estimate_intrinsics
from dexsdk.calib.plane_scale import compute_plane_scale
import os
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
        self.anchor = AnchorHelper()
        self._anchor_name = ""
        self._last_anchor_pose = None

        def _get_mm_scale():
            data = load_k_json(self._k_path) or {}
            try:
                ps = data.get("plane_scale") or {}
                sx = float(ps.get("mm_per_px_x"))
                sy = float(ps.get("mm_per_px_y"))
                if sx > 0 and sy > 0:
                    return (sx, sy), "mm"
            except Exception:
                pass
            return (1.0, 1.0), "px"

        self._k_path = os.path.expanduser("~/.vision_sdk/K.json")
        self._robot_path = os.path.expanduser("~/.vision_sdk/robot_map.json")
        ensure_dir(os.path.dirname(self._k_path))
        self.measure = MeasureService(_get_mm_scale)
        self.robot = RobotMap()
        # try load previous robot map
        try:
            self.robot.load(self._robot_path)
        except Exception:
            pass

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

        self.led = LEDStateMachine(LEDClient(LED_HOST, LED_PORT), pre_ms=PRE_MS, post_ms=POST_MS)

        self._register_handlers()

    def _register_handlers(self):
        r = self.router

        async def h_ping(ws, msg: WSMessage):
            await ws.send(json.dumps({"type": "pong", "ts_ms": int(time.time() * 1000), **({"req_id": msg["req_id"]} if "req_id" in msg else {})}))

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
            # apply camera parameters (same as monolith)
            try:
                if "fps" in p:
                    self.camera.cam.set_framerate(float(p["fps"]))
            except Exception:
                pass
            if not p.get("auto_exposure", True):
                try:
                    self.camera.cam.set_manual_exposure(int(p.get("exposure_us", 6000)), float(p.get("gain", 2.0)))
                except Exception:
                    pass
            else:
                try:
                    self.camera.cam.set_auto_exposure(True)
                except Exception:
                    pass
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
            if p.get("focus_mode") == "manual":
                try:
                    self.camera.cam.set_lens_position(float(p.get("dioptre", 0.0)))
                except Exception:
                    pass
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

        async def h_set_anchor_source(ws, msg: WSMessage):
            self._anchor_name = str(msg.get("object", "")).strip()
            # reset origin and last pose when source changes
            try:
                self.anchor.reset_origin()
            except Exception:
                pass
            self._last_anchor_pose = None
            await r.send_acked(ws, msg)

        async def h_run_measure(ws, msg: WSMessage):
            job = msg.get("job", {}) or {}
            use_anchor = bool(msg.get("anchor", False))
            frame = self.last_proc_bgr.copy() if self.last_proc_bgr is not None else None
            if frame is None:
                await r.send_acked(ws, msg, ok=False, error="no frame")
                return
            # Apply anchoring if requested
            roi = job.get("roi")
            if use_anchor and roi is not None and self._last_anchor_pose is not None:
                if self.anchor.origin_pose is None:
                    self.anchor.set_origin(self._last_anchor_pose, tuple(int(v) for v in roi))
                else:
                    self.anchor.update(self._last_anchor_pose)
                H, W = frame.shape[:2]
                new_roi = self.anchor.warp_rect(tuple(int(v) for v in (self.anchor.base_rect or tuple(roi))), W, H)
                job = {**job, "roi": list(new_roi)}

            loop = asyncio.get_running_loop()

            def _work():
                return self.measure.run_job(frame, job)

            packet, ov = await loop.run_in_executor(None, _work)
            # Broadcast result
            jpeg = await loop.run_in_executor(None, lambda: cv2.imencode(".jpg", ov, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUAL])[1].tobytes())
            import base64

            b64 = base64.b64encode(jpeg).decode("ascii")
            await self.stream.broadcast_json(self.clients, {"type": "measures", "packet": packet, "overlay_jpeg_b64": b64})
            await r.send_acked(ws, msg)

        r.register("set_anchor_source", h_set_anchor_source)
        r.register("run_measure", h_run_measure)

        # ---- One-click calibration ---------------------------------------
        async def h_calibrate(ws, msg: WSMessage):
            await r.send_acked(ws, msg)
            frame = self.last_proc_bgr.copy() if self.last_proc_bgr is not None else None
            if frame is None:
                await self.stream.broadcast_json(self.clients, {"type": "calibration", "ok": False, "error": "no frame"})
                return
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            loop = asyncio.get_running_loop()

            def _work():
                intr = estimate_intrinsics(gray)
                plane = compute_plane_scale(gray, K_json=None, cfg={"target_width": 0, "pair_weight": 0.5, "no_bias": False})
                payload = {
                    "model": intr.get("model"),
                    "K": intr.get("K"),
                    "dist": intr.get("dist"),
                    "image_size": intr.get("image_size"),
                    "rms_reprojection_error_px": intr.get("rms_reprojection_error_px"),
                    "rmse_px": intr.get("rmse_px"),
                    "plane_scale": plane.get("summary", {}).get("plane_scale", {}),
                    "created_at_ms": int(time.time() * 1000),
                }
                # overlay: draw tag corners if present
                ov = frame.copy()
                for d in plane.get("detections", []):
                    try:
                        pts = d.get("corners_px")
                        if pts is None:
                            continue
                        arr = np.array(pts, dtype=np.float32)
                        if arr.ndim == 3:
                            arr = arr.reshape(-1, 2)
                        if arr.shape[0] >= 4:
                            q = arr.astype(np.int32)
                            cv2.polylines(ov, [q], True, (0, 255, 0), 2)
                    except Exception:
                        pass
                ps = payload["plane_scale"]
                pxmmx = ps.get("px_per_mm_x"); pxmmy = ps.get("px_per_mm_y")
                mmppx = ps.get("mm_per_px_x"); mmppy = ps.get("mm_per_px_y")
                txt = f"px/mm X={pxmmx:.3f} Y={pxmmy:.3f}\nmm/px X={mmppx:.5f} Y={mmppy:.5f}" if ps else "no plane scale"
                y0 = 24
                for i, line in enumerate(txt.split("\n")):
                    cv2.putText(ov, line, (12, y0 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(ov, line, (12, y0 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                save_k_json(self._k_path, payload)
                return payload, ov

            try:
                payload, ov = await loop.run_in_executor(None, _work)
                jpeg = await loop.run_in_executor(None, lambda: cv2.imencode(".jpg", ov, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUAL])[1].tobytes())
                b64 = base64.b64encode(jpeg).decode("ascii")
                await self.stream.broadcast_json(self.clients, {"type": "calibration", "ok": True, "saved": self._k_path, "summary": payload, "overlay_jpeg_b64": b64})
            except Exception as e:
                await self.stream.broadcast_json(self.clients, {"type": "calibration", "ok": False, "error": str(e)})

        r.register("calibrate_one_click", h_calibrate)

        # ---- Robot calibration -------------------------------------------
        async def h_robot_pairs(ws, msg: WSMessage):
            op = str(msg.get("op", "")).lower()
            if op == "add":
                self.robot.add_pair(msg.get("robot", [0, 0]), msg.get("pixel", [0, 0]))
                self.robot.save(self._robot_path)
                await r.send_acked(ws, msg)
            elif op == "clear":
                self.robot.clear(); self.robot.save(self._robot_path)
                await r.send_acked(ws, msg)
            else:
                await r.send_acked(ws, msg, ok=False, error="unknown op")

        async def h_robot_solve(ws, msg: WSMessage):
            loop = asyncio.get_running_loop()
            try:
                params = await loop.run_in_executor(None, self.robot.solve)
                self.robot.save(self._robot_path)
                await self.stream.broadcast_json(self.clients, {"type": "robot_calibration", "ok": True, "params": params})
                await r.send_acked(ws, msg)
            except Exception as e:
                await self.stream.broadcast_json(self.clients, {"type": "robot_calibration", "ok": False, "error": str(e)})
                await r.send_acked(ws, msg, ok=False, error=str(e))

        async def h_robot_apply(ws, msg: WSMessage):
            try:
                px, py = msg.get("pixel", [0, 0])
                rx, ry = self.robot.apply(float(px), float(py))
                await self.stream.broadcast_json(self.clients, {"type": "robot_apply", "ok": True, "robot": [rx, ry]})
                await r.send_acked(ws, msg)
            except Exception as e:
                await self.stream.broadcast_json(self.clients, {"type": "robot_apply", "ok": False, "error": str(e)})
                await r.send_acked(ws, msg, ok=False, error=str(e))

        r.register("robot_pairs", h_robot_pairs)
        r.register("robot_solve", h_robot_solve)
        r.register("robot_apply", h_robot_apply)

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

            # LED state update; may request capture in trigger mode
            capture_event = self.led.update("training" if self.mode == "training" else "trigger")

            # capture RGB
            rgb = self.camera.get_frame(self.view)
            bgr_small = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.last_proc_bgr = bgr_small

            # detection decision
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
                # update anchor reference if selected present
                try:
                    name_sel = getattr(self, "_anchor_name", "")
                    if name_sel:
                        for o in objs:
                            if o.get("name") == name_sel and o.get("detections"):
                                # pick first
                                pose = o["detections"][0].get("pose") or {}
                                cur_pose = {
                                    "x": float(pose.get("x", 0.0)),
                                    "y": float(pose.get("y", 0.0)),
                                    "theta_deg": float(pose.get("theta_deg", pose.get("theta", 0.0))),
                                }
                                self._last_anchor_pose = cur_pose
                                # keep updating for origin-based anchoring
                                if self.anchor.origin_pose is not None:
                                    self.anchor.update(cur_pose)
                                break
                except Exception:
                    pass

            # broadcast
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

                await self.stream.broadcast_frame(self.clients, send_bgr, send_bgr.shape[1], send_bgr.shape[0], loop=loop)
                if detect_now:
                    await self.stream.broadcast_json(
                        self.clients, {"type": "detections", "payload": {"result": {"objects": objs}}, "overlay_jpeg_b64": None}
                    )

            self.every_counter += 1
            self.frame_id += 1

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
