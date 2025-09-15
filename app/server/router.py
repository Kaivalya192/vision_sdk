from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, Optional

from websockets.server import WebSocketServerProtocol

from .types import WSMessage


Handler = Callable[[WebSocketServerProtocol, WSMessage], Awaitable[None]]


class CommandRouter:
    def __init__(self):
        self._handlers: Dict[str, Handler] = {}

    def register(self, msg_type: str, handler: Handler) -> None:
        self._handlers[msg_type.strip().lower()] = handler

    async def dispatch(self, ws: WebSocketServerProtocol, msg: WSMessage) -> None:
        t = str(msg.get("type", "")).lower()
        if t in self._handlers:
            await self._handlers[t](ws, msg)
        else:
            await self.send_acked(ws, msg, ok=False, error="unknown command")

    async def send_acked(
        self,
        ws: WebSocketServerProtocol,
        req: Optional[WSMessage],
        *,
        cmd: Optional[str] = None,
        ok: bool = True,
        error: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        payload = {"type": "ack", "cmd": cmd or (req.get("type") if req else None), "ok": ok, "error": error}
        if req and "req_id" in req:
            payload["req_id"] = req["req_id"]
        if extra:
            payload.update(extra)
        try:
            await ws.send(__import__("json").dumps(payload))
        except Exception:
            pass

