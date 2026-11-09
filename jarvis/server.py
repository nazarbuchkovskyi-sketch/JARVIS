"""Local-only bridge between the Jarvis core and its window (127.0.0.1).

Jarvis listens only while its window is connected; when the last window closes,
the whole process shuts down.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Callable, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import UI_DIST, Config
from .core import Jarvis

log = logging.getLogger("jarvis.server")

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
DEV_UI_PORT = 5173  # `npm run dev`
CLOSE_GRACE_SECONDS = 5.0  # a page reload reconnects well within this
FIRST_WINDOW_SECONDS = 120.0  # no window ever connected (Edge failed to open): exit


def _local_host(host_header: str) -> bool:
    """Host header must name this machine (defends against DNS rebinding)."""
    if not host_header:
        return False
    host = host_header.rsplit(":", 1)[0] if not host_header.startswith("[") else host_header[1:].split("]")[0]
    return host in LOCAL_HOSTS


def allowed_origins(port: int) -> set[str]:
    out = set()
    for host in ("127.0.0.1", "localhost"):
        out.add(f"http://{host}:{port}")
        out.add(f"http://{host}:{DEV_UI_PORT}")
    return out


def _origin_ok(origin: str, port: int) -> bool:
    """Only the Jarvis window itself (or its dev server) — not any other local page."""
    if not origin:
        return False
    p = urlsplit(origin)
    return f"{p.scheme}://{p.hostname}:{p.port}" in allowed_origins(port)


def create_app(config: Config, request_exit: Optional[Callable[[], None]] = None,
               expect_window: bool = True) -> FastAPI:
    jarvis = Jarvis(config)
    port = int(config["port"])
    clients = {"n": 0, "seen": False}
    exit_task: dict[str, Optional[asyncio.Task]] = {"t": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        jarvis.start()
        watchdog = None
        if expect_window and request_exit is not None:
            async def first_window() -> None:
                await asyncio.sleep(FIRST_WINDOW_SECONDS)
                if not clients["seen"]:
                    log.warning("the window never connected: shutting down")
                    request_exit()
            watchdog = asyncio.create_task(first_window())
        yield
        if watchdog is not None:
            watchdog.cancel()
        jarvis.shutdown()

    app = FastAPI(lifespan=lifespan, title="Jarvis")
    app.state.jarvis = jarvis

    @app.middleware("http")
    async def only_local(request: Request, call_next):
        if not _local_host(request.headers.get("host", "")):
            return PlainTextResponse("forbidden", status_code=403)
        return await call_next(request)

    async def _exit_if_still_closed() -> None:
        await asyncio.sleep(CLOSE_GRACE_SECONDS)
        if clients["n"] == 0:
            log.info("window closed: shutting down")
            if request_exit is not None:
                request_exit()

    def _window_changed() -> None:
        jarvis.set_window_open(clients["n"] > 0)
        if clients["n"] > 0:
            if exit_task["t"] is not None:
                exit_task["t"].cancel()
                exit_task["t"] = None
        elif clients["seen"] and request_exit is not None:
            exit_task["t"] = asyncio.create_task(_exit_if_still_closed())

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        # Only the local Jarvis window may connect (blocks other websites and other local apps).
        client = ws.client.host if ws.client else ""
        if client not in LOCAL_HOSTS or not _origin_ok(ws.headers.get("origin", ""), port) or not _local_host(ws.headers.get("host", "")):
            await ws.close(code=1008)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=500)

        def push(event: dict) -> None:
            def put() -> None:
                if q.full() and event.get("type") in ("levels", "stats"):
                    return  # drop meters rather than block on a slow window
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass
            loop.call_soon_threadsafe(put)

        unsubscribe = jarvis.bus.subscribe(push)
        clients["n"] += 1
        clients["seen"] = True
        _window_changed()
        await ws.send_text(json.dumps(jarvis.snapshot(), ensure_ascii=False))

        async def sender() -> None:
            while True:
                event = await q.get()
                await ws.send_text(json.dumps(event, ensure_ascii=False))

        async def reply_system(text: str) -> None:
            await ws.send_text(json.dumps({"type": "message", "role": "system", "text": text, "ts": loop.time()}, ensure_ascii=False))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                if not isinstance(msg, dict):
                    continue
                kind = msg.get("type")
                if kind == "text":
                    jarvis.submit(str(msg.get("text", ""))[:4000])
                elif kind == "stop":
                    await asyncio.to_thread(jarvis.stop)
                elif kind == "ptt":
                    # Fire and forget: never hold up the receive loop (Stop must stay instant).
                    loop.run_in_executor(None, jarvis.ptt_event, bool(msg.get("active")))
                elif kind == "approve":
                    jarvis.resolve_approval(str(msg.get("id", "")), bool(msg.get("allow")))
                elif kind == "set":
                    try:
                        await asyncio.to_thread(jarvis.apply_setting, str(msg.get("key")), msg.get("value"))
                    except (KeyError, ValueError, TypeError, OSError) as exc:
                        await reply_system(f"Настройка отклонена: {exc}")
        except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
            pass
        finally:
            unsubscribe()
            send_task.cancel()
            clients["n"] -= 1
            _window_changed()

    @app.post("/api/approve")
    async def approve(request: Request) -> JSONResponse:
        """Called by the Claude Code PreToolUse hook (jarvis/approve_hook.py)."""
        token = request.headers.get("x-jarvis-token", "")
        if not hmac.compare_digest(token, jarvis.approval_token):
            return JSONResponse({"decision": "deny", "reason": "bad token"}, status_code=403)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"decision": "deny", "reason": "bad request"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"decision": "deny", "reason": "bad request"}, status_code=400)
        tool = str(body.get("tool_name", ""))
        tool_input = body.get("tool_input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        allowed, reason = await asyncio.to_thread(jarvis.request_approval, tool, tool_input)
        return JSONResponse({"decision": "allow" if allowed else "deny", "reason": reason})

    @app.get("/camera.mjpg")
    async def camera(request: Request):
        # Only the Jarvis window may embed the camera (not other websites).
        if request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
            return PlainTextResponse("forbidden", status_code=403)

        async def frames():
            seq = -1
            while True:
                if await request.is_disconnected():
                    return
                eyes = jarvis.eyes
                if eyes is None or not eyes.enabled:
                    await asyncio.sleep(0.5)
                    continue
                seq, jpeg = await asyncio.to_thread(eyes.latest, seq, 1.0)
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    if (UI_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
    else:
        @app.get("/")
        async def no_ui() -> HTMLResponse:
            return HTMLResponse("<h1>Jarvis</h1><p>Интерфейс не собран: выполните <code>npm install && npm run build</code> в папке ui.</p>")

    return app
