#!/usr/bin/env python3
"""Local server: serves frontend + WebSocket proxy to Baseten with auth."""

import os
from pathlib import Path

import websockets
from aiohttp import web

BASETEN_API_KEY = os.environ.get("BASETEN_API_KEY", "")
BASETEN_WS_URL = os.environ.get("BASETEN_WS_URL", "")
PORT = int(os.environ.get("PORT", "8080"))
FRONTEND_DIR = Path(__file__).parent / "frontend"


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    headers = {"Authorization": f"Api-Key {BASETEN_API_KEY}"}

    try:
        async with websockets.connect(BASETEN_WS_URL, additional_headers=headers) as baseten_ws:
            print("[ws] Connected to Baseten")

            async def browser_to_baseten():
                try:
                    async for msg in ws:
                        if msg.type == web.WSMsgType.BINARY:
                            await baseten_ws.send(msg.data)
                        elif msg.type == web.WSMsgType.TEXT:
                            await baseten_ws.send(msg.data)
                        elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                            break
                except Exception:
                    pass

            async def baseten_to_browser():
                try:
                    async for msg in baseten_ws:
                        if isinstance(msg, bytes):
                            await ws.send_bytes(msg)
                        else:
                            await ws.send_str(msg)
                except Exception:
                    pass

            await asyncio.gather(browser_to_baseten(), baseten_to_browser())
    except Exception as e:
        print(f"[ws] Error: {e}")
    finally:
        print("[ws] Session ended")

    return ws


async def index_handler(request):
    return web.FileResponse(FRONTEND_DIR / "index.html")


def main():
    if not BASETEN_API_KEY or not BASETEN_WS_URL:
        print("Set BASETEN_API_KEY and BASETEN_WS_URL env vars")
        print("  cp .env.example .env && edit .env")
        return

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/", index_handler)
    app.router.add_static("/", FRONTEND_DIR, show_index=False)

    print(f"[server] http://localhost:{PORT}")
    print(f"[server] WebSocket proxy at ws://localhost:{PORT}/ws → Baseten")
    web.run_app(app, host="localhost", port=PORT)


if __name__ == "__main__":
    main()
