#!/usr/bin/env python3
"""Local WebSocket proxy that adds Baseten auth header.
Browser connects to ws://localhost:8765, proxy relays to Baseten with Authorization header.
"""

import asyncio
import os
import signal

import websockets

BASETEN_API_KEY = os.environ.get("BASETEN_API_KEY", "")
BASETEN_WS_URL = os.environ.get("BASETEN_WS_URL", "")
LOCAL_PORT = 8765


async def proxy_handler(browser_ws):
    print(f"[proxy] Browser connected")

    headers = {"Authorization": f"Api-Key {BASETEN_API_KEY}"}

    try:
        async with websockets.connect(BASETEN_WS_URL, additional_headers=headers) as baseten_ws:
            print(f"[proxy] Connected to Baseten")

            async def browser_to_baseten():
                try:
                    async for msg in browser_ws:
                        await baseten_ws.send(msg)
                except websockets.ConnectionClosed:
                    pass

            async def baseten_to_browser():
                try:
                    async for msg in baseten_ws:
                        await browser_ws.send(msg)
                except websockets.ConnectionClosed:
                    pass

            await asyncio.gather(browser_to_baseten(), baseten_to_browser())

    except Exception as e:
        print(f"[proxy] Error: {e}")
    finally:
        print(f"[proxy] Session ended")


async def main():
    if not BASETEN_API_KEY or not BASETEN_WS_URL:
        print("Set BASETEN_API_KEY and BASETEN_WS_URL env vars")
        return

    print(f"[proxy] Relaying ws://localhost:{LOCAL_PORT} → {BASETEN_WS_URL}")
    print(f"[proxy] Ready — point your browser to ws://localhost:{LOCAL_PORT}")

    stop = asyncio.get_event_loop().create_future()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set_result, None)

    async with websockets.serve(proxy_handler, "localhost", LOCAL_PORT):
        await stop


if __name__ == "__main__":
    asyncio.run(main())
