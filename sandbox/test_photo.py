#!/usr/bin/env python3
"""
Example: request a front photo from the Go2 over WebRTC.

Requires:
    pip install go2-webrtc-connect
"""

import asyncio
import json
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

# Topic constants
REQ_TOPIC = "rt/api/videohub/request"
RESP_TOPIC = "rt/api/videohub/response"   # where responses come back (may vary by firmware)

async def main():
    # --- Connect to robot (STA mode with IP, but you can switch to AP or Remote) ---
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.4.30")
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.4.30")

    # await conn.connect()
    print("Connected to Go2.")

    # --- Handler for responses ---
    def on_response(msg: bytes):
        try:
            txt = msg.decode("utf-8", errors="ignore")
            data = json.loads(txt)
            print("📷 Response:", json.dumps(data, indent=2))
        except Exception:
            print("Raw response (bytes):", msg[:64], "...")

    # Subscribe to the response channel
    if hasattr(conn, "subscribe"):
        conn.subscribe(RESP_TOPIC, on_response)
    elif hasattr(conn, "subscribe_topic"):
        conn.subscribe_topic(RESP_TOPIC, on_response)

    # --- Send request ---
    req_msg = {
        "header": {"name": "FRONT_PHOTO_REQ"},
        "body": {}
    }
    payload = json.dumps(req_msg).encode("utf-8")

    print("Sending FRONT_PHOTO_REQ…")
    await conn.send_message(REQ_TOPIC, payload)

    # Wait for the response (a few seconds)
    await asyncio.sleep(5)

    await conn.close()
    print("Closed connection.")

if __name__ == "__main__":
    asyncio.run(main())
