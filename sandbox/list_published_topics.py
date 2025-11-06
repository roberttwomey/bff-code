#!/usr/bin/env python3
"""
Discover which Unitree Go2 LiDAR/Mapping topics are actually publishing via go2_webrtc_connect.

It:
  1) Connects over WebRTC (AP / STA / Remote)
  2) Attaches a generic message handler
  3) Listens for N seconds
  4) Prints:
      - Which of your target topics published
      - Any other topics observed

Usage examples:
  python list_published_topics.py --method sta --ip 192.168.123.161
  python list_published_topics.py --method ap --listen-seconds 8
  python list_published_topics.py --method remote --serial ABC123 --username you@u.edu --password '***'

Notes:
  - Keep the Unitree mobile app closed (WebRTC is typically single-client).
  - If you’re in STA and know the robot IP, use --ip. Otherwise use --serial.
"""

import asyncio
import argparse
import sys
import time
from typing import Dict, Set, Optional, Callable

# --- Import driver ---
try:
    from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
except Exception:
    print("Could not import go2_webrtc_connect driver. Install the repo or package first.")
    print("  pip install go2-webrtc-connect  (or clone and pip install -e .)")
    sys.exit(1)

# Optional constants; present in some versions
try:
    from go2_webrtc_driver.constants import RTC_TOPIC  # noqa: F401
except Exception:
    RTC_TOPIC = None

# --- Target topics to check ---
TARGET_TOPICS = [
    "rt/utlidar/switch",
    "rt/utlidar/voxel_map",
    "rt/utlidar/voxel_map_compressed",
    "rt/utlidar/lidar_state",
    "rt/utlidar/robot_pose",
    "rt/utlidar/scan",
    "rt/utlidar/cloud",
    "rt/utlidar/imu",
    "rt/utlidar/robot_odom",
    "rt/utlidar/foot_position",
    "rt/utlidar/cloud_deskewed",
    "rt/utlidar/heigh_map_array",   # as provided
    "rt/mapping/grid_map",
]

class TopicSniffer:
    """
    Attach a generic message handler and record any topics seen.
    Works across driver revisions by trying a few handler methods.
    """
    def __init__(self, conn: Go2WebRTCConnection):
        self.conn = conn
        self.seen_topics: Set[str] = set()

    def _generic_handler_payload_only(self, payload: bytes):
        # Some APIs provide payload-only hooks (no topic) — not useful for listing,
        # so we ignore those unless we have a topic context, which we don't here.
        pass

    def _generic_handler_topic_payload(self, *args, **kwargs):
        # Expect either (topic, payload) or (topic, payload, …)
        if not args:
            return
        topic = args[0]
        if isinstance(topic, str):
            self.seen_topics.add(topic)

    def attach(self) -> Optional[str]:
        """
        Try to register a handler that receives (topic, payload).
        Returns the method name used (for info) or None if none worked.
        """
        # Prefer handlers that give (topic, payload)
        for m in ["on_message", "add_message_handler"]:
            if hasattr(self.conn, m):
                try:
                    getattr(self.conn, m)(self._generic_handler_topic_payload)
                    return m
                except Exception:
                    pass

        # Some drivers only allow per-topic callbacks; as a fallback,
        # subscribe to all TARGET_TOPICS with a topic-aware wrapper.
        for m in ["subscribe", "subscribe_topic", "add_topic_listener", "add_subscriber", "on"]:
            if hasattr(self.conn, m):
                used = m
                for t in TARGET_TOPICS:
                    try:
                        def make_cb(topic_str: str) -> Callable:
                            def cb(payload):
                                self.seen_topics.add(topic_str)
                            return cb
                        getattr(self.conn, m)(t, make_cb(t))
                    except Exception:
                        # ignore per-topic subscription failures; we’ll still catch others if possible
                        pass
                return f"{used}-per-topic"
        return None


async def connect_from_args(args) -> Go2WebRTCConnection:
    if args.method == "ap":
        return Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    elif args.method == "sta":
        if not (args.ip or args.serial):
            print("STA requires --ip or --serial", file=sys.stderr)
            sys.exit(2)
        if args.ip:
            return Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)
        else:
            return Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=args.serial)
    else:  # remote
        if not (args.serial and args.username and args.password):
            print("Remote requires --serial --username --password", file=sys.stderr)
            sys.exit(2)
        return Go2WebRTCConnection(
            WebRTCConnectionMethod.Remote,
            serialNumber=args.serial,
            username=args.username,
            password=args.password,
        )

async def main():
    parser = argparse.ArgumentParser(description="List which Go2 LiDAR/Mapping topics are publishing.")
    # parser.add_argument("--method", choices=["ap", "sta", "remote"], default="sta")
    # parser.add_argument("--ip", type=str, help="Robot IP for STA")
    # parser.add_argument("--serial", type=str, help="Robot serial (STA or Remote)")
    # parser.add_argument("--username", type=str, help="Unitree account (Remote)")
    # parser.add_argument("--password", type=str, help="Unitree account password (Remote)")
    parser.add_argument("--listen-seconds", type=float, default=6.0, help="How long to listen for traffic")
    parser.add_argument("--show-all", action="store_true", help="Also show non-target topics seen")
    args = parser.parse_args()

    # conn = await connect_from_args(args)
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.4.30")

    # print(f"Connecting via {args.method.upper()}...")
    await conn.connect()
    print("Connected.")

    sniffer = TopicSniffer(conn)
    used = sniffer.attach()
    if used is None:
        print("Could not attach a message handler to detect topics. Exiting.", file=sys.stderr)
        try:
            await conn.close()
        finally:
            sys.exit(3)
    else:
        print(f"Attached handler via: {used}")

    # Some drivers require enabling topics explicitly; try enabling all targets if supported
    enabled_any = False
    for m in ["set_topic_enabled", "enable_topic"]:
        if hasattr(conn, m):
            for t in TARGET_TOPICS:
                try:
                    getattr(conn, m)(t, True)
                    enabled_any = True
                except Exception:
                    pass
    if enabled_any:
        print("Requested topics to be enabled where supported.")

    # Listen window
    wait_s = max(0.5, args.listen_seconds)
    print(f"Listening for {wait_s:.1f}s...")
    await asyncio.sleep(wait_s)

    # Report
    seen = sniffer.seen_topics
    target_set = set(TARGET_TOPICS)
    seen_targets = sorted(seen & target_set)
    unseen_targets = sorted(target_set - seen)
    other_topics = sorted(seen - target_set)

    print("\n=== Published TARGET topics ===")
    if seen_targets:
        for t in seen_targets:
            print(f"  ✓ {t}")
    else:
        print("  (none observed)")

    print("\n=== TARGET topics with no messages observed ===")
    if unseen_targets:
        for t in unseen_targets:
            print(f"  • {t}")
    else:
        print("  (all target topics seen)")

    if args.show_all:
        print("\n=== OTHER topics observed ===")
        if other_topics:
            for t in other_topics:
                print(f"  - {t}")
        else:
            print("  (none)")

    # Clean up
    try:
        await conn.close()
    except Exception:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
