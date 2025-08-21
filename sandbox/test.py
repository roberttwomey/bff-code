import cv2, numpy as np, asyncio, logging, threading, time, sys
from queue import Queue
from aiortc import MediaStreamTrack
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC

logging.basicConfig(level=logging.FATAL)

def main():
    frame_q = Queue()
    latest_state = {"imu": (0,0,0), "soc": None, "power_v": None}
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.4.30")

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            if not frame_q.full():
                frame_q.put(img)

    def lowstate_callback(msg):
        data = msg["data"]
        rpy = tuple(data["imu_state"]["rpy"])
        soc = data["bms_state"]["soc"]
        power_v = data.get("power_v")
        latest_state["imu"], latest_state["soc"], latest_state["power_v"] = rpy, soc, power_v

    def run_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            await conn.connect()                    # 1 peer connection
            conn.video.switchVideoChannel(True)     # add video track(s)
            conn.video.add_track_callback(recv_camera_stream)
            conn.datachannel.pub_sub.subscribe(     # subscribe over DataChannel
                RTC_TOPIC["LOW_STATE"], lowstate_callback
            )
        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=run_loop, args=(loop,), daemon=True)
    t.start()

    # Simple OpenCV viewer + HUD
    h, w = 720, 1280
    cv2.namedWindow("Go2 Video", cv2.WINDOW_NORMAL)
    blank = np.zeros((h, w, 3), np.uint8)
    first = True
    try:
        while True:
            if first: 
                last_img = blank.copy()
                first = False
            else:
                last_img = img.copy()
            img = frame_q.get() if not frame_q.empty() else last_img.copy()
            r, p, y = latest_state["imu"]
            soc = latest_state["soc"]
            pv = latest_state["power_v"]
            hud = f"IMU RPY: {r:.2f}, {p:.2f}, {y:.2f} | SOC: {soc}% | V: {pv}V"
            cv2.putText(img, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("Go2 Video", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.005)
    finally:
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        t.join()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting…"); sys.exit(0)
