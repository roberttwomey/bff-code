import cv2, numpy as np, asyncio, logging, threading, time, sys
from queue import Queue
from aiortc import MediaStreamTrack
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC
import open3d as o3d
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.FATAL)

def main():
    frame_q = Queue()
    lidar_q = Queue()
    latest_state = {"imu": (0,0,0), "soc": None, "power_v": None}
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.4.30")

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            
            # # Grab planar Y, U, V from PyAV without copies
            # h, w = frame.height, frame.width
            # y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(h, w)
            # u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(h//2, w//2)
            # v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(h//2, w//2)

            # # Repack to I420 layout (Y full, then U, then V) for OpenCV
            # # Stack U and V each upsampled in height to match OpenCV’s expected memory layout:
            # uv = np.vstack([u, v])                       # (h, w//2) total when stacked
            # yuv_i420 = np.vstack([y, cv2.resize(uv, (w, h//2), interpolation=cv2.INTER_NEAREST)])

            # # Convert YUV(I420) -> BGR (CPU, vectorized)
            # img = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_I420)
            
            if not frame_q.full():
                frame_q.put(img)

    def lowstate_callback(msg):
        data = msg["data"]
        rpy = tuple(data["imu_state"]["rpy"])
        soc = data["bms_state"]["soc"]
        power_v = data.get("power_v")
        latest_state["imu"], latest_state["soc"], latest_state["power_v"] = rpy, soc, power_v

    def lidar_callback(msg):
        try:
            print(f"Lidar message received: {type(msg)}")
            print(f"Message keys: {msg.keys() if hasattr(msg, 'keys') else 'No keys'}")
            
            data = msg["data"]["data"]
            positions = data.get("positions", [])
            print(f"Positions length: {len(positions)}")
            
            if len(positions) > 0:
                # Convert positions to numpy array
                points = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)
                print(f"Converted to {len(points)} points")
                if not lidar_q.full():
                    lidar_q.put(points)
                    print(f"Added {len(points)} points to queue")
        except Exception as e:
            print(f"Lidar callback error: {e}")
            import traceback
            traceback.print_exc()

    def run_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            await conn.connect()                    # 1 peer connection
            conn.video.switchVideoChannel(True)     # add video track(s)
            conn.video.add_track_callback(recv_camera_stream)
            conn.datachannel.pub_sub.subscribe(     # subscribe over DataChannel
                RTC_TOPIC["LOW_STATE"], lowstate_callback
            )
            # Enable lidar stream
            await conn.datachannel.disableTrafficSaving(True)
            conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
            
            # Subscribe to lidar data
            conn.datachannel.pub_sub.subscribe(
                "rt/utlidar/voxel_map_compressed", lidar_callback
            )
        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=run_loop, args=(loop,), daemon=True)
    t.start()

    # Setup Open3D for 3D lidar visualization
    print("Creating Open3D visualizer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window("Go2 Lidar 3D Point Cloud", width=800, height=600, left=50, top=50)
    print("Open3D window created")
    
    # Create initial point cloud
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    print("Added point cloud geometry")
    
    # Set initial view for voxel grid coordinates
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([64, 64, 20])  # Center of the voxel grid
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.01)  # Much smaller zoom for large coordinate ranges
    print("Set initial camera view")
    
    # Add some test points to verify visualization works
    test_points = np.array([
        [64, 64, 20],  # Center point
        [64, 64, 25],  # Above center
        [64, 64, 15],  # Below center
        [74, 64, 20],  # Right of center
        [54, 64, 20],  # Left of center
        [64, 74, 20],  # Forward of center
        [64, 54, 20],  # Back of center
    ], dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(test_points)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((len(test_points), 3)) * 0.5)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    print("Added test points to verify visualization")
    
    # Force a render to make sure the window shows something
    vis.poll_events()
    vis.update_renderer()
    print("Forced initial render")
    
    # Also create a simple matplotlib fallback window
    plt.ion()
    fig_2d, ax_2d = plt.subplots(figsize=(8, 6))
    ax_2d.set_xlabel('X')
    ax_2d.set_ylabel('Y')
    ax_2d.set_title('Go2 Lidar 2D View (Fallback)')
    ax_2d.grid(True, alpha=0.3)
    scatter_2d = ax_2d.scatter([], [], c=[], cmap='viridis', s=1, alpha=0.7)
    plt.colorbar(scatter_2d, ax=ax_2d, label='Height (Z)')
    print("Created matplotlib fallback window")
    
    # Force the Open3D window to be visible
    print("Forcing Open3D window to be visible...")
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)  # Give it time to render
    print("Open3D window should now be visible")
    

    # Simple OpenCV viewer + HUD
    h, w = 720, 1280
    cv2.namedWindow("Go2 Video", cv2.WINDOW_NORMAL)
    blank = np.zeros((h, w, 3), np.uint8)
    first = True
    
    def update_lidar_plot():
        if not lidar_q.empty():
            points = lidar_q.get()
            print(f"Processing {len(points)} points in update_lidar_plot")
            if len(points) > 0:
                # Debug: show point ranges
                print(f"Point ranges - X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
                print(f"Point ranges - Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
                print(f"Point ranges - Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
                
                # For voxel grid coordinates, use appropriate filtering
                mask = (points[:, 0] >= 0) & (points[:, 0] <= 128) & (points[:, 1] >= 0) & (points[:, 1] <= 128) & (points[:, 2] >= 0) & (points[:, 2] <= 50)
                filtered_points = points[mask]
                print(f"After filtering: {len(filtered_points)} points")
                
                if len(filtered_points) > 0:
                    # Update point cloud geometry
                    pcd.points = o3d.utility.Vector3dVector(filtered_points)
                    
                    # Color points by height (Z coordinate) for better visualization
                    colors = np.zeros((len(filtered_points), 3))
                    z_normalized = (filtered_points[:, 2] - filtered_points[:, 2].min()) / (filtered_points[:, 2].max() - filtered_points[:, 2].min() + 1e-6)
                    colors[:, 0] = z_normalized  # Red channel based on height
                    colors[:, 1] = 1 - z_normalized  # Green channel (inverse height)
                    colors[:, 2] = 0.3  # Blue channel constant
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # Update visualization
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    print(f"Updated visualization with {len(filtered_points)} points")
                    
                    # Also update matplotlib fallback
                    scatter_2d.set_offsets(filtered_points[:, :2])
                    scatter_2d.set_array(filtered_points[:, 2])
                    fig_2d.canvas.draw()
                    fig_2d.canvas.flush_events()
                    print(f"Updated matplotlib fallback with {len(filtered_points)} points")
                else:
                    print("No points after filtering - trying without any filtering")
                    # Try without filtering as a fallback
                    pcd.points = o3d.utility.Vector3dVector(points)
                    z_normalized = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-6)
                    colors = np.zeros((len(points), 3))
                    colors[:, 0] = z_normalized
                    colors[:, 1] = 1 - z_normalized
                    colors[:, 2] = 0.3
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    print(f"Updated visualization with {len(points)} unfiltered points")
                    
                    # Also update matplotlib fallback
                    scatter_2d.set_offsets(points[:, :2])
                    scatter_2d.set_array(points[:, 2])
                    fig_2d.canvas.draw()
                    fig_2d.canvas.flush_events()
                    print(f"Updated matplotlib fallback with {len(points)} unfiltered points")
            else:
                print("No points to process")
        else:
            print("No lidar data in queue")
    
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
            
            # Update lidar plot
            update_lidar_plot()
            
            # Also update Open3D window periodically
            try:
                vis.poll_events()
                vis.update_renderer()
                # Check if window is still valid
                if not vis.poll_events():
                    print("Open3D window may have been closed")
            except Exception as e:
                print(f"Open3D update error: {e}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.005)
    finally:
        cv2.destroyAllWindows()
        vis.destroy_window()
        plt.close('all')
        loop.call_soon_threadsafe(loop.stop)
        t.join()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting…"); sys.exit(0)
