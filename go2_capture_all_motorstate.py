import cv2, numpy as np, asyncio, logging, threading, time, sys
from queue import Queue
from aiortc import MediaStreamTrack
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC
import open3d as o3d
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.FATAL)

# Add this function after the existing imports
def create_motor_state_overlay(img, motor_state):
    """Create an overlay showing motor states in a compact visual format."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    padding = 10
    line_height = 20

    # Starting position for motor info (left side, below IMU)
    x = 20
    y = 70  # Below the IMU/SOC HUD

    # Draw motor states
    for i, motor in enumerate(motor_state):
        # Color based on temperature (green to red)
        temp = motor['temperature']
        if temp < 45:
            color = (0, 255, 0)  # Green
        elif temp < 60:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        # Format motor info
        motor_text = f"M{i+1}: {motor['q']:.2f}° {temp}°C"
        if motor['lost']:
            motor_text += " [LOST]"
            color = (0, 0, 255)  # Red if lost

        cv2.putText(img, motor_text, (x, y + i * line_height), 
                font, font_scale, color, thickness)

    return img

# Open3D visualization variables
vis = None
pcd = None
visualization_running = False

# Constants for point rotation
ROTATE_X_ANGLE = np.pi / 2  # 90 degrees
ROTATE_Z_ANGLE = np.pi      # 180 degrees

def adjust_view_to_fit_data(vis, points):
    """Adjust the view to fit all the point cloud data."""
    if len(points) == 0:
        return
    
    try:
        # Calculate the bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center = (min_coords + max_coords) / 2
        extent = max_coords - min_coords
        max_extent = np.max(extent)
        
        # Get view control
        ctr = vis.get_view_control()
        
        # Set the look-at point to the center of the data
        ctr.set_lookat(center)
        
        # Calculate a reasonable zoom level based on the extent
        # Add some padding (1.5x) to ensure all points are visible
        zoom_factor = 1.0 / (max_extent * 1.5)
        ctr.set_zoom(zoom_factor)
        
        # print(f"Adjusted view: center={center}, zoom_factor={zoom_factor:.6f}")  # Suppressed debug message
        
    except Exception as e:
        print(f"Error adjusting view: {e}")

def update_visualization(points, scalars):
    """Update the point cloud visualization with new data."""
    global pcd, vis
    
    if not visualization_running or vis is None:
        return
    
    try:
        if len(points) == 0:
            # Clear existing point cloud if no points
            if pcd is not None:
                pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                pcd.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
                vis.update_geometry(pcd)
        else:
            # Calculate and display data extent
            extent_info = calculate_data_extent(points)
            
            # Update existing point cloud instead of recreating
            if pcd is None:
                # Create point cloud only once
                pcd = o3d.geometry.PointCloud()
                vis.add_geometry(pcd, False)
            
            # Update points
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Color points by scalar values with better contrast
            colors = np.zeros((len(points), 3))
            if len(scalars) > 0:
                max_scalar = np.max(scalars)
                min_scalar = np.min(scalars)
                if max_scalar > min_scalar:
                    normalized_scalars = (scalars - min_scalar) / (max_scalar - min_scalar)
                    # Use a more vibrant color scheme
                    colors[:, 0] = normalized_scalars  # Red for distance
                    colors[:, 1] = 0.5 + 0.5 * normalized_scalars  # Green
                    colors[:, 2] = 1.0 - normalized_scalars  # Blue inverse
                else:
                    colors[:, 0] = 1.0  # Default red
                    colors[:, 1] = 0.5  # Default green
                    colors[:, 2] = 0.0  # Default blue
            else:
                colors[:, 0] = 1.0  # Default red
                colors[:, 1] = 0.5  # Default green
                colors[:, 2] = 0.0  # Default blue
            
            # Update colors
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Update the visualizer (only once per frame)
            vis.update_geometry(pcd)
            
            # Adjust view to fit all points (only on first update or when needed)
            # adjust_view_to_fit_data(vis, points)
        
        # print(f"Updated visualization with {len(points)} points")  # Suppressed debug message
        
    except Exception as e:
        print(f"Error updating visualization: {e}")
        import traceback
        traceback.print_exc()

def setup_visualization():
    """Initialize Open3D visualization."""
    global vis, pcd, visualization_running
    
    try:
        print("Setting up Open3D visualization...")
        vis = o3d.visualization.Visualizer()
        vis.create_window("Go2 Lidar 3D Point Cloud", width=1200, height=800)
        
        # Set rendering options for larger points
        opt = vis.get_render_option()
        opt.point_size = 4.0  # Make points larger
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
        opt.show_coordinate_frame = True
        
        # Create initial empty point cloud (will be reused)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
        
        # Set view with much more zoom out capability
        ctr = vis.get_view_control()
        ctr.set_front([-1, 1, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.001)  # Even more zoom out
        
        # Add coordinate frame - make it much larger to match the scale
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)  # Much larger coordinate frame
        vis.add_geometry(coord_frame)
        
        visualization_running = True
        print("Open3D visualization window created successfully.")
        print("Point size set to: 4.0")
        print("Press 'q' to close the window.")
        print("Use mouse wheel to zoom in/out, drag to rotate, right-click to pan")
        print("The view will automatically adjust to show all LIDAR data")
        
        # Test the window is working
        vis.poll_events()
        vis.update_renderer()
        
    except Exception as e:
        print(f"Error setting up visualization: {e}")
        visualization_running = False
        raise

def calculate_data_extent(points):
    """Calculate the full extent of the point cloud data."""
    if len(points) == 0:
        return None
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (min_coords + max_coords) / 2
    extent = max_coords - min_coords
    max_extent = np.max(extent)
    
    # Suppressed debug messages
    # print(f"Data extent: X[{min_coords[0]:.2f}, {max_coords[0]:.2f}], Y[{min_coords[1]:.2f}, {max_coords[1]:.2f}], Z[{min_coords[2]:.2f}, {max_coords[2]:.2f}]")
    # print(f"Center: {center}")
    # print(f"Max extent: {max_extent:.2f}")
    
    return {
        'min': min_coords,
        'max': max_coords,
        'center': center,
        'extent': extent,
        'max_extent': max_extent
    }

def rotate_points(points, x_angle, z_angle):
    """Rotate points around the x and z axes by given angles."""
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle)],
        [0, np.sin(x_angle), np.cos(x_angle)]
    ])
    
    rotation_matrix_z = np.array([
        [np.cos(z_angle), -np.sin(z_angle), 0],
        [np.sin(z_angle), np.cos(z_angle), 0],
        [0, 0, 1]
    ])
    
    points = points @ rotation_matrix_x.T
    points = points @ rotation_matrix_z.T
    return points

def main():
    frame_q = Queue()
    lidar_q = Queue()
    # latest_state = {"imu": (0,0,0), "soc": None, "power_v": None}
    latest_state = {
        "imu": (0,0,0), 
        "soc": None, 
        "power_v": None,
        "motor_state": []
    }
    
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

    def display_lowstate_data(message):

        # Extracting data from the message
        imu_state = message['imu_state']['rpy']
        motor_state = message['motor_state']
        bms_state = message['bms_state']
        foot_force = message['foot_force']
        temperature_ntc1 = message['temperature_ntc1']
        power_v = message['power_v']

        # Clear the entire screen and reset cursor position to top
        sys.stdout.write("\033[H\033[J")

        # Print the Go2 Robot Status
        print("Go2 Robot Status (LowState)")
        print("===========================")

        # IMU State (RPY)
        print(f"IMU - RPY: Roll: {imu_state[0]}, Pitch: {imu_state[1]}, Yaw: {imu_state[2]}")

        # Compact Motor States Display (Each motor on one line)
        print("\nMotor States (q, Temperature, Lost):")
        print("------------------------------------------------------------")
        for i, motor in enumerate(motor_state):
            # Display motor info in a single line
            print(f"Motor {i + 1:2}: q={motor['q']:.4f}, Temp={motor['temperature']}°C, Lost={motor['lost']}")

        # BMS (Battery Management System) State
        print("\nBattery Management System (BMS) State:")
        print(f"  Version: {bms_state['version_high']}.{bms_state['version_low']}")
        print(f"  SOC (State of Charge): {bms_state['soc']}%")
        print(f"  Current: {bms_state['current']} mA")
        print(f"  Cycle Count: {bms_state['cycle']}")
        print(f"  BQ NTC: {bms_state['bq_ntc']}°C")
        print(f"  MCU NTC: {bms_state['mcu_ntc']}°C")

        # Foot Force
        print(f"\nFoot Force: {foot_force}")

        # Additional Sensors
        print(f"Temperature NTC1: {temperature_ntc1}°C")
        print(f"Power Voltage: {power_v}V")

        # Optionally, flush to ensure immediate output
        sys.stdout.flush()

    def lowstate_callback(msg):
        data = msg["data"]
        rpy = tuple(data["imu_state"]["rpy"])
        soc = data["bms_state"]["soc"]
        power_v = data.get("power_v")
        motor_state = data["motor_state"]
        latest_state["imu"] = rpy
        latest_state["soc"] = soc
        latest_state["power_v"] = power_v
        latest_state["motor_state"] = motor_state
        
        # optionally display to console
        # display_lowstate_data(data)

    def lidar_callback(msg):
        try:
            # print(f"Lidar message received: {type(msg)}")
            # print(f"Message keys: {msg.keys() if hasattr(msg, 'keys') else 'No keys'}")
            
            data = msg["data"]["data"]
            positions = data.get("positions", [])
            # print(f"Positions length: {len(positions)}")
            
            if len(positions) > 0:
                # Convert positions to numpy array
                points = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)
                # print(f"Converted to {len(points)} points")
                if not lidar_q.full():
                    lidar_q.put(points)
                    # print(f"Added {len(points)} points to queue")
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

    # Setup Open3D visualization
    setup_visualization()
    
    # Simple OpenCV viewer + HUD
    h, w = 360, 640  # Reduced from 720x1280 to 480x640
    cv2.namedWindow("Go2 Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Go2 Video", w, h)  # Explicitly set window size
    blank = np.zeros((h, w, 3), np.uint8)
    first = True
    
    def update_lidar_plot():
        if not lidar_q.empty():
            points = lidar_q.get()
            # print(f"Processing {len(points)} points in update_lidar_plot")  # Suppressed debug message
            if len(points) > 0:
                # Debug: show point ranges (suppressed)
                # print(f"Point ranges - X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
                # print(f"Point ranges - Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
                # print(f"Point ranges - Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
                
                # Process points using the Open3D functions
                # Rotate points
                rotated_points = rotate_points(points, ROTATE_X_ANGLE, ROTATE_Z_ANGLE)
                
                # Filter points if needed (you can adjust these values)
                minYValue = 0
                maxYValue = 100
                if len(rotated_points) > 0:
                    filtered_points = rotated_points[(rotated_points[:, 1] >= minYValue) & (rotated_points[:, 1] <= maxYValue)]
                    
                    # Only center if we have points
                    if len(filtered_points) > 0:
                        center_x = float(np.mean(filtered_points[:, 0]))
                        center_y = float(np.mean(filtered_points[:, 1]))
                        center_z = float(np.mean(filtered_points[:, 2]))
                        offset_points = filtered_points - np.array([center_x, center_y, center_z])
                    else:
                        offset_points = filtered_points
                else:
                    offset_points = rotated_points
                
                # Calculate scalars for coloring
                scalars = np.linalg.norm(offset_points, axis=1)
                
                # Update Open3D visualization
                update_visualization(offset_points, scalars)
                
                # print(f"Updated visualization with {len(offset_points)} processed points")  # Suppressed debug message
            else:
                # print("No points to process")  # Suppressed debug message
                pass
        else:
            # print("No lidar data in queue")  # Suppressed debug message
            pass
    
    try:
        while True:
            if first: 
                last_img = blank.copy()
                first = False
            else:
                last_img = img.copy()
            img = frame_q.get() if not frame_q.empty() else last_img.copy()
            
            # Draw IMU/Battery HUD
            r, p, y = latest_state["imu"]
            soc = latest_state["soc"]
            pv = latest_state["power_v"]
            hud = f"IMU RPY: {r:.2f}, {p:.2f}, {y:.2f} | SOC: {soc}% | V: {pv}V"
            cv2.putText(img, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            
            # Add motor state overlay if available
            if "motor_state" in latest_state:
                img = create_motor_state_overlay(img, latest_state["motor_state"])
            
            cv2.imshow("Go2 Video", img)
            
            # Update lidar plot
            update_lidar_plot()
            
            # Update Open3D window
            try:
                if vis and visualization_running:
                    vis.poll_events()
                    vis.update_renderer()
            except Exception as e:
                print(f"Open3D update error: {e}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.005)
    finally:
        cv2.destroyAllWindows()
        if vis:
            vis.destroy_window()
        loop.call_soon_threadsafe(loop.stop)
        t.join()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting…"); sys.exit(0)
