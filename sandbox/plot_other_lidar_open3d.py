# filepath: plot_lidar_stream_rt.py
""" @MrRobotoW at The RoboVerse Discord """
""" robert.wagoner@gmail.com """
""" 01/30/2025 """
""" Inspired from lidar_stream.py by @legion1581 at The RoboVerse Discord """

VERSION = "1.0.19"

import asyncio
import logging
import csv
import numpy as np
import open3d as o3d
import time
import argparse
from datetime import datetime
import os
import sys
import ast
import threading
import json

# --- Dependency: go2_webrtc_connect as submodule or installed package ---
try:
    from go2_webrtc_connect.go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
except ImportError:
    try:
        from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
    except ImportError:
        raise ImportError(
            "Could not import Go2WebRTCConnection. "
            "Please install the go2_webrtc_connect package or add it as a submodule.\n"
            "For submodule: git submodule add https://github.com/robwa/go2_webrtc_connect.git\n"
            "For pip: pip install go2-webrtc-connect"
        )

# Increase the field size limit for CSV reading
csv.field_size_limit(sys.maxsize)

logging.basicConfig(level=logging.FATAL)

# Constants to enable/disable features
ENABLE_POINT_CLOUD = True
SAVE_LIDAR_DATA = True

# File paths
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LIDAR_CSV_FILE = f"lidar_data_{timestamp}.csv"
STREAM_DATA_FILE = f"stream_data_{timestamp}.json"

# Global variables
lidar_csv_file = None
lidar_csv_writer = None
lidar_buffer = []
message_count = 0  # Counter for processed LIDAR messages
reconnect_interval = 5  # Time (seconds) before retrying connection

# Stream tracking
stream_data = {}
stream_messages = {}

# Constants
MAX_RETRY_ATTEMPTS = 10

ROTATE_X_ANGLE = np.pi / 2  # 90 degrees
ROTATE_Z_ANGLE = np.pi      # 90 degrees

minYValue = 0
maxYValue = 100

# Open3D visualization variables
vis = None
pcd = None
visualization_running = False

# Define all LiDAR and mapping streams to monitor
LIDAR_STREAMS = [
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
    "rt/utlidar/heigh_map_array",
    "rt/mapping/grid_map"
]

# Parse command-line arguments
parser = argparse.ArgumentParser(description=f"LIDAR Viz v{VERSION}")
parser.add_argument("--version", action="version", version=f"LIDAR Viz v{VERSION}")
parser.add_argument("--cam-center", action="store_true", help="Put Camera at the Center")
parser.add_argument("--type-voxel", action="store_true", help="Voxel View")
parser.add_argument("--csv-read", type=str, help="Read from CSV files instead of WebRTC")
parser.add_argument("--csv-write", action="store_true", help="Write CSV data file")
parser.add_argument("--skip-mod", type=int, default=1, help="Skip messages using modulus (default: 1, no skipping)")
parser.add_argument('--minYValue', type=int, default=0, help='Minimum Y value for the plot')
parser.add_argument('--maxYValue', type=int, default=100, help='Maximum Y value for the plot')
parser.add_argument('--point-size', type=float, default=4.0, help='Point size for visualization (default: 8.0)')
parser.add_argument('--streams', nargs='+', choices=LIDAR_STREAMS, default=LIDAR_STREAMS, help='Specific streams to monitor')
args = parser.parse_args()

minYValue = args.minYValue
maxYValue = args.maxYValue
SAVE_LIDAR_DATA = args.csv_write

# Initialize stream tracking
for stream in args.streams:
    stream_data[stream] = []
    stream_messages[stream] = 0

def log_stream_message(stream_name, message):
    """Log stream message data for analysis."""
    global stream_data, stream_messages
    
    try:
        # Extract basic info from message
        message_info = {
            'timestamp': time.time(),
            'stream': stream_name,
            'message_type': type(message).__name__,
            'data_keys': list(message.keys()) if hasattr(message, 'keys') else [],
            'data_type': type(message.get('data', {})).__name__ if hasattr(message, 'get') else 'unknown'
        }
        
        # Add specific data based on stream type
        if 'voxel_map' in stream_name or 'cloud' in stream_name:
            if 'data' in message and 'positions' in message['data']:
                positions = message['data']['positions']
                message_info['point_count'] = len(positions) // 3 if isinstance(positions, list) else 0
                message_info['has_positions'] = True
            else:
                message_info['has_positions'] = False
        
        stream_data[stream_name].append(message_info)
        stream_messages[stream_name] += 1
        
        # Keep only last 1000 messages per stream to prevent memory issues
        if len(stream_data[stream_name]) > 1000:
            stream_data[stream_name] = stream_data[stream_name][-1000:]
        
        print(f"[{stream_name}] Message {stream_messages[stream_name]}: {message_info}")
        
    except Exception as e:
        print(f"Error logging stream message for {stream_name}: {e}")

def save_stream_data():
    """Save collected stream data to JSON file."""
    try:
        with open(STREAM_DATA_FILE, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'stream_summary': {stream: {
                    'message_count': stream_messages[stream],
                    'last_message': stream_data[stream][-1] if stream_data[stream] else None,
                    'message_types': list(set(msg.get('message_type', 'unknown') for msg in stream_data[stream]))
                } for stream in args.streams},
                'stream_data': stream_data
            }, f, indent=2, default=str)
        print(f"Stream data saved to {STREAM_DATA_FILE}")
        
        # Print summary
        print("\n=== STREAM SUMMARY ===")
        total_messages = sum(stream_messages.values())
        print(f"Total messages across all streams: {total_messages}")
        
        active_streams = [stream for stream in args.streams if stream_messages[stream] > 0]
        inactive_streams = [stream for stream in args.streams if stream_messages[stream] == 0]
        
        if active_streams:
            print(f"\nActive streams ({len(active_streams)}):")
            for stream in active_streams:
                print(f"  {stream}: {stream_messages[stream]} messages")
        
        if inactive_streams:
            print(f"\nInactive streams ({len(inactive_streams)}):")
            for stream in inactive_streams:
                print(f"  {stream}: No messages received")
        
        print("=" * 20)
        
    except Exception as e:
        print(f"Error saving stream data: {e}")

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
        
        print(f"Adjusted view: center={center}, zoom_factor={zoom_factor:.6f}")
        
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
        
        print(f"Updated visualization with {len(points)} points")
        
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
        vis.create_window("LIDAR Visualization", width=1200, height=800)
        
        # Set rendering options for larger points
        opt = vis.get_render_option()
        opt.point_size = args.point_size  # Make points larger
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
        opt.show_coordinate_frame = True
        
        # Create initial empty point cloud (will be reused)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
        
        # Set view with much more zoom out capability
        ctr = vis.get_view_control()
        if args.cam_center:
            ctr.set_front([0, 0, -1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.001)  # Even more zoom out
        else:
            ctr.set_front([-1, 1, -1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.001)  # Even more zoom out
        
        # Add coordinate frame - make it much larger to match the scale
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)  # Much larger coordinate frame
        vis.add_geometry(coord_frame)
        
        visualization_running = True
        print("Open3D visualization window created successfully.")
        print(f"Point size set to: {args.point_size}")
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

def run_visualization():
    """Run the Open3D visualization loop."""
    global vis, visualization_running
    
    print("Starting visualization loop...")
    try:
        while visualization_running and vis is not None:
            try:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in visualization loop: {e}")
                break
    except KeyboardInterrupt:
        print("Visualization interrupted by user")
    finally:
        if vis:
            try:
                vis.destroy_window()
            except:
                pass
        visualization_running = False
        print("Visualization window closed.")

def setup_csv_output():
    """Set up CSV files for LIDAR output."""
    global lidar_csv_file, lidar_csv_writer

    if SAVE_LIDAR_DATA:
        lidar_csv_file = open(LIDAR_CSV_FILE, mode='w', newline='', encoding='utf-8')
        lidar_csv_writer = csv.writer(lidar_csv_file)
        lidar_csv_writer.writerow(['stamp', 'frame_id', 'resolution', 'src_size', 'origin', 'width', 
                                   'point_count', 'positions'])
        lidar_csv_file.flush()  # Ensure the header row is flushed to disk

def close_csv_output():
    """Close CSV files."""
    global lidar_csv_file

    if lidar_csv_file:
        lidar_csv_file.close()
        lidar_csv_file = None

def calculate_data_extent(points):
    """Calculate the full extent of the point cloud data."""
    if len(points) == 0:
        return None
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (min_coords + max_coords) / 2
    extent = max_coords - min_coords
    max_extent = np.max(extent)
    
    print(f"Data extent: X[{min_coords[0]:.2f}, {max_coords[0]:.2f}], Y[{min_coords[1]:.2f}, {max_coords[1]:.2f}], Z[{min_coords[2]:.2f}, {max_coords[2]:.2f}]")
    print(f"Center: {center}")
    print(f"Max extent: {max_extent:.2f}")
    
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

async def lidar_webrtc_connection():
    """Connect to WebRTC and process LIDAR data from multiple streams."""
    global lidar_buffer, message_count
    retry_attempts = 0

    while retry_attempts < MAX_RETRY_ATTEMPTS:
        try:
            # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.12.1")  # WebRTC IP
            conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.4.30")  # WebRTC IP

            logging.info("Connecting to WebRTC...")
            await conn.connect()
            logging.info("Connected to WebRTC.")
            retry_attempts = 0  # Reset retry attempts on successful connection

            await conn.datachannel.disableTrafficSaving(True)
            conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
            setup_csv_output()

            # Track messages for this connection
            local_message_count = 0

            async def generic_stream_handler(stream_name):
                """Generic handler for any stream type."""
                async def handler(message):
                    nonlocal local_message_count
                    global message_count
                    
                    try:
                        # Log the message for analysis
                        log_stream_message(stream_name, message)
                        
                        # Handle different stream types
                        if 'imu' in stream_name:
                            # Handle IMU data (different structure)
                            print(f"[{stream_name}] IMU data received: {message}")
                            return
                        elif 'robot_pose' in stream_name or 'robot_odom' in stream_name:
                            # Handle robot pose/odometry data
                            print(f"[{stream_name}] Pose/Odom data received: {message}")
                            return
                        elif 'foot_position' in stream_name:
                            # Handle foot position data
                            print(f"[{stream_name}] Foot position data received: {message}")
                            return
                        elif 'switch' in stream_name:
                            # Handle switch/status data
                            print(f"[{stream_name}] Switch data received: {message}")
                            return
                        elif 'grid_map' in stream_name:
                            # Handle grid map data
                            print(f"[{stream_name}] Grid map data received: {message}")
                            return
                        
                        # Only process point cloud data for visualization
                        if not ENABLE_POINT_CLOUD or ('voxel_map' not in stream_name and 'cloud' not in stream_name):
                            return

                        if message_count % args.skip_mod != 0:
                            message_count += 1
                            return

                        # Handle different message structures for point cloud data
                        positions = []
                        if 'data' in message:
                            if 'positions' in message['data']:
                                positions = message['data']['positions']
                            elif 'data' in message['data'] and 'positions' in message['data']['data']:
                                positions = message['data']['data']['positions']

                        if not positions:
                            print(f"[{stream_name}] No positions data found")
                            return

                        points = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)
                        total_points = len(points)
                        
                        # Don't remove duplicates immediately - show more data
                        unique_points = points  # Changed from np.unique(points, axis=0)

                        if SAVE_LIDAR_DATA and lidar_csv_writer:
                            # Extract additional data if available
                            origin = message.get('origin', message['data'].get('origin', []))
                            stamp = message.get('stamp', message['data'].get('stamp', time.time()))
                            frame_id = message.get('frame_id', message['data'].get('frame_id', 0))
                            resolution = message.get('resolution', message['data'].get('resolution', 0.05))
                            src_size = message.get('src_size', message['data'].get('src_size', 0))
                            width = message.get('width', message['data'].get('width', [128, 128, 38]))
                            
                            lidar_csv_writer.writerow([
                                stamp,
                                frame_id,
                                resolution,
                                src_size,
                                str(origin),
                                str(width),
                                len(unique_points),
                                unique_points.tolist()
                            ])
                            lidar_csv_file.flush()

                        points = rotate_points(unique_points, ROTATE_X_ANGLE, ROTATE_Z_ANGLE)
                        
                        # Less aggressive filtering - show more points
                        if len(points) > 0:
                            points = points[(points[:, 1] >= minYValue) & (points[:, 1] <= maxYValue)]
                            
                            # Only center if we have points
                            if len(points) > 0:
                                center_x = float(np.mean(points[:, 0]))
                                center_y = float(np.mean(points[:, 1]))
                                center_z = float(np.mean(points[:, 2]))
                                offset_points = points - np.array([center_x, center_y, center_z])
                            else:
                                offset_points = points
                        else:
                            offset_points = points

                        message_count += 1
                        local_message_count += 1
                        print(f"[{stream_name}] Message {message_count}: Total points={total_points}, Filtered points={len(offset_points)}")

                        scalars = np.linalg.norm(offset_points, axis=1)
                        
                        # Update Open3D visualization
                        update_visualization(offset_points, scalars)

                    except Exception as e:
                        logging.error(f"Error in {stream_name} callback: {e}")
                        print(f"Error processing {stream_name}: {e}")
                
                return handler

            # Subscribe to all specified streams
            print(f"Subscribing to {len(args.streams)} streams: {', '.join(args.streams)}")
            
            for stream_name in args.streams:
                try:
                    handler = await generic_stream_handler(stream_name)
                    conn.datachannel.pub_sub.subscribe(
                        stream_name,
                        lambda msg, stream=stream_name: asyncio.create_task(handler(msg))
                    )
                    print(f"Subscribed to {stream_name}")
                except Exception as e:
                    print(f"Error subscribing to {stream_name}: {e}")

            # Keep the connection active
            print("Monitoring streams... Press Ctrl+C to stop")
            while True:
                await asyncio.sleep(1.0)
                # Print periodic summary
                if local_message_count % 10 == 0 and local_message_count > 0:
                    total_streams = sum(stream_messages.values())
                    print(f"Total messages across all streams: {total_streams}")
                    for stream, count in stream_messages.items():
                        if count > 0:
                            print(f"  {stream}: {count} messages")

        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.info(f"Reconnecting in {reconnect_interval} seconds... (Attempt {retry_attempts + 1}/{MAX_RETRY_ATTEMPTS})")
            close_csv_output()
            try:
                await conn.disconnect()
            except Exception as e:
                logging.error(f"Error during disconnect: {e}")
            await asyncio.sleep(reconnect_interval)
            retry_attempts += 1

    # Save stream data before exiting
    save_stream_data()
    
    if retry_attempts >= MAX_RETRY_ATTEMPTS:
        logging.error("Max retry attempts reached. Exiting.")

async def read_csv_and_update(csv_file):
    """Continuously read CSV files and update visualization."""
    global message_count

    while True:  # Infinite loop to restart at EOF
        try:
            total_messages = sum(1 for _ in open(csv_file)) - 1  # Calculate total messages

            with open(csv_file, mode='r', newline='', encoding='utf-8') as lidar_file:
                lidar_reader = csv.DictReader(lidar_file)

                for lidar_row in lidar_reader:
                    if message_count % args.skip_mod == 0:
                        try:
                            # Extract and validate positions
                            positions = ast.literal_eval(lidar_row.get("positions", "[]"))
                            if isinstance(positions, list) and all(isinstance(item, list) and len(item) == 3 for item in positions):
                                points = np.array(positions, dtype=np.float32)
                            else:
                                points = np.array([item for item in positions if isinstance(item, list) and len(item) == 3], dtype=np.float32)

                            # Extract and compute origin, resolution, width, and center
                            origin = np.array(eval(lidar_row.get("origin", "[]")), dtype=np.float32)
                            resolution = float(lidar_row.get("resolution", 0.05))
                            width = np.array(eval(lidar_row.get("width", "[128, 128, 38]")), dtype=np.float32)
                            center = origin + (width * resolution) / 2

                            # Process points
                            if points.size > 0:
                                points = rotate_points(points, ROTATE_X_ANGLE, ROTATE_Z_ANGLE)
                                points = points[(points[:, 1] >= minYValue) & (points[:, 1] <= maxYValue)]
                                
                                # Don't remove duplicates - show more data
                                unique_points = points  # Changed from np.unique(points, axis=0)
                                
                                # Calculate center coordinates
                                center_x = float(np.mean(unique_points[:, 0]))
                                center_y = float(np.mean(unique_points[:, 1]))
                                center_z = float(np.mean(unique_points[:, 2]))

                                # Offset points by center coordinates
                                offset_points = unique_points - np.array([center_x, center_y, center_z])
                            else:
                                unique_points = np.empty((0, 3), dtype=np.float32)
                                offset_points = unique_points

                            # Update visualization
                            scalars = np.linalg.norm(offset_points, axis=1)
                            update_visualization(offset_points, scalars)

                            # Print message details
                            print(f"LIDAR Message {message_count}/{total_messages}: Filtered points={len(offset_points)}")

                        except Exception as e:
                            logging.error(f"Exception during processing: {e}")

                    # Increment message count
                    message_count += 1

            # Restart file reading when EOF is reached
            message_count = 0  # Reset counter if needed

        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")

def start_webrtc():
    """Run WebRTC connection in a separate asyncio loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(lidar_webrtc_connection())

if __name__ == "__main__":
    print(f"LIDAR Visualization v{VERSION}")
    print("Using Open3D for visualization")
    
    try:
        # Setup visualization in main thread first
        setup_visualization()
        
        # Start data processing threads
        if args.csv_read:
            print(f"Reading from CSV file: {args.csv_read}")
            csv_thread = threading.Thread(target=lambda: asyncio.run(read_csv_and_update(args.csv_read)), daemon=True)
            csv_thread.start()
        else:
            print("Starting WebRTC connection...")
            webrtc_thread = threading.Thread(target=start_webrtc, daemon=True)
            webrtc_thread.start()

        # Run visualization in main thread (not in separate thread)
        print("Starting visualization...")
        run_visualization()
        
    except KeyboardInterrupt:
        print("Shutting down...")
        visualization_running = False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        visualization_running = False
        # Save any collected stream data
        save_stream_data()