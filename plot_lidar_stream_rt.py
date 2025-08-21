# filepath: plot_lidar_stream_rt.py
""" @MrRobotoW at The RoboVerse Discord """
""" robert.wagoner@gmail.com """
""" 01/30/2025 """
""" Inspired from lidar_stream.py by @legion1581 at The RoboVerse Discord """

VERSION = "1.0.18"

import asyncio
import logging
import csv
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO
import argparse
from datetime import datetime
import os
import sys
import ast
import base64

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

# Flask app and SocketIO setup
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

logging.basicConfig(level=logging.FATAL)

# Constants to enable/disable features
ENABLE_POINT_CLOUD = True
SAVE_LIDAR_DATA = True

# File paths
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LIDAR_CSV_FILE = f"lidar_data_{timestamp}.csv"

# Global variables
lidar_csv_file = None
lidar_csv_writer = None

lidar_buffer = []
message_count = 0  # Counter for processed LIDAR messages
reconnect_interval = 5  # Time (seconds) before retrying connection

# Constants
MAX_RETRY_ATTEMPTS = 10

ROTATE_X_ANGLE = np.pi / 2  # 90 degrees
ROTATE_Z_ANGLE = np.pi      # 90 degrees

minYValue = 0
maxYValue = 100

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
args = parser.parse_args()

minYValue = args.minYValue
maxYValue = args.maxYValue
SAVE_LIDAR_DATA = args.csv_write

@socketio.on('check_args')
def handle_check_args():
    typeFlag = 0b0101 # default iso cam & point cloud
    if args.cam_center:
        typeFlag |= 0b0010
    if args.type_voxel:
        typeFlag &= ~0b1000 # disable point cloud
        typeFlag |=  0b1000  # Set voxel flag
    typeFlagBinary = format(typeFlag, "04b")
    socketio.emit("check_args_ack", {"type": typeFlagBinary})

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
    """Connect to WebRTC and process LIDAR data."""
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

            async def lidar_callback_task(message):
                nonlocal local_message_count
                global message_count
                if not ENABLE_POINT_CLOUD:
                    return

                try:
                    if message_count % args.skip_mod != 0:
                        message_count += 1
                        return

                    positions = message["data"]["data"].get("positions", [])
                    origin = message["data"].get("origin", [])
                    points = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)
                    total_points = len(points)
                    unique_points = np.unique(points, axis=0)

                    if SAVE_LIDAR_DATA and lidar_csv_writer:
                        lidar_csv_writer.writerow([
                            message["data"]["stamp"],
                            message["data"]["frame_id"],
                            message["data"]["resolution"],
                            message["data"]["src_size"],
                            message["data"]["origin"],
                            message["data"]["width"],
                            len(unique_points),
                            unique_points.tolist()
                        ])
                        lidar_csv_file.flush()

                    points = rotate_points(unique_points, ROTATE_X_ANGLE, ROTATE_Z_ANGLE)
                    points = points[(points[:, 1] >= minYValue) & (points[:, 1] <= maxYValue)]

                    center_x = float(np.mean(points[:, 0]))
                    center_y = float(np.mean(points[:, 1]))
                    center_z = float(np.mean(points[:, 2]))
                    offset_points = points - np.array([center_x, center_y, center_z])

                    message_count += 1
                    local_message_count += 1
                    print(f"LIDAR Message {message_count}: Total points={total_points}, Unique points={len(unique_points)}")

                    scalars = np.linalg.norm(offset_points, axis=1)
                    socketio.emit("lidar_data", {
                        "points": offset_points.tolist(),
                        "scalars": scalars.tolist(),
                        "center": {"x": center_x, "y": center_y, "z": center_z}
                    })

                except Exception as e:
                    logging.error(f"Error in LIDAR callback: {e}")

            # Subscribe to LIDAR voxel map messages
            conn.datachannel.pub_sub.subscribe(
                "rt/utlidar/voxel_map_compressed",
                lambda message: asyncio.create_task(lidar_callback_task(message))
            )

            # Keep the connection active, but break after 50 messages
            while True:
                await asyncio.sleep(0.1)
                if local_message_count >= 50:
                    print("Restarting WebRTC connection after 50 messages.")
                    break

            close_csv_output()
            try:
                await conn.disconnect()
            except Exception as e:
                logging.error(f"Error during disconnect: {e}")

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

    logging.error("Max retry attempts reached. Exiting.")

async def read_csv_and_emit(csv_file):
    """Continuously read CSV files and emit data without delay."""
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
                                unique_points = np.unique(points, axis=0)
                                # Calculate center coordinates
                                center_x = float(np.mean(unique_points[:, 0]))
                                center_y = float(np.mean(unique_points[:, 1]))
                                center_z = float(np.mean(unique_points[:, 2]))

                                # Offset points by center coordinates
                                offset_points = unique_points - np.array([center_x, center_y, center_z])
                            else:
                                unique_points = np.empty((0, 3), dtype=np.float32)
                                offset_points = unique_points

                            # Emit data to Socket.IO
                            scalars = np.linalg.norm(offset_points, axis=1)
                            socketio.emit("lidar_data", {
                                "points": offset_points.tolist(),
                                "scalars": scalars.tolist(),
                                "center": {"x": center_x, "y": center_y, "z": center_z}
                            })

                            # Print message details
                            print(f"LIDAR Message {message_count}/{total_messages}: Unique points={len(unique_points)}")

                        except Exception as e:
                            logging.error(f"Exception during processing: {e}")

                    # Increment message count
                    message_count += 1

            # Restart file reading when EOF is reached
            message_count = 0  # Reset counter if needed

        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")

def start_webrtc_video():
    """Run WebRTC connection for video streaming and emit frames to the web UI."""
    import asyncio
    import numpy as np
    from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
    from go2_webrtc_driver.constants import RTC_TOPIC

    async def video_channel():
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.4.30")
        await conn.connect()
        print("Video WebRTC connected.")

        def video_callback(message):
            try:
                # message['data'] is a JPEG byte array
                jpg_bytes = message['data']
                # Encode to base64 for web transmission
                jpg_b64 = base64.b64encode(jpg_bytes).decode('utf-8')
                socketio.emit("video_frame", {"image": jpg_b64})
            except Exception as e:
                print(f"Video callback error: {e}")

        conn.datachannel.pub_sub.subscribe(RTC_TOPIC['CAMERA_STREAM'], video_callback)

        try:
            while True:
                await asyncio.sleep(0.1)
        finally:
            await conn.disconnect()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(video_channel())

@app.route("/")
def index():
    # HTML template with floating LowState and Video subwindows over the 3D pointcloud view
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LIDAR, LowState & Video Viewer</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body { margin: 0; overflow: hidden; }
            #three-canvas { width: 100vw; height: 100vh; display: block; }
            #lowstate-window {
                position: fixed;
                top: 24px;
                right: 24px;
                width: 340px;
                background: rgba(34,34,34,0.97);
                color: #eee;
                border-radius: 10px;
                box-shadow: 0 4px 24px #0008;
                padding: 18px 18px 12px 18px;
                z-index: 1000;
                font-family: monospace;
                font-size: 14px;
                max-height: 80vh;
                overflow-y: auto;
            }
            #lowstate-window h2 {
                margin-top: 0;
                font-size: 1.2em;
                color: #ffb300;
            }
            #lowstate-window table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 8px;
            }
            #lowstate-window th, #lowstate-window td {
                border: 1px solid #888;
                padding: 2px 6px;
                text-align: center;
            }
            #video-window {
                position: fixed;
                bottom: 24px;
                right: 24px;
                width: 340px;
                background: rgba(20,20,20,0.97);
                color: #eee;
                border-radius: 10px;
                box-shadow: 0 4px 24px #0008;
                padding: 10px;
                z-index: 1000;
                text-align: center;
            }
            #video-window h2 {
                margin: 0 0 8px 0;
                font-size: 1.1em;
                color: #90caf9;
            }
            #video-frame {
                width: 320px;
                height: 240px;
                background: #111;
                border-radius: 6px;
                object-fit: contain;
            }
        </style>
    </head>
    <body>
        <div id="lowstate-window">
            <h2>Go2 Robot Status (LowState)</h2>
            <div id="lowstate-content">Waiting for data...</div>
        </div>
        <div id="video-window">
            <h2>Go2 Camera Stream</h2>
            <img id="video-frame" src="" alt="Video stream will appear here"/>
        </div>
        <canvas id="three-canvas"></canvas>
        <script>
            // --- LowState display ---
            var socket = io();
            socket.on("lowstate_data", function(data) {
                let html = "";
                if(data.error) {
                    html = "<b>Error:</b> " + data.error;
                } else {
                    html += "<b>IMU RPY:</b> " + data.imu_rpy.join(", ") + "<br>";
                    html += "<b>Motors:</b><br><table><tr><th>#</th><th>q</th><th>Temp</th><th>Lost</th></tr>";
                    data.motors.forEach((m, i) => {
                        html += `<tr><td>${i+1}</td><td>${m.q}</td><td>${m.temperature}°C</td><td>${m.lost}</td></tr>`;
                    });
                    html += "</table>";
                    html += "<b>BMS:</b> Version " + data.bms.version + ", SOC: " + data.bms.soc + "%, Current: " + data.bms.current + "mA, Cycle: " + data.bms.cycle + "<br>";
                    html += "BQ NTC: " + data.bms.bq_ntc + "°C, MCU NTC: " + data.bms.mcu_ntc + "°C<br>";
                    html += "<b>Foot Force:</b> " + JSON.stringify(data.foot_force) + "<br>";
                    html += "<b>Temperature NTC1:</b> " + data.temperature_ntc1 + "°C<br>";
                    html += "<b>Power Voltage:</b> " + data.power_v + "V<br>";
                }
                document.getElementById("lowstate-content").innerHTML = html;
            });

            // --- Video display ---
            socket.on("video_frame", function(data) {
                if(data.image) {
                    document.getElementById("video-frame").src = "data:image/jpeg;base64," + data.image;
                }
            });

            // --- 3D Pointcloud viewer (minimal setup, you can expand as needed) ---
            let scene = new THREE.Scene();
            let camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
            let renderer = new THREE.WebGLRenderer({canvas: document.getElementById('three-canvas'), antialias: true});
            renderer.setClearColor(0x111111);
            renderer.setSize(window.innerWidth, window.innerHeight);

            let controls = new THREE.OrbitControls(camera, renderer.domElement);
            camera.position.set(0, 0, 5);
            controls.update();

            let pointCloud = null;

            function render() {
                requestAnimationFrame(render);
                controls.update();
                renderer.render(scene, camera);
            }
            render();

            socket.on("lidar_data", function(data) {
                if (!data.points) return;
                if (pointCloud) {
                    scene.remove(pointCloud);
                }
                let geometry = new THREE.BufferGeometry();
                let positions = new Float32Array(data.points.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                let colors = new Float32Array(data.scalars.map(s => [s/10, 0.5, 1-s/10]).flat());
                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                let material = new THREE.PointsMaterial({size: 0.05, vertexColors: true});
                pointCloud = new THREE.Points(geometry, material);
                scene.add(pointCloud);
            });

            window.addEventListener('resize', function() {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
        </script>
    </body>
    </html>
    """, version=VERSION)

def start_webrtc():
    """Run WebRTC connection in a separate asyncio loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(lidar_webrtc_connection())

if __name__ == "__main__":
    import threading
    if args.csv_read:
        csv_thread = threading.Thread(target=lambda: asyncio.run(read_csv_and_emit(args.csv_read)), daemon=True)
        csv_thread.start()
    else:
        webrtc_thread = threading.Thread(target=start_webrtc, daemon=True)
        webrtc_thread.start()

    socketio.run(app, host="127.0.0.1", port=8080, debug=False)