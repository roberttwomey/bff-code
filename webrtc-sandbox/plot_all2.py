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
import time
from aiortc import MediaStreamTrack
from queue import Queue
import cv2
import base64

# Create an OpenCV window and display a blank image
# height, width = 720, 1280  # Adjust the size as needed
# img = np.zeros((height, width, 3), dtype=np.uint8)
# cv2.imshow('Video', img)
# cv2.waitKey(1)  # Ensure the window is created

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

# Queue for frames received from video channel
frame_queue = Queue()


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

@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>LIDAR Viz v{{ version }}</title>
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
                max-height: 90vh;
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
                top: 24px;
                left: 24px;
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
            let scene, camera, renderer, controls, pointCloud, voxelMesh;
            let voxelSize = 1.0;
            let transparency = .5;
            let wireframe = false;
            let lightIntensity = .5;
            let pointCloudEnable = 1;
            let pollingInterval;
            const socket = io();
                                  
            // --- LowState display ---
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
                                  
            document.addEventListener("DOMContentLoaded", () => {                                 
                function init() {
                    // Initialize the scene
                    const scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x333333);

                    const sceneRotationDegrees = -90;  // Change this to any angle (e.g., 90, 180, -90)
                    scene.rotation.y = THREE.MathUtils.degToRad(sceneRotationDegrees); // Convert to radians

                    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
                    camera.position.set(-100, 100, -100); // Adjust camera for rotated scene
                    camera.lookAt(0, 0, 0); // Ensure it's looking at the center

                    // Initialize the renderer
                    // const renderer = new THREE.WebGLRenderer({ antialias: true });
                    let renderer = new THREE.WebGLRenderer({canvas: document.getElementById('three-canvas'), antialias: true});

                    renderer.setSize(window.innerWidth, window.innerHeight);
                    document.body.appendChild(renderer.domElement);

                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.target.set(0, 0, 0); // Ensure the rotation works around the center
                    controls.enableDamping = true; // Smooth movement
                    controls.dampingFactor = 0.05;
                    controls.maxPolarAngle = Math.PI; // Allow full rotation
                    controls.screenSpacePanning = true;
                    controls.update();

                    const ambientLight = new THREE.AmbientLight(0x555555, 0.5); // Soft background light
                    scene.add(ambientLight);

                    const directionalLight = new THREE.DirectionalLight(0xffffff, 1); // Main light source
                    directionalLight.position.set(0, 100, 0); // Position above the scene
                    directionalLight.castShadow = true;
                    scene.add(directionalLight);

                    const axesHelper = new THREE.AxesHelper(5);
                    scene.add(axesHelper);
                                                                                                                              
                    socket.on("connect", () => {
                        console.log("Socket connected...");
                        pollArgs();
                    });     
                                  
                    socket.on("check_args_ack", (data) => {
                        console.log("Received check_args event:", data);
                         const typeFlag = parseInt(data.type, 2);                         
                        if (typeFlag & 0b0001) {  
                            camera.position.set(-100, 100, -100); // Adjust camera for rotated scene
                            camera.lookAt(0, 0, 0); // Ensure it's looking at the center
                        }
                        if (typeFlag & 0b0010) {  
                            camera.position.set(0, 0, 10); // Set camera at the center of the scene
                            camera.lookAt(0, 0, -1); // Look slightly forward       
                         }
                        if (typeFlag & 0b0100) {  
                            pointCloudEnable = 1;
                            console.log("ptcloud:", pointCloudEnable);
                        }
                        if (typeFlag & 0b1000) {  
                            pointCloudEnable = 0;
                            console.log("ptcloud:", pointCloudEnable);
                        }
                        controls.update();
                        clearInterval(pollingInterval);
                     });
                                  
                    // Handle LIDAR data
                    socket.on("lidar_data", (data) => {
                        if (!data.handled) {
                            data.handled = true; // Prevent re-triggering
                            console.log("Received LIDAR data");
                            const points = data.points || [];
                            const scalars = data.scalars || [];

                            if (pointCloudEnable > 0) {
                                if (pointCloud) scene.remove(pointCloud);
                                if (voxelMesh) {
                                    scene.remove(voxelMesh);
                                    voxelMesh = null;
                                }

                                const geometry = new THREE.BufferGeometry();
                                const vertices = new Float32Array(points.flat());
                                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

                                const colors = new Float32Array(scalars.length * 3);
                                const maxScalar = Math.max.apply(null, scalars);
                                scalars.forEach((scalar, i) => {
                                    const color = new THREE.Color();
                                    color.setHSL(scalar / maxScalar, 1.0, 0.5);
                                    colors.set([color.r, color.g, color.b], i * 3);
                                });

                                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

                                //const material = new THREE.PointsMaterial({ size: 0.3, vertexColors: true });
                                const material = new THREE.PointsMaterial({ size: 1.0, vertexColors: true });
                                pointCloud = new THREE.Points(geometry, material);

                                scene.add(pointCloud);
                            } else {
                                if (voxelMesh) scene.remove(voxelMesh);
                                voxelMesh = createVoxelMesh(points, scalars, voxelSize, Infinity);
                                if (voxelMesh instanceof THREE.Object3D) {
                                    scene.add(voxelMesh);
                                }
                                // Remove any existing point cloud
                                if (pointCloud) {
                                    scene.remove(pointCloud);
                                    pointCloud = null;
                                }
                            }
                        }
                    });

                    function animate() {
                        requestAnimationFrame(animate);
                        controls.update();
                        renderer.render(scene, camera);
                    }

                    animate();
                }

                init();
            });

            function pollArgs() {
                pollingInterval = setInterval(() => {
                    socket.emit('check_args');
                }, 1000); // Poll every second
            }                     
                                     
            /**
            * Creates a voxel mesh from point data and scalar data.
            */
            function createVoxelMesh(points, scalars, voxelSize, maxVoxelsToShow = Infinity) {
                const geometry = new THREE.BufferGeometry();

                try {
                    // Precompute cube vertex offsets
                    const halfSize = voxelSize / 2;
                    const cubeVertexOffsets = [
                        [-halfSize, -halfSize, -halfSize],
                        [halfSize, -halfSize, -halfSize],
                        [halfSize, halfSize, -halfSize],
                        [-halfSize, halfSize, -halfSize],
                        [-halfSize, -halfSize, halfSize],
                        [halfSize, -halfSize, halfSize],
                        [halfSize, halfSize, halfSize],
                        [-halfSize, halfSize, halfSize]
                    ];

                    // Precompute indices for a unit cube
                    const cubeIndices = [
                        0, 1, 2, 2, 3, 0, // Back
                        4, 5, 6, 6, 7, 4, // Front
                        0, 1, 5, 5, 4, 0, // Bottom
                        2, 3, 7, 7, 6, 2, // Top
                        0, 3, 7, 7, 4, 0, // Left
                        1, 2, 6, 6, 5, 1  // Right
                    ];

                    const maxVoxels = Math.min(maxVoxelsToShow, points.length);
                    const maxScalar = Math.max(...scalars);

                    // Typed arrays for better performance
                    const positions = new Float32Array(maxVoxels * 8 * 3); // 8 vertices * 3 coords per voxel
                    const colors = new Float32Array(maxVoxels * 8 * 3);    // 8 vertices * 3 color channels per voxel
                    const indices = new Uint32Array(maxVoxels * 36);       // 12 triangles (36 indices) per voxel

                    let positionOffset = 0;
                    let colorOffset = 0;
                    let indexOffset = 0;

                    for (let i = 0; i < maxVoxels; i++) {
                        const centerX = points[i][0];
                        const centerY = points[i][1];
                        const centerZ = points[i][2];

                        // Compute color based on scalar
                        const normalizedScalar = scalars[i] / maxScalar;
                        const color = new THREE.Color();
                        color.setHSL(normalizedScalar * 0.7, 1.0, 0.5);

                        // Add vertices and colors
                        for (let j = 0; j < 8; j++) {
                            const [dx, dy, dz] = cubeVertexOffsets[j];
                            positions[positionOffset++] = centerX + dx;
                            positions[positionOffset++] = centerY + dy;
                            positions[positionOffset++] = centerZ + dz;

                            colors[colorOffset++] = color.r;
                            colors[colorOffset++] = color.g;
                            colors[colorOffset++] = color.b;
                        }

                        // Add indices with offsets
                        for (let j = 0; j < cubeIndices.length; j++) {
                            indices[indexOffset++] = cubeIndices[j] + i * 8;
                        }
                    }

                    // Set attributes and indices in the geometry
                    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
                    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
                    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

                    } catch (error) {
                                  THREE.Cache.clear();
                        if (error instanceof RangeError) {
                            console.error("🚨 Array buffer allocation failed:", error);

                            // Clear existing memory to free up space
                            THREE.Cache.clear();

                            // Optional: Trigger garbage collection (only works in Chrome DevTools)
                            if (window.gc) window.gc();

                            // Return an empty geometry to prevent further errors
                            return new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
                        } else {
                            throw error;  // Re-throw other errors
                        }
                    }

                // Create the material
                const material = new THREE.MeshBasicMaterial({
                    vertexColors: true,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: transparency,
                    wireframe: wireframe
                });

                // Return the voxel mesh
                return new THREE.Mesh(geometry, material);
            }
        </script>
    </body>
    </html>
    """, version=VERSION)

def display_data(message):

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

def start_webrtc():
    """Run WebRTC connection in a separate asyncio loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(webrtc_lidar_and_lowstate())

async def webrtc_lidar_and_lowstate():
    """Connect to WebRTC and process both LIDAR and LowState data using the same connection."""
    global lidar_buffer, message_count
    retry_attempts = 0

    while retry_attempts < MAX_RETRY_ATTEMPTS:
        try:
            # Adjust IP as needed for your robot
            conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.4.30")
            logging.info("Connecting to WebRTC...")
            await conn.connect()
            logging.info("Connected to WebRTC.")
            retry_attempts = 0  # Reset retry attempts on successful connection

            await conn.datachannel.disableTrafficSaving(True)
            conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
            setup_csv_output()

            # Track messages for this connection
            local_message_count = 0

            # --- LIDAR callback ---
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

                    scalars = np.linalg.norm(offset_points, axis=1);
                    socketio.emit("lidar_data", {
                        "points": offset_points.tolist(),
                        "scalars": scalars.tolist(),
                        "center": {"x": center_x, "y": center_y, "z": center_z}
                    });

                except Exception as e:
                    logging.error(f"Error in LIDAR callback: {e}")

            # --- LowState callback ---
            def lowstate_callback(message):
                try:
                    current_message = message['data']
                    # Extract and format lowstate data for web display
                    imu_state = current_message['imu_state']['rpy']
                    motor_state = current_message['motor_state']
                    bms_state = current_message['bms_state']
                    foot_force = current_message['foot_force']
                    temperature_ntc1 = current_message['temperature_ntc1']
                    power_v = current_message['power_v']

                    motors = [
                        {
                            "q": round(motor['q'], 4),
                            "temperature": motor['temperature'],
                            "lost": motor['lost']
                        }
                        for motor in motor_state
                    ]

                    lowstate_dict = {
                        "imu_rpy": [round(x, 3) for x in imu_state],
                        "motors": motors,
                        "bms": {
                            "version": f"{bms_state['version_high']}.{bms_state['version_low']}",
                            "soc": bms_state['soc'],
                            "current": bms_state['current'],
                            "cycle": bms_state['cycle'],
                            "bq_ntc": bms_state['bq_ntc'],
                            "mcu_ntc": bms_state['mcu_ntc'],
                        },
                        "foot_force": foot_force,
                        "temperature_ntc1": temperature_ntc1,
                        "power_v": power_v
                    }
                    socketio.emit("lowstate_data", lowstate_dict)
                    # print(f"lowstate_data: {lowstate_dict}");
                    display_data(current_message)

                except Exception as e:
                    logging.error(f"Error in LowState callback: {e}")

            # Subscribe to LIDAR voxel map messages
            conn.datachannel.pub_sub.subscribe(
                "rt/utlidar/voxel_map_compressed",
                lambda message: asyncio.create_task(lidar_callback_task(message))
            )

            # Subscribe to LowState messages
            from go2_webrtc_driver.constants import RTC_TOPIC
            conn.datachannel.pub_sub.subscribe(
                RTC_TOPIC['LOW_STATE'],
                lambda message: asyncio.create_task(lowstate_callback(message))
                # lowstate_callback
            )

            # --- Video setup & callback (same connection) ---
            async def recv_camera_stream(track: MediaStreamTrack):
                while True:
                    frame = await track.recv()
                    img = frame.to_ndarray(format="bgr24")
                    frame_queue.put(img)

                    # Correctly encode the frame (img) as JPEG and then to base64
                    try:
                        # img is already in BGR format suitable for OpenCV
                        ret, jpg_bytes = cv2.imencode('.jpg', img)
                        if ret:
                            frame_bytes = jpg_bytes.tobytes()
                            jpg_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                            socketio.emit("video_frame", {"image": jpg_b64})
                    except Exception as e:
                        print(f"JPEG encoding error: {e}")

            # Turn on video channel and add callback
            try:
                # Enable video channel and register track callback
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as e:
                logging.error(f"Error enabling video channel: {e}")


            # Keep the connection active, but restart after 50 LIDAR messages
            while True:
                await asyncio.sleep(0.1)
                if local_message_count >= 50:
                    console.log("Restarting WebRTC connection after 50 messages.");
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

    logging.error("Max retry attempts reached. Exiting.");


def video_display_loop():
    """Continuously read frames from the queue and display them using OpenCV."""
    try:
        while True:
            if not frame_queue.empty():
                img = frame_queue.get()
                cv2.imshow('Go2 Camera Stream', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import threading
    if args.csv_read:
        csv_thread = threading.Thread(target=lambda: asyncio.run(read_csv_and_emit(args.csv_read)), daemon=True)
        csv_thread.start()
    else:
        webrtc_thread = threading.Thread(target=start_webrtc, daemon=True)
        webrtc_thread.start()

        # # Start OpenCV display loop for WebRTC video frames
        # video_thread = threading.Thread(target=video_display_loop, daemon=True)
        # video_thread.start()

        # Optional: legacy JPEG stream via datachannel topic
        # webrtc_video_thread = threading.Thread(target=start_webrtc_video, daemon=True)
        # webrtc_video_thread.start()

    socketio.run(app, host="127.0.0.1", port=8080, debug=False)