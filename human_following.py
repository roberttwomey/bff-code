"""
Human Following Robot Controller

Video Output Configuration (via .env file):
    VIDEO_OUTPUT=display    - Local window using cv2.imshow (default)
    VIDEO_OUTPUT=mjpeg      - Remote MJPEG streaming (view in browser)
    
    VIDEO_STREAM_PORT=8080  - Port for MJPEG streaming (default: 8080)
    
    UNITREE_GO2_IP=192.168.4.30  - Robot IP address

Example .env:
    VIDEO_OUTPUT=mjpeg
    VIDEO_STREAM_PORT=8080
    UNITREE_GO2_IP=192.168.4.30
"""

import cv2
import numpy as np
import asyncio
import logging
import threading
import time
import os
from queue import Queue
from dotenv import load_dotenv
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
from ultralytics import YOLO
import json

# DDS imports for local connection
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    DDS_AVAILABLE = True
except ImportError:
    DDS_AVAILABLE = False
    print("Note: unitree_sdk2py not available. DDS movement will not be used.")

# Optional streaming support
try:
    from streaming_utils import MJPEGStreamer, RTSPStreamer
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    print("Note: streaming_utils not available. Install Flask for web streaming.")

# Load environment variables from .env file
load_dotenv()

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

# Ethernet interface name for local connection
ETHERNET_INTERFACE = "enP8p1s0"

def is_interface_active(interface_name):
    """Check if the network interface exists and is active (UP)."""
    operstate_path = f"/sys/class/net/{interface_name}/operstate"
    if not os.path.exists(operstate_path):
        return False
    
    try:
        with open(operstate_path, 'r') as f:
            operstate = f.read().strip()
        return operstate == "up"
    except (IOError, OSError):
        return False

def check_local_connection(interface_name=ETHERNET_INTERFACE, timeout=2.0):
    """
    Check if robot is locally connected via ethernet.
    This is a quick check that only verifies the interface is active.
    The actual channel factory initialization should be done separately.
    Returns True if interface is active, False otherwise.
    """
    if not DDS_AVAILABLE:
        return False
    
    return is_interface_active(interface_name)

def verify_dds_connection(interface_name=ETHERNET_INTERFACE, timeout=2.0):
    """
    Verify DDS connection by checking for lowstate messages.
    Channel factory must be initialized before calling this.
    Returns True if messages are received, False otherwise.
    """
    if not DDS_AVAILABLE:
        return False
    
    try:
        # Set up subscriber to check for messages
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        message_received = False
        
        def LowStateHandler(msg: LowState_):
            nonlocal message_received
            message_received = True
        
        sub.Init(LowStateHandler, 10)
        
        # Wait for a message with a timeout
        start_time = time.time()
        while not message_received and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        return message_received
    except Exception as e:
        print(f"Error verifying DDS connection: {e}")
        return False

class HumanFollower:
    """
    Human following robot controller using YOLO detection and visual servoing.
    
    Coordinate System:
    - x: Forward/Backward (positive = forward, negative = backward)
    - y: Left/Right (positive = left, negative = right) 
    - z: Rotation (positive = turn left, negative = turn right)
    
    Movement Behavior:
    - Aggressive forward movement towards detected humans
    - Fast, responsive turning for quick centering
    - Continuous forward movement even when close to humans
    - High-speed control loop (20Hz) for smooth operation
    """
    def __init__(self, ip=None, stream_mode=None, stream_port=None):
        """
        Initialize HumanFollower
        
        Args:
            ip: Robot IP address (overrides .env)
            stream_mode: 'display' (cv2.imshow) or 'mjpeg' (HTTP stream) (overrides .env)
            stream_port: Port for MJPEG streaming (overrides .env)
        """
        # Read IP from environment variable, fallback to provided ip or default
        self.ip = ip or os.getenv('UNITREE_GO2_IP', '192.168.4.30')
        self.frame_queue = Queue()
        self.conn = None
        self.loop = None
        self.asyncio_thread = None
        
        # DDS connection state
        self.use_dds = False
        self.sport_client = None
        self.ethernet_interface = ETHERNET_INTERFACE
        
        # Streaming setup - read from .env if not provided
        # VIDEO_OUTPUT can be 'display' (local window) or 'mjpeg' (remote streaming)
        env_video_output = os.getenv('VIDEO_OUTPUT', 'display').lower()
        self.stream_mode = stream_mode or env_video_output
        
        # Validate stream_mode
        if self.stream_mode not in ['display', 'mjpeg']:
            print(f"Warning: Invalid VIDEO_OUTPUT '{self.stream_mode}', defaulting to 'display'")
            self.stream_mode = 'display'
        
        # Read port from .env if not provided
        env_port = os.getenv('VIDEO_STREAM_PORT', '8080')
        try:
            self.stream_port = stream_port or int(env_port)
        except ValueError:
            print(f"Warning: Invalid VIDEO_STREAM_PORT '{env_port}', using default 8080")
            self.stream_port = 8080
        
        self.streamer = None
        
        # Initialize streaming based on mode
        if self.stream_mode == 'mjpeg':
            if STREAMING_AVAILABLE:
                self.streamer = MJPEGStreamer(port=self.stream_port)
                self.streamer.start()
                print(f"Video output: MJPEG streaming at http://localhost:{self.stream_port}")
                print(f"  View in browser: http://localhost:{self.stream_port}")
            else:
                print("Warning: MJPEG streaming requested but Flask not available.")
                print("  Install Flask: pip install flask")
                print("  Falling back to local display mode.")
                self.stream_mode = 'display'
        elif self.stream_mode == 'display':
            print("Video output: Local window (cv2.imshow)")
        
        # Load YOLO model for human detection
        self.model = YOLO('yolov8n.pt')  # Using nano model for speed
        
        # Visual servoing parameters
        self.target_center_x = 640  # Target center of frame (assuming 1280x720)
        self.center_tolerance = 5  # Reduced tolerance for more precise centering
        self.turn_speed = 1.5  # Increased angular velocity for faster turning
        self.move_speed = 1.25#1.5  # Increased forward movement speed for faster approach
        self.min_human_confidence = 0.75  # Minimum confidence for human detection
        
        # Control flags
        self.is_following = False
        self.last_human_detected = None
        self.current_movement = {'x': 0, 'y': 0, 'z': 0}
        self.movement_task = None
        
        # Movement control parameters
        self.command_rate_hz = 10  # Commands per second (lower = slower, more stable)
        # Command rate options:
        # - 20 Hz (0.05s): Very responsive, may be jerky
        # - 10 Hz (0.1s): Balanced responsiveness and stability (default)
        # - 5 Hz (0.2s): Slower, more stable movement
        # - 2 Hz (0.5s): Very slow, very stable
        
    async def connect(self):
        """Connect to the Go2 robot - tries DDS first, falls back to WebRTC"""
        # Check for local connection first
        if DDS_AVAILABLE:
            print("Checking for local ethernet connection...")
            if check_local_connection(self.ethernet_interface):
                print(f"Ethernet interface {self.ethernet_interface} is active, initializing DDS...")
                try:
                    # Initialize DDS channel factory
                    ChannelFactoryInitialize(0, self.ethernet_interface)
                    
                    # Verify connection by checking for messages
                    if verify_dds_connection(self.ethernet_interface):
                        print("DDS connection verified")
                        # Initialize SportClient for movement
                        self.sport_client = SportClient()
                        self.sport_client.SetTimeout(10.0)
                        self.sport_client.Init()
                        
                        self.use_dds = True
                        print("DDS movement control initialized")
                        
                        # Still need WebRTC for video streaming
                        print("Connecting via WebRTC for video streaming...")
                        self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
                        await self.conn.connect()
                        print("WebRTC connected for video streaming")
                        
                        return True
                    else:
                        print("DDS connection verification failed - no messages received")
                        self.use_dds = False
                except Exception as e:
                    print(f"Failed to initialize DDS: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Falling back to WebRTC for both video and movement...")
                    self.use_dds = False
            else:
                print("No local ethernet connection detected, using WebRTC for both video and movement...")
        
        # Fallback to WebRTC for both video and movement
        try:
            self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
            await self.conn.connect()
            print("Connected to Go2 robot via WebRTC")
            
            # Switch to normal mode for movement control
            await self.switch_to_normal_mode()
            
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    async def switch_to_normal_mode(self):
        """Switch robot to normal motion mode"""
        try:
            print("Checking current motion mode...")
            # Get current motion mode
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], 
                {"api_id": 1001}
            )
            
            print(f"Motion mode response: {response}")
            
            if response and 'data' in response and 'header' in response['data']:
                status = response['data']['header']['status']['code']
                if status == 0:
                    data = json.loads(response['data']['data'])
                    current_mode = data['name']
                    print(f"Current motion mode: {current_mode}")
                    
                    # Switch to normal mode if not already
                    if current_mode != "normal":
                        print("Switching to normal motion mode...")
                        switch_response = await self.conn.datachannel.pub_sub.publish_request_new(
                            RTC_TOPIC["MOTION_SWITCHER"], 
                            {
                                "api_id": 1002,
                                "parameter": {"name": "normal"}
                            }
                        )
                        print(f"Switch response: {switch_response}")
                        await asyncio.sleep(5)  # Wait for mode switch
                        print("Switched to normal mode")
                    else:
                        print("Already in normal mode")
                else:
                    print(f"Failed to get motion mode, status: {status}")
            else:
                print("Invalid response format from motion mode request")
                
        except Exception as e:
            print(f"Error switching motion mode: {e}")
            import traceback
            traceback.print_exc()
    
    async def move_robot(self, x=0, y=0, z=0):
        """Move the robot with specified velocities - uses DDS if available, otherwise WebRTC"""
        try:
            # Ensure all values are Python native types
            x, y, z = float(x), float(y), float(z)
            
            # Use DDS if available and initialized
            if self.use_dds and self.sport_client:
                # DDS movement (synchronous, non-blocking)
                try:
                    code = self.sport_client.Move(x, y, z)
                    if code == 0:
                        print(f"DDS movement command: x={x:.3f}, y={y:.3f}, z={z:.3f}")
                    else:
                        print(f"DDS movement command failed with code: {code}")
                except Exception as e:
                    print(f"Error in DDS movement: {e}")
                    # Fallback to WebRTC if DDS fails
                    self.use_dds = False
                    await self._move_robot_webrtc(x, y, z)
            else:
                # WebRTC movement
                await self._move_robot_webrtc(x, y, z)
                
        except Exception as e:
            print(f"Error moving robot: {e}")
            import traceback
            traceback.print_exc()
    
    async def _move_robot_webrtc(self, x=0, y=0, z=0):
        """Move the robot using WebRTC (internal method)"""
        try:
            print(f"Sending WebRTC movement command: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], 
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": x, "y": y, "z": z}
                }
            )
            
            # Check if the command was successful
            if response and 'data' in response and 'header' in response['data']:
                status = response['data']['header']['status']['code']
                if status == 0:
                    print(f"WebRTC movement command successful: x={x:.3f}, y={y:.3f}, z={z:.3f}")
                else:
                    print(f"WebRTC movement command failed with status: {status}")
            else:
                print("No response received from WebRTC movement command")
                
        except Exception as e:
            print(f"Error in WebRTC movement: {e}")
            import traceback
            traceback.print_exc()
    
    async def stop_robot(self):
        """Stop the robot movement"""
        print("Stopping robot movement")
        self.current_movement = {'x': 0, 'y': 0, 'z': 0}
        await self.move_robot(0, 0, 0)
    
    async def test_movement(self):
        """Test basic movement to verify robot can move"""
        print("Testing robot movement...")
        
        # Test forward movement at higher speed
        print("Testing forward movement...")
        await self.move_robot(1.0, 0, 0)
        await asyncio.sleep(2)
        
        # Test turning at higher speed
        print("Testing left turn...")
        await self.move_robot(0, 0, 1.0)
        await asyncio.sleep(2)
        
        # Test right turn at higher speed
        print("Testing right turn...")
        await self.move_robot(0, 0, -1.0)
        await asyncio.sleep(2)
        
        # Test combined movement (forward + turn)
        print("Testing combined movement (forward + turn)...")
        await self.move_robot(1.0, 0, 0.5)
        await asyncio.sleep(2)
        
        # Stop
        await self.stop_robot()
        print("Movement test completed")
    
    async def continuous_movement_task(self):
        """Task that continuously sends movement commands"""
        while self.is_following:
            try:
                # Send current movement command
                if (abs(self.current_movement['x']) > 0.01 or 
                    abs(self.current_movement['y']) > 0.01 or 
                    abs(self.current_movement['z']) > 0.01):
                    
                    # For DDS, we can call it directly (it's non-blocking)
                    if self.use_dds and self.sport_client:
                        try:
                            self.sport_client.Move(
                                self.current_movement['x'],
                                self.current_movement['y'], 
                                self.current_movement['z']
                            )
                        except Exception as e:
                            print(f"Error in DDS movement task: {e}")
                            # Fallback to WebRTC
                            self.use_dds = False
                            await self._move_robot_webrtc(
                                self.current_movement['x'],
                                self.current_movement['y'], 
                                self.current_movement['z']
                            )
                    else:
                        # WebRTC movement (async)
                        await self.move_robot(
                            self.current_movement['x'],
                            self.current_movement['y'], 
                            self.current_movement['z']
                        )
                
                # Send commands at regular intervals
                await asyncio.sleep(1.0 / self.command_rate_hz)  # Configurable command rate
                
            except Exception as e:
                print(f"Error in continuous movement task: {e}")
                await asyncio.sleep(0.1)
    
    def detect_humans(self, frame):
        """Detect humans in the frame using YOLO"""
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            humans = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detected object is a person (class 0 in COCO dataset)
                        if box.cls == 0 and box.conf > self.min_human_confidence:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            humans.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                'confidence': float(confidence),
                                'area': (x2 - x1) * (y2 - y1)
                            })
            
            return humans
        except Exception as e:
            print(f"Error in human detection: {e}")
            return []
    
    def calculate_control_signals(self, humans):
        """Calculate control signals based on detected humans"""
        if not humans:
            return 0, 0, 0  # No movement
        
        # Find the largest human (closest to camera)
        largest_human = max(humans, key=lambda h: h['area'])
        
        # Calculate horizontal offset from center
        center_x = largest_human['center'][0]
        offset_x = center_x - self.target_center_x
        
        # Calculate control signals
        turn_z = 0
        move_x = 0
        
        # Turn towards human if not centered - more aggressive turning
        if abs(offset_x) > self.center_tolerance:
            # More aggressive turn speed calculation for faster response
            turn_z = -1.0 * np.clip(offset_x / self.target_center_x * self.turn_speed * 1.5, -self.turn_speed, self.turn_speed)
        else:
            # Small corrections even when close to center
            turn_z = -0.3 * np.clip(offset_x / self.target_center_x, -0.5, 0.5)
        
        # Always move forward (positive x direction) when human is detected
        # More aggressive forward movement
        area_ratio = largest_human['area'] / (1280 * 720)  # Normalize by frame area
        
        if area_ratio < 0.25:  # Increased threshold to allow closer approach
            # Base forward speed
            base_speed = self.move_speed
            
            # Less aggressive slowdown as human gets closer
            distance_factor = 1 - area_ratio * 2  # Reduced from 3 to 2
            
            # Less penalty for not being perfectly centered
            centering_factor = 1 - (abs(offset_x) / self.target_center_x) * 0.3  # Reduced from 0.5 to 0.3
            
            # Combine factors to get final forward speed (always positive)
            move_x = base_speed * distance_factor * centering_factor
            
            # Higher minimum forward movement for more aggressive approach
            move_x = max(move_x, 0.3)  # Increased from 0.05 to 0.3
        else:
            # Even when close, maintain some forward movement
            move_x = 0.1
        
        # Convert numpy types to Python native types for JSON serialization
        return float(move_x), 0.0, float(turn_z)
    
    async def recv_camera_stream(self, track: MediaStreamTrack):
        """Receive video frames and process them"""
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                self.frame_queue.put(img)
            except Exception as e:
                print(f"Error receiving frame: {e}")
                break
    
    def run_asyncio_loop(self, loop):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(loop)
        
        async def setup():
            try:
                # Switch video channel on and start receiving video frames
                self.conn.video.switchVideoChannel(True)
                
                # Add callback to handle received video frames
                self.conn.video.add_track_callback(self.recv_camera_stream)
                
                print("Video stream started")
            except Exception as e:
                print(f"Error in WebRTC connection: {e}")
        
        loop.run_until_complete(setup())
        loop.run_forever()
    
    async def start_following(self):
        """Start the human following behavior"""
        print("Starting human following...")
        self.is_following = True
        
        # Create a new event loop for the asyncio code
        self.loop = asyncio.new_event_loop()
        
        # Start the asyncio event loop in a separate thread
        self.asyncio_thread = threading.Thread(target=self.run_asyncio_loop, args=(self.loop,))
        self.asyncio_thread.start()
        
        # Wait a bit for video stream to start
        await asyncio.sleep(2)
        
        # Start continuous movement task
        self.movement_task = asyncio.create_task(self.continuous_movement_task())
        
        try:
            while self.is_following:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Detect humans
                    humans = self.detect_humans(frame)
                    
                    if humans:
                        # Calculate control signals
                        move_x, move_y, turn_z = self.calculate_control_signals(humans)
                        
                        # Update current movement for continuous task (ensure Python native types)
                        self.current_movement = {'x': float(move_x), 'y': float(move_y), 'z': float(turn_z)}
                        
                        # Debug: Print types to ensure they're Python native
                        print(f"Movement types - x: {type(move_x)}, y: {type(move_y)}, z: {type(turn_z)}")
                        
                        # Log movement commands
                        if abs(move_x) > 0.01 or abs(turn_z) > 0.01:
                            print(f"Movement command: x={move_x:.2f} (forward), y={move_y:.2f}, z={turn_z:.2f}")
                            if move_x > 0:
                                print(f"  -> Moving FORWARD (positive x) at speed {move_x:.2f}")
                            if abs(turn_z) > 0.01:
                                direction = "LEFT" if turn_z > 0 else "RIGHT"
                                print(f"  -> Turning {direction} at speed {abs(turn_z):.2f}")
                        
                        # Draw detection on frame
                        for human in humans:
                            x1, y1, x2, y2 = human['bbox']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Human: {human['confidence']:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        self.last_human_detected = time.time()
                    else:
                        # No humans detected, stop moving
                        if self.last_human_detected and time.time() - self.last_human_detected > 2:
                            self.current_movement = {'x': 0, 'y': 0, 'z': 0}
                            print("No humans detected, stopped moving")
                    
                    # Display or stream the frame
                    if self.stream_mode == 'display':
                        cv2.imshow('Human Following', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    elif self.stream_mode == 'mjpeg' and self.streamer:
                        # Stream the frame via MJPEG
                        self.streamer.update_frame(frame)
                else:
                    await asyncio.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("Stopping human following...")
        finally:
            # Cancel movement task
            if self.movement_task:
                self.movement_task.cancel()
                try:
                    await self.movement_task
                except asyncio.CancelledError:
                    pass
            
            await self.stop_robot()
            if self.stream_mode == 'display':
                cv2.destroyAllWindows()
            if self.streamer:
                self.streamer.stop()
            self.is_following = False
            
            # Stop the asyncio event loop
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.asyncio_thread:
                self.asyncio_thread.join()
    
    async def run(self):
        """Main run method"""
        if await self.connect():
            await self.start_following()

async def main():
    # Video output mode is read from .env file:
    # VIDEO_OUTPUT=display  -> Local window (cv2.imshow)
    # VIDEO_OUTPUT=mjpeg    -> Remote streaming (HTTP, view in browser)
    # VIDEO_STREAM_PORT=8080 -> Port for MJPEG streaming (default: 8080)
    
    follower = HumanFollower()  # Settings read from .env
    
    # Ask user if they want to test movement first
    # print("\n1. Test robot movement first")
    # print("2. Start human following directly")
    
    try:
        # choice = input("Enter your choice (1 or 2): ").strip()
        
        # if choice == "1":
        #     print("\nTesting robot movement...")
        #     if await follower.connect():
        #         await follower.test_movement()
        #         print("\nMovement test completed. Starting human following...")
        #         await follower.start_following()
        # elif choice == "2":
        #     print("\nStarting human following...")
        #     await follower.run()
        # else:
        #     print("Invalid choice. Starting human following...")
        #     await follower.run()

        print("Starting human following...")
        await follower.run()
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
