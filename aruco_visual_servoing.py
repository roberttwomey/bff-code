import cv2
import cv2.aruco as aruco
import numpy as np
import asyncio
import logging
import threading
import time
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
import json

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

class ArucoVisualServoing:
    """
    Aruco tag visual servoing controller for Go2 robot.
    
    Coordinate System:
    - x: Forward/Backward (positive = forward, negative = backward)
    - y: Left/Right (positive = left, negative = right) 
    - z: Rotation (positive = turn left, negative = turn right)
    
    Behavior:
    - Searches for Aruco tag by rotating
    - Once found, centers the tag horizontally
    - Moves forward/backward to achieve 1 meter distance
    - Maintains perpendicular orientation to the tag
    """
    def __init__(self, ip="192.168.4.30"):
        self.ip = ip
        self.frame_queue = Queue()
        self.conn = None
        self.loop = None
        self.asyncio_thread = None
        
        # Initialize Aruco detector
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Visual servoing parameters
        self.target_center_x = 640  # Target center of frame (assuming 1280x720)
        self.center_tolerance = 30  # Tolerance for centering
        self.target_distance = 1.0  # Target distance in meters
        self.distance_tolerance = 0.1  # Distance tolerance in meters
        self.tag_size = 0.2  # Physical size of Aruco tag in meters (adjust as needed)
        
        # Control parameters
        self.turn_speed = 0.8  # Angular velocity for turning
        self.move_speed = 0.6  # Forward/backward movement speed
        self.search_speed = 1.0  # Speed for searching rotation
        
        # Control flags
        self.is_running = False
        self.current_movement = {'x': 0, 'y': 0, 'z': 0}
        self.movement_task = None
        
        # State tracking
        self.state = "searching"  # "searching", "centering", "approaching", "positioned"
        self.last_tag_detected = None
        self.command_rate_hz = 10  # Commands per second (slower for debugging)
        
    async def connect(self):
        """Connect to the Go2 robot"""
        try:
            self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
            await self.conn.connect()
            print("Connected to Go2 robot")
            
            # Switch to MCF mode for movement control
            await self.switch_to_mcf_mode()
            
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    async def switch_to_mcf_mode(self):
        """Switch robot to MCF (Manual Control Foot) motion mode"""
        try:
            print("Checking current motion mode...")
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], 
                {"api_id": 1001}
            )
            
            if response and 'data' in response and 'header' in response['data']:
                status = response['data']['header']['status']['code']
                if status == 0:
                    data = json.loads(response['data']['data'])
                    current_mode = data['name']
                    print(f"Current motion mode: {current_mode}")
                    
                    if current_mode != "mcf":
                        print("Switching to MCF motion mode...")
                        switch_response = await self.conn.datachannel.pub_sub.publish_request_new(
                            RTC_TOPIC["MOTION_SWITCHER"], 
                            {
                                "api_id": 1002,
                                "parameter": {"name": "mcf"}
                            }
                        )
                        await asyncio.sleep(5)  # Wait for mode switch
                        print("Switched to MCF mode")
                    else:
                        print("Already in MCF mode")
                else:
                    print(f"Failed to get motion mode, status: {status}")
            else:
                print("Invalid response format from motion mode request")
                
        except Exception as e:
            print(f"Error switching to MCF motion mode: {e}")
    
    async def move_robot(self, x=0, y=0, z=0):
        """Move the robot with specified velocities"""
        try:
            x, y, z = float(x), float(y), float(z)
            print(f"Sending movement command: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], 
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": x, "y": y, "z": z}
                }
            )
            
            if response and 'data' in response and 'header' in response['data']:
                status = response['data']['header']['status']['code']
                if status == 0:
                    print(f"Movement command successful: x={x:.3f}, y={y:.3f}, z={z:.3f}")
                else:
                    print(f"Movement command failed with status: {status}")
            else:
                print("No response received from movement command")
                
        except Exception as e:
            print(f"Error moving robot: {e}")
    
    async def stop_robot(self):
        """Stop the robot movement"""
        print("Stopping robot movement")
        self.current_movement = {'x': 0, 'y': 0, 'z': 0}
        await self.move_robot(0, 0, 0)
    
    async def test_movement(self):
        """Test basic movement to verify robot can move"""
        print("Testing robot movement...")
        
        # Test turning
        print("Testing left turn...")
        await self.move_robot(0, 0, 1.0)
        await asyncio.sleep(2)
        
        print("Testing right turn...")
        await self.move_robot(0, 0, -1.0)
        await asyncio.sleep(2)
        
        # Stop
        await self.stop_robot()
        print("Movement test completed")
    
    async def continuous_movement_task(self):
        """Task that continuously sends movement commands"""
        while self.is_running:
            try:
                # Debug: Print current movement values
                print(f"Movement task - Current: x={self.current_movement['x']:.3f}, y={self.current_movement['y']:.3f}, z={self.current_movement['z']:.3f}")
                
                if (abs(self.current_movement['x']) > 0.01 or 
                    abs(self.current_movement['y']) > 0.01 or 
                    abs(self.current_movement['z']) > 0.01):
                    
                    print(f"Sending movement command: x={self.current_movement['x']:.3f}, y={self.current_movement['y']:.3f}, z={self.current_movement['z']:.3f}")
                    await self.move_robot(
                        self.current_movement['x'],
                        self.current_movement['y'], 
                        self.current_movement['z']
                    )
                else:
                    print("No movement - values too small")
                
                await asyncio.sleep(1.0 / self.command_rate_hz)
                
            except Exception as e:
                print(f"Error in continuous movement task: {e}")
                await asyncio.sleep(0.1)
    
    def detect_aruco_tags(self, frame):
        """Detect Aruco tags in the frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            tags = []
            if ids is not None:
                for i, corner in enumerate(corners):
                    tag_id = ids[i][0]
                    corners_reshaped = corner.reshape((4, 2))
                    
                    # Calculate tag center
                    center_x = int(np.mean(corners_reshaped[:, 0]))
                    center_y = int(np.mean(corners_reshaped[:, 1]))
                    
                    # Calculate tag size in pixels for distance estimation
                    width = np.linalg.norm(corners_reshaped[0] - corners_reshaped[1])
                    height = np.linalg.norm(corners_reshaped[1] - corners_reshaped[2])
                    tag_size_pixels = (width + height) / 2
                    
                    # Estimate distance using tag size
                    # distance = (tag_size * focal_length) / tag_size_pixels
                    # For simplicity, we'll use a rough estimation
                    # Assuming focal length of ~1000 pixels for typical camera
                    focal_length = 1000
                    estimated_distance = (self.tag_size * focal_length) / tag_size_pixels
                    
                    tags.append({
                        'id': tag_id,
                        'center': (center_x, center_y),
                        'corners': corners_reshaped,
                        'size_pixels': tag_size_pixels,
                        'estimated_distance': estimated_distance
                    })
            
            return tags
            
        except Exception as e:
            print(f"Error in Aruco detection: {e}")
            return []
    
    def calculate_control_signals(self, tags):
        """Calculate control signals based on detected Aruco tags"""
        if not tags:
            # No tags detected - search by rotating
            print("No Aruco tags detected - searching...")
            return 0, 0, self.search_speed
        
        # Use the first detected tag
        tag = tags[0]
        center_x, center_y = tag['center']
        estimated_distance = tag['estimated_distance']
        
        print(f"Tag detected - ID: {tag['id']}, Center: ({center_x}, {center_y}), Distance: {estimated_distance:.2f}m")
        
        # Calculate horizontal offset for centering
        offset_x = center_x - self.target_center_x
        
        # Determine state and control signals
        if abs(offset_x) > self.center_tolerance:
            # Need to center the tag horizontally
            self.state = "centering"
            
            # Simplified turning logic - more direct approach
            if offset_x > 0:
                # Tag is to the right, turn right (negative z)
                turn_z = -self.turn_speed
            else:
                # Tag is to the left, turn left (positive z)
                turn_z = self.turn_speed
                
            move_x = 0  # Don't move forward/backward while centering
            print(f"Centering tag - Offset: {offset_x:.1f}, Turn speed: {turn_z:.3f}")
            print(f"Tag at {center_x}, target at {self.target_center_x}, turning {'right' if turn_z < 0 else 'left'}")
            
        elif abs(estimated_distance - self.target_distance) > self.distance_tolerance:
            # Tag is centered, need to adjust distance
            self.state = "approaching"
            if estimated_distance > self.target_distance:
                # Too far - move forward
                move_x = self.move_speed
                print(f"Moving forward - Distance: {estimated_distance:.2f}m, Target: {self.target_distance}m")
            else:
                # Too close - move backward
                move_x = -self.move_speed
                print(f"Moving backward - Distance: {estimated_distance:.2f}m, Target: {self.target_distance}m")
            turn_z = 0  # Keep current orientation
            
        else:
            # Tag is centered and at correct distance
            self.state = "positioned"
            move_x = 0
            turn_z = 0
            print(f"Positioned correctly - Distance: {estimated_distance:.2f}m")
        
        return float(move_x), 0.0, float(turn_z)
    
    def draw_aruco_info(self, frame, tags):
        """Draw Aruco tag information on the frame"""
        for tag in tags:
            # Draw tag corners
            corners = tag['corners'].astype(np.int32)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            
            # Draw tag center
            center_x, center_y = tag['center']
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Draw tag ID
            cv2.putText(frame, f"ID: {tag['id']}", (center_x + 10, center_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw distance
            cv2.putText(frame, f"Dist: {tag['estimated_distance']:.2f}m", (center_x + 10, center_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw target center line
        cv2.line(frame, (self.target_center_x, 0), (self.target_center_x, frame.shape[0]), (0, 255, 255), 2)
        
        # Draw tolerance zone
        tol_left = self.target_center_x - self.center_tolerance
        tol_right = self.target_center_x + self.center_tolerance
        cv2.line(frame, (tol_left, 0), (tol_left, frame.shape[0]), (0, 255, 0), 1)
        cv2.line(frame, (tol_right, 0), (tol_right, frame.shape[0]), (0, 255, 0), 1)
    
    def draw_status(self, frame):
        """Draw current status on the frame"""
        status_text = f"State: {self.state.upper()}"
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw target distance
        target_text = f"Target Distance: {self.target_distance}m"
        cv2.putText(frame, target_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
                self.conn.video.switchVideoChannel(True)
                self.conn.video.add_track_callback(self.recv_camera_stream)
                print("Video stream started")
            except Exception as e:
                print(f"Error in WebRTC connection: {e}")
        
        loop.run_until_complete(setup())
        loop.run_forever()
    
    async def start_visual_servoing(self):
        """Start the Aruco visual servoing behavior"""
        print("Starting Aruco visual servoing...")
        self.is_running = True
        
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
            while self.is_running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Detect Aruco tags
                    tags = self.detect_aruco_tags(frame)
                    
                    if tags:
                        # Calculate control signals
                        move_x, move_y, turn_z = self.calculate_control_signals(tags)
                        
                        # Debug: Print control signals
                        print(f"Control signals calculated - move_x: {move_x:.3f}, move_y: {move_y:.3f}, turn_z: {turn_z:.3f}")
                        
                        # Update current movement
                        self.current_movement = {'x': float(move_x), 'y': float(move_y), 'z': float(turn_z)}
                        
                        # Debug: Print updated movement
                        print(f"Updated movement - x: {self.current_movement['x']:.3f}, y: {self.current_movement['y']:.3f}, z: {self.current_movement['z']:.3f}")
                        
                        self.last_tag_detected = time.time()
                        
                        # Draw tag information
                        self.draw_aruco_info(frame, tags)
                        
                    else:
                        # No tags detected - search mode
                        self.current_movement = {'x': 0, 'y': 0, 'z': self.search_speed}
                        self.state = "searching"
                    
                    # Draw status
                    self.draw_status(frame)
                    
                    # Display the frame
                    cv2.imshow('Aruco Visual Servoing', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    await asyncio.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("Stopping visual servoing...")
        finally:
            # Cancel movement task
            if self.movement_task:
                self.movement_task.cancel()
                try:
                    await self.movement_task
                except asyncio.CancelledError:
                    pass
            
            await self.stop_robot()
            cv2.destroyAllWindows()
            self.is_running = False
            
            # Stop the asyncio event loop
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.asyncio_thread:
                self.asyncio_thread.join()
    
    async def run(self):
        """Main run method"""
        if await self.connect():
            await self.start_visual_servoing()

async def main():
    servoing = ArucoVisualServoing()
    
    print("Aruco Visual Servoing Robot")
    print("===========================")
    print("1. Test movement")
    print("2. Start visual servoing")
    print("Enter choice (1 or 2): ", end="")
    
    try:
        choice = input().strip()
        
        if await servoing.connect():
            if choice == "1":
                await servoing.test_movement()
            else:
                print("Starting Aruco tag detection and positioning...")
                print("Press 'q' to quit")
                await servoing.start_visual_servoing()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
