import cv2
import numpy as np
import asyncio
import logging
import threading
import time
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
from ultralytics import YOLO
import json

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

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
    def __init__(self, ip="192.168.4.30"):
        self.ip = ip
        self.frame_queue = Queue()
        self.conn = None
        self.loop = None
        self.asyncio_thread = None
        
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
        """Connect to the Go2 robot"""
        try:
            self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
            await self.conn.connect()
            print("Connected to Go2 robot")
            
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
        """Move the robot with specified velocities"""
        try:
            # Ensure all values are Python native types
            x, y, z = float(x), float(y), float(z)
            print(f"Sending movement command: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            
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
                    print(f"Movement command successful: x={x:.3f}, y={y:.3f}, z={z:.3f}")
                else:
                    print(f"Movement command failed with status: {status}")
            else:
                print("No response received from movement command")
                
        except Exception as e:
            print(f"Error moving robot: {e}")
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
                    
                    # Display the frame
                    cv2.imshow('Human Following', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
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
            cv2.destroyAllWindows()
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
    follower = HumanFollower()
    
    # Ask user if they want to test movement first
    print("Human Following Robot")
    print("====================")
    print("1. Test robot movement first")
    print("2. Start human following directly")
    
    try:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\nTesting robot movement...")
            if await follower.connect():
                await follower.test_movement()
                print("\nMovement test completed. Starting human following...")
                await follower.start_following()
        elif choice == "2":
            print("\nStarting human following...")
            await follower.run()
        else:
            print("Invalid choice. Starting human following...")
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
