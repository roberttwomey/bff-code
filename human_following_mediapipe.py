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
import mediapipe as mp
import json

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

class HumanFollowerMediaPipe:
    """
    Human following robot controller using MediaPipe pose detection and visual servoing.
    
    Coordinate System:
    - x: Forward/Backward (positive = forward, negative = backward)
    - y: Left/Right (positive = left, negative = right) 
    - z: Rotation (positive = turn left, negative = turn right)
    
    Movement Behavior:
    - Conservative forward movement towards detected humans (maintains greater distance)
    - Slower, more controlled turning for stable centering
    - Stops forward movement when too close to maintain safe distance
    - Uses skeleton center (multiple keypoints) for more stable human tracking
    """
    def __init__(self, ip="192.168.4.30"):
        self.ip = ip
        self.frame_queue = Queue()
        self.conn = None
        self.loop = None
        self.asyncio_thread = None
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Visual servoing parameters
        self.target_center_x = 640  # Target center of frame (assuming 1280x720)
        self.center_tolerance = 5  # Reduced tolerance for more precise centering
        self.turn_speed = 0.8  # Reduced angular velocity for slower turning
        self.move_speed = 0.6  # Reduced forward movement speed for slower approach
        self.min_pose_confidence = 0.5  # Minimum confidence for pose detection
        
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
            # await self.switch_to_normal_mode()
            await self.switch_to_mcf_mode()
            
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
    
    async def switch_to_mcf_mode(self):
        """Switch robot to MCF (Manual Control Foot) motion mode"""
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
                    
                    # Switch to MCF mode if not already
                    if current_mode != "mcf":
                        print("Switching to MCF motion mode...")
                        switch_response = await self.conn.datachannel.pub_sub.publish_request_new(
                            RTC_TOPIC["MOTION_SWITCHER"], 
                            {
                                "api_id": 1002,
                                "parameter": {"name": "mcf"}
                            }
                        )
                        print(f"Switch response: {switch_response}")
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
    
    def detect_humans_mediapipe(self, frame):
        """Detect humans using MediaPipe pose detection"""
        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            humans = []
            if results.pose_landmarks:
                # Get frame dimensions
                h, w = frame.shape[:2]
                
                # Extract key landmarks for tracking
                landmarks = results.pose_landmarks.landmark
                
                # Get nose position (center of face)
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                
                # Get left and right shoulders for width estimation
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                
                # Get hip position for height estimation
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                
                # Calculate bounding box based on pose landmarks for whole body
                # Get more body landmarks for comprehensive bounding box
                left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
                right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
                
                # Calculate body dimensions
                shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
                body_height = abs(left_ear.y - left_ankle.y) * h  # Full body height from ear to ankle
                
                # Estimate comprehensive bounding box
                bbox_width = max(shoulder_width * 2.5, 120)  # Wider to include arms
                bbox_height = body_height * 1.2  # Full body with some margin
                
                # Center the bounding box on the body center (not just nose)
                body_center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
                body_center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
                
                x1 = max(0, int(body_center_x * w - bbox_width / 2))
                y1 = max(0, int(body_center_y * h - bbox_height / 2))
                x2 = min(w, int(body_center_x * w + bbox_width / 2))
                y2 = min(h, int(body_center_y * h + bbox_height / 2))
                
                # Calculate confidence based on landmark visibility
                visible_landmarks = sum(1 for lm in landmarks if lm.visibility > 0.5)
                confidence = visible_landmarks / len(landmarks)
                
                # Calculate confidence for key tracking landmarks
                key_landmarks = [
                    self.mp_pose.PoseLandmark.NOSE,
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_HIP
                ]
                
                key_confidence = sum(landmarks[i].visibility for i in key_landmarks) / len(key_landmarks)
                
                if confidence > self.min_pose_confidence:
                    humans.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (nose_x, nose_y),
                        'confidence': confidence,
                        'key_confidence': key_confidence,  # Confidence for key tracking points
                        'area': bbox_width * bbox_height,
                        'landmarks': results.pose_landmarks,
                        'nose_pos': (nose_x, nose_y),
                        'shoulder_width': shoulder_width,
                        'body_height': body_height,
                        'landmark_visibilities': {i: landmarks[i].visibility for i in range(len(landmarks))}
                    })
            
            return humans
            
        except Exception as e:
            print(f"Error in MediaPipe human detection: {e}")
            return []
    
    def calculate_control_signals(self, humans):
        """Calculate control signals based on detected humans using pose information"""
        if not humans:
            return 0, 0, 0  # No movement
        
        # Find the human with highest key confidence (most stable tracking points)
        best_human = max(humans, key=lambda h: h['key_confidence'])
        
        # Calculate skeleton center using multiple key points for more stable tracking
        landmarks = best_human['landmarks'].landmark
        h, w = 720, 1280  # Frame dimensions
        
        # Get key skeleton points for center calculation
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate skeleton center (average of key points)
        skeleton_center_x = int((nose.x + left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) * w / 5)
        skeleton_center_y = int((nose.y + left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) * h / 5)
        
        # Use skeleton center for tracking instead of just nose
        center_x = skeleton_center_x
        offset_x = center_x - self.target_center_x
        
        # Calculate control signals
        turn_z = 0.0
        move_x = 0.0
        
        # Turn towards human if not centered - more aggressive turning
        if abs(offset_x) > self.center_tolerance:
            # More aggressive turn speed calculation for faster response
            turn_z = -1.0 * np.clip(offset_x / self.target_center_x * self.turn_speed * 1.5, -self.turn_speed, self.turn_speed)
        else:
            # Small corrections even when close to center
            turn_z = -0.3 * np.clip(offset_x / self.target_center_x, -0.5, 0.5)
        
        # Always move forward (positive x direction) when human is detected
        # Use shoulder width and body height for better distance estimation
        area_ratio = best_human['area'] / (1280 * 720)  # Normalize by frame area
        
        if area_ratio < 0.15:  # Reduced threshold to maintain greater distance
            # Base forward speed
            base_speed = self.move_speed
            
            # More aggressive slowdown as human gets closer (maintains greater distance)
            distance_factor = 1 - area_ratio * 4  # Increased from 2 to 4 for more aggressive slowdown
            
            # Less penalty for not being perfectly centered
            centering_factor = 1 - (abs(offset_x) / self.target_center_x) * 0.3
            
            # Combine factors to get final forward speed (always positive)
            move_x = base_speed * distance_factor * centering_factor
            
            # Lower minimum forward movement for more conservative approach
            move_x = max(move_x, 0.15)  # Reduced from 0.3 to 0.15
        else:
            # When too close, stop forward movement to maintain distance
            move_x = 0.0
        
        # Convert numpy types to Python native types for JSON serialization
        return float(move_x), 0.0, float(turn_z)
    
    def draw_pose_landmarks(self, frame, humans):
        """Draw MediaPipe pose landmarks on the frame"""
        for human in humans:
            if 'landmarks' in human:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    human['landmarks'],
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Draw bounding box
                x1, y1, x2, y2 = human['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence and position info
                cv2.putText(frame, f"Overall: {human['confidence']:.2f} | Key: {human['key_confidence']:.2f}", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw skeleton center (tracking center)
                landmarks = human['landmarks'].landmark
                h, w = frame.shape[:2]
                
                # Calculate and draw skeleton center
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                
                skeleton_center_x = int((nose.x + left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) * w / 5)
                skeleton_center_y = int((nose.y + left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) * h / 5)
                
                # Draw skeleton center with larger circle
                cv2.circle(frame, (skeleton_center_x, skeleton_center_y), 8, (255, 0, 0), -1)
                cv2.putText(frame, "CENTER", (skeleton_center_x+10, skeleton_center_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Also draw nose for reference
                nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                cv2.circle(frame, (nose_x, nose_y), 3, (0, 255, 255), -1)
    
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
        print("Starting human following with MediaPipe pose detection...")
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
                    
                    # Detect humans using MediaPipe
                    humans = self.detect_humans_mediapipe(frame)
                    
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
                        
                        # Log confidence information for debugging
                        if humans:
                            best_human = max(humans, key=lambda h: h['key_confidence'])
                            print(f"Tracking confidence - Overall: {best_human['confidence']:.3f}, Key points: {best_human['key_confidence']:.3f}")
                        
                        # Draw pose landmarks and detection info
                        self.draw_pose_landmarks(frame, humans)
                        
                        self.last_human_detected = time.time()
                    else:
                        # No humans detected, stop moving
                        if self.last_human_detected and time.time() - self.last_human_detected > 2:
                            self.current_movement = {'x': 0, 'y': 0, 'z': 0}
                            print("No humans detected, stopped moving")
                    
                    # Display the frame
                    cv2.imshow('Human Following (MediaPipe)', frame)
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
    follower = HumanFollowerMediaPipe()
    
    # Start human following directly (removed menu)
    print("Human Following Robot (MediaPipe)")
    print("=================================")
    print("Starting human following...")
    
    try:
        await follower.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

# Test option code kept for reference (uncomment to use):
# async def test_movement_only():
#     """Test robot movement without starting human following"""
#     follower = HumanFollowerMediaPipe()
#     if await follower.connect():
#         await follower.test_movement()
#         print("Movement test completed.")
#     else:
#         print("Failed to connect to robot")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
