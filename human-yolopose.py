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
    Human following robot controller using YOLO pose detection and visual servoing.
    
    This implementation uses the ultralytics YOLO pose model (yolo11n-pose.pt) for more
    accurate human tracking by detecting pose keypoints and using torso center for
    precise positioning.
    
    Coordinate System:
    - x: Forward/Backward (positive = forward, negative = backward)
    - y: Left/Right (positive = left, negative = right) 
    - z: Rotation (positive = turn left, negative = turn right)
    
    Enhanced Features:
    - Pose-based human detection with keypoint tracking
    - Torso center calculation for more accurate centering
    - Pose stability assessment for adaptive movement control
    - Multi-human selection based on pose quality and size
    - Lateral movement adjustment based on pose orientation
    - Visual pose skeleton overlay for debugging
    
    Movement Behavior:
    - Adaptive forward movement based on pose stability
    - Precise turning using torso center positioning
    - Enhanced tracking with pose keypoint confidence
    - Intelligent human selection when multiple people detected
    """
    def __init__(self, ip="192.168.4.30"):
        self.ip = ip
        self.frame_queue = Queue()
        self.conn = None
        self.loop = None
        self.asyncio_thread = None
        
        # Load YOLO pose model for human pose detection
        self.model = YOLO('yolo11n-pose.pt')  # Using nano pose model for speed
        
        # Visual servoing parameters
        self.target_center_x = 640  # Target center of frame (assuming 1280x720)
        self.center_tolerance = 5  # Reduced tolerance for more precise centering
        self.turn_speed = 1.5  # Increased angular velocity for faster turning
        self.move_speed = 1.25#1.5  # Increased forward movement speed for faster approach
        self.min_human_confidence = 0.85  # Minimum confidence for human detection
        
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
        """Detect humans in the frame using YOLO pose model"""
        try:
            # Run YOLO pose inference
            results = self.model(frame, verbose=False)
            
            humans = []
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                
                if boxes is not None and keypoints is not None:
                    for i, box in enumerate(boxes):
                        # Check if detected object is a person (class 0 in COCO dataset)
                        if box.cls == 0 and box.conf > self.min_human_confidence:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            # Get pose keypoints for this person
                            person_keypoints = None
                            if i < len(keypoints.data):
                                person_keypoints = keypoints.data[i].cpu().numpy()
                            
                            # Calculate center from bounding box
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Try to get a more accurate center from keypoints if available
                            if person_keypoints is not None:
                                # Use center of torso keypoints (shoulders and hips) for better centering
                                # COCO pose keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
                                # 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist,
                                # 11=left_hip, 12=right_hip, 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
                                torso_keypoints = [5, 6, 11, 12]  # shoulders and hips
                                valid_keypoints = []
                                
                                for kp_idx in torso_keypoints:
                                    if kp_idx < len(person_keypoints):
                                        kp = person_keypoints[kp_idx]
                                        if len(kp) >= 3 and kp[2] > 0.5:  # confidence threshold for keypoint
                                            valid_keypoints.append([kp[0], kp[1]])
                                
                                if valid_keypoints:
                                    # Use average of valid torso keypoints for center
                                    valid_keypoints = np.array(valid_keypoints)
                                    center_x = int(np.mean(valid_keypoints[:, 0]))
                                    center_y = int(np.mean(valid_keypoints[:, 1]))
                            
                            humans.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'center': (center_x, center_y),
                                'confidence': float(confidence),
                                'area': (x2 - x1) * (y2 - y1),
                                'keypoints': person_keypoints,
                                'torso_center': (center_x, center_y)  # More accurate center from pose
                            })
            
            return humans
        except Exception as e:
            print(f"Error in human pose detection: {e}")
            return []
    
    def calculate_control_signals(self, humans):
        """Calculate control signals based on detected humans using pose information"""
        if not humans:
            return 0, 0, 0  # No movement
        
        # Find the best human to follow based on pose quality and size
        best_human = self.select_best_human(humans)
        
        # Use torso center for more accurate tracking
        center_x, center_y = best_human['torso_center']
        offset_x = center_x - self.target_center_x
        
        # Calculate control signals
        turn_z = 0
        move_x = 0
        move_y = 0
        
        # Enhanced turning based on pose stability
        pose_stability = self.calculate_pose_stability(best_human)
        
        # Turn towards human if not centered - adaptive turning based on pose quality
        if abs(offset_x) > self.center_tolerance:
            # Adjust turn speed based on pose stability
            stability_factor = max(0.5, pose_stability)  # Minimum 0.5, max 1.0
            turn_z = -1.0 * np.clip(offset_x / self.target_center_x * self.turn_speed * stability_factor, -self.turn_speed, self.turn_speed)
        else:
            # Small corrections even when close to center
            turn_z = -0.3 * np.clip(offset_x / self.target_center_x, -0.5, 0.5)
        
        # Enhanced forward movement based on pose information
        area_ratio = best_human['area'] / (1280 * 720)  # Normalize by frame area
        
        if area_ratio < 0.25:  # Threshold for close approach
            # Base forward speed
            base_speed = self.move_speed
            
            # Distance factor - slower when closer
            distance_factor = 1 - area_ratio * 2
            
            # Centering factor - slower when not well centered
            centering_factor = 1 - (abs(offset_x) / self.target_center_x) * 0.3
            
            # Pose stability factor - move more confidently when pose is stable
            pose_factor = max(0.7, pose_stability)
            
            # Combine factors to get final forward speed
            move_x = base_speed * distance_factor * centering_factor * pose_factor
            
            # Minimum forward movement
            move_x = max(move_x, 0.3)
        else:
            # When very close, slow down but maintain some movement
            move_x = 0.1
        
        # Add lateral movement based on pose orientation (optional)
        # This could help the robot position itself better relative to the human
        if best_human['keypoints'] is not None:
            lateral_offset = self.calculate_lateral_offset(best_human)
            if abs(lateral_offset) > 50:  # Significant lateral offset
                move_y = np.clip(lateral_offset / 100.0, -0.3, 0.3)
        
        # Convert numpy types to Python native types for JSON serialization
        return float(move_x), float(move_y), float(turn_z)
    
    def select_best_human(self, humans):
        """Select the best human to follow based on pose quality and size"""
        if len(humans) == 1:
            return humans[0]
        
        # Score each human based on multiple factors
        scored_humans = []
        for human in humans:
            score = 0
            
            # Size factor (larger = closer = better)
            size_score = human['area'] / (1280 * 720)
            score += size_score * 2
            
            # Pose quality factor
            pose_score = self.calculate_pose_stability(human)
            score += pose_score
            
            # Confidence factor
            score += human['confidence']
            
            # Center proximity factor (prefer humans closer to center)
            center_distance = abs(human['center'][0] - self.target_center_x)
            center_score = 1 - (center_distance / self.target_center_x)
            score += center_score * 0.5
            
            scored_humans.append((score, human))
        
        # Return the human with the highest score
        scored_humans.sort(key=lambda x: x[0], reverse=True)
        return scored_humans[0][1]
    
    def calculate_pose_stability(self, human):
        """Calculate how stable/complete the pose detection is"""
        if human['keypoints'] is None:
            return 0.0
        
        keypoints = human['keypoints']
        visible_keypoints = 0
        total_keypoints = len(keypoints)
        
        # Count visible keypoints (confidence > 0.5)
        for kp in keypoints:
            if len(kp) >= 3 and kp[2] > 0.5:
                visible_keypoints += 1
        
        # Return ratio of visible keypoints
        return visible_keypoints / max(total_keypoints, 1)
    
    def calculate_lateral_offset(self, human):
        """Calculate lateral offset based on pose orientation"""
        if human['keypoints'] is None:
            return 0
        
        keypoints = human['keypoints']
        
        # Try to get shoulder positions for lateral alignment
        left_shoulder = None
        right_shoulder = None
        
        if len(keypoints) > 6:
            if len(keypoints[5]) >= 3 and keypoints[5][2] > 0.5:  # left shoulder
                left_shoulder = keypoints[5]
            if len(keypoints[6]) >= 3 and keypoints[6][2] > 0.5:  # right shoulder
                right_shoulder = keypoints[6]
        
        if left_shoulder is not None and right_shoulder is not None:
            # Calculate shoulder center
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            return shoulder_center_x - self.target_center_x
        
        return 0
    
    def draw_pose_keypoints(self, frame, keypoints):
        """Draw pose keypoints on the frame"""
        if keypoints is None:
            return
        
        # Define keypoint connections (skeleton structure)
        # COCO pose connections
        connections = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),  # nose to eyes, eyes to ears
            # Torso
            (5, 6), (5, 11), (6, 12), (11, 12),  # shoulders and hips
            # Left arm
            (5, 7), (7, 9),  # left shoulder to elbow to wrist
            # Right arm
            (6, 8), (8, 10),  # right shoulder to elbow to wrist
            # Left leg
            (11, 13), (13, 15),  # left hip to knee to ankle
            # Right leg
            (12, 14), (14, 16),  # right hip to knee to ankle
        ]
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if len(kp) >= 3 and kp[2] > 0.3:  # confidence threshold
                x, y = int(kp[0]), int(kp[1])
                confidence = kp[2]
                
                # Color based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                cv2.circle(frame, (x, y), 3, color, -1)
                
                # Draw keypoint number for debugging (optional)
                if i in [5, 6, 11, 12]:  # Only show torso keypoint numbers
                    cv2.putText(frame, str(i), (x + 5, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw connections
        for connection in connections:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                len(keypoints[pt1_idx]) >= 3 and len(keypoints[pt2_idx]) >= 3 and
                keypoints[pt1_idx][2] > 0.3 and keypoints[pt2_idx][2] > 0.3):
                
                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
    
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
                        
                        # Draw detection and pose on frame
                        for human in humans:
                            x1, y1, x2, y2 = human['bbox']
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw confidence and pose stability
                            pose_stability = self.calculate_pose_stability(human)
                            cv2.putText(frame, f"Human: {human['confidence']:.2f} | Pose: {pose_stability:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Draw pose keypoints if available
                            if human['keypoints'] is not None:
                                self.draw_pose_keypoints(frame, human['keypoints'])
                            
                            # Draw torso center
                            torso_x, torso_y = human['torso_center']
                            cv2.circle(frame, (torso_x, torso_y), 5, (255, 0, 0), -1)
                            cv2.putText(frame, "Torso", (torso_x + 10, torso_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        
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
