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

class HumanLegsTracker:
    """
    Human legs tracking robot controller using MediaPipe pose detection.
    Focuses only on the lower half of the skeleton (hips, knees, ankles, feet).
    
    Coordinate System:
    - x: Forward/Backward (positive = forward, negative = backward)
    - y: Left/Right (positive = left, negative = right) 
    - z: Rotation (positive = turn left, negative = turn right)
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
        
        # Configure MediaPipe Pose with segmentation enabled
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks=True,
            enable_segmentation=True,  # Enable segmentation for fallback
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.6
        )
        
        # Visual servoing parameters
        self.target_center_x = 640  # Target center of frame (assuming 1280x720)
        self.center_tolerance = 30  # Tolerance for centering
        self.turn_speed = 0.8  # Angular velocity for turning
        self.move_speed = 0.6  # Forward movement speed
        self.min_hip_visibility = 0.4  # Minimum hip visibility to trust lower body
        self.min_lower_body_confidence = 0.3  # Minimum confidence for lower body tracking (reduced since we're tracking more points)
        
        # Control flags
        self.is_following = False
        self.last_human_detected = None
        self.current_movement = {'x': 0, 'y': 0, 'z': 0}
        self.movement_task = None
        
        # Tracking state
        self.tracking_state = "searching"  # "tracking", "lost", "searching"
        self.legs_lost_time = None
        self.legs_lost_threshold = 1.0  # Seconds to wait before considering legs lost
        
        # Movement control parameters
        self.command_rate_hz = 5  # Commands per second
        
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
            
            print(f"Motion mode response: {response}")
            
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
                        print(f"Switch response: {switch_response}")
                        await asyncio.sleep(5)
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
    
    def detect_lower_body(self, frame):
        """Detect lower body using MediaPipe pose detection"""
        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            h, w = frame.shape[:2]
            overlay = frame.copy()
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get hip landmarks
                LHIP = self.mp_pose.PoseLandmark.LEFT_HIP
                RHIP = self.mp_pose.PoseLandmark.RIGHT_HIP
                
                # Calculate hip visibility
                hips_vis = (landmarks[LHIP].visibility + landmarks[RHIP].visibility) / 2.0
                hip_y = int(((landmarks[LHIP].y + landmarks[RHIP].y) / 2) * h)
                
                # If hips are visible enough, trust lower-body joints
                if hips_vis > self.min_hip_visibility:
                    # Define lower body landmarks - include full lower body from hips down
                    lower_idxs = [
                        # Hips (core of lower body)
                        self.mp_pose.PoseLandmark.LEFT_HIP,
                        self.mp_pose.PoseLandmark.RIGHT_HIP,
                        # Thighs
                        self.mp_pose.PoseLandmark.LEFT_KNEE,
                        self.mp_pose.PoseLandmark.RIGHT_KNEE,
                        # Lower legs
                        self.mp_pose.PoseLandmark.LEFT_ANKLE,
                        self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                        # Feet
                        self.mp_pose.PoseLandmark.LEFT_HEEL,
                        self.mp_pose.PoseLandmark.RIGHT_HEEL,
                        self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                        self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
                    ]
                    
                    # Track visible lower body points
                    visible_points = []
                    for idx in lower_idxs:
                        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                        if 0 <= x < w and 0 <= y < h and landmarks[idx].visibility > 0.5:
                            cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)
                            visible_points.append((x, y))
                    
                    # Calculate lower body center and confidence
                    if visible_points:
                        center_x = int(np.mean([p[0] for p in visible_points]))
                        center_y = int(np.mean([p[1] for p in visible_points]))
                        
                        # Calculate confidence based on number of visible points
                        # Weight hips more heavily since they're the core of lower body
                        hip_points = 0
                        other_points = 0
                        
                        for idx in lower_idxs:
                            if idx in [self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP]:
                                if landmarks[idx].visibility > 0.5:
                                    hip_points += 1
                            else:
                                if landmarks[idx].visibility > 0.5:
                                    other_points += 1
                        
                        # Weighted confidence: hips are 40% of total confidence, other points 60%
                        hip_confidence = hip_points / 2.0  # 2 hip points
                        other_confidence = other_points / 8.0  # 8 other lower body points
                        confidence = (hip_confidence * 0.4) + (other_confidence * 0.6)
                        
                        # Calculate bounding box for lower body
                        if len(visible_points) >= 2:
                            x_coords = [p[0] for p in visible_points]
                            y_coords = [p[1] for p in visible_points]
                            
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            
                            # Add some margin to the bounding box
                            margin = 20
                            x1 = max(0, x1 - margin)
                            y1 = max(0, y1 - margin)
                            x2 = min(w, x2 + margin)
                            y2 = min(h, y2 + margin)
                            
                            return {
                                'center': (center_x, center_y),
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'visible_points': visible_points,
                                'hip_visibility': hips_vis,
                                'landmarks': results.pose_landmarks
                            }
                
                # Fall back to segmentation for lower-half tracking if pose landmarks are poor
                if results.segmentation_mask is not None:
                    seg = (results.segmentation_mask * 255).astype(np.uint8)
                    person = cv2.threshold(seg, 128, 255, cv2.THRESH_BINARY)[1]
                    
                    # Focus on lower half of the segmentation
                    h_seg, w_seg = person.shape
                    lower_half = person[h_seg//2:, :]  # Take lower half of the frame
                    
                    # Find contours in lower half
                    contours, _ = cv2.findContours(lower_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Find the largest contour (likely the person)
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        if cv2.contourArea(largest_contour) > 1000:  # Minimum area threshold
                            # Get bounding rectangle
                            x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
                            
                            # Adjust y coordinate for lower half
                            y += h_seg // 2
                            
                            # Calculate center
                            center_x = x + w_rect // 2
                            center_y = y + h_rect // 2
                            
                            return {
                                'center': (center_x, center_y),
                                'bbox': (x, y, x + w_rect, y + h_rect),
                                'confidence': 0.3,  # Lower confidence for segmentation fallback
                                'visible_points': [],
                                'hip_visibility': 0.0,
                                'landmarks': results.pose_landmarks,
                                'method': 'segmentation'
                            }
            
            return None
            
        except Exception as e:
            print(f"Error in lower body detection: {e}")
            return None
    
    def calculate_control_signals(self, lower_body_data):
        """Calculate control signals based on detected lower body"""
        if not lower_body_data:
            return 0, 0, 0  # No movement
        
        center_x, center_y = lower_body_data['center']
        confidence = lower_body_data['confidence']
        
        # Check if confidence is high enough
        if confidence < self.min_lower_body_confidence:
            return 0, 0, 0
        
        # Calculate horizontal offset from target center
        offset_x = center_x - self.target_center_x
        
        # Calculate control signals
        turn_z = 0.0
        move_x = 0.0
        
        # Turn towards lower body if not centered
        if abs(offset_x) > self.center_tolerance:
            turn_z = -1.0 * np.clip(offset_x / self.target_center_x * self.turn_speed, -self.turn_speed, self.turn_speed)
            print(f"Turning to center lower body - Offset: {offset_x:.1f}, Turn speed: {turn_z:.3f}")
        else:
            # Small corrections when close to center
            turn_z = -0.2 * np.clip(offset_x / self.target_center_x, -0.3, 0.3)
        
        # Move forward based on distance (using bounding box area as proxy)
        bbox = lower_body_data['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_ratio = area / (1280 * 720)  # Normalize by frame area
        
        # Move forward when lower body is not too close
        if area_ratio < 0.3:  # Threshold for maintaining distance
            move_x = self.move_speed * (1 - area_ratio * 2)  # Slow down as lower body gets closer
            move_x = max(move_x, 0.1)  # Minimum forward movement
        else:
            move_x = 0.0  # Stop when too close
        
        return float(move_x), 0.0, float(turn_z)
    
    def draw_lower_body_debug(self, frame, lower_body_data):
        """Draw lower body detection debug information"""
        if not lower_body_data:
            return
        
        # Draw bounding box
        x1, y1, x2, y2 = lower_body_data['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw center point
        center_x, center_y = lower_body_data['center']
        cv2.circle(frame, (center_x, center_y), 8, (255, 0, 0), -1)
        cv2.putText(frame, "LOWER BODY CENTER", (center_x + 10, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw confidence and method info
        confidence = lower_body_data['confidence']
        method = lower_body_data.get('method', 'pose')
        cv2.putText(frame, f"Confidence: {confidence:.2f} | Method: {method}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw target center line
        cv2.line(frame, (self.target_center_x, 0), (self.target_center_x, frame.shape[0]), 
                (0, 255, 255), 2)
        
        # Draw tolerance zone
        tol_left = self.target_center_x - self.center_tolerance
        tol_right = self.target_center_x + self.center_tolerance
        cv2.line(frame, (tol_left, 0), (tol_left, frame.shape[0]), (0, 255, 0), 1)
        cv2.line(frame, (tol_right, 0), (tol_right, frame.shape[0]), (0, 255, 0), 1)
        
        # Draw offset indicator
        offset = center_x - self.target_center_x
        if abs(offset) > self.center_tolerance:
            arrow_color = (0, 0, 255) if offset > 0 else (255, 0, 0)
            arrow_text = "TURN RIGHT" if offset > 0 else "TURN LEFT"
            cv2.putText(frame, arrow_text, (center_x + 20, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
    
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
            import traceback
            traceback.print_exc()
    
    async def stop_robot(self):
        """Stop the robot movement"""
        print("Stopping robot movement")
        self.current_movement = {'x': 0, 'y': 0, 'z': 0}
        await self.move_robot(0, 0, 0)
    
    async def continuous_movement_task(self):
        """Task that continuously sends movement commands"""
        while self.is_following:
            try:
                if (abs(self.current_movement['x']) > 0.01 or 
                    abs(self.current_movement['y']) > 0.01 or 
                    abs(self.current_movement['z']) > 0.01):
                    
                    await self.move_robot(
                        self.current_movement['x'],
                        self.current_movement['y'], 
                        self.current_movement['z']
                    )
                
                await asyncio.sleep(1.0 / self.command_rate_hz)
                
            except Exception as e:
                print(f"Error in continuous movement task: {e}")
                await asyncio.sleep(0.1)
    
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
    
    async def start_following(self):
        """Start the legs following behavior"""
        print("Starting legs following with MediaPipe...")
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
                    
                    # Detect lower body
                    lower_body_data = self.detect_lower_body(frame)
                    
                    if lower_body_data and lower_body_data['confidence'] > self.min_lower_body_confidence:
                        # Good lower body detection
                        if self.tracking_state != "tracking":
                            print(f"Lower body tracking started - Confidence: {lower_body_data['confidence']:.2f}")
                            self.tracking_state = "tracking"
                            self.legs_lost_time = None
                        
                        # Calculate control signals
                        move_x, move_y, turn_z = self.calculate_control_signals(lower_body_data)
                        
                        # Update current movement
                        self.current_movement = {'x': float(move_x), 'y': float(move_y), 'z': float(turn_z)}
                        
                        # Log movement commands
                        if abs(move_x) > 0.01 or abs(turn_z) > 0.01:
                            print(f"Movement command: x={move_x:.2f} (forward), y={move_y:.2f}, z={turn_z:.2f}")
                        
                        # Draw debug information
                        self.draw_lower_body_debug(frame, lower_body_data)
                        
                        self.last_human_detected = time.time()
                    else:
                        # No lower body detected or poor confidence
                        if self.tracking_state == "tracking":
                            self.tracking_state = "lost"
                            self.legs_lost_time = time.time()
                            print("Lower body tracking lost")
                        
                        # Stop movement when lower body is lost
                        self.current_movement = {'x': 0, 'y': 0, 'z': 0}
                        
                        # Check if we should start searching
                        if (self.legs_lost_time and 
                            time.time() - self.legs_lost_time > self.legs_lost_threshold):
                            self.tracking_state = "searching"
                            print("Starting search mode - looking for lower body...")
                    
                    # Draw tracking status
                    status_text = f"STATUS: {self.tracking_state.upper()}"
                    cv2.putText(frame, status_text, (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display the frame
                    cv2.imshow('Lower Body Following (MediaPipe)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    await asyncio.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("Stopping legs following...")
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
    tracker = HumanLegsTracker()
    
    print("Human Legs Following Robot (MediaPipe)")
    print("=====================================")
    print("Starting legs following...")
    
    try:
        await tracker.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
