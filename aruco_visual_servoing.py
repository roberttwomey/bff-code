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
        self.center_tolerance = 50  # Increased tolerance for centering to reduce oscillation
        self.target_distance = 1.0  # Target distance in meters
        self.distance_tolerance = 0.1  # Distance tolerance in meters
        self.tag_size = 0.2  # Physical size of Aruco tag in meters (adjust as needed)
        
        # Control parameters
        self.turn_speed = 0.5  # Angular velocity for turning
        self.move_speed = 0.6  # Forward/backward movement speed
        self.sideways_speed = 0.4  # Sideways movement speed
        self.search_speed = 1.0  # Speed for searching rotation
        
        # Proportional control gains
        self.centering_gain = 0.0005  # Reduced gain for lateral movement (was 0.001)
        self.perpendicularity_gain = 0.3  # Reduced gain for forward/backward movement (was 0.5)
        self.rotation_gain = 0.2  # Reduced gain for rotation (was 0.3)
        self.distance_gain = 0.2  # Reduced gain for distance control (was 0.3)
        
        # Dead zone parameters
        self.min_movement_threshold = 0.01  # Reduced minimum movement threshold (was 0.03)
        self.dead_zone_multiplier = 0.6  # Reduced dead zone multiplier (was 0.8)
        
        # Control flags
        self.is_running = False
        self.current_movement = {'x': 0, 'y': 0, 'z': 0}
        self.movement_task = None
        
        # State tracking
        self.state = "searching"  # "searching", "centering", "orienting", "approaching", "positioned"
        self.last_tag_detected = None
        self.command_rate_hz = 5  # Commands per second (slower for debugging)
        
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
        
        # Test forward/backward
        print("Testing forward movement...")
        await self.move_robot(1.0, 0, 0)
        await asyncio.sleep(2)
        
        print("Testing backward movement...")
        await self.move_robot(-1.0, 0, 0)
        await asyncio.sleep(2)
        
        # Test sideways movement
        print("Testing left movement...")
        await self.move_robot(0, 1.0, 0)
        await asyncio.sleep(2)
        
        print("Testing right movement...")
        await self.move_robot(0, -1.0, 0)
        await asyncio.sleep(2)
        
        # Stop
        await self.stop_robot()
        print("Movement test completed")
    
    async def direct_move(self, direction, duration=1.0):
        """Execute a direct movement command"""
        print(f"Executing direct move: {direction}")
        
        if direction == "forward":
            await self.move_robot(self.move_speed, 0, 0)
        elif direction == "backward":
            await self.move_robot(-self.move_speed, 0, 0)
        elif direction == "left":
            await self.move_robot(0, self.sideways_speed, 0)
        elif direction == "right":
            await self.move_robot(0, -self.sideways_speed, 0)
        elif direction == "rotate_left":
            await self.move_robot(0, 0, self.turn_speed)
        elif direction == "rotate_right":
            await self.move_robot(0, 0, -self.turn_speed)
        else:
            print(f"Unknown direction: {direction}")
            return
        
        await asyncio.sleep(duration)
        await self.stop_robot()
        print(f"Direct move {direction} completed")
    
    async def continuous_movement_task(self):
        """Task that continuously sends movement commands"""
        while self.is_running:
            try:
                # Debug: Print current movement values
                print(f"Movement task - Current: x={self.current_movement['x']:.3f}, y={self.current_movement['y']:.3f}, z={self.current_movement['z']:.3f}")
                
                # Determine movement type for clearer logging
                movement_type = self.get_movement_type()
                print(f"Movement type: {movement_type}")
                
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
    
    def get_movement_type(self):
        """Get a human-readable description of the current movement with priority order"""
        x, y, z = self.current_movement['x'], self.current_movement['y'], self.current_movement['z']
        
        # Priority order: Lateral (left/right) -> Forward/Backward -> Rotation
        movements = []
        
        # Check lateral movement first (highest priority)
        if abs(y) > 0.01:
            movements.append("LEFT" if y > 0 else "RIGHT")
        
        # Check forward/backward movement second
        if abs(x) > 0.01:
            movements.append("FORWARD" if x > 0 else "BACKWARD")
        
        # Check rotation last (lowest priority)
        if abs(z) > 0.01:
            movements.append("ROTATE LEFT" if z > 0 else "ROTATE RIGHT")
        
        if len(movements) == 0:
            return "STOPPED"
        elif len(movements) == 1:
            return movements[0]
        else:
            return " + ".join(movements)
    
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
        """Calculate control signals based on detected Aruco tags using combined proportional control"""
        if not tags:
            # No tags detected - search by rotating
            print("No Aruco tags detected - searching...")
            return 0, 0, self.search_speed
        
        # Use the first detected tag
        tag = tags[0]
        center_x, center_y = tag['center']
        estimated_distance = tag['estimated_distance']
        corners = tag['corners']
        
        print(f"Tag detected - ID: {tag['id']}, Center: ({center_x}, {center_y}), Distance: {estimated_distance:.2f}m")
        
        # Calculate horizontal offset for centering
        offset_x = center_x - self.target_center_x
        
        # Calculate perpendicularity using perspective distortion
        perpendicularity_score, orientation_info = self.calculate_perpendicularity(corners)
        
        print(f"Perpendicularity score: {perpendicularity_score:.3f}, Orientation: {orientation_info}")
        print(f"Centering offset: {offset_x:.1f} pixels, Tolerance: {self.center_tolerance}")
        print(f"Distance: {estimated_distance:.2f}m, Target: {self.target_distance}m, Tolerance: {self.distance_tolerance}")
        
        # Initialize movement commands
        move_x = 0
        move_y = 0
        turn_z = 0
        
        # Calculate distance error
        distance_error = estimated_distance - self.target_distance
        
        # COMBINED CONTROL STRATEGY:
        # Always calculate all corrections and apply them simultaneously
        
        # 1. Centering correction (lateral movement) - ALWAYS active
        centering_gain = self.centering_gain
        move_y = -centering_gain * offset_x  # Negative because positive offset means move right
        move_y = np.clip(move_y, -self.sideways_speed, self.sideways_speed)
        
        if abs(offset_x) > self.center_tolerance:
            print(f"Centering - Lateral movement: {move_y:.3f} (offset: {offset_x:.1f})")
        else:
            print(f"Centering - Minimal lateral movement: {move_y:.3f} (offset: {offset_x:.1f})")
        
        # 2. Perpendicularity correction - ALWAYS active
        if perpendicularity_score < 0.9:
            perp_gain = self.perpendicularity_gain
            
            if orientation_info == "tilted_up":
                # Move backward to become perpendicular
                perp_move = -perp_gain * (0.9 - perpendicularity_score)
                perp_move = np.clip(perp_move, -self.move_speed, 0)
                move_x += perp_move
                print(f"Orienting - Moving backward: {perp_move:.3f} (perp score: {perpendicularity_score:.3f})")
                
            elif orientation_info == "tilted_down":
                # Move forward to become perpendicular
                perp_move = perp_gain * (0.9 - perpendicularity_score)
                perp_move = np.clip(perp_move, 0, self.move_speed)
                move_x += perp_move
                print(f"Orienting - Moving forward: {perp_move:.3f} (perp score: {perpendicularity_score:.3f})")
                
            elif orientation_info in ["tilted_left", "tilted_right"]:
                # Use rotation for left/right tilt
                turn_gain = self.rotation_gain
                if orientation_info == "tilted_left":
                    perp_turn = turn_gain * (0.9 - perpendicularity_score)
                    perp_turn = np.clip(perp_turn, 0, self.turn_speed)
                    turn_z += perp_turn
                else:
                    perp_turn = -turn_gain * (0.9 - perpendicularity_score)
                    perp_turn = np.clip(perp_turn, -self.turn_speed, 0)
                    turn_z += perp_turn
                print(f"Orienting - Turning: {perp_turn:.3f} (perp score: {perpendicularity_score:.3f})")
            else:
                print(f"Orienting - No correction needed (perp score: {perpendicularity_score:.3f})")
        else:
            print("Tag is perpendicular - no orientation correction needed")
        
        # 3. Distance correction (forward/backward movement) - ALWAYS active
        if abs(distance_error) > self.distance_tolerance:
            distance_gain = self.distance_gain
            
            if distance_error > 0:
                # Too far - move forward
                distance_move = distance_gain * distance_error
                distance_move = np.clip(distance_move, 0, self.move_speed)
                move_x += distance_move
                print(f"Approaching - Moving forward: {distance_move:.3f} (distance error: {distance_error:.2f}m)")
            else:
                # Too close - move backward
                distance_move = distance_gain * distance_error
                distance_move = np.clip(distance_move, -self.move_speed, 0)
                move_x += distance_move
                print(f"Approaching - Moving backward: {distance_move:.3f} (distance error: {distance_error:.2f}m)")
        else:
            print("Distance is correct - no distance correction needed")
        
        # 4. Combined movement strategy for maintaining centering during perpendicularity correction
        # When tag is roughly centered but needs perpendicularity correction, use rotation to help maintain centering
        if (abs(offset_x) < self.center_tolerance * 2 and  # Roughly centered
            perpendicularity_score < 0.9 and  # Needs perpendicularity correction
            orientation_info in ["tilted_left", "tilted_right"]):  # Left/right tilt
            
            # Add a small rotation component to help maintain centering
            centering_rotation_gain = 0.1  # Small gain for centering rotation
            centering_turn = centering_rotation_gain * offset_x  # Use offset to determine rotation direction
            centering_turn = np.clip(centering_turn, -self.turn_speed * 0.3, self.turn_speed * 0.3)  # Limit to 30% of max turn speed
            turn_z += centering_turn
            print(f"Combined control - Adding centering rotation: {centering_turn:.3f} (offset: {offset_x:.1f})")
        
        # 5. Special handling for when tag is centered but needs perpendicularity correction
        # This prevents the robot from overshooting by using more conservative movements
        if (abs(offset_x) <= self.center_tolerance and  # Tag is centered
            perpendicularity_score < 0.9):  # But needs perpendicularity correction
            
            # Use a less aggressive safety factor to preserve necessary movements
            centering_safety_factor = 0.8  # Only reduce speed by 20% when centered but not perpendicular
            move_x *= centering_safety_factor
            move_y *= centering_safety_factor
            turn_z *= centering_safety_factor
            print(f"Centered but not perpendicular - Reducing speeds by {((1-centering_safety_factor)*100):.0f}%")
            
            # For left/right tilts when centered, use more rotation and less lateral movement
            if orientation_info in ["tilted_left", "tilted_right"]:
                # Increase rotation component and reduce lateral movement
                rotation_boost = 1.5  # Boost rotation by 50%
                lateral_reduction = 0.3  # Reduce lateral movement by 70%
                turn_z *= rotation_boost
                move_y *= lateral_reduction
                print(f"Centered with left/right tilt - Boosting rotation, reducing lateral movement")
        
        # 6. Check if all conditions are met (positioned correctly)
        if (abs(offset_x) <= self.center_tolerance and 
            perpendicularity_score >= 0.9 and 
            abs(distance_error) <= self.distance_tolerance):
            
            self.state = "positioned"
            # Stop all movement
            move_x = 0
            move_y = 0
            turn_z = 0
            print(f"POSITIONED - All conditions met! Stopping movement.")
        else:
            # Determine current state for display
            if abs(offset_x) > self.center_tolerance:
                self.state = "centering"
            elif perpendicularity_score < 0.9:
                self.state = "orienting"
            elif abs(distance_error) > self.distance_tolerance:
                self.state = "approaching"
            else:
                self.state = "fine_tuning"
        
        # Apply dead zone multiplier
        if abs(offset_x) < self.center_tolerance * self.dead_zone_multiplier:
            move_y = 0
        if abs(distance_error) < self.distance_tolerance * self.dead_zone_multiplier:
            move_x = 0
        
        # Apply safety speed reduction when close to target
        safety_factor = 1.0
        if (abs(offset_x) < self.center_tolerance * 1.5 and 
            abs(distance_error) < self.distance_tolerance * 1.5 and
            perpendicularity_score > 0.8):
            safety_factor = 0.2  # Reduce speed by 80% when close to target (was 0.3)
            print(f"Safety mode: Reducing movement speed by {((1-safety_factor)*100):.0f}%")
        elif (abs(offset_x) < self.center_tolerance * 2.5 and 
              abs(distance_error) < self.distance_tolerance * 2.5 and
              perpendicularity_score > 0.7):
            safety_factor = 0.5  # Reduce speed by 50% when moderately close to target
            print(f"Moderate safety mode: Reducing movement speed by {((1-safety_factor)*100):.0f}%")
        
        move_x *= safety_factor
        move_y *= safety_factor
        turn_z *= safety_factor
        
        # Apply minimum movement threshold to prevent jitter
        min_threshold = self.min_movement_threshold
        if abs(move_x) < min_threshold:
            move_x = 0
        if abs(move_y) < min_threshold:
            move_y = 0
        if abs(turn_z) < min_threshold:
            turn_z = 0
        
        # Special case: If very close to target distance, allow small movements
        if abs(distance_error) < self.distance_tolerance * 1.5 and abs(distance_error) > self.distance_tolerance * 0.5:
            # Allow smaller movements when close to target
            close_threshold = 0.005  # Very small threshold for close movements
            if abs(move_x) < close_threshold and abs(distance_error) > self.distance_tolerance:
                # Restore small distance correction
                small_correction = 0.01 if distance_error > 0 else -0.01
                move_x = small_correction
                print(f"Close to target - Allowing small distance correction: {move_x:.3f}")
        
        # Debug output
        print(f"Final movement commands - x: {move_x:.3f}, y: {move_y:.3f}, z: {turn_z:.3f}")
        print(f"State: {self.state}")
        
        # Additional debug info
        print(f"Distance error: {distance_error:.3f}m, Target: {self.target_distance}m")
        print(f"Perpendicularity score: {perpendicularity_score:.3f}, Threshold: 0.9")
        print(f"Centering offset: {offset_x:.1f}px, Tolerance: {self.center_tolerance}")
        
        return float(move_x), float(move_y), float(turn_z)
    
    def calculate_perpendicularity(self, corners):
        """
        Calculate how perpendicular the robot is to the Aruco tag surface.
        Returns a score (0-1) and orientation information.
        """
        try:
            # corners should be a 4x2 array of corner coordinates
            if corners.shape != (4, 2):
                print(f"Invalid corners shape: {corners.shape}")
                return 0.5, "unknown"
            
            # Calculate the sides of the tag
            # Assuming corners are in order: top-left, top-right, bottom-right, bottom-left
            top_left = corners[0]
            top_right = corners[1]
            bottom_right = corners[2]
            bottom_left = corners[3]
            
            # Calculate side lengths
            top_side = np.linalg.norm(top_right - top_left)
            bottom_side = np.linalg.norm(bottom_right - bottom_left)
            left_side = np.linalg.norm(bottom_left - top_left)
            right_side = np.linalg.norm(bottom_right - top_right)
            
            # Calculate aspect ratio (should be close to 1.0 for a square tag when perpendicular)
            horizontal_ratio = top_side / bottom_side
            vertical_ratio = left_side / right_side
            
            # Calculate center lines to detect tilt
            top_center = (top_left + top_right) / 2
            bottom_center = (bottom_left + bottom_right) / 2
            left_center = (top_left + bottom_left) / 2
            right_center = (top_right + bottom_right) / 2
            
            # Calculate tilt angles
            horizontal_tilt = np.arctan2(bottom_center[1] - top_center[1], bottom_center[0] - top_center[0])
            vertical_tilt = np.arctan2(right_center[0] - left_center[0], right_center[1] - left_center[1])
            
            # Convert to degrees
            horizontal_tilt_deg = np.degrees(horizontal_tilt)
            vertical_tilt_deg = np.degrees(vertical_tilt)
            
            # Calculate perpendicularity score
            # Perfect perpendicularity would have:
            # - horizontal_ratio = 1.0 (no perspective distortion)
            # - vertical_ratio = 1.0 (no perspective distortion)
            # - horizontal_tilt = 0° (no rotation)
            # - vertical_tilt = 0° (no tilt)
            
            ratio_score = min(horizontal_ratio, vertical_ratio) / max(horizontal_ratio, vertical_ratio)
            tilt_score = 1.0 - (abs(horizontal_tilt_deg) + abs(vertical_tilt_deg)) / 90.0  # Normalize to 0-1
            
            perpendicularity_score = (ratio_score + tilt_score) / 2.0
            
            # Determine orientation information (simplified - no hysteresis)
            orientation_info = "perpendicular"
            
            # Use a lower threshold for more responsive control
            responsive_threshold = 10  # degrees
            
            if abs(horizontal_tilt_deg) > responsive_threshold:
                if horizontal_tilt_deg > 0:
                    orientation_info = "tilted_left"
                else:
                    orientation_info = "tilted_right"
            elif abs(vertical_tilt_deg) > responsive_threshold:
                if vertical_tilt_deg > 0:
                    orientation_info = "tilted_down"
                else:
                    orientation_info = "tilted_up"
            else:
                orientation_info = "perpendicular"
            
            # Debug information
            print(f"Perpendicularity analysis:")
            print(f"  Horizontal ratio: {horizontal_ratio:.3f}, Vertical ratio: {vertical_ratio:.3f}")
            print(f"  Horizontal tilt: {horizontal_tilt_deg:.1f}°, Vertical tilt: {vertical_tilt_deg:.1f}°")
            print(f"  Ratio score: {ratio_score:.3f}, Tilt score: {tilt_score:.3f}")
            print(f"  Responsive threshold: {responsive_threshold}°")
            print(f"  Current orientation: {orientation_info}")
            
            return perpendicularity_score, orientation_info
            
        except Exception as e:
            print(f"Error calculating perpendicularity: {e}")
            return 0.5, "error"
    
    def draw_aruco_info(self, frame, tags):
        """Draw Aruco tag information on the frame"""
        for tag in tags:
            # Draw tag corners
            corners = tag['corners'].astype(np.int32)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            
            # Draw tag center
            center_x, center_y = tag['center']
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Calculate and display perpendicularity information
            perpendicularity_score, orientation_info = self.calculate_perpendicularity(tag['corners'])
            
            # Color code based on perpendicularity
            if perpendicularity_score > 0.9:
                color = (0, 255, 0)  # Green for good perpendicularity
            elif perpendicularity_score > 0.7:
                color = (0, 255, 255)  # Yellow for moderate perpendicularity
            else:
                color = (0, 0, 255)  # Red for poor perpendicularity
            
            # Draw tag ID
            cv2.putText(frame, f"ID: {tag['id']}", (center_x + 10, center_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw distance
            cv2.putText(frame, f"Dist: {tag['estimated_distance']:.2f}m", (center_x + 10, center_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw perpendicularity information
            cv2.putText(frame, f"Perp: {perpendicularity_score:.2f}", (center_x + 10, center_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Orient: {orientation_info}", (center_x + 10, center_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw orientation indicators
            if orientation_info == "tilted_left":
                cv2.arrowedLine(frame, (center_x - 30, center_y), (center_x - 10, center_y), (0, 0, 255), 2)
            elif orientation_info == "tilted_right":
                cv2.arrowedLine(frame, (center_x + 10, center_y), (center_x + 30, center_y), (0, 0, 255), 2)
            elif orientation_info == "tilted_up":
                cv2.arrowedLine(frame, (center_x, center_y - 30), (center_x, center_y - 10), (0, 0, 255), 2)
            elif orientation_info == "tilted_down":
                cv2.arrowedLine(frame, (center_x, center_y + 10), (center_x, center_y + 30), (0, 0, 255), 2)
        
        # Draw target center line
        cv2.line(frame, (self.target_center_x, 0), (self.target_center_x, frame.shape[0]), (0, 255, 255), 2)
        
        # Draw tolerance zone
        tol_left = self.target_center_x - self.center_tolerance
        tol_right = self.target_center_x + self.center_tolerance
        cv2.line(frame, (tol_left, 0), (tol_left, frame.shape[0]), (0, 255, 0), 1)
        cv2.line(frame, (tol_right, 0), (tol_right, frame.shape[0]), (0, 255, 0), 1)
    
    def draw_status(self, frame):
        """Draw current status on the frame"""
        # Color code the state
        if self.state == "positioned":
            color = (0, 255, 0)  # Green for positioned
        elif self.state == "searching":
            color = (0, 255, 255)  # Yellow for searching
        else:
            color = (0, 0, 255)  # Red for other states
        
        status_text = f"State: {self.state.upper()}"
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw current movement command
        movement_type = self.get_movement_type()
        movement_text = f"Movement: {movement_type}"
        cv2.putText(frame, movement_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw target distance
        target_text = f"Target Distance: {self.target_distance}m"
        cv2.putText(frame, target_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw perpendicularity target
        perp_text = f"Target Perpendicularity: 0.9+"
        cv2.putText(frame, perp_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
    print("1. Test basic movement")
    print("2. Test direct movement commands")
    print("3. Start visual servoing")
    print("Enter choice (1, 2, or 3): ", end="")
    
    try:
        choice = input().strip()
        
        if await servoing.connect():
            if choice == "1":
                await servoing.test_movement()
            elif choice == "2":
                print("Testing direct movement commands...")
                await servoing.direct_move("forward", 2.0)
                await servoing.direct_move("backward", 2.0)
                await servoing.direct_move("left", 2.0)
                await servoing.direct_move("right", 2.0)
                await servoing.direct_move("rotate_left", 2.0)
                await servoing.direct_move("rotate_right", 2.0)
                print("Direct movement test completed")
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
