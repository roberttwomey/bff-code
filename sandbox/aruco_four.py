import cv2
import cv2.aruco as aruco
import numpy as np
import asyncio
import time
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
import json

# Camera calibration parameters
GO2_CAM_K = np.array([
    [818.18507419, 0.0, 637.94628188],
    [0.0, 815.32431463, 338.3480119],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

GO2_CAM_D = np.array([[-0.07203219],
                      [-0.05228525],
                      [ 0.05415833],
                      [-0.02288355]], dtype=np.float32)

class ImprovedArUcoController:
    def __init__(self, ip="192.168.4.30"):
        self.ip = ip
        self.conn = None
        self.frame_queue = asyncio.Queue()
        
        # ArUco detection setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.tag_size = 0.2  # Physical tag size in meters (20cm)
        
        # Control parameters
        self.target_distance = 0.75  # Final target distance in meters
        self.min_distance = 0.4      # Minimal approach distance in meters
        self.center_tolerance = 50   # Pixels tolerance for centering
        self.distance_tolerance = 0.05  # Distance tolerance in meters
        self.perpendicular_tolerance = np.deg2rad(5)  # Perpendicular tolerance in radians
        
        # Control gains
        self.k_approach = 2.0        # Approach control gain
        self.k_yaw = 2.0             # Yaw control gain
        self.k_strafe = 3.0          # Strafe control gain
        self.k_backoff = 1.5         # Backoff control gain
        
        # Velocity limits
        self.max_linear_vel = 1.5   # m/s
        self.max_angular_vel = 0.5  # rad/s
        
        # State
        self.is_running = False
        self.last_command_time = 0
        self.command_rate = 5  # 5 Hz
        self.current_phase = 1  # Track current phase
        self.phase_switch_time = 0  # Time of last phase switch
        self.phase_switch_delay = 1.0  # Minimum time between phase switches (seconds)
        
        # Angle filtering for stability
        self.last_horizontal_angle = 0.0
        self.angle_filter_alpha = 0.9  # Low-pass filter coefficient
        
    async def connect(self):
        """Connect to Go2 robot"""
        try:
            self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
            await self.conn.connect()
            print("Connected to Go2 robot")
            
            # Switch to MCF mode
            await self.switch_to_mcf_mode()
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    async def switch_to_mcf_mode(self):
        """Switch robot to MCF motion mode"""
        try:
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], 
                {"api_id": 1001}
            )
            
            if response and 'data' in response and 'header' in response['data']:
                status = response['data']['header']['status']['code']
                if status == 0:
                    data = json.loads(response['data']['data'])
                    current_mode = data['name']
                    
                    if current_mode != "mcf":
                        await self.conn.datachannel.pub_sub.publish_request_new(
                            RTC_TOPIC["MOTION_SWITCHER"], 
                            {"api_id": 1002, "parameter": {"name": "mcf"}}
                        )
                        await asyncio.sleep(2)
                        print("Switched to MCF mode")
                    else:
                        print("Already in MCF mode")
        except Exception as e:
            print(f"Error switching to MCF mode: {e}")
    
    async def cmd_vel(self, vx: float, vy: float, wz: float):
        """Send velocity command to robot"""
        try:
            print(f"Sending cmd_vel: vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}")
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], 
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": vx, "y": vy, "z": wz}
                }
            )
            
            if response and 'data' in response and 'header' in response['data']:
                status = response['data']['header']['status']['code']
                if status == 0:
                    print(f"Movement command sent successfully")
                else:
                    print(f"Movement command failed with status: {status}")
            else:
                print(f"No response received from movement command")
        except Exception as e:
            print(f"Error sending movement command: {e}")
            import traceback
            traceback.print_exc()
    
    async def recv_camera_stream(self, track: MediaStreamTrack):
        """Receive video frames and process them"""
        print(f"Video track callback started: {track.kind}")
        frame_count = 0
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                await self.frame_queue.put(img)
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Received {frame_count} frames")
            except Exception as e:
                print(f"Error receiving frame: {e}")
                break
    
    def detect_aruco(self, frame):
        """Detect ArUco tag and return pose information"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            if ids is not None and len(ids) > 0:
                # Use first detected tag
                corner = corners[0]
                
                # Define 3D points of the ArUco tag (TL,TR,BR,BL order)
                obj_points = np.array([
                    [-self.tag_size/2,  self.tag_size/2, 0],  # TL
                    [ self.tag_size/2,  self.tag_size/2, 0],  # TR
                    [ self.tag_size/2, -self.tag_size/2, 0],  # BR
                    [-self.tag_size/2, -self.tag_size/2, 0]   # BL
                ], dtype=np.float32)
                
                # Solve PnP using IPPE_SQUARE solver (best for planar squares)
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, corner, GO2_CAM_K, GO2_CAM_D,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if success:
                    # Convert rotation vector to rotation matrix
                    R_ct, _ = cv2.Rodrigues(rvec)
                    t_ct = tvec.flatten()
                    
                    # Calculate distance
                    distance = np.linalg.norm(t_ct)
                    
                    # Calculate center offset (how far tag is from image center)
                    center_x = frame.shape[1] / 2
                    center_y = frame.shape[0] / 2
                    
                    # Project tag center to image plane
                    tag_center_3d = np.array([[0, 0, 0]], dtype=np.float32)  # Tag center in tag frame
                    tag_center_2d, _ = cv2.projectPoints(
                        tag_center_3d, rvec, tvec, GO2_CAM_K, GO2_CAM_D
                    )
                    tag_center_2d = tag_center_2d[0][0]
                    
                    center_offset_x = tag_center_2d[0] - center_x
                    center_offset_y = tag_center_2d[1] - center_y
                    
                    # Calculate tag's normal vector in camera frame
                    tag_normal_camera = R_ct @ np.array([0, 0, 1])  # Tag's Z-axis in camera frame
                    
                    # Calculate perpendicular angle (rotation around Y-axis in XZ plane)
                    perpendicular_angle = np.arctan2(R_ct[2,0], R_ct[0,0])
                    
                    # Normalize angle to [-π, π] range to avoid wrapping issues
                    perpendicular_angle = np.arctan2(np.sin(perpendicular_angle), np.cos(perpendicular_angle))
                    
                    # Apply low-pass filter to reduce noise
                    filtered_angle = (self.angle_filter_alpha * self.last_horizontal_angle + 
                                    (1 - self.angle_filter_alpha) * perpendicular_angle)
                    self.last_horizontal_angle = filtered_angle
                    
                    return {
                        'detected': True,
                        'distance': distance,
                        'center_offset_x': center_offset_x,
                        'center_offset_y': center_offset_y,
                        'corners': corner,
                        'id': ids[0][0],
                        'perpendicular_angle': filtered_angle,
                        'tag_normal': tag_normal_camera,
                        'rvec': rvec,
                        'tvec': tvec
                    }
            
            return {'detected': False}
            
        except Exception as e:
            print(f"Error in ArUco detection: {e}")
            return {'detected': False}
    
    def clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))
    
    def draw_detection_info(self, frame, detection):
        """Draw detection information on frame"""
        if detection['detected']:
            # Draw tag corners
            cv2.aruco.drawDetectedMarkers(frame, [detection['corners']])
            
            # Draw distance and center offset
            cv2.putText(frame, f"Distance: {detection['distance']:.3f}m", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Center X: {detection['center_offset_x']:.1f}px", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Center Y: {detection['center_offset_y']:.1f}px", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Tag ID: {detection['id']}", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if 'perpendicular_angle' in detection:
                cv2.putText(frame, f"Perp Angle: {np.rad2deg(detection['perpendicular_angle']):.1f}°", 
                           (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw coordinate frame axes on the tag
            if 'rvec' in detection and 'tvec' in detection:
                cv2.drawFrameAxes(frame, GO2_CAM_K, GO2_CAM_D, 
                                detection['rvec'], detection['tvec'], 0.1)
        else:
            cv2.putText(frame, "No tag detected", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    async def run_aruco_controller(self):
        """Main ArUco controller loop with 4-phase strategy"""
        print("Starting Improved ArUco controller...")
        self.is_running = True
        
        # Setup video stream
        self.conn.video.switchVideoChannel(True)
        self.conn.video.add_track_callback(self.recv_camera_stream)
        
        # Wait for video stream to start
        print("Waiting for video stream to initialize...")
        await asyncio.sleep(3)
        
        # Check if we're receiving frames
        if self.frame_queue.empty():
            print("Warning: No frames received yet. Video stream may not be working.")
        else:
            print("Video stream working!")
        
        try:
            while self.is_running:
                if not self.frame_queue.empty():
                    frame = await self.frame_queue.get()
                    
                    # Detect ArUco tag
                    detection = self.detect_aruco(frame)
                    
                    if detection['detected']:
                        distance = detection['distance']
                        center_offset_x = detection['center_offset_x']
                        center_offset_y = detection['center_offset_y']
                        perpendicular_angle = detection['perpendicular_angle']
                        
                        print(f"Tag detected - Distance: {distance:.3f}m, "
                              f"Center offset: ({center_offset_x:.1f}, {center_offset_y:.1f})px, "
                              f"Perp angle: {np.rad2deg(perpendicular_angle):.1f}°")
                        
                        # Add hysteresis to prevent rapid phase switching
                        current_time = time.time()
                        can_switch_phase = (current_time - self.phase_switch_time) > self.phase_switch_delay
                        
                        # Determine target phase based on conditions
                        target_phase = self.current_phase
                        
                        if self.current_phase == 1:  # Centering
                            if abs(center_offset_x) <= self.center_tolerance:
                                target_phase = 2
                        elif self.current_phase == 2:  # Approaching
                            if abs(distance - self.min_distance) <= self.distance_tolerance:
                                target_phase = 3
                        elif self.current_phase == 3:  # Perpendicular alignment
                            if (abs(perpendicular_angle) <= self.perpendicular_tolerance and 
                                abs(center_offset_x) <= self.center_tolerance):
                                target_phase = 4
                        elif self.current_phase == 4:  # Backing off
                            if abs(distance - self.target_distance) <= self.distance_tolerance:
                                target_phase = 5  # Final position
                        
                        # Only switch phases if enough time has passed
                        if can_switch_phase and target_phase != self.current_phase:
                            self.current_phase = target_phase
                            self.phase_switch_time = current_time
                            print(f"Switching to Phase {self.current_phase}")
                        
                        # Execute current phase
                        if self.current_phase == 1:
                            # Phase 1: Center with yaw rotation only
                            vx = 0  # No forward/backward
                            vy = 0  # No strafing
                            
                            # Yaw to center the tag
                            center_ratio = abs(center_offset_x) / (self.center_tolerance * 1.5)
                            adaptive_gain = self.k_yaw * min(center_ratio, 1.0)
                            
                            wz = self.clamp(-adaptive_gain * center_offset_x * 0.02, 
                                          -self.max_angular_vel, self.max_angular_vel)
                            
                            # Ensure minimum movement
                            min_wz = 0.1
                            if abs(wz) > 0.01 and abs(wz) < min_wz:
                                wz = min_wz if wz > 0 else -min_wz
                            
                            print(f"Phase 1 - Centering: center_offset={center_offset_x:.1f}px, wz={wz:.3f}")
                            cv2.putText(frame, "PHASE 1 - CENTERING", 
                                       (frame.shape[1]//2 - 150, frame.shape[0] - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                            
                        elif self.current_phase == 2:
                            # Phase 2: Approach to minimal distance
                            vy = 0  # No strafing
                            wz = 0  # No yaw rotation
                            
                            distance_error = distance - self.min_distance
                            distance_ratio = abs(distance_error) / (self.distance_tolerance * 1.5)
                            adaptive_gain = self.k_approach * min(distance_ratio, 1.0)
                            
                            vx = self.clamp(adaptive_gain * distance_error, 
                                          -self.max_linear_vel, self.max_linear_vel)
                            
                            # Ensure minimum movement
                            min_vx = 0.15
                            if abs(vx) > 0.01 and abs(vx) < min_vx:
                                vx = min_vx if vx > 0 else -min_vx
                            
                            print(f"Phase 2 - Approaching: distance_error={distance_error:.3f}m, vx={vx:.3f}")
                            cv2.putText(frame, "PHASE 2 - APPROACHING", 
                                       (frame.shape[1]//2 - 150, frame.shape[0] - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                            
                        elif self.current_phase == 3:
                            # Phase 3: Strafe and yaw to perpendicular alignment
                            vx = 0  # No forward/backward
                            
                            # Strafe to center laterally
                            current_pos_camera = detection['tvec'].flatten()
                            lateral_error = current_pos_camera[1]
                            vy = self.clamp(-self.k_strafe * lateral_error * 1.0, 
                                          -self.max_linear_vel, self.max_linear_vel)
                            
                            # Ensure minimum strafe movement
                            min_vy = 0.15
                            if abs(vy) > 0.01 and abs(vy) < min_vy:
                                vy = min_vy if vy > 0 else -min_vy
                            
                            # Yaw to achieve perpendicular alignment
                            wz = self.clamp(-self.k_yaw * perpendicular_angle * 0.5, 
                                          -self.max_angular_vel, self.max_angular_vel)
                            
                            print(f"Phase 3 - Perpendicular: perp_angle={np.rad2deg(perpendicular_angle):.1f}°, lateral_error={lateral_error:.3f}m, vy={vy:.3f}, wz={wz:.3f}")
                            cv2.putText(frame, "PHASE 3 - PERPENDICULAR", 
                                       (frame.shape[1]//2 - 150, frame.shape[0] - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                            
                        elif self.current_phase == 4:
                            # Phase 4: Back off to desired distance
                            vy = 0  # No strafing
                            wz = 0  # No yaw rotation
                            
                            distance_error = distance - self.target_distance
                            distance_ratio = abs(distance_error) / (self.distance_tolerance * 1.5)
                            adaptive_gain = self.k_backoff * min(distance_ratio, 1.0)
                            
                            vx = self.clamp(adaptive_gain * distance_error, 
                                          -self.max_linear_vel, self.max_linear_vel)
                            
                            # Ensure minimum movement
                            min_vx = 0.15
                            if abs(vx) > 0.01 and abs(vx) < min_vx:
                                vx = min_vx if vx > 0 else -min_vx
                            
                            print(f"Phase 4 - Backing off: distance_error={distance_error:.3f}m, vx={vx:.3f}")
                            cv2.putText(frame, "PHASE 4 - BACKING OFF", 
                                       (frame.shape[1]//2 - 150, frame.shape[0] - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            
                        elif self.current_phase == 5:
                            # Phase 5: Final position - maintain
                            vx, vy, wz = 0, 0, 0
                            print(f"Phase 5 - Final position achieved! Distance: {distance:.3f}m, Perp angle: {np.rad2deg(perpendicular_angle):.1f}°")
                            cv2.putText(frame, "FINAL POSITION - MAINTAINING", 
                                       (frame.shape[1]//2 - 150, frame.shape[0] - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        
                        # Send velocity command
                        print(f"Final commands - vx: {vx:.3f}, vy: {vy:.3f}, wz: {wz:.3f}")
                        
                        # Check if enough time has passed since last command
                        current_time = time.time()
                        command_interval = 1.0 / self.command_rate
                        if current_time - self.last_command_time >= command_interval:
                            if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(wz) > 0.01:
                                print(f"Sending non-zero commands - vx: {vx:.3f}, vy: {vy:.3f}, wz: {wz:.3f}")
                                await self.cmd_vel(vx, vy, wz)
                            else:
                                print("Commands too small, sending zero velocity")
                                await self.cmd_vel(0, 0, 0)
                            
                            self.last_command_time = current_time
                        else:
                            print(f"Skipping command, too soon since last command ({current_time - self.last_command_time:.3f}s)")
                    else:
                        # No tag detected - stop
                        await self.cmd_vel(0, 0, 0)
                        print("No tag detected - stopping")
                    
                    # Draw detection info on frame
                    self.draw_detection_info(frame, detection)
                    
                    # Display the frame
                    cv2.imshow('Improved ArUco Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("No frames available, waiting...")
                    await asyncio.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        finally:
            self.is_running = False
            await self.cmd_vel(0, 0, 0)
            cv2.destroyAllWindows()

async def main():
    """Main function"""
    controller = ImprovedArUcoController()
    
    if await controller.connect():
        try:
            await controller.run_aruco_controller()
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
            await controller.cmd_vel(0, 0, 0)
        finally:
            await controller.cmd_vel(0, 0, 0)

if __name__ == "__main__":
    asyncio.run(main())
