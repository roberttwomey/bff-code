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

class SimpleArUcoController:
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
        self.target_distance = 0.75  # Target distance in meters
        self.center_tolerance = 30   # Pixels tolerance for centering
        self.distance_tolerance = 0.05  # Distance tolerance in meters
        self.perpendicular_tolerance = np.deg2rad(10)  # Perpendicular tolerance in radians (10 degrees)
        
        # Control gains
        self.k_approach = 5.0       # Approach control gain (increased for 10Hz)
        self.k_yaw = 0.5            # Yaw control gain (increased for 10Hz)
        self.k_strafe = 1.0        # Strafe control gain for perpendicular alignment (increased for 10Hz)
        
        # Velocity limits
        self.max_linear_vel = 1.5   # m/s
        self.max_angular_vel = 0.5  # rad/s
        
        # State
        self.is_running = False
        self.last_command_time = 0
        self.command_rate = 5  # 10 Hz
        
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
                    
                    # Calculate tag orientation for perpendicular alignment
                    # Get the tag's normal vector in camera frame
                    tag_normal_camera = R_ct @ np.array([0, 0, 1])  # Tag's Z-axis in camera frame
                    
                    # Calculate horizontal angle to tag normal (for perpendicular alignment)
                    # We want the robot's X-axis to be perpendicular to the tag's normal
                    horizontal_angle = np.arctan2(tag_normal_camera[1], tag_normal_camera[0])
                    
                    return {
                        'detected': True,
                        'distance': distance,
                        'center_offset_x': center_offset_x,
                        'center_offset_y': center_offset_y,
                        'corners': corner,
                        'id': ids[0][0],
                        'horizontal_angle': horizontal_angle,
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
            if 'horizontal_angle' in detection:
                cv2.putText(frame, f"Angle: {np.rad2deg(detection['horizontal_angle']):.1f}°", 
                           (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw tag normal vector
            if 'tag_normal' in detection:
                # Get tag center position in image
                corners = detection['corners']
                # Calculate center from the 4 corner points
                tag_center_x = int(np.mean(corners[:, 0]))
                tag_center_y = int(np.mean(corners[:, 1]))
                
                # Scale the normal vector for visualization (make it visible on screen)
                scale = 100  # pixels
                normal_x = detection['tag_normal'][0] * scale
                normal_y = detection['tag_normal'][1] * scale
                
                # Draw the normal vector as an arrow from tag center
                end_x = int(tag_center_x + normal_x)
                end_y = int(tag_center_y + normal_y)

                
                # Draw normal vector components
                cv2.putText(frame, f"Normal: [{detection['tag_normal'][0]:.2f}, {detection['tag_normal'][1]:.2f}, {detection['tag_normal'][2]:.2f}]", 
                           (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Draw coordinate frame axes on the tag
                if 'rvec' in detection and 'tvec' in detection:
                    cv2.drawFrameAxes(frame, GO2_CAM_K, GO2_CAM_D, 
                                    detection['rvec'], detection['tvec'], 0.1)
        else:
            cv2.putText(frame, "No tag detected", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    async def run_aruco_controller(self):
        """Main ArUco controller loop"""
        print("Starting ArUco controller...")
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
                        
                        print(f"Tag detected - Distance: {distance:.3f}m, "
                              f"Center offset: ({center_offset_x:.1f}, {center_offset_y:.1f})px")
                        
                        # Sequential control: first center with yaw, then approach
                        yaw_tolerance = 0.05  # radians (about 3 degrees)
                        center_tolerance_pixels = 30  # pixels
                        
                        # Check if yaw is centered (tag is in center of image)
                        if abs(center_offset_x) > center_tolerance_pixels:
                            # Phase 1: Center with yaw rotation only
                            vy = 0  # No strafing
                            vx = 0  # No forward/backward
                            
                            # Decrease yaw amplitude as robot gets closer to centered
                            # Use a non-linear scaling that reduces gain near center
                            center_ratio = abs(center_offset_x) / (center_tolerance_pixels * 1.5)
                            adaptive_gain = self.k_yaw * min(center_ratio, 1.0)  # Reduce gain when close to center
                            
                            wz = self.clamp(-adaptive_gain * center_offset_x * 0.01, 
                                          -self.max_angular_vel, self.max_angular_vel)
                            
                            print(f"Phase 1 - Centering with yaw: center_offset={center_offset_x:.1f}px, wz={wz:.3f}")
                            
                            # Draw phase status on frame
                            cv2.putText(frame, "PHASE 1 - CENTERING WITH YAW", 
                                       (frame.shape[1]//2 - 150, frame.shape[0] - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                            
                        elif abs(distance - self.target_distance) > self.distance_tolerance:
                            # Phase 2: Approach with forward/backward motion only
                            vy = 0  # No strafing
                            wz = 0  # No yaw rotation
                            distance_error = distance - self.target_distance
                            
                            # Decrease forward/backward amplitude as robot gets closer to target distance
                            # Use a non-linear scaling that reduces gain near target distance
                            distance_ratio = abs(distance_error) / (self.distance_tolerance * 1.5)
                            adaptive_gain = self.k_approach * min(distance_ratio, 1.0)  # Reduce gain when close to target
                            
                            vx = self.clamp(adaptive_gain * distance_error, 
                                          -self.max_linear_vel, self.max_linear_vel)
                            
                            print(f"Phase 2 - Approaching: distance_error={distance_error:.3f}m, adaptive_gain={adaptive_gain:.3f}, vx={vx:.3f}")
                            
                            # Draw phase status on frame
                            cv2.putText(frame, "PHASE 2 - APPROACHING", 
                                       (frame.shape[1]//2 - 150, frame.shape[0] - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                            
                        else:
                            # Phase 3: Perpendicular alignment using strafe and yaw
                            horizontal_angle = detection['horizontal_angle']
                            
                            if abs(horizontal_angle) > self.perpendicular_tolerance:
                                # Phase 3a: Align perpendicular to tag
                                vx = 0  # No forward/backward
                                
                                # Calculate ideal position that is target_distance away and perpendicular
                                tag_normal = detection['tag_normal']
                                current_pos_camera = detection['tvec'].flatten()  # Current position in camera frame
                                
                                # Ideal position: target_distance away from tag center, perpendicular to tag normal
                                # The ideal position should be: tag_center + target_distance * perpendicular_direction
                                # Perpendicular direction is perpendicular to tag normal in horizontal plane
                                perpendicular_dir = np.array([-tag_normal[1], tag_normal[0], 0])  # Perpendicular in horizontal plane
                                perpendicular_dir = perpendicular_dir / np.linalg.norm(perpendicular_dir)  # Normalize
                                
                                ideal_pos_camera = self.target_distance * perpendicular_dir
                                
                                # Calculate error between current and ideal position
                                position_error = current_pos_camera - ideal_pos_camera
                                
                                # Project error to camera plane for strafe control
                                # We care about the lateral (Y) component of the error
                                ideal_position_error = position_error[1]  # Y component for strafe
                                
                                # Use ideal position error for strafe, horizontal angle for yaw
                                vy = self.clamp(self.k_strafe * ideal_position_error * 0.5, 
                                              -self.max_linear_vel, self.max_linear_vel)
                                
                                # # Yaw to rotate toward perpendicular orientation
                                # wz = self.clamp(-self.k_yaw * horizontal_angle * 0.5, 
                                #               -self.max_angular_vel, self.max_angular_vel)
                                wz = 0

                                print(f"Phase 3a - Perpendicular alignment: angle={np.rad2deg(horizontal_angle):.1f}°, ideal_error={ideal_position_error:.3f}m, vy={vy:.3f}, wz={wz:.3f}")
                                
                                # Draw phase status on frame
                                cv2.putText(frame, "PHASE 3A - PERPENDICULAR ALIGNMENT", 
                                           (frame.shape[1]//2 - 150, frame.shape[0] - 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                                
                            else:
                                # Phase 3b: Perpendicular and at target distance - maintain position
                                vx, vy, wz = 0, 0, 0
                                print(f"Phase 3b - Perpendicular alignment achieved! Angle: {np.rad2deg(horizontal_angle):.1f}°")
                                
                                # Draw status on frame
                                self.draw_detection_info(frame, detection)
                                cv2.putText(frame, "PERPENDICULAR - MAINTAINING", 
                                           (frame.shape[1]//2 - 150, frame.shape[0] - 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        
                        # Send velocity command
                        print(f"Final commands - vx: {vx:.3f}, vy: {vy:.3f}, wz: {wz:.3f}")
                        
                        # Check if enough time has passed since last command
                        current_time = time.time()
                        command_interval = 1.0 / self.command_rate  # Calculate interval from rate
                        if current_time - self.last_command_time >= command_interval:
                            # Check if commands are non-zero
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
                    cv2.imshow('ArUco Detection', frame)
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
    controller = SimpleArUcoController()
    
    if await controller.connect():
        try:
            await controller.run_aruco_controller()
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
            await controller.cmd_vel(0, 0, 0)
        finally:
            await controller.cmd_vel(0, 0, 0)

async def test_movement():
    """Test function to verify robot can move"""
    controller = SimpleArUcoController()
    
    if await controller.connect():
        print("Testing basic movement...")
        
        # Test forward movement
        print("Testing forward movement (0.2 m/s)...")
        await controller.cmd_vel(0.2, 0, 0)
        await asyncio.sleep(2)
        
        # Stop
        print("Stopping...")
        await controller.cmd_vel(0, 0, 0)
        await asyncio.sleep(1)
        
        # Test rotation
        print("Testing rotation (0.3 rad/s)...")
        await controller.cmd_vel(0, 0, 0.3)
        await asyncio.sleep(2)
        
        # Stop
        print("Stopping...")
        await controller.cmd_vel(0, 0, 0)
        print("Movement test completed")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_movement())
    else:
        asyncio.run(main())
