import cv2
import cv2.aruco as aruco
import numpy as np
import asyncio
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
import json

@dataclass
class DetectionResult:
    """Result from Aruco detection"""
    R_ct: np.ndarray  # 3x3 rotation matrix from camera to tag
    t_ct: np.ndarray  # 3x1 translation vector from camera to tag
    ok: bool          # Whether detection was successful
    stable: bool = False  # Whether detection is stable (N consecutive frames)

class PBVSController:
    """
    Pose-Based Visual Servoing (PBVS) Controller with State Machine
    
    Implements a simple PBVS controller that sequences:
    S0: Search (rotate slowly)
    S1: Yaw lock (align with tag normal)
    S2: Lateral center (strafe to center)
    S3: Approach (move forward/back to 1m)
    S4: Fine settle (fine positioning)
    """
    
    def __init__(self, ip="192.168.4.30"):
        self.ip = ip
        self.conn = None
        self.loop = None
        self.asyncio_thread = None
        self.frame_queue = Queue()
        
        # State machine variables
        self.state = "S0"
        self.lost_count = 0
        self.stable_count = 0
        
        # Detection stability parameters
        self.K = 10  # Max consecutive lost detections before returning to S0
        self.M = 5   # Consecutive stable frames needed for state transition
        self.N = 3   # Consecutive detections needed for "stable" flag
        
        # Control gains
        self.k_psi = 0.5   # Yaw control gain
        self.k_y = 0.5      # Lateral control gain  
        self.k_x = 0.3      # Forward/back control gain
        
        # Velocity limits
        self.psi_max = 0.5  # Max yaw velocity
        self.vx_max = 0.5   # Max forward velocity
        self.vy_max = 0.4   # Max lateral velocity
        
        # Camera calibration parameters (from aruco_three.py)
        self.camera_matrix = np.array([
            [818.18507419, 0.0, 637.94628188],
            [0.0, 815.32431463, 338.3480119],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array([[-0.07203219],
                                    [-0.05228525],
                                    [ 0.05415833],
                                    [-0.02288355]], dtype=np.float32)
        
        # Camera to base transform (calibrated values)
        self.R_bc = np.eye(3)  # Identity for now - should be calibrated
        self.t_bc = np.array([0, 0, 0.3])  # Camera 30cm above base
        
        # Aruco detection setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.tag_size = 0.2  # Physical tag size in meters
        
        # Detection history for stability
        self.detection_history = []
        
        # Control flags
        self.is_running = False
        
        # Control rate limiting
        self.control_rate_hz = 10.0
        self.last_command_time = 0.0
        
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
        """Send velocity command to robot at controlled rate"""
        current_time = time.time()
        time_since_last = current_time - self.last_command_time
        min_interval = 1.0 / self.control_rate_hz
        
        if time_since_last < min_interval:
            return  # Skip command if not enough time has passed
        
        try:
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], 
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": vx, "y": vy, "z": wz}
                }
            )
            
            if response and 'data' in response and 'header' in response['data']:
                status = response['data']['header']['status']['code']
                if status != 0:
                    print(f"Movement command failed with status: {status}")
            
            self.last_command_time = current_time
        except Exception as e:
            print(f"Error sending movement command: {e}")
    
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
    
    def detect_aruco(self, frame) -> DetectionResult:
        """Detect Aruco tag and return pose information"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            if ids is not None and len(ids) > 0:
                # Use first detected tag
                corner = corners[0]
                
                # Estimate pose using solvePnP with calibrated camera parameters
                obj_points = np.array([
                    [-self.tag_size/2, -self.tag_size/2, 0],
                    [self.tag_size/2, -self.tag_size/2, 0],
                    [self.tag_size/2, self.tag_size/2, 0],
                    [-self.tag_size/2, self.tag_size/2, 0]
                ], dtype=np.float32)
                
                # Solve PnP with calibrated camera matrix and distortion coefficients
                # Use IPPE_SQUARE for better accuracy with square markers like Aruco tags
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, corner, self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if success:
                    # Convert rotation vector to rotation matrix
                    R_ct, _ = cv2.Rodrigues(rvec)
                    t_ct = tvec.flatten()
                    
                    # Check stability
                    self.detection_history.append((R_ct, t_ct))
                    if len(self.detection_history) > self.N:
                        self.detection_history.pop(0)
                    
                    stable = len(self.detection_history) >= self.N
                    
                    return DetectionResult(R_ct=R_ct, t_ct=t_ct, ok=True, stable=stable)
            
            return DetectionResult(R_ct=np.eye(3), t_ct=np.zeros(3), ok=False, stable=False)
            
        except Exception as e:
            print(f"Error in Aruco detection: {e}")
            return DetectionResult(R_ct=np.eye(3), t_ct=np.zeros(3), ok=False, stable=False)
    

    
    def wrap_to_pi(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))
    
    def draw_status(self, frame):
        """Draw status information on frame"""
        # Draw state
        cv2.putText(frame, f"State: {self.state}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw lost count
        cv2.putText(frame, f"Lost: {self.lost_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw stable count
        cv2.putText(frame, f"Stable: {self.stable_count}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

    
    async def run_pbvs_controller(self):
        """Main PBVS controller loop"""
        print("Starting PBVS controller...")
        self.is_running = True
        
        # Create a new event loop for the asyncio code
        self.loop = asyncio.new_event_loop()
        
        # Start the asyncio event loop in a separate thread
        self.asyncio_thread = threading.Thread(target=self.run_asyncio_loop, args=(self.loop,))
        self.asyncio_thread.start()
        
        # Wait for video stream to start
        await asyncio.sleep(2)
        
        try:
            while self.is_running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Detect Aruco tag
                    det = self.detect_aruco(frame)
                    
                    if not det.ok:
                        self.lost_count += 1
                        if self.lost_count > self.K:
                            await self.cmd_vel(0, 0, 0)  # Stop
                            self.state = "S0"
                            print(f"Lost detection for {self.lost_count} frames, returning to S0")
                        continue
                    
                    self.lost_count = 0
                    
                    # Transform pose to base frame
                    R_bt = self.R_bc @ det.R_ct
                    t_bt = self.R_bc @ det.t_ct + self.t_bc
                    
                    # Calculate tag normal and center in base frame
                    n_b = R_bt @ np.array([0, 0, 1.0])  # Tag normal in base frame
                    p_b = t_bt  # Tag center in base frame
                    
                    # Calculate desired pose
                    p_des = p_b - 1.0 * n_b  # Desired camera position (1m away on normal)
                    
                    # Calculate desired yaw to align robot's +x axis with direction to tag
                    # We want to face the tag, so align x-axis with the direction from robot to tag
                    # The tag position in base frame is t_bt, so direction to tag is t_bt normalized
                    direction_to_tag = t_bt / np.linalg.norm(t_bt)
                    psi_des = np.arctan2(direction_to_tag[1], direction_to_tag[0])  # Align +x with direction to tag
                    
                    # Calculate errors (approximate from camera frame)
                    e_y = t_bt[1]  # Left/right error
                    e_x = np.linalg.norm(t_bt) - 1.0  # Forward/back distance error
                    
                    # Calculate relative rotation between tag normal and robot's current orientation
                    # The tag normal in base frame gives us the relative orientation
                    tag_normal_angle = np.arctan2(n_b[1], n_b[0])  # Angle of tag normal in base frame
                    e_psi = self.wrap_to_pi(tag_normal_angle)  # Yaw error relative to robot's current orientation
                    
                    # Coordinate system: positive z = turn left, negative z = turn right
                    # If e_psi > 0, we need to turn left (positive z)
                    # If e_psi < 0, we need to turn right (negative z)
                    # So the sign is already correct, no need to negate
                    
                    print(f"State: {self.state}, Errors: x={e_x:.3f}, y={e_y:.3f}, psi={np.rad2deg(e_psi):.1f}°")
                    
                    # State machine
                    if self.state == "S0":
                        # Slow spin to search for tag
                        await self.cmd_vel(0, 0, 0.3)
                        if det.stable:
                            self.state = "S1"
                            print("S0→S1: Yaw lock")
                    
                    elif self.state == "S1":
                        # Yaw lock - align robot's X-axis to be perpendicular to the tag's normal
                        # Calculate horizontal angle to tag normal (for perpendicular alignment)
                        # We want the robot's X-axis to be perpendicular to the tag's normal
                        # The tag normal in camera frame is R_ct @ [0,0,1]
                        tag_normal_camera = det.R_ct @ np.array([0, 0, 1])
                        horizontal_angle = np.arctan2(tag_normal_camera[1], tag_normal_camera[0])
                        e_psi_s1 = self.wrap_to_pi(horizontal_angle)
                        
                        vx, vy = 0.0, 0.0
                        wz = self.clamp(self.k_psi * e_psi_s1, -self.psi_max, self.psi_max)
                        await self.cmd_vel(vx, vy, wz)
                        
                        if abs(e_psi_s1) < np.deg2rad(10):
                            self.stable_count += 1
                            if self.stable_count > self.M:
                                self.state = "S2"
                                self.stable_count = 0
                                print("S1→S2: Lateral center")
                        else:
                            self.stable_count = 0
                    
                    elif self.state == "S2":
                        # Lateral center - strafe to center
                        vx = 0.0
                        vy = self.clamp(self.k_y * e_y, -self.vy_max, self.vy_max)
                        # wz = self.clamp(self.k_psi * e_psi * 0.5, -self.psi_max/2, self.psi_max/2)
                        await self.cmd_vel(vx, vy, wz)
                        
                        if abs(e_y) < 0.05 and abs(e_psi) < np.deg2rad(5):
                            self.stable_count += 1
                            if self.stable_count > self.M:
                                self.state = "S3"
                                self.stable_count = 0
                                print("S2→S3: Approach")
                        else:
                            self.stable_count = 0
                    
                    elif self.state == "S3":
                        # Approach - move to 1m distance
                        vx = self.clamp(self.k_x * e_x, -self.vx_max, self.vx_max)
                        # vy = self.clamp(self.k_y * e_y * 0.5, -self.vy_max/2, self.vy_max/2)
                        # wz = self.clamp(self.k_psi * e_psi * 0.5, -self.psi_max/2, self.psi_max/2)
                        await self.cmd_vel(vx, vy, wz)
                        
                        if abs(e_x) < 0.05:
                            self.stable_count += 1
                            if self.stable_count > self.M:
                                self.state = "S4"
                                self.stable_count = 0
                                print("S3→S4: Fine settle")
                        else:
                            self.stable_count = 0
                    
                    elif self.state == "S4":
                        # Fine settle - precise positioning
                        if abs(e_x) < 0.03 and abs(e_y) < 0.03 and abs(e_psi) < np.deg2rad(3):
                            # Target achieved - maintain position
                            await self.cmd_vel(0, 0, 0)
                            print(f"Maintaining position - Errors: x={e_x:.3f}, y={e_y:.3f}, psi={np.rad2deg(e_psi):.1f}°", end='\r')
                        else:
                            # Still fine-tuning
                            vx = self.clamp(self.k_x * e_x * 0.3, -0.1, 0.1)
                            vy = self.clamp(self.k_y * e_y * 0.3, -0.1, 0.1)
                            wz = self.clamp(self.k_psi * e_psi * 0.3, -0.2, 0.2)
                            await self.cmd_vel(vx, vy, wz)
                    
                    # Draw status on frame
                    self.draw_status(frame)
                    
                    # Display the frame
                    cv2.imshow('PBVS Controller', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # No frames available, add small delay to prevent busy waiting
                    await asyncio.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        finally:
            self.is_running = False
            await self.cmd_vel(0, 0, 0)
            cv2.destroyAllWindows()
            
            # Stop the asyncio event loop
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.asyncio_thread:
                self.asyncio_thread.join()

# Example usage
async def main():
    """Example usage of PBVS controller"""
    controller = PBVSController()
    
    if await controller.connect():
        try:
            await controller.run_pbvs_controller()
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
            await controller.cmd_vel(0, 0, 0)
        finally:
            await controller.cmd_vel(0, 0, 0)

async def test_rotation_direction():
    """Test function to verify rotation direction"""
    controller = PBVSController()
    
    if await controller.connect():
        print("Testing rotation direction...")
        print("Positive z should turn left, negative z should turn right")
        
        # Test left turn
        print("Testing left turn (positive z)...")
        await controller.cmd_vel(0, 0, 0.3)
        await asyncio.sleep(3)
        
        # Stop
        await controller.cmd_vel(0, 0, 0)
        await asyncio.sleep(1)
        
        # Test right turn
        print("Testing right turn (negative z)...")
        await controller.cmd_vel(0, 0, -0.3)
        await asyncio.sleep(3)
        
        # Stop
        await controller.cmd_vel(0, 0, 0)
        print("Rotation test completed")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_rotation_direction())
    else:
        asyncio.run(main())
