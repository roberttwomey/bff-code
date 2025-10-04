"""
GPU-Accelerated YOLO Human Following
===================================

This script shows how to use GPU-accelerated YOLO for human following,
similar to your existing MediaPipe implementation but with YOLO.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

class YOLOGPUHumanFollower:
    """
    Human following using GPU-accelerated YOLO pose detection
    """
    
    def __init__(self, model_path="yolo11n-pose.pt"):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.target_center_x = 640  # Target center of frame
        self.center_tolerance = 50
        self.turn_speed = 0.8
        self.move_speed = 0.6
        self.min_confidence = 0.5
        
        # Initialize GPU setup
        self.setup_gpu()
        
    def setup_gpu(self):
        """Setup GPU acceleration"""
        print("=== GPU Setup ===")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = 'cpu'
            print("Using CPU (GPU not available)")
        
        # Load model
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        print(f"Model loaded on: {self.device}")
        
        # Enable half precision for better performance
        if self.device == 'cuda':
            self.model.half()
            print("Half precision (FP16) enabled")
        
    def detect_humans(self, frame):
        """Detect humans using YOLO pose detection"""
        try:
            # Run YOLO inference
            results = self.model.track(
                frame,
                device=self.device,
                conf=self.min_confidence,
                tracker="bytetrack.yaml",
                verbose=False
            )
            
            humans = []
            
            if results and len(results) > 0:
                result = results[0]
                
                # Check if we have pose keypoints
                if result.keypoints is not None and len(result.keypoints) > 0:
                    keypoints = result.keypoints.data[0]  # First person's keypoints
                    
                    # YOLO pose keypoints (17 points):
                    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
                    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
                    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
                    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
                    
                    # Get key points for human tracking
                    nose = keypoints[0] if keypoints[0][2] > 0.5 else None  # visibility > 0.5
                    left_shoulder = keypoints[5] if keypoints[5][2] > 0.5 else None
                    right_shoulder = keypoints[6] if keypoints[6][2] > 0.5 else None
                    left_hip = keypoints[11] if keypoints[11][2] > 0.5 else None
                    right_hip = keypoints[12] if keypoints[12][2] > 0.5 else None
                    
                    # Calculate human center and confidence
                    visible_points = [p for p in [nose, left_shoulder, right_shoulder, left_hip, right_hip] if p is not None]
                    
                    if len(visible_points) >= 3:  # Need at least 3 key points
                        # Calculate center
                        center_x = int(np.mean([p[0] for p in visible_points]))
                        center_y = int(np.mean([p[1] for p in visible_points]))
                        
                        # Calculate confidence
                        confidence = len(visible_points) / 5.0
                        
                        # Calculate bounding box
                        if result.boxes is not None and len(result.boxes) > 0:
                            box = result.boxes.xyxy[0]  # Get first bounding box
                            x1, y1, x2, y2 = map(int, box)
                            
                            humans.append({
                                'center': (center_x, center_y),
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'keypoints': keypoints,
                                'track_id': result.boxes.id[0].item() if result.boxes.id is not None else None
                            })
            
            return humans
            
        except Exception as e:
            print(f"Error in YOLO human detection: {e}")
            return []
    
    def calculate_control_signals(self, humans):
        """Calculate control signals based on detected humans"""
        if not humans:
            return 0, 0, 0
        
        # Find the human with highest confidence
        best_human = max(humans, key=lambda h: h['confidence'])
        
        center_x, center_y = best_human['center']
        
        # Calculate horizontal offset
        offset_x = center_x - self.target_center_x
        
        # Calculate control signals
        turn_z = 0.0
        move_x = 0.0
        
        # Turn towards human if not centered
        if abs(offset_x) > self.center_tolerance:
            turn_z = -1.0 * np.clip(offset_x / self.target_center_x * self.turn_speed, -self.turn_speed, self.turn_speed)
            print(f"Turning to center human - Offset: {offset_x:.1f}, Turn speed: {turn_z:.3f}")
        else:
            turn_z = -0.2 * np.clip(offset_x / self.target_center_x, -0.3, 0.3)
        
        # Move forward based on distance
        bbox = best_human['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_ratio = area / (1280 * 720)
        
        if area_ratio < 0.3:
            move_x = self.move_speed * (1 - area_ratio * 2)
            move_x = max(move_x, 0.1)
        else:
            move_x = 0.0
        
        return float(move_x), 0.0, float(turn_z)
    
    def draw_detection_debug(self, frame, humans):
        """Draw detection debug information"""
        for human in humans:
            # Draw bounding box
            x1, y1, x2, y2 = human['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = human['center']
            cv2.circle(frame, (center_x, center_y), 8, (255, 0, 0), -1)
            cv2.putText(frame, "HUMAN CENTER", (center_x + 10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw confidence
            confidence = human['confidence']
            cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw track ID if available
            if human['track_id'] is not None:
                cv2.putText(frame, f"ID: {human['track_id']}", 
                           (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw target center line
        cv2.line(frame, (self.target_center_x, 0), (self.target_center_x, frame.shape[0]), 
                (0, 255, 255), 2)
        
        # Draw tolerance zone
        tol_left = self.target_center_x - self.center_tolerance
        tol_right = self.target_center_x + self.center_tolerance
        cv2.line(frame, (tol_left, 0), (tol_left, frame.shape[0]), (0, 255, 0), 1)
        cv2.line(frame, (tol_right, 0), (tol_right, frame.shape[0]), (0, 255, 0), 1)
    
    def run_webcam_tracking(self):
        """Run human tracking on webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting YOLO GPU human tracking...")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect humans
                humans = self.detect_humans(frame)
                
                if humans:
                    # Calculate control signals
                    move_x, move_y, turn_z = self.calculate_control_signals(humans)
                    
                    # Log movement commands
                    if abs(move_x) > 0.01 or abs(turn_z) > 0.01:
                        print(f"Movement: x={move_x:.2f}, y={move_y:.2f}, z={turn_z:.2f}")
                    
                    # Draw debug information
                    self.draw_detection_debug(frame, humans)
                else:
                    print("No humans detected")
                
                # Show frame
                cv2.imshow('YOLO GPU Human Following', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Stopping tracking...")
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function"""
    follower = YOLOGPUHumanFollower()
    follower.run_webcam_tracking()

if __name__ == "__main__":
    main()

