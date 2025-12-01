import time
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2
import numpy as np
from ultralytics import YOLO

ethernet_interface = "enP8p1s0"

def get_camera_frame(client):
    """Capture a single frame from the camera"""
    code, data = client.GetImageSample()
    
    if code != 0:
        print(f"Get image sample error. code: {code}")
        return None
    
    # Convert to numpy image
    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    return image

def detect_objects(model, frame):
    """Run YOLO object detection on a frame"""
    if frame is None:
        return []
    
    # Run YOLO inference
    results = model(frame, verbose=False)
    
    # Extract detections
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls]
            detections.append({
                'class': cls_name,
                'confidence': conf,
                'class_id': cls
            })
    
    return detections

def print_detections(angle, detections):
    """Print detected objects in a formatted way"""
    if not detections:
        print(f"[Angle: {angle:>6.1f}°] No objects detected")
        return
    
    # Group by class
    class_counts = {}
    for det in detections:
        cls = det['class']
        if cls not in class_counts:
            class_counts[cls] = []
        class_counts[cls].append(det['confidence'])
    
    # Print summary
    summary_parts = [f"[Angle: {angle:>6.1f}°]"]
    for cls, confs in sorted(class_counts.items()):
        avg_conf = sum(confs) / len(confs)
        summary_parts.append(f"{cls}({len(confs)}, conf:{avg_conf:.2f})")
    
    print(" | ".join(summary_parts))

def scan_panorama(sport_client, video_client, model, max_angle=90, step_angle=15):
    """
    Rotate the robot left and right to scan the environment with YOLO detection
    
    Args:
        sport_client: SportClient for movement control
        video_client: VideoClient for camera access
        model: YOLO model for object detection
        max_angle: Maximum angle to rotate (degrees) in each direction
        step_angle: Angle increment between scans (degrees)
    """
    print(f"\nStarting panorama scan: ±{max_angle}° in {step_angle}° steps")
    print("=" * 70)
    
    # Calculate sequence of angles
    # Sweep pattern: start at -90°, go to +90° in steps
    # This ensures no backtracking and good coverage
    angles = []
    for angle in range(-max_angle, max_angle + 1, step_angle):
        angles.append(angle)
    
    print(f"Scan sequence: {angles}")
    print("=" * 70)
    
    # Stand up the robot
    print("\nStanding up robot...")
    sport_client.StandUp()
    time.sleep(2)
    
    # Rotate to starting position and scan
    current_angle = 0
    for target_angle in angles:
        # Calculate rotation needed
        rotation_needed = target_angle - current_angle
        
        # Rotate the robot
        # Using angular velocity (rad/s) for rotation
        # 0.5 rad/s is a moderate rotation speed
        rotation_duration = abs(rotation_needed) * np.pi / 180 / 0.5  # Convert to seconds
        
        print(f"\nRotating from {current_angle:.1f}° to {target_angle:.1f}° " +
              f"({rotation_needed:+.1f}° over {rotation_duration:.2f}s)")
        
        if rotation_needed > 0:
            # Rotate clockwise (right)
            sport_client.Move(0, 0, 0.5)
        else:
            # Rotate counter-clockwise (left)
            sport_client.Move(0, 0, -0.5)
        
        # Wait for rotation to complete
        time.sleep(rotation_duration)
        
        # Stop movement
        sport_client.StopMove()
        time.sleep(0.5)  # Settling time
        
        # Capture and detect
        frame = get_camera_frame(video_client)
        detections = detect_objects(model, frame)
        print_detections(target_angle, detections)
        
        current_angle = target_angle
    
    print("\n" + "=" * 70)
    print("Panorama scan complete!")
    
    # Return to center
    if current_angle != 0:
        print(f"\nReturning to center from {current_angle:.1f}°")
        rotation_needed = -current_angle
        rotation_duration = abs(rotation_needed) * np.pi / 180 / 0.5
        
        if rotation_needed > 0:
            sport_client.Move(0, 0, 0.5)
        else:
            sport_client.Move(0, 0, -0.5)
        
        time.sleep(rotation_duration)
        sport_client.StopMove()
        time.sleep(0.5)

if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    
    # Initialize channel
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, ethernet_interface)
    
    # Initialize sport and video clients
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    
    video_client = VideoClient()
    video_client.SetTimeout(3.0)
    video_client.Init()
    
    # Load YOLO model (using default YOLOv8n)
    print("\nLoading YOLO model (YOLOv8n)...")
    try:
        model = YOLO('yolov8n.pt')  # This will auto-download on first run
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Make sure ultralytics is installed: pip install ultralytics")
        sys.exit(1)
    
    try:
        # Perform panorama scan
        # Parameters: max_angle=90 (scan ±90°), step_angle=15 (15° increments)
        scan_panorama(sport_client, video_client, model, max_angle=90, step_angle=15)
        
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user")
    except Exception as e:
        print(f"\n\nError during scan: {e}")
    finally:
        # Cleanup
        print("\nStopping robot...")
        sport_client.StopMove()
        sport_client.Damp()
        print("Done.")

