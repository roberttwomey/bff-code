# Human Following Robot Script (MediaPipe Version)

This script enables the Go2 robot to detect and follow humans using **MediaPipe pose detection** instead of YOLO. MediaPipe provides detailed skeleton tracking that offers several advantages for human following applications.

## Key Advantages of MediaPipe over YOLO

### 1. **Skeleton-Based Tracking**
- **33 Body Landmarks**: Tracks nose, shoulders, hips, knees, ankles, and more
- **Pose Confidence**: Uses landmark visibility for more reliable detection
- **Better Occlusion Handling**: Can track humans even when partially visible

### 2. **More Accurate Following**
- **Nose-Centered Tracking**: Uses nose position instead of bounding box center
- **Body Dimension Awareness**: Estimates human size using shoulder width and body height
- **Smoother Movement**: More stable tracking leads to smoother robot movement

### 3. **Performance Benefits**
- **Lighter Model**: MediaPipe is optimized for real-time performance
- **CPU-Friendly**: Works well on devices without GPU acceleration
- **Lower Latency**: Faster inference for more responsive following

## Features

- **Pose Detection**: Uses MediaPipe's 33-point body landmark detection
- **Nose-Centered Tracking**: Tracks the nose position for precise centering
- **Body Size Estimation**: Calculates distance using shoulder width and body height
- **Visual Feedback**: Displays skeleton landmarks and tracking information
- **Aggressive Movement**: Fast, responsive following behavior
- **Configurable Speed**: Adjustable command rate for smooth operation

## Prerequisites

1. **Go2 Robot**: A Go2 robot with WebRTC capabilities
2. **Network Connection**: Robot must be accessible via IP address
3. **Python 3.8+**: Required for async/await support

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_mediapipe.txt
```

2. MediaPipe will be automatically downloaded on first run

## Configuration

Edit the `ip` parameter in the `HumanFollowerMediaPipe` class constructor:

```python
follower = HumanFollowerMediaPipe(ip="192.168.4.30")  # Replace with your robot's IP
```

## Usage

Run the script:
```bash
python human_following_mediapipe.py
```

### Controls

- **Q key**: Quit the program
- **Ctrl+C**: Stop the program gracefully

## How It Works

### 1. **MediaPipe Pose Detection**
- Processes video frames using MediaPipe's pose model
- Detects 33 body landmarks including nose, shoulders, hips, etc.
- Calculates confidence based on landmark visibility
- Estimates human size using shoulder width and body height

### 2. **Nose-Centered Tracking**
- Uses nose position as the primary tracking point
- More accurate than bounding box center
- Better handles partial occlusions and body rotations

### 3. **Enhanced Distance Estimation**
- Combines shoulder width and body height for better size calculation
- More accurate distance estimation than bounding box area
- Adaptive speed control based on actual human dimensions

### 4. **Visual Servoing**
- **Turning**: Calculates horizontal offset from nose position and turns robot accordingly
- **Forward Movement**: Always moves forward when humans are detected
- **Speed Control**: Adjusts speed based on distance and centering quality

## Control Parameters

```python
self.target_center_x = 640      # Target center of frame
self.center_tolerance = 5       # Pixels tolerance for center
self.turn_speed = 1.5          # Angular velocity for turning
self.move_speed = 1.25         # Forward movement speed
self.min_pose_confidence = 0.5 # Minimum confidence for pose detection
self.command_rate_hz = 10      # Commands per second
```

## MediaPipe Model Configuration

```python
self.pose = self.mp_pose.Pose(
    static_image_mode=False,      # Video mode
    model_complexity=1,           # 0=Lite, 1=Full, 2=Heavy
    smooth_landmarks=True,        # Smooth landmark positions
    enable_segmentation=False,    # Disable for performance
    smooth_segmentation=True,     # Smooth segmentation
    min_detection_confidence=0.5, # Minimum detection confidence
    min_tracking_confidence=0.5   # Minimum tracking confidence
)
```

## Performance Tuning

### **Model Complexity**
- **0 (Lite)**: Fastest, good for real-time applications
- **1 (Full)**: Balanced performance and accuracy (default)
- **2 (Heavy)**: Highest accuracy, slower performance

### **Confidence Thresholds**
- **Detection**: Lower values detect humans more easily but may have false positives
- **Tracking**: Higher values provide more stable tracking but may lose tracking more easily

### **Command Rate**
- **Higher rates (20Hz)**: More responsive but potentially jerky
- **Lower rates (5Hz)**: Smoother but less responsive

## Comparison with YOLO Version

| Feature | YOLO Version | MediaPipe Version |
|---------|--------------|-------------------|
| **Detection Method** | Bounding box | Skeleton landmarks |
| **Tracking Point** | Box center | Nose position |
| **Distance Estimation** | Box area | Body dimensions |
| **Occlusion Handling** | Basic | Advanced |
| **Performance** | GPU-dependent | CPU-optimized |
| **Accuracy** | Good | Excellent |
| **Smoothness** | Moderate | High |

## Troubleshooting

### Common Issues

1. **No Pose Detection**: Check lighting conditions and ensure person is fully visible
2. **Jumpy Tracking**: Reduce command rate or increase smoothing parameters
3. **Poor Performance**: Lower model complexity or reduce confidence thresholds
4. **High CPU Usage**: Lower command rate or use Lite model complexity

### Performance Optimization

- Use `model_complexity=0` for maximum speed
- Reduce `command_rate_hz` for smoother movement
- Adjust confidence thresholds based on your environment
- Ensure good lighting for better landmark detection

## Customization

### **Adding New Behaviors**
Modify the `calculate_control_signals` method to use additional pose information:

```python
def calculate_control_signals(self, humans):
    # Access additional pose data:
    # human['shoulder_width'] - shoulder width in pixels
    # human['body_height'] - body height in pixels
    # human['landmarks'] - full MediaPipe landmarks object
    
    # Add your custom logic here
    pass
```

### **Pose-Based Behaviors**
Use pose landmarks for advanced behaviors:

```python
# Check if person is raising their hand
left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
if left_wrist.y < left_shoulder.y:
    print("Person is raising left hand!")

# Check if person is walking
hip_movement = calculate_hip_movement(landmarks)
if hip_movement > threshold:
    print("Person is moving!")
```

## License

This script is provided as-is for educational and research purposes. Use at your own risk and always ensure proper safety measures when operating robots.
