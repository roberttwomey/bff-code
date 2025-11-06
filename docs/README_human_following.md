# Human Following Robot Script

This script enables the Go2 robot to detect humans using YOLO (You Only Look Once) object detection and perform visual servoing to follow them.

## Features

- **Human Detection**: Uses YOLOv8 nano model for real-time human detection
- **Visual Servoing**: Automatically turns towards detected humans and moves towards them
- **Safety Features**: Slows down as it gets closer to humans and stops when no humans are detected
- **Real-time Video**: Displays the camera feed with detection overlays

## Prerequisites

1. **Go2 Robot**: A Go2 robot with WebRTC capabilities
2. **Network Connection**: Robot must be accessible via IP address
3. **Python 3.8+**: Required for async/await support

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_human_following.txt
```

2. Download the YOLOv8 model (will be downloaded automatically on first run):
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Configuration

Edit the `ip` parameter in the `HumanFollower` class constructor in `human_following.py`:

```python
follower = HumanFollower(ip="192.168.8.181")  # Replace with your robot's IP
```

## Usage

Run the script:
```bash
cd sandbox
python human_following.py
```

### Controls

- **Q key**: Quit the program
- **Ctrl+C**: Stop the program gracefully

## How It Works

### 1. Human Detection
- Uses YOLOv8 nano model to detect humans in each video frame
- Filters detections by confidence threshold (default: 0.5)
- Identifies the largest human (assumed to be closest to camera)

### 2. Visual Servoing
- **Turning**: Calculates horizontal offset from frame center and turns robot accordingly
- **Forward Movement**: Always moves forward (positive x direction) when humans are detected, with speed adjusted based on distance and centering
- **Safety**: Automatically stops when no humans are detected

### 3. Control Parameters

```python
self.target_center_x = 640      # Target center of frame
self.center_tolerance = 50      # Pixels tolerance for center
self.turn_speed = 0.3          # Angular velocity for turning
self.move_speed = 0.2          # Forward movement speed (positive x direction)
self.min_human_confidence = 0.5 # Minimum confidence for detection
```

### 4. Movement Behavior

The robot will:
- **Always move forward** (positive x direction) when humans are detected
- **Turn left/right** to center the human in the frame
- **Slow down** as it gets closer to humans (distance-based speed control)
- **Reduce forward speed** when humans are not well centered
- **Maintain minimum forward movement** (0.05) to ensure continuous approach
- **Stop completely** when no humans are detected

## Safety Considerations

- **Distance Control**: Robot slows down as it approaches humans
- **Automatic Stop**: Stops moving when no humans are detected
- **Emergency Stop**: Use Ctrl+C to immediately stop all movement
- **Supervision**: Always supervise the robot during operation

## Troubleshooting

### Common Issues

1. **Connection Failed**: Check robot IP address and network connectivity
2. **No Video Stream**: Ensure robot is in WebRTC mode and camera is enabled
3. **Poor Detection**: Adjust confidence threshold or lighting conditions
4. **Erratic Movement**: Tune control parameters for smoother operation

### Performance Optimization

- Use a more powerful YOLO model (e.g., `yolov8s.pt` or `yolov8m.pt`) for better accuracy
- Reduce frame processing rate if performance is poor
- Adjust control parameters based on your specific use case

## Customization

### Adding New Behaviors

You can extend the script by modifying the `calculate_control_signals` method:

```python
def calculate_control_signals(self, humans):
    # Add your custom logic here
    # Return (move_x, move_y, turn_z) values
    pass
```

### Different Detection Models

Change the YOLO model in the constructor:

```python
self.model = YOLO('yolov8s.pt')  # Small model for better accuracy
self.model = YOLO('yolov8m.pt')  # Medium model for balanced performance
```

## License

This script is provided as-is for educational and research purposes. Use at your own risk and always ensure proper safety measures when operating robots.
