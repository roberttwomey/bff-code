# Aruco Visual Servoing for Go2 Robot

This program enables the Go2 robot to detect Aruco tags and automatically position itself 1 meter away and perpendicular to the tag using visual servoing.

## Features

- **Aruco Tag Detection**: Uses OpenCV's Aruco module to detect 6x6 Aruco tags
- **Visual Servoing**: Automatically controls robot movement to achieve target position
- **Perpendicularity Detection**: Analyzes tag perspective distortion to determine orientation
- **Distance Estimation**: Estimates distance to tag using tag size in pixels
- **Multi-axis Movement**: Supports forward/backward, left/right, and rotational movement
- **State Machine**: Implements different behaviors for searching, centering, orienting, and approaching
- **Real-time Display**: Shows detection results, perpendicularity analysis, and robot state in real-time

## Requirements

- Go2 robot with camera
- Python 3.7+
- Required packages (see `requirements_aruco.txt`)

## Installation

1. Install the required packages:
```bash
pip install -r requirements_aruco.txt
```

2. Ensure your Go2 robot is connected to the same network as your computer.

## Usage

1. **Prepare an Aruco Tag**: 
   - Use a 6x6 Aruco tag (dictionary DICT_6X6_250)
   - Print it on paper or display it on a screen
   - The default tag size is assumed to be 0.1 meters (10 cm) - adjust `tag_size` in the code if needed

2. **Run the Program**:
```bash
python aruco_visual_servoing.py
```

3. **Robot Behavior**:
   - **Searching**: Robot rotates to find an Aruco tag
   - **Centering**: Once found, robot moves left/right to center the tag horizontally
   - **Orienting**: Robot moves forward/backward to become perpendicular to the tag surface
   - **Approaching**: Robot moves forward/backward to achieve 1-meter distance
   - **Positioned**: Robot maintains position when perpendicular and at target distance

**Movement Priority Order:**
1. **Lateral Movement** (left/right) - for horizontal centering
2. **Forward/Backward** - for distance and vertical orientation
3. **Rotation** (left/right) - as fallback for orientation adjustments

4. **Controls**:
   - Press 'q' to quit the program
   - The robot will automatically stop when the program exits

## Configuration

You can adjust these parameters in the `ArucoVisualServoing` class:

- `target_distance`: Target distance to tag (default: 1.0 meters)
- `distance_tolerance`: Acceptable distance range (default: ±0.1 meters)
- `center_tolerance`: Horizontal centering tolerance (default: ±30 pixels)
- `tag_size`: Physical size of Aruco tag in meters (default: 0.2 meters)
- `turn_speed`: Angular velocity for turning (default: 0.8)
- `move_speed`: Forward/backward movement speed (default: 0.6)
- `sideways_speed`: Sideways movement speed (default: 0.4)
- `search_speed`: Rotation speed when searching (default: 1.0)

## Visual Feedback

The program displays:
- **Green rectangle**: Detected Aruco tag boundaries
- **Blue circle**: Tag center point
- **Yellow line**: Target center position
- **Green lines**: Centering tolerance zone
- **Color-coded text**: Perpendicularity score and orientation information
- **Red arrows**: Direction indicators for orientation corrections
- **Text overlay**: Tag ID, distance, perpendicularity score, and robot state

## Troubleshooting

1. **No tags detected**: 
   - Ensure the Aruco tag is visible and well-lit
   - Check that the tag size matches the `tag_size` parameter
   - Try adjusting lighting conditions

2. **Poor distance estimation**:
   - Calibrate the `focal_length` parameter for your camera
   - Measure and set the correct `tag_size`

3. **Robot not moving**:
   - Check network connection to the robot
   - Ensure the robot is in MCF mode
   - Verify the robot's IP address is correct

## Safety Notes

- Always ensure the robot has clear space to move
- Keep the Aruco tag at a safe distance during testing
- Be ready to stop the program if needed
- The robot will stop automatically when the program exits

## Technical Details

- Uses OpenCV's Aruco detector with DICT_6X6_250 dictionary
- Distance estimation based on tag size in pixels and assumed focal length
- Continuous movement commands sent at 10 Hz
- WebRTC video streaming for real-time camera feed
- Asynchronous movement control for smooth operation
