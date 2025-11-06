# Simple ArUco Detection and Approach Program

This program detects an ArUco tag using the Go2 robot's camera, centers it in the field of view, approaches until 1 meter away, and then pauses.

## Features

- **ArUco Tag Detection**: Uses OpenCV's ArUco detector with 6x6 dictionary
- **Camera Calibration**: Uses provided camera intrinsic and distortion parameters
- **Visual Servoing**: Simple proportional control for centering and approaching
- **Real-time Display**: Shows detection information and camera feed
- **Safe Operation**: Velocity limits and emergency stop functionality

## Requirements

Install the required dependencies:

```bash
pip install -r requirements_aruco_simple.txt
```

## Usage

1. **Prepare an ArUco Tag**: Print or display a 6x6 ArUco tag (20cm physical size)
2. **Connect to Go2**: Make sure your computer is connected to the Go2 robot's WiFi network
3. **Run the Program**:

```bash
python aruco_detection_simple.py
```

## How It Works

1. **Connection**: Connects to the Go2 robot and switches to MCF motion mode
2. **Video Stream**: Starts receiving camera frames from the robot
3. **Tag Detection**: Continuously detects ArUco tags in the camera feed
4. **Centering**: Moves laterally to center the tag in the image
5. **Approach**: Moves forward/backward to reach 1 meter distance
6. **Pause**: Stops and waits when target distance is reached

## Control Parameters

- **Target Distance**: 1.0 meters
- **Distance Tolerance**: 0.05 meters
- **Center Tolerance**: 50 pixels
- **Max Linear Velocity**: 0.3 m/s
- **Max Angular Velocity**: 0.5 rad/s

## Camera Calibration

The program uses the provided camera calibration parameters:

```python
GO2_CAM_K = np.array([
    [818.18507419, 0.0, 637.94628188],
    [0.0, 815.32431463, 338.3480119],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

GO2_CAM_D = np.array([[-0.07203219],
                      [-0.05228525],
                      [ 0.05415833],
                      [-0.02288355]], dtype=np.float32)
```

## ArUco Tag Specifications

- **Dictionary**: DICT_6X6_1000
- **Physical Size**: 20cm x 20cm
- **Detection**: Any tag ID from the dictionary will work

## Safety Features

- **Velocity Limits**: Prevents excessive speeds
- **Emergency Stop**: Press 'q' to quit or Ctrl+C to interrupt
- **No Tag Detection**: Robot stops when no tag is visible
- **Distance Monitoring**: Continuous distance checking

## Troubleshooting

1. **No Video Stream**: Check WiFi connection and robot IP address
2. **No Tag Detection**: Ensure proper lighting and tag visibility
3. **Poor Tracking**: Adjust control gains in the code if needed
4. **Connection Issues**: Verify robot is powered on and in range

## Customization

You can modify the following parameters in the code:

- `target_distance`: Change the approach distance
- `tag_size`: Adjust for different tag sizes
- Control gains (`k_centering`, `k_approach`, `k_yaw`): Tune robot behavior
- Velocity limits: Adjust for different speed requirements
