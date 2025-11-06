# Pose-Based Visual Servoing (PBVS) Controller

A simple pose-based visual servoing controller with a state machine that sequences rotate/strafe/forward movements for stable control even with jittery Aruco tag detections.

## Overview

This controller implements a 5-state pose-based visual servoing system:

- **S0 (Search)**: Slow rotation to find Aruco tag
- **S1 (Yaw Lock)**: Align robot yaw with tag normal
- **S2 (Lateral Center)**: Strafe to center tag laterally
- **S3 (Approach)**: Move forward/back to achieve 1m distance
- **S4 (Fine Settle)**: Fine positioning for precise alignment

## Features

- **Stable Detection**: Uses N consecutive detections to establish stability
- **State Machine**: Sequential control prevents oscillation and ensures systematic approach
- **Robust to Jitter**: State transitions require M consecutive stable frames
- **Automatic Recovery**: Returns to search mode if detection is lost for K frames
- **Real-time Visualization**: Shows current state and error metrics

## Requirements

Install dependencies:
```bash
pip install -r requirements_pbvs.txt
```

## Usage

1. **Setup Aruco Tag**: Place a 6x6 Aruco tag (from DICT_6X6_1000) in your environment
2. **Configure Robot IP**: Update the IP address in the controller if needed
3. **Run Controller**:
   ```bash
   python pbvs_controller.py
   ```

## Configuration

### Key Parameters

- `K = 10`: Max consecutive lost detections before returning to S0
- `M = 5`: Consecutive stable frames needed for state transition  
- `N = 3`: Consecutive detections needed for "stable" flag
- `tag_size = 0.2`: Physical size of Aruco tag in meters

### Control Gains

- `k_psi = 1.0`: Yaw control gain
- `k_y = 0.5`: Lateral control gain
- `k_x = 0.3`: Forward/back control gain

### Velocity Limits

- `psi_max = 1.0`: Max yaw velocity (rad/s)
- `vx_max = 0.5`: Max forward velocity (m/s)
- `vy_max = 0.4`: Max lateral velocity (m/s)

## State Machine Details

### S0: Search
- **Action**: Slow spin (0.3 rad/s)
- **Transition**: When tag detected and stable for N frames
- **Purpose**: Find Aruco tag in field of view

### S1: Yaw Lock  
- **Action**: Pure yaw control to align with tag normal
- **Transition**: When yaw error < 10° for M consecutive frames
- **Purpose**: Establish proper orientation

### S2: Lateral Center
- **Action**: Strafe to center tag, maintain yaw alignment
- **Transition**: When lateral error < 5cm and yaw error < 5° for M frames
- **Purpose**: Center tag in robot's field of view

### S3: Approach
- **Action**: Move forward/back to 1m distance, maintain centering
- **Transition**: When distance error < 5cm for M consecutive frames
- **Purpose**: Achieve target distance

### S4: Fine Settle
- **Action**: Fine positioning with reduced gains
- **Transition**: When all errors < tolerance (3cm, 3°)
- **Purpose**: Precise final positioning

## Error Calculations

The controller calculates three key errors:

1. **Lateral Error (e_y)**: Left/right offset from tag center
2. **Distance Error (e_x)**: Forward/back distance to 1m target
3. **Yaw Error (e_psi)**: Angular alignment with tag normal

## Coordinate System

- **x**: Forward/backward (positive = forward)
- **y**: Left/right (positive = left)  
- **z**: Rotation (positive = turn left)

## Camera Calibration

The controller uses approximate camera parameters. For better performance:

1. **Calibrate Camera**: Use OpenCV calibration tools
2. **Update Camera Matrix**: Replace `fx`, `fy`, `cx`, `cy` with calibrated values
3. **Calibrate Transform**: Measure and update `R_bc`, `t_bc` (camera to base transform)

## IMU Integration

The `psi_curr()` method currently returns 0 as a placeholder. To integrate with robot IMU:

1. **Access IMU Data**: Get yaw from robot's IMU sensor
2. **Update Method**: Replace placeholder with actual IMU reading
3. **Coordinate Frame**: Ensure IMU yaw matches robot's coordinate system

## Troubleshooting

### No Tag Detection
- Check tag size configuration matches physical tag
- Verify tag is from DICT_6X6_1000 dictionary
- Ensure adequate lighting

### Oscillatory Behavior
- Reduce control gains (`k_psi`, `k_y`, `k_x`)
- Increase stability thresholds (`M`, `N`)
- Check camera calibration

### Slow Convergence
- Increase control gains
- Reduce stability thresholds
- Verify tag size and camera parameters

## Safety Features

- **Automatic Stop**: Robot stops if detection lost for K frames
- **Velocity Limits**: All commands are clamped to safe limits
- **Graceful Shutdown**: Clean shutdown on Ctrl+C
- **Emergency Stop**: Robot stops immediately on interruption

## Example Output

```
Connected to Go2 robot
Switched to MCF mode
Video stream started
Starting PBVS controller...
State: S0, Errors: x=0.000, y=0.000, psi=0.0°
Tag detected and stable, transitioning to S1 (yaw lock)
State: S1, Errors: x=0.500, y=0.100, psi=15.2°
Yaw aligned, transitioning to S2 (lateral center)
State: S2, Errors: x=0.450, y=0.080, psi=2.1°
Laterally centered, transitioning to S3 (approach)
State: S3, Errors: x=0.200, y=0.020, psi=1.5°
Distance achieved, transitioning to S4 (fine settle)
State: S4, Errors: x=0.025, y=0.015, psi=1.2°
Target position achieved!
```
