# YOLO GPU Acceleration Setup

This guide explains how to set up GPU acceleration for YOLO tracking in your robot following system.

## Prerequisites

### 1. NVIDIA GPU Setup
- **NVIDIA GPU** with CUDA support (GTX 1060 or better recommended)
- **NVIDIA Drivers** (latest version)
- **CUDA Toolkit** (11.8 or 12.x)
- **cuDNN** (compatible with your CUDA version)

### 2. Python Environment
```bash
# Create virtual environment
python -m venv yolo_gpu_env
source yolo_gpu_env/bin/activate  # On Windows: yolo_gpu_env\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install YOLO and dependencies
pip install ultralytics
pip install opencv-python
pip install numpy
```

## Installation Steps

### 1. Check GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### 2. Install Requirements
```bash
pip install -r requirements_yolo_gpu.txt
```

### 3. Test GPU Acceleration
```python
from ultralytics import YOLO
import torch

# Load model
model = YOLO("yolo11n-pose.pt")

# Move to GPU
if torch.cuda.is_available():
    model.to('cuda')
    print("Model loaded on GPU")
    
    # Test inference
    results = model.track("https://youtu.be/LNwODJXcvt4", device='cuda')
    print("GPU acceleration working!")
else:
    print("GPU not available")
```

## Usage Examples

### Basic GPU Tracking
```python
from ultralytics import YOLO
import torch

# Load model
model = YOLO("yolo11n-pose.pt")

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Track with GPU
results = model.track(
    "video.mp4",
    show=True,
    device=device,
    tracker="bytetrack.yaml"
)
```

### Webcam GPU Tracking
```python
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
model.to('cuda')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run tracking on GPU
    results = model.track(frame, device='cuda', tracker="bytetrack.yaml")
    
    # Draw results
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO GPU Tracking', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Performance Optimization

### 1. Half Precision (FP16)
```python
# Enable half precision for faster inference
model.half()
results = model.track(frame, device='cuda', half=True)
```

### 2. Memory Management
```python
import torch

# Clear GPU cache
torch.cuda.empty_cache()

# Monitor GPU memory
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### 3. Batch Processing
```python
# Process multiple frames at once
frames = [frame1, frame2, frame3]
results = model.track(frames, device='cuda')
```

## Integration with Robot Following

### 1. Replace MediaPipe with YOLO
```python
# Instead of MediaPipe pose detection
# Use YOLO pose detection with GPU acceleration

def detect_humans_yolo_gpu(self, frame):
    results = self.model.track(
        frame,
        device='cuda',
        conf=0.5,
        tracker="bytetrack.yaml"
    )
    
    # Process results similar to MediaPipe
    # Extract keypoints and calculate control signals
```

### 2. Performance Benefits
- **Speed**: 3-5x faster inference on GPU
- **Accuracy**: Better pose detection with YOLO
- **Tracking**: Built-in object tracking with ByteTrack
- **Memory**: Efficient GPU memory usage

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or use half precision
   model.half()
   torch.cuda.empty_cache()
   ```

2. **CUDA Not Available**
   ```bash
   # Reinstall PyTorch with CUDA support
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Slow Performance**
   ```python
   # Use half precision
   model.half()
   
   # Optimize model
   model.optimize()
   ```

### Performance Monitoring
```python
import time
import torch

# Benchmark inference time
start_time = time.time()
results = model.track(frame, device='cuda')
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.3f}s")

# Monitor GPU usage
print(f"GPU utilization: {torch.cuda.utilization()}%")
```

## Files Created

- `sandbox/test_yolo_pose.py` - Basic YOLO GPU setup
- `sandbox/yolo_gpu_tracking.py` - Comprehensive GPU examples
- `sandbox/yolo_gpu_human_following.py` - Robot following with YOLO GPU
- `requirements_yolo_gpu.txt` - GPU requirements
- `README_yolo_gpu.md` - This documentation

## Next Steps

1. **Test GPU Setup**: Run `python sandbox/test_yolo_pose.py`
2. **Benchmark Performance**: Use `sandbox/yolo_gpu_tracking.py`
3. **Integrate with Robot**: Modify your existing following code
4. **Optimize Settings**: Adjust confidence thresholds and tracking parameters

## Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [ByteTrack Tracker](https://github.com/ifzhang/ByteTrack)

