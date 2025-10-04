"""
YOLO GPU Tracking Examples
=========================

This script demonstrates how to use GPU acceleration with YOLO for various tracking tasks.
"""

from ultralytics import YOLO
import torch
import cv2
import time

def check_gpu_setup():
    """Check GPU availability and setup"""
    print("=== GPU Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"Current GPU: {torch.cuda.current_device()}")
    else:
        print("No GPU available - will use CPU")
    print()

def basic_gpu_tracking():
    """Basic GPU tracking example"""
    print("=== Basic GPU Tracking ===")
    
    # Load model
    model = YOLO("yolo11n-pose.pt")
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on: {device}")
    
    # Track with GPU
    results = model.track(
        "https://youtu.be/LNwODJXcvt4",
        show=True,
        device=device,
        tracker="bytetrack.yaml"
    )

def webcam_gpu_tracking():
    """Webcam tracking with GPU acceleration"""
    print("=== Webcam GPU Tracking ===")
    
    # Load model
    model = YOLO("yolo11n-pose.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print(f"Starting webcam tracking on {device}")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run tracking
            results = model.track(frame, device=device, tracker="bytetrack.yaml")
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Show frame
            cv2.imshow('YOLO GPU Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping webcam tracking...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def performance_comparison():
    """Compare CPU vs GPU performance"""
    print("=== Performance Comparison ===")
    
    model = YOLO("yolo11n-pose.pt")
    
    # Test image
    test_image = "https://ultralytics.com/images/bus.jpg"
    
    # CPU test
    print("Testing CPU performance...")
    start_time = time.time()
    for i in range(10):
        results = model.track(test_image, device='cpu', verbose=False)
    cpu_time = time.time() - start_time
    print(f"CPU time for 10 runs: {cpu_time:.2f}s")
    
    if torch.cuda.is_available():
        # GPU test
        print("Testing GPU performance...")
        start_time = time.time()
        for i in range(10):
            results = model.track(test_image, device='cuda', verbose=False)
        gpu_time = time.time() - start_time
        print(f"GPU time for 10 runs: {gpu_time:.2f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("GPU not available for comparison")

def advanced_gpu_tracking():
    """Advanced GPU tracking with custom settings"""
    print("=== Advanced GPU Tracking ===")
    
    model = YOLO("yolo11n-pose.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Custom tracking parameters
    tracking_params = {
        'device': device,
        'tracker': 'bytetrack.yaml',
        'conf': 0.5,  # Confidence threshold
        'iou': 0.7,   # IoU threshold
        'max_det': 10,  # Maximum detections
        'half': True,   # Use half precision (FP16) for faster inference
        'dnn': False,   # Use OpenCV DNN backend
    }
    
    print(f"Using device: {device}")
    print(f"Half precision: {tracking_params['half']}")
    
    # Track with custom parameters
    results = model.track(
        "https://youtu.be/LNwODJXcvt4",
        show=True,
        **tracking_params
    )

def memory_optimization():
    """GPU memory optimization techniques"""
    print("=== GPU Memory Optimization ===")
    
    if not torch.cuda.is_available():
        print("GPU not available")
        return
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Load model with memory optimization
    model = YOLO("yolo11n-pose.pt")
    model.to('cuda')
    
    # Use half precision to save memory
    model.half()
    print("Model converted to half precision (FP16)")
    
    # Track with memory-efficient settings
    results = model.track(
        "https://youtu.be/LNwODJXcvt4",
        show=True,
        device='cuda',
        half=True,  # Use half precision
        tracker="bytetrack.yaml"
    )

if __name__ == "__main__":
    # Check GPU setup
    check_gpu_setup()
    
    # Choose which example to run
    print("Available examples:")
    print("1. Basic GPU tracking")
    print("2. Webcam GPU tracking")
    print("3. Performance comparison")
    print("4. Advanced GPU tracking")
    print("5. Memory optimization")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        basic_gpu_tracking()
    elif choice == "2":
        webcam_gpu_tracking()
    elif choice == "3":
        performance_comparison()
    elif choice == "4":
        advanced_gpu_tracking()
    elif choice == "5":
        memory_optimization()
    else:
        print("Invalid choice")

