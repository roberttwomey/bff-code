from ultralytics import YOLO

# Load an official or custom model
# model = YOLO("yolo11n.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model

# Choose input source
USE_WEBCAM = True  # Set to True for webcam, False for YouTube

if USE_WEBCAM:
    # Use webcam (device 0 is usually the default camera)
    input_source = 0
    print("Using webcam as input source")
else:
    # Use YouTube video
    input_source = "https://youtu.be/LNwODJXcvt4"
    print("Using YouTube video as input source")

# Perform tracking with the model
results = model.track(input_source, show=True, device='mps')  # Tracking with default tracker
# results = model.track(input_source, show=True, tracker="bytetrack.yaml")  # with ByteTrack