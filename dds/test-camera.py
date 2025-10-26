from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2
import numpy as np
import sys
from flask import Flask, render_template_string, Response
import threading
from datetime import datetime
import time

ethernet_interface = "enP8p1s0"

app = Flask(__name__)

# Global variables
latest_frame = None
frame_lock = threading.Lock()
video_writer = None
is_recording = False

# HTML template for the webpage
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Go2 Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .video-container {
            text-align: center;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        img {
            max-width: 100%;
            border: 2px solid #333;
            border-radius: 5px;
        }
        .info {
            margin-top: 20px;
            padding: 10px;
            background-color: #e8f4f8;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>Go2 Robot Camera Feed</h1>
    <div class="video-container">
        <img src="{{ url_for('video_feed') }}" alt="Camera Stream">
        <div class="info">
            <p><strong>Status:</strong> Live Stream Active</p>
            <p><strong>Note:</strong> Video is being saved locally with timestamps</p>
        </div>
    </div>
</body>
</html>
"""

def generate_frames():
    """Generator function to stream frames to the web page"""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)
                continue
            frame = latest_frame.copy()
        
        # Resize to 540p (960x540) for web streaming
        frame_540p = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        
        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame_540p, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the main webpage"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    """Run Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

def camera_loop(client):
    """Main camera loop to capture and process frames"""
    global latest_frame, video_writer, is_recording
    
    # Setup video writer with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"go2_recording_{timestamp}.mp4"
    
    # Video writer parameters for low filesize
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
    fps = 15.0
    frame_size = None
    
    code, data = client.GetImageSample()
    
    # Request normal when code==0
    while code == 0:
        # Get Image data from Go2 robot
        code, data = client.GetImageSample()
        
        # Convert to numpy image
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is None:
            continue
        
        # Resize to 720p (1280x720)
        # image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
        
        # Initialize video writer on first frame
        if video_writer is None and image is not None:
            frame_size = (image.shape[1], image.shape[0])
            video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
            is_recording = True
            print(f"Recording started: {output_filename}")
            print(f"Frame size: {frame_size}")
        
        # Update latest frame for web streaming
        with frame_lock:
            latest_frame = image
        
        # Write frame to video file
        if video_writer is not None and is_recording:
            video_writer.write(image)
        
        time.sleep(0.01)  # Small delay to control frame rate
    
    if code != 0:
        print("Get image sample error. code:", code)
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
        print(f"Recording saved: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, ethernet_interface)

    client = VideoClient()  # Create a video client
    client.SetTimeout(3.0)
    client.Init()
    
    print("Starting camera stream...")
    print("Web interface available at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    try:
        # Run camera loop in main thread
        camera_loop(client)
    except KeyboardInterrupt:
        print("\nStopping camera stream...")
    finally:
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
