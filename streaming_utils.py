"""
Utility classes for streaming cv2 images via various protocols.
"""
import cv2
import threading
import time
from flask import Flask, Response, render_template_string
from queue import Queue, Empty


class MJPEGStreamer:
    """
    Stream cv2 images as MJPEG over HTTP.
    Works in any web browser - just open http://localhost:PORT
    """
    def __init__(self, port=8080, quality=85, max_fps=30):
        self.port = port
        self.quality = quality
        self.max_fps = max_fps
        self.frame_queue = Queue(maxsize=2)  # Keep only latest frames
        self.app = Flask(__name__)
        self.running = False
        self.server_thread = None
        
        # Setup Flask routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes for streaming"""
        
        HTML_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Stream</title>
            <style>
                body { margin: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; }
                img { max-width: 100%; max-height: 100%; }
            </style>
        </head>
        <body>
            <img src="{{ url_for('video_feed') }}" alt="Video Stream">
        </body>
        </html>
        """
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def _generate_frames(self):
        """Generator function to stream frames"""
        frame_time = 1.0 / self.max_fps
        
        while self.running:
            try:
                # Get latest frame (non-blocking)
                frame = self.frame_queue.get(timeout=0.1)
                
                # Encode as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           frame_bytes + b'\r\n')
                
                time.sleep(frame_time)
                
            except Empty:
                # No frame available, send a placeholder or wait
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in frame generation: {e}")
                time.sleep(0.1)
    
    def update_frame(self, frame):
        """Update the current frame to stream"""
        if not self.running:
            return
        
        # Clear old frames and add new one
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        self.frame_queue.put(frame.copy())
    
    def start(self):
        """Start the streaming server"""
        if self.running:
            return
        
        self.running = True
        self.server_thread = threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=self.port, 
                                       debug=False, threaded=True, use_reloader=False),
            daemon=True
        )
        self.server_thread.start()
        print(f"MJPEG stream started at http://localhost:{self.port}")
    
    def stop(self):
        """Stop the streaming server"""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=2)


class RTSPStreamer:
    """
    Stream cv2 images via RTSP using GStreamer/FFmpeg.
    Requires: pip install opencv-python[headless] (with GStreamer support)
    Or use subprocess with ffmpeg
    """
    def __init__(self, rtsp_url="rtsp://localhost:8554/stream", fps=30):
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.writer = None
        self.writer_thread = None
    
    def _writer_loop(self):
        """Write frames to RTSP stream"""
        # Try to use OpenCV VideoWriter with GStreamer pipeline
        pipeline = (
            f"appsrc ! video/x-raw,format=BGR ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc speed-preset=ultrafast tune=zerolatency ! "
            f"rtph264pay name=pay0 pt=96"
        )
        
        try:
            fourcc = cv2.VideoWriter.fourcc(*'H264')
            # Note: This requires GStreamer support in OpenCV
            # Alternative: Use subprocess with ffmpeg
            self.writer = cv2.VideoWriter(
                self.rtsp_url, 
                fourcc, 
                self.fps, 
                (640, 480)  # Will be set from first frame
            )
        except Exception as e:
            print(f"RTSP via OpenCV failed: {e}")
            print("Falling back to FFmpeg subprocess method...")
            self._use_ffmpeg_subprocess()
            return
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if self.writer and self.writer.isOpened():
                    self.writer.write(frame)
            except Empty:
                continue
            except Exception as e:
                print(f"Error writing RTSP frame: {e}")
    
    def _use_ffmpeg_subprocess(self):
        """Use FFmpeg subprocess for RTSP streaming"""
        import subprocess
        import numpy as np
        
        # FFmpeg command for RTSP server
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '640x480',  # Adjust based on your frame size
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-f', 'rtsp',
            self.rtsp_url
        ]
        
        try:
            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    process.stdin.write(frame.tobytes())
                except Empty:
                    continue
                except Exception as e:
                    print(f"Error in FFmpeg streaming: {e}")
                    break
            
            process.stdin.close()
            process.wait()
        except FileNotFoundError:
            print("FFmpeg not found. Please install FFmpeg for RTSP streaming.")
        except Exception as e:
            print(f"FFmpeg subprocess error: {e}")
    
    def update_frame(self, frame):
        """Update the current frame to stream"""
        if not self.running:
            return
        
        # Keep only latest frame
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        self.frame_queue.put(frame.copy())
    
    def start(self):
        """Start RTSP streaming"""
        if self.running:
            return
        
        self.running = True
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        print(f"RTSP stream started at {self.rtsp_url}")
    
    def stop(self):
        """Stop RTSP streaming"""
        self.running = False
        if self.writer:
            self.writer.release()
        if self.writer_thread:
            self.writer_thread.join(timeout=2)
