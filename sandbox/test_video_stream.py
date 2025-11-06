import cv2
import cv2.aruco as aruco
import numpy as np
import asyncio
import threading
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

class VideoStreamTest:
    def __init__(self, ip="192.168.4.30"):
        self.ip = ip
        self.conn = None
        self.frame_queue = Queue()
        self.is_running = False
        
    async def connect(self):
        """Connect to Go2 robot"""
        try:
            self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
            await self.conn.connect()
            print("Connected to Go2 robot")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    async def recv_camera_stream(self, track: MediaStreamTrack):
        """Receive video frames and process them"""
        print(f"Video track callback started: {track.kind}")
        frame_count = 0
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                self.frame_queue.put(img)
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Received {frame_count} frames, queue size: {self.frame_queue.qsize()}")
            except Exception as e:
                print(f"Error receiving frame: {e}")
                break
    
    def run_asyncio_loop(self, loop):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(loop)
        
        async def setup():
            try:
                print("Setting up video stream...")
                self.conn.video.switchVideoChannel(True)
                print("Video channel switched on")
                
                self.conn.video.add_track_callback(self.recv_camera_stream)
                print("Video track callback added")
                
                print("Setup completed successfully")
            except Exception as e:
                print(f"Error in WebRTC connection: {e}")
                import traceback
                traceback.print_exc()
        
        loop.run_until_complete(setup())
        loop.run_forever()
    
    async def test_video_stream(self):
        """Test video stream"""
        print("Starting video stream test...")
        self.is_running = True
        
        # Create a new event loop for the asyncio code
        loop = asyncio.new_event_loop()
        
        # Start the asyncio event loop in a separate thread
        asyncio_thread = threading.Thread(target=self.run_asyncio_loop, args=(loop,))
        asyncio_thread.start()
        
        # Wait a bit for video stream to start
        print("Waiting for video stream to initialize...")
        await asyncio.sleep(3)
        
        # Check if we're receiving frames
        if self.frame_queue.empty():
            print("Warning: No frames received yet. Video stream may not be working.")
        else:
            print(f"Video stream working! Queue has {self.frame_queue.qsize()} frames")
        
        try:
            while self.is_running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    print(f"Processing frame, queue size: {self.frame_queue.qsize()}")
                    
                    # Display the frame
                    cv2.imshow('Video Stream Test', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("No frames available, waiting...")
                    await asyncio.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        finally:
            self.is_running = False
            cv2.destroyAllWindows()

async def main():
    """Test video stream"""
    test = VideoStreamTest()
    
    if await test.connect():
        try:
            await test.test_video_stream()
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
