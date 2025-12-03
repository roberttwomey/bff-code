#!/usr/bin/env python3
"""
Lightweight web server that displays:
- Current dreams (most recent synthesized image/video) if in dreaming mode
- Camera view otherwise

Independent of chat-manager.py and dream-manager.py
"""
import os
import sys
import time
import threading
import requests
from pathlib import Path
from flask import Flask, render_template_string, Response, send_file
import cv2
import numpy as np

# Robot camera imports (optional - will fail gracefully if not available)
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient
    ROBOT_CAMERA_AVAILABLE = True
except ImportError:
    ROBOT_CAMERA_AVAILABLE = False
    print("Warning: Robot camera libraries not available. Camera view will not work.")

# Configuration
ETHERNET_INTERFACE = "enP8p1s0"
SD_SERVER = "http://127.0.0.1:7860"
OUTPUTS_DIR = "outputs"
DREAM_CHECK_INTERVAL = 60.0  # seconds between dreamstate checks
PORT = 8080

app = Flask(__name__)

# Global state
latest_camera_frame = None
camera_frame_lock = threading.Lock()
video_client = None
is_dreaming = False
recent_dreams = []  # List of (path, type) tuples, sorted by modification time (newest first)
dream_lock = threading.Lock()
channel_factory_initialized = False
CYCLE_INTERVAL = 5.0  # seconds between cycling to next image/video
MAX_DREAMS = 60  # maximum number of recent dreams to track

# Subtitle mappings for animation files
SUBTITLES = {
    0: "(BFF) is a new media artwork exploring intimacy, embodiment, intelligence, and alignment in human-AI relationships",
    1: "through co-parenting two robot dogs.",
    2: "Documented as an experimental film",
    3: "he project follows two-artist researchers",
    4: "each paired with an identical robot dog and local LLM",
    5: "as they cultivate emotional bonds, train model behavior",
    6: "and dialogue on questions of mind, embodiment, and relationality in the age of generative AI",
    7: "Structured as a metalogue, it blends dialogue with rich multi-modal imagery",
    8: "drawn from a range of human and machine perspectives.",
    9: "Working with LIDAR scans, 360° video, gaussian splats",
    10: "and snapshots of internal internal model states",
    11: "The film constructs a hybrid cinematic language",
    12: "that toggles between perception and affect, embodiment, computation, and language.",
    13: "As the collaborators exchange and evolve the AI's mind",
    14: "BFF documents this distributed act of care and co-creation",
    15: "The film interrogates the boundaries between simulation and authenticity",
    16: "emotional labor and machine learning, human complexity and synthetic intelligence",
    17: "offering a poetic meditation on what we aspire to, what we search for in relation to our machine kin."
}

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BFF Dream Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background: #000;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .media-container {
            width: 100vw;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        .media-item {
            width: 100vw;
            height: 100vh;
            object-fit: contain;
            display: none;
        }
        .media-item.active {
            display: block;
        }
        img, video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .subtitle-overlay {
            position: fixed;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%);
            width: 70%;
            max-width: 1200px;
            text-align: left;
            color: #FFD700;
            font-size: 48px;
            font-family: Arial, sans-serif;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
            z-index: 1000;
            padding: 20px;
            line-height: 1.4;
        }
    </style>
</head>
<body>
    {% if is_dreaming %}
        <div class="media-container" id="mediaContainer">
            <!-- Media items will be populated by JavaScript -->
        </div>
        <div class="subtitle-overlay" id="subtitleOverlay"></div>
    {% else %}
        <div class="media-container">
            <img src="{{ url_for('camera_feed') }}" alt="Camera Stream" style="display: block;">
        </div>
    {% endif %}
    
    <script>
        {% if is_dreaming %}
        let currentIndex = 0;
        let dreams = [];
        const cycleInterval = {{ cycle_interval }} * 1000; // Convert to milliseconds
        
        // Subtitle mappings for animation files
        const subtitles = {
            0: "(BFF) is a new media artwork exploring intimacy, embodiment, intelligence, and alignment in human-AI relationships",
            1: "through co-parenting two robot dogs.",
            2: "Documented as an experimental film",
            3: "the project follows two-artist researchers",
            4: "each paired with an identical robot dog and local LLM",
            5: "as they cultivate emotional bonds, train model behavior",
            6: "and dialogue on questions of mind, embodiment, and relationality in the age of generative AI",
            7: "Structured as a metalogue, it blends dialogue with rich multi-modal imagery",
            8: "drawn from a range of human and machine perspectives.",
            9: "Working with LIDAR scans, 360° video, gaussian splats",
            10: "and snapshots of internal internal model states",
            11: "The film constructs a hybrid cinematic language",
            12: "that toggles between perception and affect, embodiment, computation, and language.",
            13: "As the collaborators exchange and evolve the AI's mind",
            14: "BFF documents this distributed act of care and co-creation",
            15: "The film interrogates the boundaries between simulation and authenticity",
            16: "emotional labor and machine learning, human complexity and synthetic intelligence",
            17: "offering a poetic meditation on what we aspire to, what we search for in relation to our machine kin."
        };
        
        // Extract animation number from filename
        // Supports: animation_1764765844_14.gif or animation_9_seed_2447328037_1764780844.gif
        function getAnimationNumber(filename) {
            // Try pattern: animation_{timestamp}_{number}.ext
            let match = filename.match(/animation_(\d+)_(\d+)/);
            if (match) {
                const num1 = parseInt(match[1], 10);
                const num2 = parseInt(match[2], 10);
                // Timestamp is the larger number (>1 billion), animation number is smaller (0-17)
                if (num1 > 1000000000) {
                    return num2;  // num1 is timestamp, num2 is animation number
                } else {
                    return num1;  // num1 is animation number, num2 is timestamp
                }
            }
            
            // Try pattern: animation_{number}_seed_{seed}_{timestamp}.ext
            match = filename.match(/animation_(\d+)_seed_\d+_(\d+)/);
            if (match) {
                return parseInt(match[1], 10);  // First number is animation number
            }
            
            return null;
        }
        
        // Update subtitle based on current dream filename
        function updateSubtitle(filename) {
            const subtitleDiv = document.getElementById('subtitleOverlay');
            if (!subtitleDiv) return;
            
            const animNum = getAnimationNumber(filename);
            if (animNum !== null && subtitles[animNum]) {
                subtitleDiv.textContent = subtitles[animNum];
                subtitleDiv.style.display = 'block';
            } else {
                subtitleDiv.style.display = 'none';
            }
        }
        
        // Fetch list of dreams
        async function loadDreams() {
            try {
                const response = await fetch('/api/dreams');
                const data = await response.json();
                const newDreams = data.dreams || [];
                
                // Only re-render if the number of dreams changed
                if (newDreams.length !== dreams.length) {
                    dreams = newDreams;
                    // Reset index if current index is out of bounds
                    if (currentIndex >= dreams.length) {
                        currentIndex = 0;
                    }
                    renderDreams();
                } else {
                    dreams = newDreams;
                }
            } catch (error) {
                console.error('Error loading dreams:', error);
            }
        }
        
        // Render all dreams in the container
        function renderDreams() {
            const container = document.getElementById('mediaContainer');
            container.innerHTML = '';
            
            if (dreams.length === 0) {
                container.innerHTML = '<div style="color: white; font-size: 24px; text-align: center;">No dreams found yet. Waiting for synthesis...</div>';
                return;
            }
            
            dreams.forEach((dream, index) => {
                const mediaItem = document.createElement(dream.type === 'image' ? 'img' : 'video');
                mediaItem.className = 'media-item';
                mediaItem.src = `/dream/${index}`;
                // Store dream info as data attributes
                mediaItem.dataset.dreamType = dream.type;
                mediaItem.dataset.dreamFilename = dream.filename;
                mediaItem.dataset.dreamIndex = index;
                
                if (dream.type === 'video') {
                    mediaItem.autoplay = true;
                    mediaItem.loop = false; // Don't loop - let it play once
                    mediaItem.muted = true;
                }
                container.appendChild(mediaItem);
            });
            
            // Show current item (preserve position) - showItem will handle scheduling
            if (dreams.length > 0) {
                showItem(currentIndex);
            }
        }
        
        // Show specific item and set up appropriate timing
        function showItem(index) {
            const items = document.querySelectorAll('.media-item');
            let currentFilename = '';
            
            items.forEach((item, i) => {
                if (i === index) {
                    item.classList.add('active');
                    currentFilename = item.dataset.dreamFilename || '';
                    // If it's a video, restart it
                    if (item.tagName === 'VIDEO') {
                        item.currentTime = 0;
                        item.play();
                    }
                } else {
                    item.classList.remove('active');
                    // Pause videos when hidden
                    if (item.tagName === 'VIDEO') {
                        item.pause();
                    }
                }
            });
            
            // Update subtitle based on filename
            updateSubtitle(currentFilename);
            
            // Schedule next item based on media type
            scheduleNextForItem(index);
        }
        
        // Cycle to next item
        function nextItem() {
            if (dreams.length === 0) return;
            currentIndex = (currentIndex + 1) % dreams.length;
            showItem(currentIndex);
        }
        
        let cyclingTimeout = null;
        let videoEndListener = null;
        
        // Schedule next item based on current item type
        function scheduleNextForItem(index) {
            // Clear any existing timeout/listener
            if (cyclingTimeout) {
                clearTimeout(cyclingTimeout);
                cyclingTimeout = null;
            }
            if (videoEndListener) {
                const currentItem = document.querySelector('.media-item.active');
                if (currentItem && currentItem.tagName === 'VIDEO') {
                    currentItem.removeEventListener('ended', videoEndListener);
                }
                videoEndListener = null;
            }
            
            if (dreams.length <= 1) return;
            
            const currentItem = document.querySelector('.media-item.active');
            if (!currentItem) return;
            
            const dreamType = currentItem.dataset.dreamType;
            const filename = currentItem.dataset.dreamFilename || '';
            const isGif = filename.toLowerCase().endsWith('.gif');
            
            // For videos and GIFs, wait for them to finish
            if (dreamType === 'video') {
                // Wait for video to end
                videoEndListener = () => {
                    nextItem();
                };
                currentItem.addEventListener('ended', videoEndListener, { once: true });
            } else if (isGif) {
                // For GIFs, wait for a reasonable duration (GIFs don't have duration property)
                // Most GIFs are 2-5 seconds, so we'll wait a bit longer to let them play
                const gifDuration = 5000; // 5 seconds default for GIFs
                cyclingTimeout = setTimeout(() => {
                    nextItem();
                }, gifDuration);
            } else {
                // For regular images, use random interval
                const minInterval = 0.7 * cycleInterval;
                const maxInterval = 1.5 * cycleInterval;
                const randomInterval = minInterval + Math.random() * (maxInterval - minInterval);
                
                cyclingTimeout = setTimeout(() => {
                    nextItem();
                }, randomInterval);
            }
        }
        
        // Start cycling
        function startCycling() {
            // Clear any existing timeout/listener
            if (cyclingTimeout) {
                clearTimeout(cyclingTimeout);
                cyclingTimeout = null;
            }
            if (videoEndListener) {
                const currentItem = document.querySelector('.media-item.active');
                if (currentItem && currentItem.tagName === 'VIDEO') {
                    currentItem.removeEventListener('ended', videoEndListener);
                }
                videoEndListener = null;
            }
            
            // Only start if we have more than one dream
            if (dreams.length > 1) {
                // Schedule next for current item
                const items = document.querySelectorAll('.media-item');
                if (items.length > 0) {
                    scheduleNextForItem(currentIndex);
                }
            }
        }
        
        // Initial load and periodic refresh
        loadDreams().then(() => {
            startCycling();
        });
        setInterval(() => {
            loadDreams().then(() => {
                startCycling(); // Restart cycling in case count changed
            });
        }, 5000); // Refresh dream list every 5 seconds
        {% endif %}
    </script>
</body>
</html>
"""


def check_dreamstate():
    """Check if robot is in dreaming mode by checking if SD API is accessible."""
    try:
        response = requests.get(f"{SD_SERVER}/sdapi/v1/options", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def parse_animation_info(filename):
    """
    Parse animation number and timestamp from filename.
    
    Supports patterns:
    - animation_1764765844_14.gif -> (14, 1764765844)
    - animation_9_seed_2447328037_1764780844.gif -> (9, 1764780844)
    
    Returns tuple: (animation_number, timestamp) or (None, None) if not found
    """
    import re
    
    # Try pattern: animation_{timestamp}_{number}.ext
    match = re.search(r'animation_(\d+)_(\d+)', filename)
    if match:
        timestamp = int(match.group(1))
        anim_num = int(match.group(2))
        # Check which is more likely the timestamp (larger number, ~10 digits)
        if timestamp > 1000000000:  # Unix timestamp range
            return (anim_num, timestamp)
        else:
            return (timestamp, anim_num)
    
    # Try pattern: animation_{number}_seed_{seed}_{timestamp}.ext
    match = re.search(r'animation_(\d+)_seed_\d+_(\d+)', filename)
    if match:
        anim_num = int(match.group(1))
        timestamp = int(match.group(2))
        return (anim_num, timestamp)
    
    return (None, None)


def find_recent_dreams(outputs_dir=OUTPUTS_DIR, max_count=MAX_DREAMS):
    """Find the most recent synthesized images and videos in outputs directory.
    
    Groups animations into batches (sets generated close together in time), then sorts:
    1. Batch - descending (newest batch first, identified by max timestamp in batch)
    2. Animation number (0-17) - ascending (within each batch)
    """
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        return []
    
    # Supported image and video extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    video_extensions = {'.gif', '.mp4', '.webm', '.mov'}
    
    all_files = []
    
    # Search for all image and video files
    for ext in image_extensions | video_extensions:
        for file_path in outputs_path.glob(f"*{ext}"):
            if file_path.is_file():
                filename = file_path.name
                anim_num, timestamp = parse_animation_info(filename)
                
                if ext in image_extensions:
                    file_type = 'image'
                else:
                    file_type = 'video'
                
                # Store: (timestamp, animation_number, path, file_type)
                # Use 0 for None timestamp to sort them last
                sort_timestamp = timestamp if timestamp is not None else 0
                sort_anim_num = anim_num if anim_num is not None else 9999
                all_files.append((sort_timestamp, sort_anim_num, str(file_path), file_type))
    
    # Group animations into batches based on animation number patterns
    # Each batch should contain at most ONE of each animation number (0-17)
    # Sort by timestamp descending to process newest first
    all_files.sort(key=lambda x: -x[0])
    
    batches = []
    current_batch = []
    seen_anim_nums = set()
    batch_max_timestamp = None
    
    for timestamp, anim_num, path, file_type in all_files:
        if timestamp == 0:  # Skip files without timestamps
            continue
        
        # If we've already seen this animation number in the current batch,
        # start a new batch (this indicates a new generation run)
        if anim_num in seen_anim_nums:
            # Save current batch
            if current_batch:
                batches.append((batch_max_timestamp, current_batch))
            # Start new batch
            current_batch = [(timestamp, anim_num, path, file_type)]
            seen_anim_nums = {anim_num}
            batch_max_timestamp = timestamp
        else:
            # Add to current batch
            if batch_max_timestamp is None:
                batch_max_timestamp = timestamp
            current_batch.append((timestamp, anim_num, path, file_type))
            seen_anim_nums.add(anim_num)
    
    # Add last batch
    if current_batch:
        batches.append((batch_max_timestamp, current_batch))
    
    # Sort each batch by animation number
    sorted_files = []
    for batch_max_ts, batch_files in batches:
        # Sort by animation number within batch
        batch_files.sort(key=lambda x: x[1])
        sorted_files.extend(batch_files)
    
    # Debug: Print sort order
    print("\n" + "="*80)
    print("DEBUG: Dream files grouped by batch, sorted by animation number:")
    print("="*80)
    batch_num = 0
    items_printed = 0
    for batch_max_ts, batch_files in batches:
        if items_printed >= max_count:
            break
        batch_num += 1
        print(f"\n--- Batch {batch_num} (max timestamp: {batch_max_ts}, {len(batch_files)} animations) ---")
        for timestamp, anim_num, path, file_type in batch_files:
            if items_printed >= max_count:
                break
            filename = os.path.basename(path)
            print(f"  Timestamp: {timestamp:>12} | Anim #{anim_num:>2} | Type: {file_type:>5} | {filename}")
            items_printed += 1
    print("\n" + "="*80 + "\n")
    
    # Return top max_count, keeping only path and file_type
    return [(path, file_type) for _, _, path, file_type in sorted_files[:max_count]]


def dreamstate_monitor():
    """Monitor dreamstate and update recent dreams list."""
    global is_dreaming, recent_dreams
    
    while True:
        new_dreamstate = check_dreamstate()
        
        with dream_lock:
            is_dreaming = new_dreamstate
            
            # If in dreaming mode, find recent dreams
            if is_dreaming:
                dreams = find_recent_dreams()
                if dreams:
                    recent_dreams = dreams
                    print(f"Dreaming mode: Found {len(dreams)} recent dream(s)")
            else:
                recent_dreams = []
        
        time.sleep(DREAM_CHECK_INTERVAL)


def camera_loop():
    """Capture camera frames in a loop."""
    global latest_camera_frame, video_client
    
    if not ROBOT_CAMERA_AVAILABLE:
        print("Camera not available - skipping camera loop")
        return
    
    try:
        video_client = VideoClient()
        video_client.SetTimeout(3.0)
        video_client.Init()
        
        print("Camera initialized successfully")
        
        code, data = video_client.GetImageSample()
        
        while code == 0:
            code, data = video_client.GetImageSample()
            
            if code != 0:
                break
            
            # Convert to numpy image
            try:
                image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                
                if image is not None:
                    with camera_frame_lock:
                        latest_camera_frame = image
            except Exception as e:
                print(f"Error processing camera frame: {e}")
                continue
            
            time.sleep(0.033)  # ~30 FPS
        
        if code != 0:
            print(f"Get image sample error. code: {code}")
    
    except Exception as e:
        print(f"Camera loop error: {e}")
        print("Camera view will not be available")


@app.route('/')
def index():
    """Render the main webpage."""
    with dream_lock:
        is_dreaming_state = is_dreaming
    
    return render_template_string(
        HTML_TEMPLATE,
        is_dreaming=is_dreaming_state,
        cycle_interval=CYCLE_INTERVAL
    )


@app.route('/api/dreams')
def api_dreams():
    """API endpoint to get list of recent dreams."""
    with dream_lock:
        dreams = recent_dreams.copy()
    
    # Return list of dreams with their indices
    dream_list = []
    for i, (dream_path, dream_type) in enumerate(dreams):
        dream_list.append({
            'index': i,
            'type': dream_type,
            'filename': os.path.basename(dream_path)
        })
    
    return {'dreams': dream_list}


@app.route('/camera_feed')
def camera_feed():
    """Video streaming route for camera."""
    def generate_frames():
        while True:
            with camera_frame_lock:
                if not channel_factory_initialized:
                    # Return a frame with "DOG NOT CONNECTED" message
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "DOG NOT CONNECTED", (120, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    frame = blank
                elif latest_camera_frame is None:
                    # Return a blank frame if no camera available
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Camera not available", (150, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    frame = blank
                else:
                    frame = latest_camera_frame.copy()
            
            # Resize for web streaming
            frame_resized = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/dream/<int:index>')
def serve_dream(index):
    """Serve a dream by index (image or video)."""
    with dream_lock:
        dreams = recent_dreams.copy()
    
    if index < 0 or index >= len(dreams):
        return "Dream index out of range", 404
    
    dream_path, dream_type = dreams[index]
    
    if not os.path.exists(dream_path):
        return "Dream file not found", 404
    
    # Determine MIME type
    if dream_type == 'image':
        mime_type = 'image/png'
        if dream_path.lower().endswith('.jpg') or dream_path.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        elif dream_path.lower().endswith('.gif'):
            mime_type = 'image/gif'
        elif dream_path.lower().endswith('.webp'):
            mime_type = 'image/webp'
    else:  # video
        mime_type = 'video/mp4'
        if dream_path.lower().endswith('.gif'):
            mime_type = 'image/gif'
        elif dream_path.lower().endswith('.webm'):
            mime_type = 'video/webm'
        elif dream_path.lower().endswith('.mov'):
            mime_type = 'video/quicktime'
    
    return send_file(dream_path, mimetype=mime_type)


def main():
    """Main entry point."""
    global video_client, channel_factory_initialized
    
    # Initialize robot camera if available
    if ROBOT_CAMERA_AVAILABLE:
        try:
            if len(sys.argv) > 1:
                ChannelFactoryInitialize(0, sys.argv[1])
            else:
                ChannelFactoryInitialize(0, ETHERNET_INTERFACE)
            channel_factory_initialized = True
            print("Channel factory initialized successfully")
        except Exception as e:
            channel_factory_initialized = False
            print(f"Channel factory initialization failed: {e}")
            print("Camera view will show 'DOG NOT CONNECTED' placeholder")
        
        # Start camera loop in a separate thread
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
        print("Camera thread started")
    else:
        print("Warning: Robot camera libraries not available")
    
    # Start dreamstate monitor in a separate thread
    monitor_thread = threading.Thread(target=dreamstate_monitor, daemon=True)
    monitor_thread.start()
    print("Dreamstate monitor started")
    
    # Create outputs directory if it doesn't exist
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"BFF Dream Viewer Server")
    print(f"{'='*60}")
    print(f"Server running at: http://0.0.0.0:{PORT}")
    print(f"Outputs directory: {OUTPUTS_DIR}")
    print(f"Dream check interval: {DREAM_CHECK_INTERVAL}s")
    print(f"{'='*60}\n")
    print("Press Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        if video_client:
            # Cleanup if needed
            pass


if __name__ == "__main__":
    main()

