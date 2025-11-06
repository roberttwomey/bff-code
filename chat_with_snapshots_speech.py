#!/usr/bin/env python3
import os, json, base64, sys, glob, math
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
import threading
import queue
import time

import numpy as np
from PIL import Image

# Official OpenAI SDK (Responses API)
# pip install openai>=1.40
from openai import OpenAI
# import json
import dotenv

# Speech recognition and synthesis
import speech_recognition as sr
import pyttsx3

# =============== Config ===============
dotenv.load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# with open("config.json", "r") as f:
    # config = json.load(f)
SNAPSHOT_DIR = os.environ.get("GO2_SNAPSHOT_DIR", "snapshots")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # vision-capable, efficient

SYSTEM_PROMPT = """You are Snapper, an embodied AI in a robot dog. Use body state (proprioception through IMU, motors, battery)
and environment (through exteroception with camera and LIDAR point cloud) to converse with the user. Do not speculate about
what has happened outside your sensory inputs. Do not communicate like a helpful assistant, you are just having a conversation
as a reliable partner, companion, or human's best friend."""

# Depth-map rendering parameters
DEPTH_IMG_SIZE = (128, 128)     # (width, height) of the depth map image
DEPTH_CROP_PAD = 1.05           # add 5% padding around XY bounds so edges aren't clipped
NEAR_BIAS = 0.0                 # meters (clamp)
FAR_BIAS = 0.0                  # meters (extra headroom)

# Speech settings
SPEECH_RATE = 150  # words per minute
SPEECH_VOLUME = 0.9  # 0.0 to 1.0
VOICE_ID = None  # Set to specific voice ID if desired, None for default

# =============== Speech Recognition ===============
class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.has_printed_listening = False
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def start_listening(self):
        """Start listening for speech in a background thread"""
        self.is_listening = True
        self.has_printed_listening = False  # Reset so "listening" prints when we start
        threading.Thread(target=self._listen_loop, daemon=True).start()
    
    def stop_listening(self):
        """Stop listening for speech"""
        self.is_listening = False
    
    def _listen_loop(self):
        """Background thread that continuously listens for speech"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    if not self.has_printed_listening:
                        print("🎤 Listening... (speak now)")
                        self.has_printed_listening = True
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    self.audio_queue.put(audio)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Microphone error: {e}")
                time.sleep(0.1)
    
    def get_speech(self) -> Optional[str]:
        """Get the next recognized speech from the queue"""
        try:
            audio = self.audio_queue.get_nowait()
            text = self.recognizer.recognize_google(audio)
            print(f"🎤 Heard: {text}")
            self.has_printed_listening = False  # Reset so "listening" prints again for next utterance
            return text
        except queue.Empty:
            return None
        except sr.UnknownValueError:
            print("🎤 Could not understand audio")
            self.has_printed_listening = False  # Reset even on error so we can try again
            return None
        except sr.RequestError as e:
            print(f"🎤 Speech recognition error: {e}")
            self.has_printed_listening = False  # Reset even on error so we can try again
            return None

# =============== Text-to-Speech ===============
class SpeechSynthesizer:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', SPEECH_RATE)
        self.engine.setProperty('volume', SPEECH_VOLUME)
        
        # Set voice if specified
        if VOICE_ID:
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if VOICE_ID in voice.id:
                    self.engine.setProperty('voice', voice.id)
                    break
    
    def speak(self, text: str, speech_recognizer=None):
        """Speak the given text, optionally pausing speech recognition during speech"""
        print(f"🔊 Speaking: {text}")
        
        # Pause speech recognition while speaking
        if speech_recognizer:
            speech_recognizer.stop_listening()
            print("🎤 Paused listening while speaking...")
        
        self.engine.say(text)
        self.engine.runAndWait()
        
        # Resume speech recognition after speaking
        if speech_recognizer:
            speech_recognizer.start_listening()
            print("🎤 Resumed listening...")

# =============== Snapshot helpers ===============
def latest_snapshot_pair(folder: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]], Optional[Path]]:
    """
    Returns (json_path, json_data, image_path) for the most recent snapshot in folder.
    Requires JSON files produced by your snapshotter (with files.photo_jpg).
    """
    folder = Path(folder)
    candidates = sorted(folder.glob("*.json"))
    if not candidates:
        return None, None, None
    json_path = candidates[-1]
    with open(json_path, "r") as f:
        data = json.load(f)
    img_rel = data.get("files", {}).get("photo_jpg")
    img_path = Path(img_rel) if img_rel else None
    return json_path, data, img_path

def find_pointcloud_for_snapshot(snapshot_json: Dict[str, Any], folder: Path) -> Optional[Path]:
    """
    Strategy:
      1) If JSON manifest has files.pointcloud_ply and it exists, use it.
      2) Else, pick the most recent *.ply in the same folder.
    """
    candidate = snapshot_json.get("files", {}).get("pointcloud_ply")
    if candidate:
        p = Path(candidate)
        if not p.is_absolute():
            p = (folder / p).resolve()
        if p.exists():
            return p
    # fallback: newest ply in folder
    plys = sorted(folder.glob("*.ply"))
    return plys[-1] if plys else None

# =============== PLY loading (Open3D optional; ASCII fallback) ===============
def load_ply_points(ply_path: Path) -> np.ndarray:
    """
    Returns Nx3 float32 points. Prefers Open3D if available; otherwise reads ASCII PLY
    with x/y/z (and ignores other properties). Raises on failure.
    """
    try:
        import open3d as o3d  # optional
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts
    except Exception:
        # Minimal ASCII PLY reader (x y z [intensity] ...)
        with open(ply_path, "r") as f:
            header = []
            line = f.readline().strip()
            if line != "ply":
                raise ValueError("Not a PLY file")
            header.append(line)
            fmt = None
            num_vertices = None
            prop_names: List[str] = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected EOF in header")
                s = line.strip()
                header.append(s)
                if s.startswith("format "):
                    fmt = s.split()[1]
                    if fmt != "ascii":
                        raise ValueError("Only ASCII PLY supported without Open3D")
                if s.startswith("element vertex "):
                    num_vertices = int(s.split()[-1])
                if s.startswith("property "):
                    parts = s.split()
                    # e.g., ['property','float','x']
                    if len(parts) >= 3:
                        prop_names.append(parts[-1])
                if s == "end_header":
                    break
            if num_vertices is None:
                raise ValueError("No vertex count in PLY")
            # Determine x/y/z indices
            try:
                ix = prop_names.index("x")
                iy = prop_names.index("y")
                iz = prop_names.index("z")
            except ValueError:
                raise ValueError("PLY missing x/y/z properties")
            pts = np.empty((num_vertices, 3), dtype=np.float32)
            for i in range(num_vertices):
                vals = f.readline().strip().split()
                pts[i, 0] = float(vals[ix])
                pts[i, 1] = float(vals[iy])
                pts[i, 2] = float(vals[iz])
            return pts

# =============== Depth-map rendering (top-down) ===============
def render_topdown_depth(points: np.ndarray,
                         img_size=(128, 128),
                         crop_pad: float = 1.05,
                         near_bias: float = 0.0,
                         far_bias: float = 0.0) -> Image.Image:
    """
    Renders a top-down height map from a list of 3D points.
    x and y are treated as pixel locations, and z as the height.

    Args:
        points: A list of (x, y, z) tuples.

    Returns:
        A PIL Image object of the top-down render, or None if no points are given.
    """
    if points.size == 0:
        return None

    # Convert coordinates to integers for pixel locations
    int_points = [(int(p[0]), int(p[1]), int(p[2])) for p in points]

    height_map = np.full(img_size, 0, dtype=np.float32)

    # Populate the height map. For overlapping points, keep the highest z-value.
    for x, y, z in int_points:
        if z > height_map[x, y]:
            height_map[x, y] = z

    # Create a PIL image
    image = Image.fromarray(height_map, 'L')

    return image

# =============== Encoding helpers ===============
def encode_image_base64(p: Path) -> str:
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def encode_pil_png_base64(im: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# =============== Prompt building ===============
def compact_state(js: Dict[str, Any], max_motors: int = 12) -> str:
    """
    Summarize robot state concisely.
    """
    imu = js.get("imu_rpy") or js.get("low_state", {}).get("imu_state", {}).get("rpy")
    soc = js.get("soc") or js.get("low_state", {}).get("bms_state", {}).get("soc")
    pv  = js.get("power_v")
    motors = js.get("motor_state") or js.get("low_state", {}).get("motor_state") or []

    # compact motor readout
    lines = []
    for i, m in enumerate(motors[:max_motors]):
        q   = m.get("q", 0.0)
        tmp = m.get("temperature", None)
        lost = m.get("lost", False)
        lines.append(f"M{i+1}: q={q:.2f}" + (f", T={tmp}C" if tmp is not None else "") + (", LOST" if lost else ""))
    if len(motors) > max_motors:
        lines.append(f"... ({len(motors)-max_motors} more motors)")

    parts = []
    if imu is not None: parts.append(f"IMU rpy≈ {tuple(round(float(x),2) for x in imu)}")
    if soc is not None: parts.append(f"Battery SOC: {soc}%")
    if pv is not None: parts.append(f"Bus V: {pv}V")
    summary_head = " | ".join(parts) if parts else "(no quick vitals)"

    motors_txt = "\n".join(lines) if lines else "(no motor telemetry)"
    return f"{summary_head}\nMotors:\n{motors_txt}"

def build_multimodal_input(user_text: str,
                           image_path: Optional[Path],
                           lidar_png_b64: Optional[str],
                           state_json: Dict[str, Any]) -> list:
    """
    Build a Responses API multimodal 'input' list with:
      1) state text
      2) RGB camera image (if present)
      3) LiDAR depth map PNG (if present)
      4) user's message
    """
    blocks = []

    # State summary first
    state_block = "Robot snapshot state:\n" + compact_state(state_json)
    blocks.append({"role": "user", "content": [{"type": "input_text", "text": state_block}]})

    # Camera image
    if image_path and image_path.exists():
        b64 = encode_image_base64(image_path)
        mime = "image/jpeg" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
        blocks.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Camera frame:"},
                {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
            ],
        })

    # LiDAR depth map
    if lidar_png_b64:
        blocks.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": "LiDAR depth map (top-down, near=bright, far=dark):"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{lidar_png_b64}"},
            ],
        })

    # User message
    if user_text.strip():
        blocks.append({"role": "user", "content": [{"type": "input_text", "text": user_text.strip()}]})

    return blocks

def print_header(json_path: Optional[Path], img_path: Optional[Path], ply_path: Optional[Path]):
    print("="*78)
    print("Embodied Chat with Speech — grounding with latest robot snapshot")
    print(f"Snapshot JSON : {json_path if json_path else '— none —'}")
    print(f"Camera image  : {img_path if img_path else '— none —'}")
    print(f"Point cloud   : {ply_path if ply_path else '— none —'}")
    print("="*78)

# =============== Speech Chat Loop ===============
def main():
    client = OpenAI(api_key=OPENAI_API_KEY)  # reads OPENAI_API_KEY
    
    # Initialize speech components
    speech_recognizer = SpeechRecognizer()
    speech_synthesizer = SpeechSynthesizer()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Looking for snapshots in: {SNAPSHOT_DIR}")
    js_path, js, img_path = latest_snapshot_pair(SNAPSHOT_DIR)
    if js is None:
        print("No snapshots found. Create some in the 'snapshots/' folder first.")
        sys.exit(1)

    # Find associated point cloud
    ply_path = find_pointcloud_for_snapshot(js, Path(SNAPSHOT_DIR))

    # Pre-render initial LiDAR depth map (we can refresh per-turn as well)
    lidar_b64 = None
    if ply_path and ply_path.exists():
        try:
            pts = load_ply_points(ply_path)
            # Optional: rotate to match your earlier pipeline (X 90°, Z 180°)
            # rot X:
            Rx = np.array([[1,0,0],
                           [0, math.cos(math.pi/2), -math.sin(math.pi/2)],
                           [0, math.sin(math.pi/2),  math.cos(math.pi/2)]], dtype=np.float32)
            # rot Z:
            Rz = np.array([[ math.cos(math.pi), -math.sin(math.pi), 0],
                           [ math.sin(math.pi),  math.cos(math.pi), 0],
                           [ 0,                  0,                 1]], dtype=np.float32)
            pts = (pts @ Rx.T) @ Rz.T

            depth_img = render_topdown_depth(
                pts, img_size=DEPTH_IMG_SIZE,
                crop_pad=DEPTH_CROP_PAD, near_bias=NEAR_BIAS, far_bias=FAR_BIAS
            )
            lidar_b64 = encode_pil_png_base64(depth_img)
            # save for inspection
            depth_img.save("latest_lidar_depth.png")
        except Exception as e:
            print(f"[warn] Could not render LiDAR depth map: {e}")

    print_header(js_path, img_path, ply_path)
    print("🎤 Speech chat mode activated!")
    print("💡 Commands:")
    print("   - Speak naturally to chat")
    print("   - Type 'quit' or 'exit' to end")
    print("   - Type 'text' to switch to text input mode")
    print("   - Type 'speech' to switch back to speech mode")
    print()

    # Start listening for speech
    speech_recognizer.start_listening()
    speech_mode = True

    while True:
        try:
            if speech_mode:
                # Speech input mode
                user_text = None
                while user_text is None:
                    user_text = speech_recognizer.get_speech()
                    time.sleep(0.1)
                    
                    # Check for quit command
                    if user_text and user_text.lower() in ['quit', 'exit', 'stop']:
                        print("👋 Goodbye!")
                        speech_recognizer.stop_listening()
                        sys.exit(0)
                    
                    # Check for mode switch
                    if user_text and user_text.lower() == 'text':
                        speech_mode = False
                        speech_recognizer.stop_listening()
                        print("📝 Switched to text input mode. Type 'speech' to switch back.")
                        break
            else:
                # Text input mode
                user_text = input("You: ").strip()
                if not user_text:
                    continue
                
                # Check for quit command
                if user_text.lower() in ['quit', 'exit', 'stop']:
                    print("👋 Goodbye!")
                    sys.exit(0)
                
                # Check for mode switch
                if user_text.lower() == 'speech':
                    speech_mode = True
                    speech_recognizer.start_listening()
                    print("🎤 Switched to speech input mode.")
                    continue

            # (Optional) Refresh to newest snapshot and newest .ply each turn:
            js_path, js, img_path = latest_snapshot_pair(SNAPSHOT_DIR)
            ply_path = find_pointcloud_for_snapshot(js, Path(SNAPSHOT_DIR))
            print_header(js_path, img_path, ply_path)

            # (Re)render LiDAR depth map if new PLY exists
            lidar_b64_turn = None
            if ply_path and ply_path.exists():
                try:
                    pts = load_ply_points(ply_path)
                    Rx = np.array([[1,0,0],
                                   [0, math.cos(math.pi/2), -math.sin(math.pi/2)],
                                   [0, math.sin(math.pi/2),  math.cos(math.pi/2)]], dtype=np.float32)
                    Rz = np.array([[ math.cos(math.pi), -math.sin(math.pi), 0],
                                   [ math.sin(math.pi),  math.cos(math.pi), 0],
                                   [ 0,                  0,                 1]], dtype=np.float32)
                    pts = (pts @ Rx.T) @ Rz.T

                    depth_img = render_topdown_depth(
                        pts, img_size=DEPTH_IMG_SIZE,
                        crop_pad=DEPTH_CROP_PAD, near_bias=NEAR_BIAS, far_bias=FAR_BIAS
                    )
                    lidar_b64_turn = encode_pil_png_base64(depth_img)
                except Exception as e:
                    print(f"[warn] LiDAR render failed this turn: {e}")

            # Build multimodal request
            input_blocks = build_multimodal_input(
                user_text=user_text,
                image_path=img_path,
                lidar_png_b64=lidar_b64_turn or lidar_b64,
                state_json=js,
            )

            resp = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=input_blocks,
                max_output_tokens=700,
            )

            assistant_response = resp.output_text.strip()
            print(f"\nAssistant: {assistant_response}\n")
            
            # Speak the response if in speech mode
            if speech_mode:
                speech_synthesizer.speak(assistant_response, speech_recognizer)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            speech_recognizer.stop_listening()
            sys.exit(0)
        except Exception as e:
            print(f"[error] {e}\n")

if __name__ == "__main__":
    main()
