#!/usr/bin/env python3
"""
Launch and manage Automatic1111 Stable Diffusion WebUI in Docker headless mode.
Provides methods to start the server, generate images/videos, and shut down.
Also controls robot state: when running, robot lies down with cyan light and lidar off.
During synthesis, cyan light blinks.

Usage: 
python dream-manager.py start
python dream-manager.py stop
python dream-manager.py image --prompt "A photo of an astronaut riding a horse"
python dream-manager.py video --prompt "A video of an astronaut riding a horse"

# Progressive video generation (still image, then 16, 32 frames)
python dream-manager.py video --prompt "A video of an astronaut" --progressive

# Custom frame progression (still image + custom stages, min 8 frames for video)
python dream-manager.py video --prompt "A video of an astronaut" --progressive --stages 16,32,48,64

# Batch generation
python dream-manager.py batch-video --prompt-file prompts.txt
python dream-manager.py batch-image --prompt-file prompts.txt

# Batch progressive generation
python dream-manager.py batch-video --prompt-file prompts.txt --progressive

# Regenerate from metadata (automatically detects image or video)
python dream-manager.py regenerate --metadata-file outputs/generated_1234567890_0_metadata.json
python dream-manager.py regenerate --metadata-file outputs/animation_seed_123_timestamp_32frames_metadata.json
"""
import subprocess
import time
import requests
import base64
import os
import signal
import sys
import json
import threading
import random
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Robot control imports
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.rpc.client import Client

ROBOT_CONTROL_AVAILABLE = True

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Configuration
DOCKER_IMAGE = "daivdl487/stable-diffusion-webui:r36.4.3"
CONTAINER_NAME = "stable-diffusion-webui"
SERVER = "http://127.0.0.1:7860"
MODEL_CHECKPOINT = "Realistic_Vision_V5.1_fp16-no-ema.safetensors"
ETHERNET_INTERFACE = "enP8p1s0"  # Robot ethernet interface

# Image/Video generation parameters from environment
DEFAULT_WIDTH = int(os.getenv('WIDTH', '256'))
DEFAULT_HEIGHT = int(os.getenv('HEIGHT', '256'))
DEFAULT_STEPS = int(os.getenv('STEPS', '20'))
DEFAULT_FPS = int(os.getenv('FPS', '8'))
DEFAULT_FRAMES = int(os.getenv('FRAMES', '32'))

# Global state
_container_id: Optional[str] = None
_webui_process: Optional[subprocess.Popen] = None
_robot_controller: Optional['RobotController'] = None


class RobotController:
    """Controls robot state: VUI LED, lidar, and standing."""
    
    def __init__(self, ethernet_interface: str = ETHERNET_INTERFACE):
        # Initialize channel factory
        ChannelFactoryInitialize(0, ethernet_interface)
        
        # VUI client for LED control
        self.vui_client = Client('vui')
        self.vui_client.SetTimeout(3.0)
        self.vui_client._RegistApi(1007, 0)
        
        # Sport client for robot commands
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # Lidar publisher
        self.lidar_publisher = ChannelPublisher("rt/utlidar/switch", String_)
        self.lidar_publisher.Init()
        self.lidar_cmd = std_msgs_msg_dds__String_()
        
        # Blinking state
        self._blinking = False
        self._blink_thread: Optional[threading.Thread] = None
        
    def set_vui_color(self, color: str, duration: int = 0) -> bool:
        """
        Set the VUI LED color.
        
        Args:
            color: Color name (e.g., "cyan", "green", "purple", "red", "blue")
            duration: Duration in seconds (0 for persistent)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            p = {"color": color, "time": duration}
            parameter = json.dumps(p)
            code, result = self.vui_client._Call(1007, parameter)
            if code != 0:
                print(f"Set color error. code: {code}, {result}")
                return False
            return True
        except Exception as e:
            print(f"Error setting VUI color: {e}")
            return False
    
    def set_lidar_state(self, status: str) -> bool:
        """
        Set lidar on or off.
        
        Args:
            status: "ON" or "OFF"
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if status not in ["ON", "OFF"]:
                print(f"Invalid lidar status: {status}")
                return False
            
            self.lidar_cmd.data = status
            self.lidar_publisher.Write(self.lidar_cmd)
            return True
        except Exception as e:
            print(f"Error setting lidar state: {e}")
            return False
    
    def stand_down(self) -> bool:
        """Make robot stand down (lie down)."""
        try:
            self.sport_client.StandDown()
            return True
        except Exception as e:
            print(f"Error in StandDown: {e}")
            return False
    
    def _blink_worker(self, blink_rate: float = 0.5):
        """Worker thread for blinking effect."""
        while self._blinking:
            # On
            self.set_vui_color("cyan", duration=1)
            time.sleep(blink_rate)
            # Off
            self.set_vui_color("off", duration=1)
            time.sleep(blink_rate)
    
    def start_blinking(self, blink_rate: float = 0.5):
        """Start blinking cyan light."""
        if self._blinking:
            return  # Already blinking
        
        self._blinking = True
        self._blink_thread = threading.Thread(target=self._blink_worker, args=(blink_rate,), daemon=True)
        self._blink_thread.start()
    
    def stop_blinking(self):
        """Stop blinking and return to solid cyan."""
        if not self._blinking:
            return
        
        self._blinking = False
        if self._blink_thread:
            self._blink_thread.join(timeout=1.0)
        
        # Return to solid cyan
        self.set_vui_color("cyan", duration=0)
    
    def enter_dreaming_state(self):
        """Put robot in dreaming state: cyan light, lidar off, stand down."""
        print("Entering robot dreaming state...")
        self.set_vui_color("cyan", duration=0)  # Solid cyan
        time.sleep(0.5)
        self.set_lidar_state("OFF")
        time.sleep(1.0)
        self.stand_down()
        time.sleep(2.0)
        print("Robot in dreaming state (cyan light, lidar off, lying down)")


def start_docker_container() -> str:
    """
    Start the Docker container for Stable Diffusion WebUI.
    Returns the container ID.
    """
    global _container_id
    
    # Check if container already exists and is running
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={CONTAINER_NAME}"],
        capture_output=True,
        text=True
    )
    if result.stdout.strip():
        _container_id = result.stdout.strip()
        print(f"Container {CONTAINER_NAME} already running with ID: {_container_id}")
        return _container_id
    
    # Check if container exists but is stopped
    result = subprocess.run(
        ["docker", "ps", "-aq", "-f", f"name={CONTAINER_NAME}"],
        capture_output=True,
        text=True
    )
    if result.stdout.strip():
        container_id = result.stdout.strip()
        print(f"Starting existing container: {container_id}")
        subprocess.run(["docker", "start", container_id], check=True)
        _container_id = container_id
        return _container_id
    
    # Create and start new container
    print(f"Starting new Docker container: {DOCKER_IMAGE}")
    
    # Use jetson-containers run to start the container in detached mode
    # Note: jetson-containers forwards args to docker run, so we use docker run syntax
    cmd = [
        "jetson-containers", "run",
        "--name", CONTAINER_NAME,
        "-d",  # detached mode
        DOCKER_IMAGE,
        "sleep", "infinity"  # Keep container running
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    # Handle case where container with same name already exists
    if result.returncode != 0:
        if "already in use" in result.stderr or "Conflict" in result.stderr:
            # Container name already exists, try to use it
            print(f"Container name {CONTAINER_NAME} already exists, using existing container...")
            result = subprocess.run(
                ["docker", "ps", "-aq", "-f", f"name={CONTAINER_NAME}"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                container_id = result.stdout.strip()
                print(f"Starting existing container: {container_id}")
                subprocess.run(["docker", "start", container_id], check=True)
                time.sleep(2)
                result = subprocess.run(
                    ["docker", "ps", "-q", "-f", f"name={CONTAINER_NAME}"],
                    capture_output=True,
                    text=True
                )
                if result.stdout.strip():
                    _container_id = result.stdout.strip()
                    print(f"Container started with ID: {_container_id}")
                    return _container_id
        else:
            raise RuntimeError(f"Failed to start container: {result.stderr}")
    
    # Wait a moment for container to be created
    time.sleep(2)
    
    # Extract container ID by checking docker ps
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={CONTAINER_NAME}"],
        capture_output=True,
        text=True
    )
    
    if not result.stdout.strip():
        # Try all containers (including stopped)
        result = subprocess.run(
            ["docker", "ps", "-aq", "-f", f"name={CONTAINER_NAME}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            container_id = result.stdout.strip()
            print(f"Container created but not running. Starting it...")
            subprocess.run(["docker", "start", container_id], check=True)
            time.sleep(1)
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={CONTAINER_NAME}"],
                capture_output=True,
                text=True
            )
    
    if not result.stdout.strip():
        raise RuntimeError(f"Failed to start container. Check docker logs for details.")
    
    _container_id = result.stdout.strip()
    print(f"Container started with ID: {_container_id}")
    return _container_id


def launch_webui_headless(container_id: Optional[str] = None) -> None:
    """
    Launch the WebUI in headless mode inside the Docker container.
    Checks if webui is already running before launching.
    """
    global _webui_process, _container_id
    
    if container_id is None:
        container_id = _container_id
        if container_id is None:
            raise RuntimeError("Container ID not set. Call start_docker_container() first.")
    
    # Check if webui is already running
    if is_server_running():
        print("WebUI is already running!")
        return
    
    # Check if launch.py process is already running in container
    result = subprocess.run(
        ["docker", "exec", container_id, "pgrep", "-f", "launch.py"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        print("WebUI launch process already running in container, waiting for API...")
        if wait_for_api():
            print("WebUI API is ready!")
            return
    
    launch_cmd = (
        "cd /opt/stable-diffusion-webui && "
        "python3 launch.py --data=/data/models/stable-diffusion "
        "--enable-insecure-extension-access --xformers --listen --port=7860 "
        "--api --nowebui"
    )
    
    print(f"Launching WebUI in container {container_id}...")
    
    # Execute command inside container using docker exec in detached mode
    cmd = [
        "docker", "exec", "-d", container_id,
        "bash", "-c", launch_cmd
    ]
    
    subprocess.run(cmd, check=True)
    
    # Wait for the API to be ready
    if not wait_for_api():
        raise RuntimeError("WebUI API did not become ready within timeout period")


def wait_for_api(timeout: int = 120) -> bool:
    """Wait for the API to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{SERVER}/sdapi/v1/options", timeout=2)
            if response.status_code == 200:
                return True
            # Non-200 status code - sleep before retrying
            time.sleep(2)
        except requests.exceptions.RequestException:
            # Request failed - sleep before retrying
            time.sleep(2)
    return False


def start_server(enable_robot_control: bool = True) -> None:
    """
    Start the Docker container and launch WebUI in headless mode.
    Also puts robot in dreaming state if robot control is available.
    
    Args:
        enable_robot_control: If True and robot control is available, control robot state
    """
    global _robot_controller
    
    # Control robot state if enabled
    if enable_robot_control:
        try:
            if _robot_controller is None:
                _robot_controller = RobotController()
            _robot_controller.enter_dreaming_state()
        except Exception as e:
            print(f"Warning: Could not control robot state: {e}")
            print("Continuing without robot control...")

    container_id = start_docker_container()
    launch_webui_headless(container_id)
    if not wait_for_api():
        raise RuntimeError("Failed to start WebUI API server")
    

def set_model(checkpoint: Optional[str] = None) -> None:
    """Switch to the specified model checkpoint."""
    if checkpoint is None:
        checkpoint = MODEL_CHECKPOINT
    
    payload = {
        "sd_model_checkpoint": checkpoint
    }
    r = requests.post(f"{SERVER}/sdapi/v1/options", json=payload)
    r.raise_for_status()
    print(f"Model switched to: {checkpoint}")


def ensure_robot_dreaming_state() -> None:
    """
    Ensure robot is in dreaming state before synthesis: lying down, lidar off, VUI cyan solid.
    Initializes robot controller if needed.
    """
    global _robot_controller
    
    if not ROBOT_CONTROL_AVAILABLE:
        return
    
    try:
        if _robot_controller is None:
            _robot_controller = RobotController()
        _robot_controller.enter_dreaming_state()
    except Exception as e:
        print(f"Warning: Could not ensure robot dreaming state: {e}")
        print("Continuing without robot control...")


def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: Optional[int] = None,
    sampler_name: str = "DPM++ 2M",
    scheduler: str = "Karras",
    cfg_scale: float = 7,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    batch_size: int = 1,
    styles: Optional[List[str]] = None,
    output_dir: str = "outputs",
    output_filename: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    Generate an image using Stable Diffusion.
    
    Args:
        prompt: The text prompt for image generation
        negative_prompt: Negative prompt (default: standard negative prompt)
        steps: Number of inference steps (None to use .env default)
        sampler_name: Sampler to use
        scheduler: Scheduler type
        cfg_scale: Guidance scale
        seed: Random seed (None for random)
        width: Image width (None to use .env default)
        height: Image height (None to use .env default)
        batch_size: Number of images to generate
        styles: List of style names
        output_dir: Directory to save output images
        output_filename: Base filename for output (None for auto-generated)
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        List of file paths to generated images
    """
    # Use defaults from environment if not specified
    if width is None:
        width = DEFAULT_WIDTH
    if height is None:
        height = DEFAULT_HEIGHT
    if steps is None:
        steps = DEFAULT_STEPS
    
    if negative_prompt is None:
        negative_prompt = (
            "painting, drawing, illustration, glitch, deformed, mutated, "
            "cross-eyed, ugly, disfigured"
        )
    
    if styles is None:
        styles = []
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "sampler_name": sampler_name,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "styles": styles,
        **kwargs
    }
    
    # Add scheduler if supported
    if scheduler and "scheduler" not in payload:
        payload["scheduler"] = scheduler
    
    # Add seed if provided
    if seed is not None:
        payload["seed"] = seed
    
    # Ensure robot is in dreaming state before synthesis (lying down, lidar off, cyan solid)
    ensure_robot_dreaming_state()
    
    # Start blinking cyan light during synthesis
    if _robot_controller:
        _robot_controller.start_blinking(blink_rate=1.0)
    
    try:
        r = requests.post(f"{SERVER}/sdapi/v1/txt2img", json=payload)
        r.raise_for_status()
        result = r.json()
    finally:
        # Stop blinking and return to solid cyan
        if _robot_controller:
            _robot_controller.stop_blinking()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the actual seed used from the result (if available in info)
    actual_seed = seed
    if "info" in result:
        try:
            info = json.loads(result["info"]) if isinstance(result["info"], str) else result["info"]
            if "seed" in info:
                actual_seed = info["seed"]
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Prepare metadata for saving
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "cfg_scale": cfg_scale,
        "seed": actual_seed,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "styles": styles,
        "timestamp": int(time.time()),
        "model_checkpoint": MODEL_CHECKPOINT,
        "generation_type": "image",
        **{k: v for k, v in kwargs.items() if k != "alwayson_scripts"}  # Exclude complex nested objects
    }
    
    saved_paths = []
    for i, img_b64 in enumerate(result.get("images", [])):
        # Decode base64 image
        img_data = base64.b64decode(img_b64.split(",", 1)[-1])
        
        if output_filename:
            if i == 0:
                out_name = output_filename
            else:
                stem, ext = os.path.splitext(output_filename)
                out_name = f"{stem}_{i}{ext}"
        else:
            timestamp = int(time.time())
            out_name = f"generated_{timestamp}_{i}.png"
        
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(img_data)
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")
        
        # Save metadata as JSON file
        stem, _ = os.path.splitext(out_name)
        metadata_path = os.path.join(output_dir, f"{stem}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
    
    return saved_paths


def regenerate_image_from_metadata(metadata_path: str, output_dir: Optional[str] = None) -> List[str]:
    """
    Regenerate an image from a saved metadata JSON file.
    
    Args:
        metadata_path: Path to the metadata JSON file
        output_dir: Optional output directory override (uses metadata if not specified)
    
    Returns:
        List of file paths to regenerated images
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract parameters from metadata
    prompt = metadata.get("prompt")
    negative_prompt = metadata.get("negative_prompt")
    steps = metadata.get("steps")
    sampler_name = metadata.get("sampler_name", "DPM++ 2M")
    scheduler = metadata.get("scheduler", "Karras")
    cfg_scale = metadata.get("cfg_scale", 7)
    seed = metadata.get("seed")
    width = metadata.get("width")
    height = metadata.get("height")
    batch_size = metadata.get("batch_size", 1)
    styles = metadata.get("styles", [])
    
    # Use specified output_dir or default to "outputs"
    if output_dir is None:
        output_dir = "outputs"
    
    # Generate new filename with timestamp
    timestamp = int(time.time())
    output_filename = f"regenerated_{timestamp}.png"
    
    print(f"Regenerating image from metadata: {metadata_path}")
    print(f"  Prompt: {prompt}")
    print(f"  Seed: {seed}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Steps: {steps}")
    
    # Call generate_image with the saved parameters
    return generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        sampler_name=sampler_name,
        scheduler=scheduler,
        cfg_scale=cfg_scale,
        seed=seed,
        width=width,
        height=height,
        batch_size=batch_size,
        styles=styles,
        output_dir=output_dir,
        output_filename=output_filename
    )


def regenerate_video_from_metadata(metadata_path: str, output_dir: Optional[str] = None) -> List[str]:
    """
    Regenerate a video from a saved metadata JSON file.
    
    Args:
        metadata_path: Path to the metadata JSON file
        output_dir: Optional output directory override (uses metadata if not specified)
    
    Returns:
        List of file paths to regenerated videos
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract parameters from metadata
    prompt = metadata.get("prompt")
    negative_prompt = metadata.get("negative_prompt")
    steps = metadata.get("steps")
    sampler_name = metadata.get("sampler_name", "DPM++ 2M Karras")
    cfg_scale = metadata.get("cfg_scale", 7)
    seed = metadata.get("seed")
    width = metadata.get("width")
    height = metadata.get("height")
    batch_size = metadata.get("batch_size", 1)
    styles = metadata.get("styles", [])
    video_length = metadata.get("video_length", 32)
    fps = metadata.get("fps", 8)
    loop_number = metadata.get("loop_number", 0)
    motion_model = metadata.get("motion_model", "mm_sd15_v3.safetensors")
    animatediff_batch_size = metadata.get("animatediff_batch_size", 16)
    stride = metadata.get("stride", 1)
    overlap = metadata.get("overlap", 4)
    
    # Use specified output_dir or default to "outputs"
    if output_dir is None:
        output_dir = "outputs"
    
    # Generate new filename with timestamp
    timestamp = int(time.time())
    output_filename = f"regenerated_{timestamp}.gif"
    
    print(f"Regenerating video from metadata: {metadata_path}")
    print(f"  Prompt: {prompt}")
    print(f"  Seed: {seed}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Steps: {steps}")
    print(f"  Frames: {video_length}")
    print(f"  FPS: {fps}")
    
    # Call generate_video with the saved parameters
    return generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        sampler_name=sampler_name,
        cfg_scale=cfg_scale,
        seed=seed,
        width=width,
        height=height,
        batch_size=batch_size,
        styles=styles,
        output_dir=output_dir,
        output_filename=output_filename,
        video_length=video_length,
        fps=fps,
        loop_number=loop_number,
        motion_model=motion_model,
        animatediff_batch_size=animatediff_batch_size,
        stride=stride,
        overlap=overlap
    )


def generate_video_progressive(
    prompt: str,
    output_dir: str = "outputs",
    base_filename: str = "animation",
    stages: Optional[List[int]] = None,
    seed: Optional[int] = None,
    **kwargs
) -> List[str]:
    """
    Generate videos progressively with increasing frame counts.
    First generates a still image, then videos with 16, 32 frames.
    Uses the same random seed for all stages for consistency.
    
    Args:
        prompt: The text prompt for video generation
        output_dir: Directory to save output videos
        base_filename: Base filename (without extension) for outputs
        stages: List of frame counts to generate (default: [16, 32])
               Note: AnimateDiff requires minimum 8 frames
        seed: Random seed to use for all stages (None = random)
        **kwargs: Additional parameters to pass to generate_video
    
    Returns:
        List of all generated file paths (image + videos)
    """
    if stages is None:
        stages = [16, 32]
    
    # Generate a random seed if not provided, use it for all stages
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    print(f"\n{'='*60}")
    print(f"  Using seed: {seed} for all stages")
    print(f"{'='*60}")
    
    all_paths = []
    timestamp = int(time.time())
    
    # First, generate a still image (0 frames)
    print(f"\n{'='*60}")
    print(f"  Generating still image (0 frames)")
    print(f"{'='*60}")
    
    try:
        image_filename = f"{base_filename}_seed_{seed}_{timestamp}_0frames.png"
        
        # Extract relevant kwargs for generate_image (if any)
        image_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['negative_prompt', 'steps', 'sampler_name', 'scheduler', 
                               'cfg_scale', 'width', 'height', 'batch_size', 'styles']}
        
        paths = generate_image(
            prompt=prompt,
            output_dir=output_dir,
            output_filename=image_filename,
            seed=seed,
            **image_kwargs
        )
        all_paths.extend(paths)
        print(f"✓ Completed still image generation")
    except Exception as e:
        print(f"✗ Error generating still image: {e}")
        # Continue with video stages even if image fails
    
    # Then generate videos with increasing frame counts
    for frame_count in stages:
        print(f"\n{'='*60}")
        print(f"  Generating {frame_count} frame(s)")
        print(f"{'='*60}")
        
        # Generate filename with frame count and seed
        output_filename = f"{base_filename}_seed_{seed}_{timestamp}_{frame_count}frames.gif"
        
        try:
            # Set animatediff_batch_size to match frame_count if it's smaller than default
            # This prevents errors when frame_count < animatediff_batch_size
            video_kwargs = kwargs.copy()
            if 'animatediff_batch_size' not in video_kwargs:
                # Use the smaller of frame_count or 16 (default batch size)
                video_kwargs['animatediff_batch_size'] = min(frame_count, 16)
            
            paths = generate_video(
                prompt=prompt,
                output_dir=output_dir,
                output_filename=output_filename,
                video_length=frame_count,
                seed=seed,
                **video_kwargs
            )
            all_paths.extend(paths)
            print(f"✓ Completed {frame_count} frame generation")
        except Exception as e:
            print(f"✗ Error generating {frame_count} frames: {e}")
            # Continue with next stage even if this one fails
    
    print(f"\n{'='*60}")
    print(f"  Progressive generation complete!")
    print(f"  Generated {len(all_paths)} file(s) total")
    print(f"  Seed used: {seed}")
    print(f"{'='*60}\n")
    
    return all_paths


def generate_video(
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: Optional[int] = None,
    sampler_name: str = "DPM++ 2M Karras",
    cfg_scale: float = 7,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    batch_size: int = 1,
    styles: Optional[List[str]] = None,
    output_dir: str = "outputs",
    output_filename: str = "animation.gif",
    video_length: Optional[int] = None,
    fps: Optional[int] = None,
    loop_number: int = 0,
    motion_model: str = "mm_sd15_v3.safetensors",
    animatediff_batch_size: int = 16,
    stride: int = 1,
    overlap: int = 4,
    **kwargs
) -> List[str]:
    """
    Generate a video/animation using AnimateDiff.
    First generates a still image (0 frames) with metadata, then generates the video.
    Uses the same seed for both image and video for consistency.
    
    Args:
        prompt: The text prompt for video generation
        negative_prompt: Negative prompt (default: standard negative prompt)
        steps: Number of inference steps (None to use .env default)
        sampler_name: Sampler to use
        cfg_scale: Guidance scale
        seed: Random seed (None for random, same seed used for image and video)
        width: Video width (None to use .env default)
        height: Video height (None to use .env default)
        batch_size: Number of videos to generate
        styles: List of style names
        output_dir: Directory to save output videos
        output_filename: Filename for output
        video_length: Number of frames (None to use .env default)
        fps: Frames per second (None to use .env default)
        loop_number: Loop number (0 = infinite loop)
        motion_model: AnimateDiff motion module
        animatediff_batch_size: Batch size for AnimateDiff
        stride: Stride for AnimateDiff
        overlap: Overlap for AnimateDiff
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        List of file paths to generated images and videos (image first, then videos)
    """
    # Use defaults from environment if not specified
    if width is None:
        width = DEFAULT_WIDTH
    if height is None:
        height = DEFAULT_HEIGHT
    if steps is None:
        steps = DEFAULT_STEPS
    if fps is None:
        fps = DEFAULT_FPS
    if video_length is None:
        video_length = DEFAULT_FRAMES
    
    # Generate seed if not provided (for consistency between image and video)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    if negative_prompt is None:
        negative_prompt = (
            "painting, drawing, illustration, glitch, deformed, mutated, "
            "cross-eyed, ugly, disfigured"
        )
    
    if styles is None:
        styles = []
    
    # Extract base filename for consistent naming
    timestamp = int(time.time())
    if output_filename == "animation.gif":
        base_filename = "animation"
    else:
        # Extract base name from custom filename (remove extension)
        base_filename = os.path.splitext(output_filename)[0]
    
    # First, generate a still image (0 frames) with metadata
    print(f"\n{'='*60}")
    print(f"  Generating still image (0 frames)")
    print(f"{'='*60}")
    
    image_paths = []
    try:
        image_filename = f"{base_filename}_seed_{seed}_{timestamp}_0frames.png"
        
        # Extract relevant kwargs for generate_image
        image_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['negative_prompt', 'steps', 'sampler_name', 'scheduler', 
                               'cfg_scale', 'width', 'height', 'batch_size', 'styles']}
        
        image_paths = generate_image(
            prompt=prompt,
            output_dir=output_dir,
            output_filename=image_filename,
            seed=seed,
            **image_kwargs
        )
        print(f"✓ Completed still image generation")
    except Exception as e:
        print(f"✗ Error generating still image: {e}")
        # Continue with video generation even if image fails
    
    # Now generate the video
    print(f"\n{'='*60}")
    print(f"  Generating {video_length} frame(s)")
    print(f"{'='*60}")
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "sampler_name": sampler_name,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "styles": styles,
        "alwayson_scripts": {
            "AnimateDiff": {
                "args": [
                    {
                        "enable": True,
                        "video_length": video_length,
                        "format": ["GIF"],
                        "loop_number": loop_number,
                        "fps": fps,
                        "model": motion_model,
                        "batch_size": animatediff_batch_size,
                        "stride": stride,
                        "overlap": overlap,
                        "interp": "Off",
                        "interp_x": 10,
                        "freeinit_enable": False,
                    }
                ]
            }
        },
        **kwargs
    }
    
    # Add seed to payload
    payload["seed"] = seed
    
    # Ensure robot is in dreaming state before synthesis (lying down, lidar off, cyan solid)
    ensure_robot_dreaming_state()
    
    # Start blinking cyan light during synthesis
    if _robot_controller:
        _robot_controller.start_blinking(blink_rate=0.5)
    
    try:
        r = requests.post(f"{SERVER}/sdapi/v1/txt2img", json=payload)
        r.raise_for_status()
        result = r.json()
    finally:
        # Stop blinking and return to solid cyan
        if _robot_controller:
            _robot_controller.stop_blinking()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the actual seed used from the result (if available in info)
    actual_seed = seed
    if "info" in result:
        try:
            info = json.loads(result["info"]) if isinstance(result["info"], str) else result["info"]
            if "seed" in info:
                actual_seed = info["seed"]
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Generate filename with seed, timestamp, and frame count if using default filename
    if output_filename == "animation.gif":
        output_filename = f"{base_filename}_seed_{actual_seed}_{timestamp}_{video_length}frames.gif"
    
    # Prepare metadata for saving
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "sampler_name": sampler_name,
        "cfg_scale": cfg_scale,
        "seed": actual_seed,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "styles": styles,
        "video_length": video_length,
        "fps": fps,
        "loop_number": loop_number,
        "motion_model": motion_model,
        "animatediff_batch_size": animatediff_batch_size,
        "stride": stride,
        "overlap": overlap,
        "timestamp": timestamp,
        "model_checkpoint": MODEL_CHECKPOINT,
        "generation_type": "video"
    }
    
    saved_paths = []
    for i, item in enumerate(result.get("images", [])):
        b64 = item.split(",", 1)[-1]
        data = base64.b64decode(b64)
        
        if i == 0:
            out_name = output_filename
        else:
            stem, ext = os.path.splitext(output_filename)
            out_name = f"{stem}_{i}{ext}"
        
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(data)
        saved_paths.append(out_path)
        print(f"Saved animation: {out_path}")
        
        # Save metadata as JSON file
        stem, _ = os.path.splitext(out_name)
        metadata_path = os.path.join(output_dir, f"{stem}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
    
    # Return both image and video paths
    all_paths = image_paths + saved_paths
    return all_paths


def stop_container(remove: bool = True) -> None:
    """Stop and optionally remove the Docker container."""
    global _container_id
    
    if _container_id is None:
        # Try to find container by name
        result = subprocess.run(
            ["docker", "ps", "-aq", "-f", f"name={CONTAINER_NAME}"],
            capture_output=True,
            text=True
        )
        if not result.stdout.strip():
            print("No container found to stop.")
            return
        _container_id = result.stdout.strip()
    
    print(f"Stopping container {_container_id}...")
    subprocess.run(["docker", "stop", _container_id], check=False)
    
    if remove:
        print(f"Removing container {_container_id}...")
        subprocess.run(["docker", "rm", _container_id], check=False)
        _container_id = None
        print("Container stopped and removed.")
    else:
        print(f"Container {_container_id} stopped (not removed).")


def shutdown_server(remove_container: bool = True, enable_robot_control: bool = True) -> None:
    """Stop the WebUI server and optionally stop/remove the container."""
    global _container_id, _robot_controller
    
    print("Shutting down WebUI server...")
    
        # Control robot state if enabled
    if enable_robot_control:
        try:
            if _robot_controller is None:
                _robot_controller = RobotController()
            _robot_controller.stop_blinking()
        except Exception as e:
            print(f"Warning: Error stopping robot blinking: {e}")

    # Stop the webui process inside the container
    container_id = _container_id
    if container_id is None:
        # Try to find container by name
        result = subprocess.run(
            ["docker", "ps", "-aq", "-f", f"name={CONTAINER_NAME}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            container_id = result.stdout.strip()
    
    if container_id:
        try:
            # Find and kill the launch.py process
            result = subprocess.run(
                ["docker", "exec", container_id, "pkill", "-f", "launch.py"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                print("WebUI process stopped.")
            else:
                print("No WebUI process found running (may already be stopped).")
        except Exception as e:
            print(f"Error stopping WebUI process: {e}")
    
    # Stop and optionally remove the container
    stop_container(remove=remove_container)


    # Stand up and enter balanced stand mode
    if _robot_controller:

        # start lidar
        _robot_controller.set_lidar_state("ON")
        print("Lidar started")
        time.sleep(2)

        print("Standing up...")
        try:
            _robot_controller.sport_client.StandUp()
            print("StandUp command sent")
            time.sleep(2)
        except Exception as e:
            print(f"Error in StandUp: {e}")
        
        print("Entering balance stand...")
        try:
            _robot_controller.sport_client.BalanceStand()
            print("BalanceStand command sent")
            time.sleep(2)
        except Exception as e:
            print(f"Error in BalanceStand: {e}")


        # set VUI to green
        _robot_controller.set_vui_color("green")
        print("VUI set to green")
        time.sleep(2)
    


def is_server_running() -> bool:
    """Check if the server is running and accessible."""
    try:
        response = requests.get(f"{SERVER}/sdapi/v1/options", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def cleanup_handler(signum, frame):
    """Handle cleanup on signal."""
    print("\nReceived shutdown signal, cleaning up...")
    shutdown_server()
    sys.exit(0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Stable Diffusion WebUI Docker container")
    parser.add_argument("command", choices=["start", "stop", "status", "image", "video", "batch-image", "batch-video", "regenerate"],
                        help="Command to execute")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")
    parser.add_argument("--prompt-file", type=str, help="File containing prompts for batch generation")
    parser.add_argument("--metadata-file", type=str, help="Metadata JSON file for regeneration")
    parser.add_argument("--append-style", type=str, help="Style string to append to all prompts")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--output-file", type=str, help="Output filename")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--progressive", action="store_true", help="Generate progressively: still image, then videos (16, 32 frames)")
    parser.add_argument("--stages", type=str, help="Comma-separated frame counts for progressive video generation (e.g., '16,32,48,64', min 8 frames)")
    
    args = parser.parse_args()
    
    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    if args.command == "start":
        start_server()
        set_model()
        print("Server started successfully!")
    
    elif args.command == "stop":
        shutdown_server()
    
    elif args.command == "status":
        if is_server_running():
            print("Server is running and accessible.")
        else:
            print("Server is not running or not accessible.")
    
    elif args.command == "image":
        if not args.prompt:
            print("Error: --prompt is required for image generation")
            sys.exit(1)
        
        prompt = args.prompt
        if args.append_style:
            prompt += " " + args.append_style
        
        if not is_server_running():
            print("Server not running. Starting server...")
            start_server()
            set_model()
        
        generate_image(
            prompt=prompt,
            output_dir=args.output_dir,
            output_filename=args.output_file,
            seed=args.seed
        )
    
    elif args.command == "video":
        if not args.prompt:
            print("Error: --prompt is required for video generation")
            sys.exit(1)
        
        prompt = args.prompt
        if args.append_style:
            prompt += " " + args.append_style
        
        if not is_server_running():
            print("Server not running. Starting server...")
            start_server()
            set_model()
        
        if args.progressive:
            # Parse custom stages if provided
            stages = None
            if args.stages:
                try:
                    stages = [int(x.strip()) for x in args.stages.split(',')]
                except ValueError:
                    print("Error: --stages must be comma-separated integers (e.g., '1,2,4,8,16')")
                    sys.exit(1)
            
            # Get base filename without extension
            if args.output_file:
                base_filename = os.path.splitext(args.output_file)[0]
            else:
                base_filename = "animation"
            
            generate_video_progressive(
                prompt=prompt,
                output_dir=args.output_dir,
                base_filename=base_filename,
                stages=stages,
                seed=args.seed
            )
        else:
            generate_video(
                prompt=prompt,
                output_dir=args.output_dir,
                output_filename=args.output_file or "animation.gif",
                seed=args.seed
            )

    elif args.command == "batch-image":
        if not args.prompt_file:
            print("Error: --prompt-file is required for batch image generation")
            sys.exit(1)
        
        if not os.path.exists(args.prompt_file):
            print(f"Error: Prompt file '{args.prompt_file}' not found")
            sys.exit(1)

        if not is_server_running():
            print("Server not running. Starting server...")
            start_server()
            set_model()
        
        try:
            with open(args.prompt_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            print(f"Found {len(prompts)} prompts in {args.prompt_file}")
            
            for i, prompt in enumerate(prompts):
                if args.append_style:
                    prompt += " " + args.append_style
                
                print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt}")
                try:
                    generate_image(
                        prompt=prompt,
                        output_dir=args.output_dir,
                        output_filename=None, # Auto-generate filename
                        seed=args.seed
                    )
                except Exception as e:
                    print(f"Error generating image for prompt '{prompt}': {e}")
                    # Continue with next prompt
                    
        except Exception as e:
            print(f"Error reading prompt file: {e}")
            sys.exit(1)


    elif args.command == "batch-video":
        if not args.prompt_file:
            print("Error: --prompt-file is required for batch video generation")
            sys.exit(1)
        
        if not os.path.exists(args.prompt_file):
            print(f"Error: Prompt file '{args.prompt_file}' not found")
            sys.exit(1)

        if not is_server_running():
            print("Server not running. Starting server...")
            start_server()
            set_model()
        
        # Parse custom stages if provided
        stages = None
        if args.progressive and args.stages:
            try:
                stages = [int(x.strip()) for x in args.stages.split(',')]
            except ValueError:
                print("Error: --stages must be comma-separated integers (e.g., '1,2,4,8,16')")
                sys.exit(1)
        
        try:
            with open(args.prompt_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            print(f"Found {len(prompts)} prompts in {args.prompt_file}")
            
            for i, prompt in enumerate(prompts):
                if args.append_style:
                    prompt += " " + args.append_style
                
                print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt}")
                try:
                    # Generate a unique base filename
                    timestamp = int(time.time())
                    base_filename = f"animation_{timestamp}_{i}"
                    
                    if args.progressive:
                        generate_video_progressive(
                            prompt=prompt,
                            output_dir=args.output_dir,
                            base_filename=base_filename,
                            stages=stages,
                            seed=args.seed
                        )
                    else:
                        output_filename = f"{base_filename}.gif"
                        generate_video(
                            prompt=prompt,
                            output_dir=args.output_dir,
                            output_filename=output_filename,
                            seed=args.seed
                        )
                except Exception as e:
                    print(f"Error generating video for prompt '{prompt}': {e}")
                    # Continue with next prompt
                    
        except Exception as e:
            print(f"Error reading prompt file: {e}")
            sys.exit(1)

    elif args.command == "regenerate":
        if not args.metadata_file:
            print("Error: --metadata-file is required for regenerate command")
            sys.exit(1)
        
        if not os.path.exists(args.metadata_file):
            print(f"Error: Metadata file '{args.metadata_file}' not found")
            sys.exit(1)
        
        if not is_server_running():
            print("Server not running. Starting server...")
            start_server()
            set_model()
        
        try:
            # Read metadata to determine if it's an image or video
            with open(args.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            generation_type = metadata.get("generation_type", "image")
            
            if generation_type == "video":
                print("Detected video metadata, regenerating video...")
                regenerate_video_from_metadata(
                    metadata_path=args.metadata_file,
                    output_dir=args.output_dir
                )
            else:
                print("Detected image metadata, regenerating image...")
                regenerate_image_from_metadata(
                    metadata_path=args.metadata_file,
                    output_dir=args.output_dir
                )
        except Exception as e:
            print(f"Error regenerating from metadata: {e}")
            sys.exit(1)
