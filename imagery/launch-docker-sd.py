#!/usr/bin/env python3
"""
Launch and manage Automatic1111 Stable Diffusion WebUI in Docker headless mode.
Provides methods to start the server, generate images/videos, and shut down.
"""
import subprocess
import time
import requests
import base64
import os
import signal
import sys
from typing import Optional, Dict, Any, List

# Configuration
DOCKER_IMAGE = "daivdl487/stable-diffusion-webui:r36.4.3"
CONTAINER_NAME = "stable-diffusion-webui"
SERVER = "http://127.0.0.1:7860"
MODEL_CHECKPOINT = "Realistic_Vision_V5.1_fp16-no-ema.safetensors"

# Global state
_container_id: Optional[str] = None
_webui_process: Optional[subprocess.Popen] = None


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
        except requests.exceptions.RequestException:
            time.sleep(2)
    return False


def start_server() -> None:
    """Start the Docker container and launch WebUI in headless mode."""
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


def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 20,
    sampler_name: str = "DPM++ 2M",
    scheduler: str = "Karras",
    cfg_scale: float = 7,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
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
        steps: Number of inference steps
        sampler_name: Sampler to use
        scheduler: Scheduler type
        cfg_scale: Guidance scale
        seed: Random seed (None for random)
        width: Image width
        height: Image height
        batch_size: Number of images to generate
        styles: List of style names
        output_dir: Directory to save output images
        output_filename: Base filename for output (None for auto-generated)
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        List of file paths to generated images
    """
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
    
    r = requests.post(f"{SERVER}/sdapi/v1/txt2img", json=payload)
    r.raise_for_status()
    result = r.json()
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    return saved_paths


def generate_video(
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 20,
    sampler_name: str = "DPM++ 2M Karras",
    cfg_scale: float = 7,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 288,
    batch_size: int = 1,
    styles: Optional[List[str]] = None,
    output_dir: str = "outputs",
    output_filename: str = "animation.gif",
    video_length: int = 32,
    fps: int = 8,
    loop_number: int = 0,
    motion_model: str = "mm_sd15_v3.safetensors",
    animatediff_batch_size: int = 16,
    stride: int = 1,
    overlap: int = 4,
    **kwargs
) -> List[str]:
    """
    Generate a video/animation using AnimateDiff.
    
    Args:
        prompt: The text prompt for video generation
        negative_prompt: Negative prompt (default: standard negative prompt)
        steps: Number of inference steps
        sampler_name: Sampler to use
        cfg_scale: Guidance scale
        seed: Random seed (None for random)
        width: Video width
        height: Video height
        batch_size: Number of videos to generate
        styles: List of style names
        output_dir: Directory to save output videos
        output_filename: Filename for output
        video_length: Number of frames
        fps: Frames per second
        loop_number: Loop number (0 = infinite loop)
        motion_model: AnimateDiff motion module
        animatediff_batch_size: Batch size for AnimateDiff
        stride: Stride for AnimateDiff
        overlap: Overlap for AnimateDiff
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        List of file paths to generated videos
    """
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
    
    # Add seed if provided
    if seed is not None:
        payload["seed"] = seed
    
    r = requests.post(f"{SERVER}/sdapi/v1/txt2img", json=payload)
    r.raise_for_status()
    result = r.json()
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    return saved_paths


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


def shutdown_server(remove_container: bool = True) -> None:
    """Stop the WebUI server and optionally stop/remove the container."""
    global _container_id
    
    print("Shutting down WebUI server...")
    
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
    parser.add_argument("command", choices=["start", "stop", "status", "image", "video"],
                        help="Command to execute")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--output-file", type=str, help="Output filename")
    parser.add_argument("--seed", type=int, help="Random seed")
    
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
        
        if not is_server_running():
            print("Server not running. Starting server...")
            start_server()
            set_model()
        
        generate_image(
            prompt=args.prompt,
            output_dir=args.output_dir,
            output_filename=args.output_file,
            seed=args.seed
        )
    
    elif args.command == "video":
        if not args.prompt:
            print("Error: --prompt is required for video generation")
            sys.exit(1)
        
        if not is_server_running():
            print("Server not running. Starting server...")
            start_server()
            set_model()
        
        generate_video(
            prompt=args.prompt,
            output_dir=args.output_dir,
            output_filename=args.output_file or "animation.gif",
            seed=args.seed
        )

