#!/usr/bin/env python3
import os, json, base64, sys, glob, math
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
from PIL import Image

# Official OpenAI SDK (Responses API)
# pip install openai>=1.40
from openai import OpenAI
# import json
import dotenv

# =============== Config ===============
dotenv.load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# with open("config.json", "r") as f:
    # config = json.load(f)
SNAPSHOT_DIR = os.environ.get("GO2_SNAPSHOT_DIR", "snapshots")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # vision-capable, efficient
# SYSTEM_PROMPT = """You are Snapper’s embodied co‑pilot. Merge proprioception (IMU, motors, battery)
# with exteroception (camera + LiDAR-derived depth map) to talk about:
# - what the robot is seeing and where it might be
# - how its body feels (temperatures, voltage, balance)
# - safety/affordances (terrain/obstacles)
# Be concrete. If uncertain, say what extra sensing/action would reduce uncertainty."""

SYSTEM_PROMPT = """You are Snapper, an embodied AI in a robot dog. Use body state (proprioception through IMU, motors, battery)
and environment (through exteroception with camera and LIDAR point cloud) to converse with the user. Do not speculate about
what has happened outside your sensory inputs. Do not communicate like a helpful assistant, you are just having a conversation
as a reliable partner, companion, or human's best friend."""

# Depth-map rendering parameters
DEPTH_IMG_SIZE = (128, 128)     # (width, height) of the depth map image
DEPTH_CROP_PAD = 1.05           # add 5% padding around XY bounds so edges aren't clipped
NEAR_BIAS = 0.0                 # meters (clamp)
FAR_BIAS = 0.0                  # meters (extra headroom)

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
# def render_topdown_depth(points: np.ndarray,
#                          img_size=(112, 112),
#                          crop_pad: float = 1.05,
#                          near_bias: float = 0.0,
#                          far_bias: float = 0.0) -> Image.Image:
#     """
#     Make a bird's-eye depth image (top-down XY grid, value = nearest range r).
#     - points: Nx3 in meters (x,y,z) in robot frame (any frame is fine for overview).
#     - depth value: r = sqrt(x^2 + y^2 + z^2) per grid cell; we keep the MIN (closest).
#     - Returns an 8-bit grayscale PIL Image (near=bright, far=dark).
#     """
#     if points.size == 0:
#         return Image.new("L", img_size, 0)

#     # Compute bounds in X,Y to define raster grid
#     x = points[:, 0]
#     y = points[:, 1]
#     z = points[:, 2]
#     r = np.sqrt(x*x + y*y + z*z)

#     xmin, xmax = np.min(x), np.max(x)
#     ymin, ymax = np.min(y), np.max(y)

#     # Pad bounds a bit so edges aren't cropped
#     x_center = 0.5 * (xmin + xmax)
#     y_center = 0.5 * (ymin + ymax)
#     x_half = 0.5 * (xmax - xmin) * crop_pad
#     y_half = 0.5 * (ymax - ymin) * crop_pad
#     xmin, xmax = x_center - x_half, x_center + x_half
#     ymin, ymax = y_center - y_half, y_center + y_half

#     W, H = img_size
#     # Map X→u (0..W-1), Y→v (0..H-1), with Y increasing downward (top-down image)
#     # So larger Y (forward) goes toward lower rows if you prefer; we'll use standard image coords.
#     # Compute indices
#     u = ( (x - xmin) / max(1e-6, (xmax - xmin)) * (W - 1) ).astype(np.int32)
#     v = ( (y - ymin) / max(1e-6, (ymax - ymin)) * (H - 1) ).astype(np.int32)

#     # Clamp within bounds
#     u = np.clip(u, 0, W - 1)
#     v = np.clip(v, 0, H - 1)

#     # For each pixel, keep the *closest* range (min r)
#     depth = np.full((H, W), np.inf, dtype=np.float32)
#     # vectorized scatter-min
#     flat_idx = v * W + u
#     # If multiple points fall into same cell, keep minimum r
#     # We'll do this by sorting and taking first
#     order = np.argsort(r)
#     flat_idx_sorted = flat_idx[order]
#     r_sorted = r[order]
#     # unique keeps first occurrence (which now corresponds to MIN r because of sorting)
#     _, first_pos = np.unique(flat_idx_sorted, return_index=True)
#     depth_flat = depth.ravel()
#     depth_flat[flat_idx_sorted[first_pos]] = r_sorted[first_pos]
#     depth = depth_flat.reshape(H, W)

#     # Handle inf (cells with no points): fill with far max
#     finite = np.isfinite(depth)
#     if not np.any(finite):
#         return Image.new("L", img_size, 0)
#     dmin = np.min(depth[finite])
#     dmax = np.max(depth[finite])
#     dmin = max(0.0, dmin - near_bias)
#     dmax = dmax + far_bias if far_bias > 0 else dmax

#     # Normalize: near -> bright (255), far -> dark (0)
#     norm = np.zeros_like(depth, dtype=np.float32)
#     scale = (dmax - dmin) if (dmax > dmin) else 1.0
#     norm[finite] = (1.0 - (depth[finite] - dmin) / scale)  # invert
#     norm[~finite] = 0.0

#     img = (norm * 255.0).astype(np.uint8)
#     return Image.fromarray(img, mode="L")

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

    # Determine the boundaries of the point cloud
    # min_x = min(p[0] for p in int_points)
    # max_x = max(p[0] for p in int_points)
    # min_y = min(p[1] for p in int_points)
    # max_y = max(p[1] for p in int_points)
    # min_z = min(p[2] for p in int_points)
    # max_z = max(p[2] for p in int_points)

    # Calculate image dimensions
    # width = max_x - min_x + 1
    # height = max_y - min_y + 1
    height_map = np.full(img_size, 0, dtype=np.float32)

    # # Populate the height map. For overlapping points, keep the highest z-value.
    for x, y, z in int_points:
        if z > height_map[x, y]:
            height_map[x, y] = z

    # # Replace -inf with the minimum z-value for normalization
    # height_map[height_map == -np.inf] = min_z

    # # Replace pixels that received no points with the minimum z-value for normalization
    # height_map[height_map < min_z] = min_z

    # Normalize the height map to a 0-255 range for an 8-bit grayscale image
    # if max_z > min_z:
    #     # Normalize to 0-1 range
    #     normalized_map = (height_map - min_z) / (max_z - min_z)

    # Create a PIL image. We flip it vertically so that the y-axis points up.
    # image = Image.fromarray(np.flipud(height_map), 'L')
    image = Image.fromarray(height_map, 'L')
    
    # # Resize to the desired output size for consistency
    # if img_size:
    #     image = image.resize(img_size, Image.Resampling.LANCZOS)

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
    print("Embodied Chat — grounding with latest robot snapshot")
    print(f"Snapshot JSON : {json_path if json_path else '— none —'}")
    print(f"Camera image  : {img_path if img_path else '— none —'}")
    print(f"Point cloud   : {ply_path if ply_path else '— none —'}")
    print("="*78)

# =============== Chat Loop ===============
def main():
    client = OpenAI(api_key=OPENAI_API_KEY)  # reads OPENAI_API_KEY
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
    print("Type your message (Ctrl+C to quit). I’ll include state + camera + LiDAR depth each turn.\n")

    while True:
        try:
            user_text = input("You: ").strip()
            if not user_text:
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

            print("\nAssistant:", resp.output_text.strip(), "\n")

        except KeyboardInterrupt:
            print("\nBye!")
            sys.exit(0)
        except Exception as e:
            print(f"[error] {e}\n")

if __name__ == "__main__":
    main()
