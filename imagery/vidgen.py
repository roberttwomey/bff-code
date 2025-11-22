import requests
import base64
import os
import json

# SERVER = "http://127.0.0.1:7860"
SERVER = "http://snapper.local:7860"

MODEL_CHECKPOINT = "Realistic_Vision_V5.1_fp16-no-ema.safetensors"  # adjust if your name differs
OUTPUT_DIR = "outputs/animatediff_clouds"
OUTPUT_FILE = "clouds_analog_anim.gif"   # AnimateDiff format=['GIF']


def set_model():
    """Switch to the Realistic_Vision_V5.1_fp16-no-ema checkpoint."""
    payload = {
        "sd_model_checkpoint": MODEL_CHECKPOINT
    }
    r = requests.post(f"{SERVER}/sdapi/v1/options", json=payload)
    r.raise_for_status()
    print(f"Model switched to: {MODEL_CHECKPOINT}")


def generate_video():
    prompt = "analog film photo clouds in the sky, desert view, photograph, 35mm, b&w"

    negative_prompt = (
        "painting, drawing, illustration, glitch, deformed, mutated, "
        "cross-eyed, ugly, disfigured"
    )

    # AnimateDiff configuration - using dictionary structure (like vidgen2.py)
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 20,
        "sampler_name": "DPM++ 2M Karras",  # Combined sampler + scheduler
        "cfg_scale": 7,
        "seed": 1486345475,
        "width": 512,
        "height": 288,
        "batch_size": 1,
        "styles": ["Analog Film"],  # Style Selector Style: Analog Film

        # AnimateDiff configuration via alwayson_scripts
        "alwayson_scripts": {
            "AnimateDiff": {
                "args": [
                    {
                        "enable": True,                 # enable AnimateDiff
                        "video_length": 32,             # number of frames
                        "format": ["GIF"],              # output format
                        "loop_number": 0,               # 0 = infinite loop
                        "fps": 8,                       # frames per second
                        "model": "mm_sd15_v3.safetensors",  # motion module
                        "batch_size": 16,
                        "stride": 1,
                        "overlap": 4,
                        "interp": "Off",
                        "interp_x": 10,
                        "freeinit_enable": False,
                    }
                ]
            }
        },
    }

    r = requests.post(f"{SERVER}/sdapi/v1/txt2img", json=payload)
    r.raise_for_status()
    result = r.json()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # AnimateDiff returns the animation as base64-encoded image/video data.
    # For format=['GIF'], this will be "data:image/gif;base64,..."
    for i, item in enumerate(result.get("images", [])):
        b64 = item.split(",", 1)[-1]
        data = base64.b64decode(b64)

        if i == 0:
            out_name = OUTPUT_FILE
        else:
            # in case multiple animations are returned
            stem, ext = os.path.splitext(OUTPUT_FILE)
            out_name = f"{stem}_{i}{ext}"

        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved animation: {out_path}")


if __name__ == "__main__":
    # Make sure webui is running:
    #   python webui.py --api --nowebui
    set_model()
    generate_video()
