import requests
import base64
import os

# SERVER = "http://127.0.0.1:7860"
SERVER = "http://snapper.local:7860"

MODEL_CHECKPOINT = "Realistic_Vision_V5.1_fp16-no-ema.safetensors"  # adjust if your name differs
OUTPUT_DIR = "outputs/analog_dog"


def set_model():
    """Switch to the Realistic_Vision_V5.1_fp16-no-ema checkpoint."""
    payload = {
        "sd_model_checkpoint": MODEL_CHECKPOINT
    }
    r = requests.post(f"{SERVER}/sdapi/v1/options", json=payload)
    r.raise_for_status()
    print(f"Model switched to: {MODEL_CHECKPOINT}")


def generate_image():
    prompt = (
        "analog film photo the dog appears to be standing on the couch, "
        "looking around the room. its position suggests curioisty or an "
        "interest in its surroundings. b&w white photo. . faded film, "
        "desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, "
        "Lomography, stained, highly detailed, found footage"
    )

    negative_prompt = (
        "painting, drawing, illustration, glitch, deformed, mutated, "
        "cross-eyed, ugly, disfigured"
    )

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 20,
        "sampler_name": "DPM++ 2M",        # sampler
        "scheduler": "Karras",             # schedule type (ignored if unsupported)
        "cfg_scale": 7,
        "seed": 3226887643,
        "width": 512,
        "height": 512,
        "batch_size": 1,
        # Style selector — in A1111 this is the "styles" field
        "styles": ["Analog Film"],         # Style Selector Style: Analog Film
    }

    r = requests.post(f"{SERVER}/sdapi/v1/txt2img", json=payload)
    r.raise_for_status()
    result = r.json()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, img_b64 in enumerate(result.get("images", [])):
        # images are base64-encoded PNGs, often prefixed with "data:image/png;base64,"
        img_data = base64.b64decode(img_b64.split(",", 1)[-1])
        out_path = os.path.join(OUTPUT_DIR, f"dog_analog_{i}.png")
        with open(out_path, "wb") as f:
            f.write(img_data)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Make sure webui is running with: python webui.py --api --nowebui
    set_model()
    generate_image()
