import requests
import base64

url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

payload = {
    "prompt": "a robot dog walking through a misty forest, cinematic lighting",
    "negative_prompt": "",
    "steps": 25,
    "cfg_scale": 7.0,
    "sampler_name": "Euler a",
    "width": 256,
    "height": 256,
    "batch_size": 1,
}

response = requests.post(url, json=payload)
response.raise_for_status()
data = response.json()

# images are base64-encoded PNGs in data["images"]
for i, b64_image in enumerate(data["images"]):
    image_bytes = base64.b64decode(b64_image.split(",", 1)[-1])
    with open(f"output_{i}.png", "wb") as f:
        f.write(image_bytes)

