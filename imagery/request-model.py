import requests

options = {
    "sd_model_checkpoint": "Realistic_Vision_V5.1_fp16-no-ema.safetensors"
}

r = requests.post(
    "http://127.0.0.1:7860/sdapi/v1/options",
    json=options
)
r.raise_for_status()
