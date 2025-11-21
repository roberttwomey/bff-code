import requests, base64

payload = {
    "prompt": "a hyperrealistic portrait of a robot dog vizsla in studio lighting, black and white, analog film, 35mm",
    "steps": 20,
    "width": 512,
    "height": 512,
}

r = requests.post(
    "http://127.0.0.1:7860/sdapi/v1/txt2img",
    json=payload
)
r.raise_for_status()
data = r.json()

# save returned image(s)
for i, b64 in enumerate(data["images"]):
    img = base64.b64decode(b64.split(",", 1)[-1])
    with open(f"output_{i}.png", "wb") as f:
        f.write(img)

