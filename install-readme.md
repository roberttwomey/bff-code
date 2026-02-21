# Installation Guide for bff-code on Jetson Orin (JetPack 6.2)

This guide covers the setup for `bff-code` and its dependencies on a fresh Jetson Orin running JetPack 6.2 (Ubuntu 22.04, CUDA 12.6).

## 1. System Prerequisites

Ensure your system is up to date and has essential build tools.

```bash
sudo apt update
sudo apt install -y python3-pip libopenblas-dev git cmake build-essential
```

## 2. Python Environment Setup

It is highly recommended to use a virtual environment or ensure your global environment is clean.

### Core Dependency Rules
> [!IMPORTANT]
> - **Numpy**: Must be `<2`. Version 2.x causes compatibility issues with some Jetson libraries.
> - **PyTorch**: Do NOT install valid `torch` from standard PyPI. You MUST use the NVIDIA Jetson wheels.
> - **CTranslate2**: Must be built from SOURCE to enable CUDA support on JetPack 6.

## 3. Core AI Library Installation

### Step 3.1: Clean up existing packages
If you have previous installations, remove them to be safe:
```bash
pip3 uninstall -y torch torchvision torchaudio faster-whisper ctranslate2 numpy
```

### Step 3.2: Install Numpy
```bash
pip3 install "numpy<2"
```

### Step 3.3: Install PyTorch (Jetson Wheels)
Install libraries compatible with JetPack 6 (CUDA 12.6).
```bash
# Download wheels from Jetson AI Lab (valid as of Jan 2026 for JP6.2)
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/81a/775c8af36ac85/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl

# Install them
pip3 install torch-2.8.0-cp310-cp310-linux_aarch64.whl
pip3 install torchvision-0.23.0-cp310-cp310-linux_aarch64.whl
pip3 install torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl

# Cleanup
rm *.whl
```

### Step 3.4: Build CTranslate2 from Source (Required for CUDA)
Standard pip install of `ctranslate2` lacks CUDA support for ARM64/Jetson.

```bash
# 1. Clone Repository (outside your project folder, e.g. in ~/code)
cd ~/code
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2

# 2. Build C++ Library with CUDA
mkdir -p build && cd build
cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP
make -j$(nproc)
sudo make install
sudo ldconfig

# 3. Install Python Bindings
cd ../python
pip3 install -r install_requirements.txt
CT_WITH_CUDA=1 pip3 install .
```

### Step 3.5: Install Faster-Whisper
```bash
pip3 install --force-reinstall faster-whisper
```

## 4. Project Dependencies

### Peer Dependencies
This project relies on sibling repositories expected to be in the same parent directory (`../`).

```bash
# Assuming directory structure:
# /home/jesse/code/
# ├── bff-code/
# ├── go2_webrtc_connect/
# └── unitree_sdk2_python/

cd /home/jesse/code/bff-code

# Install local drivers
pip3 install -e ../go2_webrtc_connect
pip3 install -e ../unitree_sdk2_python
```

### Python Packages
Install the remaining application dependencies.

```bash
pip3 install \
    opencv-python \
    aiortc \
    ultralytics \
    open3d \
    Pillow \
    openai \
    python-dotenv \
    SpeechRecognition \
    pyttsx3 \
    ollama \
    sounddevice \
    soundfile \
    piper-tts
```

## 5. Verification

Run the following python script to verify GPU access:

```python
import torch
import ctranslate2
from faster_whisper import WhisperModel

print(f"Torch GPU: {torch.cuda.is_available()}")
print(f"CTranslate2 GPU Devices: {ctranslate2.get_cuda_device_count()}")

try:
    model = WhisperModel('tiny', device='cuda', compute_type='float16')
    print("Faster-Whisper on GPU: SUCCESS")
except Exception as e:
    print(f"Faster-Whisper on GPU: FAILED ({e})")
```
