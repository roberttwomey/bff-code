# bff-code
Code for the BFF project 2025

## Setup

```zsh
pip install go2-webrtc-connect
```

# Usage

## Plot Lidar Stream

```zsh
python plot_lidar_stream_rt.py
```

## Capture All

```zsh
# 1) install the driver per README
#    git clone --recurse-submodules https://github.com/legion1581/go2_webrtc_connect.git
#    cd go2_webrtc_connect && pip install -e .
#
# 2) install deps used in this example:
pip install opencv-python numpy

# 3) same-LAN (STA) by IP, grab a photo:
python go2_capture_all.py --method sta --ip 192.168.8.181 --photo --out run1

# or STA by serial (driver will multicast-discover IP):
python go2_capture_all.py --method sta --serial B42D2000XXXXXXX --clip 3 --out run2

# or remote via Unitree TURN:
python go2_capture_all.py --method remote --serial B42D2000XXXXXXX --username you@example.com --password '***' --clip 2 --out run3
```

## Voice Assistant Setup (Jetson Orin Nano)

The voice assistant (`chat-manager.py`) uses `faster-whisper`, which relies on `CTranslate2`. on NVIDIA Jetson devices (ARM64), `CTranslate2` must be built from source to enable CUDA acceleration. Check standard PyPi wheels do not include CUDA support for aarch64.

### Building CTranslate2 from Source

1.  **Clone the Repository**:
    ```bash
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git
    cd CTranslate2
    ```

2.  **Install Build Dependencies**:
    ```bash
    pip install "pybind11>=2.2"
    ```

3.  **Build and Install**:
    Ensure `CUDA_TOOLKIT_ROOT_DIR` points to your CUDA installation (typically `/usr/local/cuda`).
    ```bash
    export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
    # Optional: Limit compilation jobs if running out of RAM
    # export MAX_JOBS=4 
    pip install .
    ```

4.  **Verify Installation**:
    ```bash
    python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
    # Should print 1 (or number of GPUs)
    ```

### Python Dependencies
```bash
pip install faster-whisper ollama sounddevice piper-tts
```