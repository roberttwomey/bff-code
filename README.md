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