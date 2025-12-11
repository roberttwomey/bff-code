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

## Setting up as a service

This guide explains how to set up `chat-manager.py` to run automatically at system startup.

### Method 1: Systemd User Service (Recommended)

This is the recommended method for production use. It runs as a **user service**, which means it runs with your user's full environment and permissions, including CUDA access. It provides proper logging, automatic restarts, and service management.

#### Step 1: Install the Service

1. Create the user systemd directory if it doesn't exist:
   ```bash
   mkdir -p ~/.config/systemd/user
   ```

2. Copy the service file to your user systemd directory:
   ```bash
   cp /home/cohab/code/bff-code/chat-manager.service ~/.config/systemd/user/
   ```

3. Reload systemd user daemon to recognize the new service:
   ```bash
   systemctl --user daemon-reload
   ```

4. Enable the service to start at boot (and when you log in):
   ```bash
   systemctl --user enable chat-manager.service
   ```

   **Note**: For the service to start at boot (before login), you may also need to enable lingering:
   ```bash
   sudo loginctl enable-linger cohab
   ```

#### Step 2: Verify Conda Environment

The service automatically uses the `bff` conda environment. Make sure it exists:
```bash
conda env list | grep bff
```

If the environment doesn't exist or is in a different location, you'll need to update the service file paths.

#### Step 3: Configure Environment Variables

Make sure your `.env` file in `/home/cohab/code/bff-code/` contains the necessary variables, especially:
- `BFF_PIPER_VOICE` - Path to your Piper voice model (e.g., `/home/cohab/code/bff-code/speech/piper/en_GB-alan-medium.onnx`)
- `BFF_OLLAMA_MODEL` - Ollama model name (default: `gemma3n:e2b`)
- `BFF_WHISPER_MODEL` - Whisper model size (default: `tiny`)
- Any other BFF_* environment variables you need

#### Step 4: Start the Service

Start the service immediately (without rebooting):
```bash
systemctl --user start chat-manager.service
```

#### Step 5: Check Status and Logs

Check if the service is running:
```bash
systemctl --user status chat-manager.service
```

View live logs:
```bash
journalctl --user -u chat-manager.service -f
```

View recent logs:
```bash
journalctl --user -u chat-manager.service -n 100
```

#### Step 6: Managing the Service

- **Stop the service:**
  ```bash
  systemctl --user stop chat-manager.service
  ```

- **Restart the service:**
  ```bash
  systemctl --user restart chat-manager.service
  ```

- **Disable startup (but keep service file):**
  ```bash
  systemctl --user disable chat-manager.service
  ```

- **Remove the service completely:**
  ```bash
  systemctl --user disable chat-manager.service
  rm ~/.config/systemd/user/chat-manager.service
  systemctl --user daemon-reload
  ```

### Method 2: Crontab (Alternative)

If you prefer a simpler approach without systemd:

1. Edit your crontab:
   ```bash
   crontab -e
   ```

2. Add this line (adjust the path and arguments as needed):
   ```cron
   @reboot sleep 30 && cd /home/cohab/code/bff-code && /usr/bin/python3 chat-manager.py --piper-voice /home/cohab/code/bff-code/speech/piper/en_GB-alan-medium.onnx >> /home/cohab/chat-manager.log 2>&1 &
   ```

   The `sleep 30` gives the system time to initialize audio and Bluetooth services.

### Troubleshooting

#### Service fails to start

1. Check the service status:
   ```bash
   systemctl --user status chat-manager.service
   ```

2. Check logs for errors:
   ```bash
   journalctl --user -u chat-manager.service -n 50
   ```

3. Verify the script works manually (with conda environment):
   ```bash
   cd /home/cohab/code/bff-code
   conda activate bff
   python3 chat-manager.py
   ```
   
   Or test the wrapper script:
   ```bash
   bash /home/cohab/code/bff-code/run-chat-manager.sh --help
   ```

#### Audio/Bluetooth issues

The service waits for `bluetooth.service` and `pulseaudio.service` to start. If you're still having issues:

1. Check if Bluetooth is enabled:
   ```bash
   systemctl status bluetooth
   ```

2. Check if PulseAudio is running:
   ```bash
   systemctl --user status pulseaudio
   ```

3. Verify audio devices are accessible:
   ```bash
   python3 -c "import sounddevice as sd; print(sd.query_devices())"
   ```

#### Permission issues

If you get permission errors, ensure the user `cohab` is in the `audio` and `pulse-access` groups:
```bash
sudo usermod -a -G audio,pulse-access cohab
```

You may need to log out and back in for group changes to take effect.

#### Customizing the Service

To modify the service (e.g., change arguments or environment variables):

1. Edit the service file:
   ```bash
   nano ~/.config/systemd/user/chat-manager.service
   ```

2. After making changes, reload and restart:
   ```bash
   systemctl --user daemon-reload
   systemctl --user restart chat-manager.service
   ```

### Notes

- **User Service**: This runs as a user service (not a system service), which means it has access to your full user environment, including CUDA/GPU devices
- The service uses a wrapper script (`run-chat-manager.sh`) that properly activates the `bff` conda environment
- The wrapper script sources conda and activates the environment, matching your manual workflow
- The service is configured to restart automatically on failure (up to 5 times within 5 minutes)
- Logs are sent to systemd journal, accessible via `journalctl --user`
- The service waits for network, audio, and Bluetooth services before starting
- Make sure Ollama is running and the required model is available before the service starts
- If your conda installation is in a different location, update the paths in `run-chat-manager.sh`
- **Important**: User services run when you log in. To start at boot (before login), enable lingering with `sudo loginctl enable-linger cohab`