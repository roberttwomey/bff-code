# Installation Guide for macOS

This guide will help you set up and run `chat-manager.py` on a MacBook computer.

## Prerequisites

- macOS 10.15 (Catalina) or later
- Administrator access (for some installations)
- Internet connection
- Microphone and speakers/headphones

## Step 1: Install Homebrew

Homebrew is the recommended package manager for macOS. If you don't have it installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. After installation, add Homebrew to your PATH:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

## Step 2: Install Python 3

Install Python 3 using Homebrew:

```bash
brew install python@3.11
```

Verify the installation:

```bash
python3 --version
```

You should see Python 3.11.x or similar.

## Step 3: Install Ollama

Ollama is required to run the language model. Install it:

```bash
brew install ollama
```

Or download the installer from [https://ollama.ai](https://ollama.ai).

Start the Ollama service:

```bash
ollama serve
```

In a new terminal window, pull the required model:

```bash
ollama pull gemma3n:e2b
```

Verify the model is available:

```bash
ollama list
```

You should see `gemma3n:e2b` in the list.

**Note:** Keep the `ollama serve` process running, or set it up to run as a background service.

## Step 4: Install Audio Dependencies

On macOS, audio support is typically built-in, but you may need to install PortAudio for `sounddevice`:

```bash
brew install portaudio
```

## Step 5: Set Up Python Virtual Environment

Navigate to the project directory:

```bash
cd /path/to/bff-code
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Your terminal prompt should now show `(venv)`.

## Step 6: Install Python Dependencies

Install the required Python packages:

```bash
pip install --upgrade pip
pip install ollama
pip install faster-whisper
pip install sounddevice
pip install soundfile
pip install numpy
pip install piper-tts
pip install torch
pip install python-dotenv
```

**Note:** If you have an Apple Silicon Mac (M1/M2/M3), PyTorch will automatically install the Apple Silicon version. For Intel Macs, it will install the Intel version.

## Step 7: Download Piper Voice Model

You need at least one Piper voice model file. Download a voice model:

```bash
# Create the piper directory if it doesn't exist
mkdir -p speech/piper
cd speech/piper

# Download a voice model (example: English UK, Alan, medium quality)
curl -L -o en_GB-alan-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx"
curl -L -o en_GB-alan-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"

# Or download a different voice from https://huggingface.co/rhasspy/piper-voices
cd ../..
```

**Alternative:** Use the `download_piper_voice.py` script if available:

```bash
python speech/download_piper_voice.py
```

## Step 8: Configure Environment Variables

Create a `.env` file in the project root (if it doesn't exist):

```bash
touch .env
```

Edit the `.env` file and add the following variables:

```bash
# Ollama model name
BFF_OLLAMA_MODEL=gemma3n:e2b

# Whisper model size (tiny, base, small, medium, large)
BFF_WHISPER_MODEL=tiny

# Path to Piper voice model
BFF_PIPER_VOICE=speech/piper/en_GB-alan-medium.onnx

# Optional: System prompt
BFF_SYSTEM_PROMPT="you are SNAPPER a robot dog. you do not say woof, whir, tail wag. answer in 2 sentences or less."

# Optional: Audio settings
BFF_SAMPLE_RATE=16000
BFF_ACTIVATION_THRESHOLD=0.03
BFF_SILENCE_THRESHOLD=0.015
BFF_SILENCE_DURATION=0.8

# Optional: Input device (leave empty to use system default)
# BFF_INPUT_DEVICE_KEYWORD=
```

Adjust the paths and settings according to your setup.

## Step 9: Grant Microphone Permissions

macOS requires explicit permission to access the microphone:

1. Go to **System Settings** (or **System Preferences** on older macOS)
2. Navigate to **Privacy & Security** → **Microphone**
3. Enable microphone access for:
   - **Terminal** (if running from terminal)
   - **Python** (if running as a standalone app)

You may be prompted when you first run the script.

## Step 10: Test the Installation

Test that everything works:

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Make sure Ollama is running (in another terminal)
# ollama serve

# Run the chat manager
python chat-manager.py --piper-voice speech/piper/en_GB-alan-medium.onnx
```

If you've set `BFF_PIPER_VOICE` in your `.env` file, you can run:

```bash
python chat-manager.py
```

## Troubleshooting

### Audio Issues

**Problem:** No audio input/output detected

**Solutions:**
- Check microphone permissions in System Settings → Privacy & Security → Microphone
- List available audio devices:
  ```bash
  python3 -c "import sounddevice as sd; print(sd.query_devices())"
  ```
- Try specifying an input device explicitly:
  ```bash
  python chat-manager.py --input-device-keyword "Built-in Microphone"
  ```

**Problem:** `sounddevice` can't find audio devices

**Solution:**
- Make sure PortAudio is installed: `brew install portaudio`
- Reinstall sounddevice: `pip install --force-reinstall sounddevice`

### Ollama Issues

**Problem:** `ollama` command not found

**Solution:**
- Make sure Ollama is installed: `brew install ollama`
- Or download from https://ollama.ai
- Make sure Ollama service is running: `ollama serve`

**Problem:** Model not found

**Solution:**
- Pull the model: `ollama pull gemma3n:e2b`
- Verify: `ollama list`
- Check the model name matches in your `.env` file

### Whisper Issues

**Problem:** Slow transcription or out of memory errors

**Solutions:**
- Use a smaller Whisper model (set `BFF_WHISPER_MODEL=tiny` in `.env`)
- On Apple Silicon, faster-whisper should use Metal acceleration automatically
- For Intel Macs, consider using CPU-only mode (default)

**Problem:** `faster-whisper` installation fails

**Solution:**
- Make sure you have Xcode Command Line Tools: `xcode-select --install`
- Try installing with: `pip install --upgrade faster-whisper`

### Piper TTS Issues

**Problem:** Voice model not found

**Solution:**
- Verify the path in `.env` or command line argument
- Make sure both `.onnx` and `.json` files are present
- Check file permissions

**Problem:** Audio playback issues

**Solution:**
- Check system audio output settings
- Try a different sample rate: `BFF_SAMPLE_RATE=22050` in `.env`
- Verify audio output device: `python3 -c "import sounddevice as sd; print(sd.query_devices())"`

### Python/Package Issues

**Problem:** `ModuleNotFoundError`

**Solution:**
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall packages: `pip install -r speech/requirements.txt` (if available)
- Or install manually: `pip install ollama faster-whisper sounddevice soundfile numpy piper-tts torch python-dotenv`

**Problem:** Python version mismatch

**Solution:**
- Use Python 3.10 or 3.11 (recommended)
- Check version: `python3 --version`
- Create new venv with specific version: `python3.11 -m venv venv`

### Bluetooth Headset (Optional)

The script supports Bluetooth headsets, but macOS handles Bluetooth differently than Linux:

- Pair your headset through System Settings → Bluetooth
- The script should automatically detect it if it's set as the default input/output device
- You can specify it explicitly: `--input-device-keyword "Your Headset Name"`

**Note:** The Linux-specific Bluetooth commands (`bluetoothctl`, `pactl`) won't work on macOS. The script will fall back to system default audio devices.

## Running as a Background Service (Optional)

On macOS, you can use `launchd` to run the script as a background service. Create a plist file:

```bash
nano ~/Library/LaunchAgents/com.bff.chat-manager.plist
```

Add the following (adjust paths as needed):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.bff.chat-manager</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/bff-code/venv/bin/python</string>
        <string>/path/to/bff-code/chat-manager.py</string>
        <string>--piper-voice</string>
        <string>/path/to/bff-code/speech/piper/en_GB-alan-medium.onnx</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/bff-code</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/chat-manager.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/chat-manager.error.log</string>
</dict>
</plist>
```

Load the service:

```bash
launchctl load ~/Library/LaunchAgents/com.bff.chat-manager.plist
```

Start it:

```bash
launchctl start com.bff.chat-manager
```

Check status:

```bash
launchctl list | grep chat-manager
```

View logs:

```bash
tail -f /tmp/chat-manager.log
```

## Quick Start Summary

Once everything is set up, you can run the chat manager with:

```bash
# Activate virtual environment
source venv/bin/activate

# Make sure Ollama is running (in another terminal)
# ollama serve

# Run the chat manager
python chat-manager.py
```

Or with explicit voice model:

```bash
python chat-manager.py --piper-voice speech/piper/en_GB-alan-medium.onnx
```

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Faster Whisper Documentation](https://github.com/guillaumekln/faster-whisper)
- [Piper TTS Documentation](https://github.com/rhasspy/piper)
- [Piper Voices](https://huggingface.co/rhasspy/piper-voices)

## System Requirements

- **RAM:** At least 8GB (16GB recommended for larger Whisper models)
- **Storage:** ~5GB for models and dependencies
- **CPU:** Any modern Intel or Apple Silicon Mac
- **GPU:** Optional, but Apple Silicon (M1/M2/M3) will use Metal acceleration automatically

## Notes

- The script uses `faster-whisper` instead of `openai-whisper` for better performance
- On Apple Silicon Macs, PyTorch and faster-whisper will use Metal acceleration
- Bluetooth headset support is limited compared to Linux (macOS handles Bluetooth through system settings)
- The script creates log files in `~/bff/logs/` by default (configurable via `BFF_LOG_ROOT`)
