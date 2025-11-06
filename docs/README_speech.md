# Speech-Enabled Chat with Snapshots

This script extends the original `chat_with_snapshots.py` with speech capabilities, allowing you to have voice conversations with the robot AI using microphone input and text-to-speech output.

## Features

- **Speech Recognition**: Uses Google's speech recognition API to convert your voice to text
- **Text-to-Speech**: Converts the AI's responses to spoken audio
- **Dual Mode**: Switch between speech and text input modes
- **All Original Features**: Maintains all the original functionality including:
  - Robot state analysis (IMU, motors, battery)
  - Camera image processing
  - LiDAR depth map visualization
  - Multimodal AI responses

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_speech.txt
```

2. **Important**: On macOS, you may need to install additional audio dependencies:
```bash
brew install portaudio
```

3. Set up your environment variables (same as original):
```bash
# Create a .env file with:
OPENAI_API_KEY=your_openai_api_key_here
GO2_SNAPSHOT_DIR=snapshots  # optional, defaults to "snapshots"
OPENAI_MODEL=gpt-4o-mini    # optional, defaults to "gpt-4o-mini"
```

## Usage

Run the speech-enabled chat:
```bash
python chat_with_snapshots_speech.py
```

### Commands

- **Speech Mode** (default):
  - Simply speak naturally to chat with the AI
  - Say "text" to switch to text input mode
  - Say "quit", "exit", or "stop" to end the session

- **Text Mode**:
  - Type your messages as in the original script
  - Type "speech" to switch back to speech mode
  - Type "quit", "exit", or "stop" to end the session

### Speech Settings

You can customize the speech behavior by modifying these variables in the script:

```python
SPEECH_RATE = 150      # Words per minute (default: 150)
SPEECH_VOLUME = 0.9    # Volume level 0.0-1.0 (default: 0.9)
VOICE_ID = None        # Specific voice ID, None for default
```

## Troubleshooting

### Microphone Issues
- Ensure your microphone is properly connected and selected as the default input device
- The script will automatically adjust for ambient noise on startup
- If you get microphone errors, try restarting the script

### Speech Recognition Issues
- Requires an internet connection (uses Google's speech recognition API)
- Speak clearly and avoid background noise
- If recognition fails, try speaking again or switch to text mode

### Text-to-Speech Issues
- On macOS, the script uses the built-in speech synthesis
- On Linux, you may need to install additional TTS engines
- Adjust `SPEECH_RATE` if the speech is too fast or slow

### Audio Dependencies
If you encounter issues with PyAudio installation:

**macOS:**
```bash
brew install portaudio
pip install PyAudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-pyaudio portaudio19-dev
pip install PyAudio
```

**Windows:**
```bash
pip install PyAudio
```

## How It Works

1. **Speech Recognition**: The `SpeechRecognizer` class runs a background thread that continuously listens for speech using your microphone
2. **Processing**: When speech is detected, it's converted to text using Google's speech recognition API
3. **AI Response**: The text is processed exactly like the original script, including robot state, camera images, and LiDAR data
4. **Text-to-Speech**: The AI's response is spoken aloud using the system's text-to-speech engine

## Switching Between Modes

The script allows seamless switching between speech and text modes:

- **Speech → Text**: Say "text" or type "text" in text mode
- **Text → Speech**: Type "speech" in text mode
- Both modes maintain the same conversation context and robot state analysis

## Example Usage

```
🎤 Speech chat mode activated!
💡 Commands:
   - Speak naturally to chat
   - Type 'quit' or 'exit' to end
   - Type 'text' to switch to text input mode
   - Type 'speech' to switch back to speech mode

🎤 Listening... (speak now)
🎤 Heard: How are you feeling today?

Assistant: I'm feeling quite good! My battery is at 85% and all my motors are responding well. I can see through my camera that we're in what looks like an indoor space, and my LiDAR is showing some interesting depth variations around us. What about you?

🔊 Speaking: I'm feeling quite good! My battery is at 85% and all my motors are responding well...
```

## Requirements

- Python 3.7+
- Microphone and speakers/headphones
- Internet connection (for speech recognition)
- OpenAI API key
- Robot snapshot data (same as original script)
