import os
import sys
import wave
from pathlib import Path

try:
    from piper import PiperVoice
except ImportError:
    print("Error: piper-tts not installed or not found in path.")
    sys.exit(1)

def load_env_var(env_path, var_name):
    """Simple .env parser to avoid checking for python-dotenv"""
    env_path = Path(env_path)
    if not env_path.exists():
        return None
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                if key.strip() == var_name:
                    return value.strip()
    return None

def main():
    # 1. Determine Voice Path
    env_path = Path(__file__).resolve().parent.parent / '.env'
    voice_path_str = load_env_var(env_path, 'BFF_PIPER_VOICE')
    
    if not voice_path_str:
        print("Warning: BFF_PIPER_VOICE not found in ../.env, checking default location.")
        # Fallback to local piper dir if variable missing entirely
        voice_path = Path(__file__).parent / "piper" / "en_GB-aru-medium.onnx"
    else:
        # Check if the absolute path from env exists (Linux vs Mac case)
        voice_path = Path(voice_path_str)
        if not voice_path.exists():
            print(f"Path from .env not found: {voice_path}")
            # Try to resolve filename relative to current script's piper/ dir
            filename = voice_path.name
            local_candidate = Path(__file__).parent / "piper" / filename
            if local_candidate.exists():
                print(f"Found local candidate: {local_candidate}")
                voice_path = local_candidate
            else:
                 # Fallback to whatever is in the piper dir if specific file not found?
                 # ideally we want the one requested, but if not there, finding ANY .onnx in piper/ might be useful
                 # for now, let's just fail if we can't find it
                 pass

    if not voice_path.exists():
         print(f"Error: Voice file not found at {voice_path}")
         return

    # 2. Read Script
    script_path = Path(__file__).parent / "script.txt"
    if not script_path.exists():
        print(f"Error: script.txt not found at {script_path}")
        return

    text = script_path.read_text().strip()
    if not text:
        print("Error: script.txt is empty")
        return

    print(f"Read script ({len(text)} chars).")

    # 3. Render Audio
    output_path = Path(__file__).parent / "script.wav"
    
    # Check for CUDA
    # Simple check: try loading with cuda if available, else cpu?
    # The sanity check script checked args. 
    # Let's try to be safe and default to CPU if we aren't sure, or try to detect?
    # standard piper usage usually detects or defaults.
    # The Test script used: use_cuda = not args.cpu
    # We will try to use CUDA if available, but piper python wrapper usually handles 'use_cuda' arg.
    # Let's import onnxruntime to check if we really want to be fancy, or just default to False for safety/compat unless explicitly requested?
    # The user request didn't specify GPU usage, but there is a test_piper_gpu.py.
    # I'll default to use_cuda=False to ensure it runs anywhere, but user might want speed.
    # Actually, looking at test_piper_gpu.py: 
    # providers = voice.session.get_providers()
    # Let's just set use_cuda=False for simplicity unless we know we have it.
    # Actually, let's try True and catch exception? No, simply use False for a render script unless performance is critical.
    # A single script.txt isn't huge.
    
    print(f"Loading voice from {voice_path}...")
    voice = PiperVoice.load(str(voice_path), use_cuda=False) # Safe default

    print(f"Synthesizing to {output_path}...")
    with wave.open(str(output_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    
    print("Done!")

if __name__ == "__main__":
    main()
