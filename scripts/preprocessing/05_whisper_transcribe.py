#!/usr/bin/env python3
# batch whisper transcription for dutch audiobook chapters
# source: https://github.com/openai/whisper/tree/main

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # mac 

import json
import sys
from pathlib import Path

try:
    import whisper
except ImportError:
    print("Error: pip install openai-whisper")
    sys.exit(1)

AUDIO_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/raw/audio")
JSON_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/whisper/json")
TEXT_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/text/chapters_from_whisper")

# Config
MODEL = "medium" # choose model from tiny, base, small, medium, large
LANG = "nl" # language
SKIP_EXISTING = True # prevent re-transcription of existing files


def whisper_transcribe():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"No audio dir: {AUDIO_DIR}")
    
    JSON_OUT.mkdir(parents=True, exist_ok=True) # create json output directory
    TEXT_OUT.mkdir(parents=True, exist_ok=True) # create text output directory
    
    # Find audio files
    exts = (".wav", ".mp3") # others can be added 
    files = sorted(f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in exts)
    print(f"Found {len(files)} audio files") # print number of audio files
    
    # Load model
    print(f"Loading Whisper {MODEL}...")
    model = whisper.load_model(MODEL)
    
    done, skip, err = 0, 0, 0 # initialize counters
    
    for audio in files:
        json_path = JSON_OUT / f"{audio.stem}_whisper.json"
        txt_path = TEXT_OUT / f"{audio.stem}_whisper.txt"
        
        if SKIP_EXISTING and json_path.exists():
            print(f"  Skip: {audio.name}")
            skip += 1 # increment skip counter
            continue
        
        print(f"  {audio.name}...", end=" ", flush=True) # print audio file name
        
        try:
            result = model.transcribe(str(audio), language=LANG, word_timestamps=True, fp16=False) # transcribe audio file
            
            # save JSON
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # save plain text
            txt_path.write_text(result.get("text", "").strip(), encoding="utf-8")
            
            print(f"{len(result.get('segments', []))} segments")
            done += 1
            
        except Exception as e:
            print(f"ERROR: {e}")
            err += 1
    
    print(f"\nDone: {done}, Skipped: {skip}, Errors: {err}")


if __name__ == "__main__":
    whisper_transcribe()
