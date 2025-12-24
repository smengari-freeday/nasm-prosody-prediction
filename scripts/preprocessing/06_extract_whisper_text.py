#!/usr/bin/env python3
# extract plain text from whisper JSON files
# source: https://github.com/openai/whisper/tree/main

import json
from pathlib import Path

JSON_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/whisper/json")
TEXT_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/text/chapters_from_whisper")


def extract_whisper_text():
    if not JSON_DIR.exists():
        raise FileNotFoundError(f"No JSON dir: {JSON_DIR}")
    
    TEXT_OUT.mkdir(parents=True, exist_ok=True) # create text output directory
    
    files = sorted(JSON_DIR.glob("*.json"))
    print(f"Found {len(files)} JSON files")
    
    for file in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        text = data.get("text", "").strip()
        
        # concatenate segments if no text
        if not text:
            text = " ".join(s.get("text", "") for s in data.get("segments", [])).strip()
        
        if not text:
            print(f"  {file.name}: no text")
            continue
        
        stem = file.stem if file.stem.endswith("_whisper") else f"{file.stem}_whisper"
        out = TEXT_OUT / f"{stem}.txt"
        out.write_text(text, encoding="utf-8") # write text to file
        print(f"  {out.name}: {len(text.split()):,} words") # print number of words
    
    print("Done.")


if __name__ == "__main__":
    main()
