#!/usr/bin/env python3
# Split full text into chapter files."""

import re
from pathlib import Path

INPUT = Path("/Users/s.mengari/Desktop/CODE2/data/raw/text/original_text.txt")
OUTPUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/text/chapters_cleaned")

# Exact Gutenberg markers
START = "*** START OF THE PROJECT GUTENBERG EBOOK 20.000 MIJLEN ONDER ZEE: OOSTELIJK HALFROND ***"
END = "*** END OF THE PROJECT GUTENBERG EBOOK 20.000 MIJLEN ONDER ZEE: OOSTELIJK HALFROND ***"


def main():
    text = INPUT.read_text(encoding="utf-8")
    
    # Find text between start and end markers
    text = text[text.find(START) + len(START):text.find(END)].strip()
    
    # Split on chapter headers 
    parts = re.split(r"^(HOOFDSTUK\s+[IVXLC]+)\s*$", text, flags=re.MULTILINE)
    
    # Create output dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    
    ch_num = 0
    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        ch_num += 1
        
        out = OUTPUT / f"ch{ch_num:02d}.txt"
        out.write_text(f"{header}\n\n{body}", encoding="utf-8")
        print(f"  {out.name}: {len(body.split()):,} words")
    
    print(f"Done. {ch_num} chapters.")


if __name__ == "__main__":
    main()
