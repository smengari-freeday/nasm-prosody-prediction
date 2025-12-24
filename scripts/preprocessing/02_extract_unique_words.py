#!/usr/bin/env python3
# extract unique words from corpus for external G2P phonemization
import re
from pathlib import Path

CHAPTERS_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/text/chapters_cleaned")
OUTPUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/lexicon/unique_words.txt")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize set of words
    words = set()
    
    for f in sorted(CHAPTERS_DIR.glob("ch*.txt")): # loop through chapters
        text = f.read_text(encoding="utf-8").lower() # read text and convert to lowercase
        # Extract words (letters, apostrophes, accented chars)
        tokens = re.findall(r"\b[a-zA-Zéèêëïîôùûüçà']+\b", text)
        words.update(tokens)
    
    # Sort all words alphabetically
    sorted_words = sorted(words)
    OUTPUT.write_text("\n".join(sorted_words), encoding="utf-8") # write to file
    # Print number of unique words
    print(f"Extracted {len(sorted_words)} unique words")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()

