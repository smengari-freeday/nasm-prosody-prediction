#!/usr/bin/env python3
# Convert hard g (ɡ) to soft g (ɣ) in phonemized dictionary.
# All words get soft g EXCEPT known loanwords.

from pathlib import Path

DICT_IN = Path("/Users/s.mengari/Desktop/CODE2/data/raw/lexicon/custom_dictionary.dict")
LOANWORDS = Path("/Users/s.mengari/Desktop/CODE2/data/raw/lexicon/hard_g_analysis/loanwords.txt")
DICT_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/raw/lexicon/custom_dictionary_soft_g.dict")


def load_dict():
    """dictionary {word: phonemes}."""
    entries = {} # init dictionary
    for line in DICT_IN.read_text(encoding="utf-8").splitlines(): # loop through lines
        line = line.strip() # strip line
        if not line or line.startswith("#") or " " not in line: # if line is empty or starts with # or contains a space
            continue # continue
        word, phonemes = line.split(" ", 1) # split line into word and phonemes
        entries[word] = phonemes # add to dictionary
    return entries

def load_loanwords():
    # load loanwords (words that keep hard g)
    if not LOANWORDS.exists():
        return set() # 
    return {w.strip().lower() for w in LOANWORDS.read_text(encoding="utf-8").splitlines() if w.strip()} # return set of loanwords


def main():
    entries = load_dict()
    print(f"Loaded {len(entries)} dictionary entries")
    
    loanwords = load_loanwords()
    print(f"Loanwords (keep hard g): {len(loanwords)}")
    
    # convert hard g → soft g for all non-loanwords
    corrected = 0
    for word, phonemes in entries.items():
        if word.lower() in loanwords:
            continue  # keep hard g for loanwords
        
        new = phonemes.replace(" ɡ ", " ɣ ")
        if new != phonemes:
            entries[word] = new
            corrected += 1
    
    with DICT_OUT.open("w", encoding="utf-8") as f:
        for word, phonemes in sorted(entries.items()):
            f.write(f"{word} {phonemes}\n")
    
    print(f"Corrected {corrected} entries (ɡ → ɣ)") # print number of corrected entries
    print(f"Output: {DICT_OUT}")


if __name__ == "__main__":
    main()
