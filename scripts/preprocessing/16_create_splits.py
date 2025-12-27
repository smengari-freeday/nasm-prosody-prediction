#!/usr/bin/env python3
# create train/val/test splits at utterance level
# splits are created at the utterance level, not the sentence level
# 80/10/10 split
# filters to only include files with complete prosody data (F0, duration, energy)

import json
import random
from pathlib import Path
from datetime import datetime

# paths
FEATURES_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/features/phoneme_level")
PROSODY_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody")
OUTPUT = FEATURES_DIR / "splits.json"

# config 
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42


def has_prosody(utt_id):
    """Check if utterance has all prosody files."""
    f0 = PROSODY_DIR / f"f0/consensus/{utt_id}_f0.npy"
    dur = PROSODY_DIR / f"durations/{utt_id}_durations.npy"
    energy = PROSODY_DIR / f"energy/{utt_id}_energy.npy"
    return all([f0.exists(), dur.exists(), energy.exists()])


def main():
    # get all feature files with prosody
    files = []
    for f in sorted(FEATURES_DIR.glob("*_features.npy")):
        utt_id = f.stem.replace("_features", "")
        if has_prosody(utt_id):
            files.append(f.name)
        else:
            print(f"Skipping {f.name} (missing prosody)")
    
    if not files:
        print(f"No valid files in {FEATURES_DIR}")
        return
    
    print(f"Found {len(files)} files with complete prosody")
    
    # shuffle
    random.seed(SEED)
    random.shuffle(files)
    
    # split
    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    
    splits = {
        "metadata": {
            "total": n,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "seed": SEED,
            "created": datetime.now().isoformat()
        },
        "splits": {
            "train": train,
            "val": val,
            "test": test
        }
    }
    
    OUTPUT.write_text(json.dumps(splits, indent=2))
    
    print(f"Train: {len(train)} ({len(train)/n:.1%})")
    print(f"Val:   {len(val)} ({len(val)/n:.1%})")
    print(f"Test:  {len(test)} ({len(test)/n:.1%})")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
