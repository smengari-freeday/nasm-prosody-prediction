#!/usr/bin/env python3
# extract phoneme-level duration from MFA TextGrids
#
# duration: phoneme offset - onset, then log-transformed: log(dur + ε)
# output: one .npy file per utterance with log-duration per phoneme

import numpy as np
from pathlib import Path
from praatio import textgrid

# paths
TEXTGRID_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/mfa/output")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody/durations")

# parameters
MIN_DUR = 0.02  # 20ms minimum phoneme duration


def get_phoneme_intervals(tg_path):
    # get phoneme labels and time intervals from TextGrid
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    tier = tg.getTier("phones")
    
    intervals = []
    for entry in tier.entries:
        label = entry.label.strip()
        if not label or label in ("", "sil", "sp", "spn"):
            continue
        dur = entry.end - entry.start
        if dur < MIN_DUR:
            continue
        intervals.append((label, entry.start, entry.end))
    
    return intervals


def extract_durations(intervals):
    # compute log-duration for each phoneme
    durations = []
    for _, start, end in intervals:
        dur = end - start
        durations.append(np.log(dur + 1e-8))  # log transform
    return np.array(durations, dtype=np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # find all TextGrids
    tg_files = list(TEXTGRID_DIR.rglob("*.TextGrid"))
    print(f"Found {len(tg_files)} TextGrid files")
    
    processed = 0
    for tg_path in tg_files:
        stem = tg_path.stem
        
        # get phoneme intervals
        intervals = get_phoneme_intervals(tg_path)
        if len(intervals) == 0:
            continue
        
        # extract log-durations
        durations = extract_durations(intervals)
        np.save(OUTPUT_DIR / f"{stem}_durations.npy", durations)
        
        processed += 1
    
    print(f"\nProcessed: {processed} utterances")
    print(f"Output: {OUTPUT_DIR}")
    
    # summary statistics
    if processed > 0:
        all_dur = []
        for f in OUTPUT_DIR.glob("*.npy"):
            all_dur.extend(np.load(f))
        all_dur = np.array(all_dur)
        print(f"\nDuration statistics (log-transformed):")
        print(f"  Mean: {np.mean(all_dur):.4f}")
        print(f"  Std: {np.std(all_dur):.4f}")
        print(f"  Range: [{np.min(all_dur):.2f}, {np.max(all_dur):.2f}]")


if __name__ == "__main__":
    main()

