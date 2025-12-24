#!/usr/bin/env python3
# extract phoneme-level prosody targets from MFA TextGrids and audio
#
# - duration: phoneme offset - onset, then log-transformed: log(dur + ε)
# - energy: average RMS (dB) per phoneme interval
# - f0: average consensus F0 (log) per phoneme interval
#
# all targets are at phoneme level (aligned to TextGrid phone tier)

import numpy as np
import librosa
from pathlib import Path
from praatio import textgrid

AUDIO_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/utterances")
TEXTGRID_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/mfa/output")
F0_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody/f0/consensus")
DURATION_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody/durations")
ENERGY_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody/energy")

# signal processing parameters
SR = 22050          # sample rate
HOP = 220           # 10ms hop (22050 * 0.01)
WIN = 551           # 25ms window (22050 * 0.025)
MIN_DUR = 0.02      # 20ms minimum phoneme duration threshold
FRAME_SHIFT = HOP / SR  # 10ms in seconds


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


def aggregate_to_phonemes(frame_values, intervals, frame_shift=FRAME_SHIFT):
    # average frame-level values per phoneme interval
    phoneme_values = []
    for _, start, end in intervals:
        start_frame = int(start / frame_shift)
        end_frame = int(end / frame_shift)
        
        if end_frame > start_frame and end_frame <= len(frame_values):
            segment = frame_values[start_frame:end_frame]
            valid = ~np.isnan(segment)
            if valid.any():
                phoneme_values.append(np.nanmean(segment))
            else:
                phoneme_values.append(np.nan)
        else:
            phoneme_values.append(np.nan)
    
    return np.array(phoneme_values, dtype=np.float32)


def extract_energy(wav_path):
    # extract frame-level RMS energy in dB
    y, _ = librosa.load(wav_path, sr=SR)
    rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP)[0]
    energy_db = 20 * np.log10(rms + 1e-10)
    return energy_db


def main():
    DURATION_OUT.mkdir(parents=True, exist_ok=True)
    ENERGY_OUT.mkdir(parents=True, exist_ok=True)
    
    wav_files = list(AUDIO_DIR.rglob("*.wav"))
    print(f"Found {len(wav_files)} wav files")
    
    processed = 0
    
    for wav in wav_files:
        stem = wav.stem
        
        # find corresponding TextGrid
        tg_path = None
        for ch_dir in TEXTGRID_DIR.glob("ch*"):
            candidate = ch_dir / f"{stem}.TextGrid"
            if candidate.exists():
                tg_path = candidate
                break
        
        if not tg_path:
            continue
        
        # get phoneme intervals from TextGrid
        intervals = get_phoneme_intervals(tg_path)
        if len(intervals) == 0:
            continue
        
        # duration: log(end - start) per phoneme
        durations = extract_durations(intervals)
        np.save(DURATION_OUT / f"{stem}_durations.npy", durations)
        
        # energy: average dB per phoneme
        frame_energy = extract_energy(wav)
        phoneme_energy = aggregate_to_phonemes(frame_energy, intervals)
        np.save(ENERGY_OUT / f"{stem}_energy.npy", phoneme_energy)
        
        # f0: average log-F0 per phoneme (load pre-computed consensus)
        f0_file = F0_DIR / f"{stem}_f0.npy"
        if f0_file.exists():
            frame_f0 = np.load(f0_file)
            phoneme_f0 = aggregate_to_phonemes(frame_f0, intervals)
            # save phoneme-level F0 (overwrite the frame-level file for consistency)
            np.save(f0_file, phoneme_f0)
        
        processed += 1
    
    print(f"\nProcessed: {processed} utterances")
    print(f"Durations: {DURATION_OUT}")
    print(f"Energy: {ENERGY_OUT}")
    print(f"F0 (phoneme-level): {F0_DIR}")


if __name__ == "__main__":
    main()
