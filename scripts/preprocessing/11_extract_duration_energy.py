#!/usr/bin/env python3
# extract phoneme-level duration and energy from MFA TextGrids and audio
#
# duration: phoneme offset - onset, then log-transformed: log(dur + ε)
# energy: Praat intensity (dB) per phoneme, z-normalized per speaker
#
# thesis methodology:
# - uses Praat's intensity extraction (via parselmouth), NOT librosa RMS
# - median intensity per phoneme (robust to outliers)
# - z-normalize per speaker/chapter (removes loudness variability)
# - silence (<-50 dB) marked as NaN
#
# based on: /Users/s.mengari/Desktop/THESIS/Thesis Final/Code/scripts/42_re_extract_energy_praat.py

import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from praatio import textgrid
from collections import defaultdict

# paths
AUDIO_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/utterances")
TEXTGRID_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/mfa/output")
DURATION_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody/durations")
ENERGY_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody/energy")

# parameters (match thesis)
MIN_DUR = 0.02           # 20ms minimum phoneme duration
INTENSITY_TIME_STEP = 0.01  # 10ms (matches F0 frame rate)
INTENSITY_MIN_PITCH = 50.0  # Hz (for intensity smoothing)
SILENCE_THRESHOLD = -50.0   # dB (cutoff for silent segments)


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


def extract_praat_intensity(wav_path):
    # extract intensity contour using Praat (thesis methodology)
    sound = parselmouth.Sound(str(wav_path))
    intensity = sound.to_intensity(
        minimum_pitch=INTENSITY_MIN_PITCH,
        time_step=INTENSITY_TIME_STEP,
        subtract_mean=True  # ensures absolute dB values
    )
    return intensity


def get_intensity_per_phoneme(intensity, intervals, statistic='median'):
    # map intensity contour to phoneme intervals
    # uses median (robust to outliers)
    phoneme_intensities = []
    
    for _, start, end in intervals:
        try:
            if statistic == 'median':
                # Get quantile 0.5 (median)
                intensity_val = call(intensity, "Get quantile", start, end, 0.5)
            else:
                # Get mean intensity
                intensity_val = call(intensity, "Get mean", start, end, "energy")
            
            # silence detection
            if intensity_val < SILENCE_THRESHOLD:
                intensity_val = np.nan
            
            phoneme_intensities.append(float(intensity_val) if not np.isnan(intensity_val) else np.nan)
        except:
            phoneme_intensities.append(np.nan)
    
    return np.array(phoneme_intensities, dtype=np.float32)


def compute_speaker_statistics(energy_dict):
    # compute mean and std per speaker/chapter for normalization
    speaker_energies = defaultdict(list)
    
    for utt_id, energy in energy_dict.items():
        # extract speaker ID (e.g., 'ch02_sent_001' -> 'ch02')
        speaker_id = utt_id.split('_')[0]
        
        # filter out NaN values
        valid_energy = energy[~np.isnan(energy)]
        if len(valid_energy) > 0:
            speaker_energies[speaker_id].extend(valid_energy.tolist())
    
    speaker_stats = {}
    for speaker_id, energies in speaker_energies.items():
        energies_array = np.array(energies)
        mean = float(np.mean(energies_array))
        std = float(np.std(energies_array))
        speaker_stats[speaker_id] = (mean, std)
        print(f"  Speaker {speaker_id}: mean={mean:.2f} dB, std={std:.2f} dB")
    
    return speaker_stats


def z_normalize_energy(energy_dict, speaker_stats):
    # z-normalize energy values per speaker/chapter
    normalized = {}
    
    for utt_id, energy in energy_dict.items():
        speaker_id = utt_id.split('_')[0]
        
        if speaker_id in speaker_stats:
            mean, std = speaker_stats[speaker_id]
            if std > 1e-8:
                normalized[utt_id] = (energy - mean) / std
            else:
                normalized[utt_id] = energy - mean
        else:
            normalized[utt_id] = energy
    
    return normalized


def main():
    DURATION_OUT.mkdir(parents=True, exist_ok=True)
    ENERGY_OUT.mkdir(parents=True, exist_ok=True)
    
    wav_files = list(AUDIO_DIR.rglob("*.wav"))
    print(f"Found {len(wav_files)} wav files")
    
    energy_dict = {}
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
        
        # energy: Praat intensity per phoneme (dB)
        try:
            intensity = extract_praat_intensity(wav)
            phoneme_energy = get_intensity_per_phoneme(intensity, intervals)
            energy_dict[stem] = phoneme_energy
        except Exception as e:
            print(f"  {stem}: energy extraction failed - {e}")
            continue
        
        processed += 1
    
    print(f"\nProcessed: {processed} utterances")
    
    # z-normalize energy per speaker/chapter
    if energy_dict:
        print("\nComputing speaker statistics...")
        speaker_stats = compute_speaker_statistics(energy_dict)
        
        print("\nZ-normalizing energy per speaker...")
        normalized_energy = z_normalize_energy(energy_dict, speaker_stats)
        
        # save normalized energy
        for utt_id, energy in normalized_energy.items():
            np.save(ENERGY_OUT / f"{utt_id}_energy.npy", energy)
        
        # summary statistics
        all_energy = np.concatenate([e[~np.isnan(e)] for e in normalized_energy.values()])
        print(f"\nEnergy statistics (after z-normalization):")
        print(f"  Mean: {np.mean(all_energy):.4f}")
        print(f"  Std: {np.std(all_energy):.4f}")
        print(f"  Range: [{np.min(all_energy):.2f}, {np.max(all_energy):.2f}]")
        print(f"  NaN phonemes: {sum(np.isnan(e).sum() for e in normalized_energy.values())}")
    
    print(f"\nOutput:")
    print(f"  Durations: {DURATION_OUT}")
    print(f"  Energy: {ENERGY_OUT}")


if __name__ == "__main__":
    main()
