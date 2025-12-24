#!/usr/bin/env python3
# Extract F0 using 3-extractor consensus (Praat, WORLD, CREPE)
# as described in my thesis Chapter 3:
# - multi-extractor fusion with weighted average
# - WORLD: w=1.2 highest weight since it is the most stable extractor, validated emperically
# - Praat: w=1.0 (baseline, occasional octave jumps)
# - CREPE: w=confidence (reduces over-prediction in non-speech) 
# - voicing decision: frame voiced if ≥2 extractors agree
# - unvoiced frames are set to NaN
# - post-processing: log transform, ±3σ clipping, interpolation 
#
# Sources: 
#   WORLD: https://github.com/mmorise/World
#   CREPE: https://github.com/maxrmorrison/torchcrepe
#   Praat: https://www.fon.hum.uva.nl/praat/

# warning: CREPE is very computationally expensive due to its neural network architecture

import numpy as np
import librosa
import pyworld as pw
import torch
import torchcrepe
from pathlib import Path
from scipy.signal import medfilt


# paths
AUDIO_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/utterances")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody/f0")

# output subdirectories for each extractor
OUTPUT_PRAAT = OUTPUT_DIR / "praat"
OUTPUT_WORLD = OUTPUT_DIR / "world"
OUTPUT_CREPE = OUTPUT_DIR / "crepe"
OUTPUT_CONSENSUS = OUTPUT_DIR / "consensus"

SR = 22050      # sample rate
HOP_MS = 10     # hop size in ms
F0_MIN = 75     # minimum F0
F0_MAX = 300    # maximum F0
CREPE_CONF_THRESH = 0.5  # CREPE confidence threshold for voicing


def extract_world_f0(y, sr):
    """WORLD vocoder F0 - stable in modal voiced regions."""
    f0, _ = pw.harvest(y.astype(np.float64), sr, frame_period=HOP_MS, f0_floor=F0_MIN, f0_ceil=F0_MAX)
    voiced = f0 > 0
    f0[~voiced] = np.nan
    return f0, voiced


def extract_crepe_f0(y, sr, device="cpu"):
    # TorchCREPE neural F0 - returns F0 and confidence
    # CREPE paper (Kim et al., 2018):
    # "The periodicity is computed as the normalized autocorrelation at the estimated period"
    audio = torch.from_numpy(y).float().unsqueeze(0).to(device)
    pitch, confidence = torchcrepe.predict(
        audio, sr, hop_length=int(sr * HOP_MS / 1000),
        fmin=F0_MIN, fmax=F0_MAX, model="tiny", device=device, return_periodicity=True
    )
    f0 = pitch.squeeze().cpu().numpy()
    conf = confidence.squeeze().cpu().numpy()
    voiced = conf >= CREPE_CONF_THRESH
    f0[~voiced] = np.nan
    return f0, conf, voiced


def extract_praat_f0(y, sr):
    # extract F0 using Praat autocorrelation 
    try:
        import parselmouth
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = sound.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX, time_step=HOP_MS/1000)
        times = pitch.xs()
        f0 = np.array([pitch.get_value_at_time(t) for t in times])
        voiced = f0 > 0
        f0[~voiced] = np.nan
        return f0, voiced
    except ImportError:
        # fallback to WORLD if parselmouth not available
        return extract_world_f0(y, sr)


# weighted average consensus as described above
def consensus_f0(f0_praat, f0_world, f0_crepe, crepe_conf, 
                 praat_voiced, world_voiced, crepe_voiced):
    n = max(len(f0_praat), len(f0_world), len(f0_crepe))
    consensus = np.full(n, np.nan)
    
    # pad arrays to same length
    def pad(arr, length):
        if len(arr) < length:
            return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)
        return arr[:length]
    

    f0_praat = pad(f0_praat, n)
    f0_world = pad(f0_world, n)
    f0_crepe = pad(f0_crepe, n)
    crepe_conf = pad(crepe_conf, n)
    praat_voiced = pad(praat_voiced.astype(float), n).astype(bool)
    world_voiced = pad(world_voiced.astype(float), n).astype(bool)
    crepe_voiced = pad(crepe_voiced.astype(float), n).astype(bool)
    
    for i in range(n):
        # voicing decision: at least 2 extractors must agree
        votes = int(praat_voiced[i]) + int(world_voiced[i]) + int(crepe_voiced[i])
        if votes < 2:
            continue  # unvoiced
        
        # weighted average
        vals, weights = [], []
        
        if not np.isnan(f0_praat[i]):
            vals.append(f0_praat[i])
            weights.append(1.0)  # praat weight
        
        if not np.isnan(f0_world[i]):
            vals.append(f0_world[i])
            weights.append(1.2)  # WORLD weight (highest)
        
        if not np.isnan(f0_crepe[i]):
            vals.append(f0_crepe[i])
            # CREPE paper: "The periodicity is computed as the normalized 
            # autocorrelation at the estimated period"
            weights.append(crepe_conf[i])  # CREPE weight = confidence
        
        if vals:
            # weighted average: F0_cons = Σ(w_k × F0_k) / Σ(w_k)
            consensus[i] = np.average(vals, weights=weights)
    
    return consensus


def correct_octave_jumps(log_f0, kernel_size=5):
    # correct octave jumps using median filter
    # octave jumps appear as sudden ±12 semitones (factor of 2 in Hz, ~0.69 in log)
    # median filter smooths these while preserving legitimate pitch movements
    valid = ~np.isnan(log_f0)
    if valid.sum() < kernel_size:
        return log_f0
    
    # apply median filter only to voiced frames
    corrected = log_f0.copy()
    voiced_values = log_f0[valid]
    smoothed = medfilt(voiced_values, kernel_size=kernel_size)
    corrected[valid] = smoothed
    
    return corrected


def clip_outliers(log_f0): 
    # clip to ±3σ per utterance to remove tracking artifacts 
    # done with the thought that 99.7% lies within three standard deviations 
    # "Any data point beyond three standard deviations from the mean is considered an outlier."
    valid = ~np.isnan(log_f0)
    if valid.sum() < 3:
        return log_f0
    
    mu = np.nanmean(log_f0)
    sigma = np.nanstd(log_f0)
    
    clipped = log_f0.copy()
    clipped = np.clip(clipped, mu - 3*sigma, mu + 3*sigma) # clip to ±3σ
    return clipped


def process_file(wav_path, stem, device="cpu"):
    
    y, _ = librosa.load(wav_path, sr=SR)
    
    # extract from all methods
    f0_praat, praat_voiced = extract_praat_f0(y, SR)
    f0_world, world_voiced = extract_world_f0(y, SR)
    f0_crepe, crepe_conf, crepe_voiced = extract_crepe_f0(y, SR, device)
    
    # save individual extractor outputs (raw F0, no post-processing)
    np.save(OUTPUT_PRAAT / f"{stem}_f0.npy", f0_praat.astype(np.float32))
    np.save(OUTPUT_WORLD / f"{stem}_f0.npy", f0_world.astype(np.float32))
    np.save(OUTPUT_CREPE / f"{stem}_f0.npy", f0_crepe.astype(np.float32))
    
    # consensus fusion (weighted average)
    f0 = consensus_f0(f0_praat, f0_world, f0_crepe, crepe_conf,
                      praat_voiced, world_voiced, crepe_voiced)
    
    # log transform: ln(F0 + 1)
    valid = ~np.isnan(f0)
    log_f0 = np.full(len(f0), np.nan)
    log_f0[valid] = np.log(f0[valid] + 1.0)
    
    # octave jump correction: median filter removes sudden ±12 semitone jumps
    log_f0 = correct_octave_jumps(log_f0, kernel_size=5)
    
    # outlier clipping: ±3σ per utterance
    log_f0 = clip_outliers(log_f0)
    
    # interpolate gaps 
    valid = ~np.isnan(log_f0)
    if valid.sum() > 2:
        f0_interp = np.interp(np.arange(len(log_f0)), np.where(valid)[0], log_f0[valid])
    else:
        f0_interp = log_f0
    
    # save consensus (post-processed: log, clipped, interpolated)
    np.save(OUTPUT_CONSENSUS / f"{stem}_f0.npy", f0_interp.astype(np.float32))
    
    return len(f0), valid.sum()


def main():
    TEXTGRID_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/mfa/output")
    
    # create output directories for each extractor and consensus
    for d in [OUTPUT_PRAAT, OUTPUT_WORLD, OUTPUT_CREPE, OUTPUT_CONSENSUS]:
        d.mkdir(parents=True, exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    wav_files = list(AUDIO_DIR.rglob("*.wav"))
    print(f"Found {len(wav_files)} wav files")
    
    processed, skipped = 0, 0
    for wav in wav_files:
        stem = wav.stem
        
        # only process files that have corresponding TextGrids (MFA alignment)
        # this ensures F0/duration/energy files are aligned
        tg_found = any((TEXTGRID_DIR / ch / f"{stem}.TextGrid").exists() 
                       for ch in TEXTGRID_DIR.glob("ch*"))
        if not tg_found:
            skipped += 1
            continue
        
        consensus_out = OUTPUT_CONSENSUS / f"{stem}_f0.npy"
        if consensus_out.exists():
            continue
        
        try:
            frames, voiced = process_file(wav, stem, device)
            print(f"  {stem}: {frames} frames, {voiced} voiced ({voiced/frames*100:.0f}%)")
            processed += 1
        except Exception as e:
            print(f"  {stem}: ERROR {e}")
    
    print(f"\nProcessed: {processed}, Skipped (no TextGrid): {skipped}")
    
    print(f"\nOutput directories:")
    print(f"  Praat:     {OUTPUT_PRAAT}")
    print(f"  WORLD:     {OUTPUT_WORLD}")
    print(f"  CREPE:     {OUTPUT_CREPE}")
    print(f"  Consensus: {OUTPUT_CONSENSUS}")


if __name__ == "__main__":
    main()
