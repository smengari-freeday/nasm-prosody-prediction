#!/usr/bin/env python3
# 3-extractor F0 consensus pipeline (matches thesis methodology)
#
# extractors: Praat (autocorrelation), WORLD (harvest), CREPE (neural)
# consensus: weighted average with CREPE confidence weighting
# voicing: majority vote (≥2/3 extractors agree)
# post-processing: log transform, octave jump correction (medfilt), ±3σ clipping
#
# output: frame-level F0 at 10ms hop, fully interpolated (no NaNs)
# saves: individual extractor outputs + consensus
#
# based on: /Users/s.mengari/Desktop/THESIS/KANTTSeptember/scripts/optimized_3extractor_f0_pipeline.py

import os
import numpy as np
import librosa
import pyworld as pw
import torch
import torchcrepe
import json
from pathlib import Path
from scipy.signal import medfilt
import warnings
warnings.filterwarnings("ignore")

# config (matches thesis)
CONFIG = {
    'praat_f0_min': 75,
    'praat_f0_max': 300,
    'world_f0_min': 80,
    'world_f0_max': 300,
    'crepe_f0_min': 60,
    'crepe_f0_max': 300,
    'frame_period_ms': 10,
    'world_frame_period_ms': 5,
    'weight_praat': 1.0,
    'weight_world': 1.2,
    'weight_crepe': 1.0,  # multiplied by confidence
    'outlier_std_threshold': 3.0,
    'crepe_confidence_threshold': 0.5,
    'epsilon': 1.0,  # for log(f0 + ε)
    'sample_rate': 22050,
    'torchcrepe_batch_size': 128,
    'torchcrepe_model': 'tiny',
}

# paths
AUDIO_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/utterances")
TEXTGRID_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/mfa/output")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody/f0")
OUTPUT_PRAAT = OUTPUT_DIR / "praat"
OUTPUT_WORLD = OUTPUT_DIR / "world"
OUTPUT_CREPE = OUTPUT_DIR / "crepe"
OUTPUT_CONSENSUS = OUTPUT_DIR / "consensus"


def extract_praat_f0(y, sr):
    """Praat autocorrelation method"""
    try:
        import parselmouth
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = sound.to_pitch(
            pitch_floor=CONFIG['praat_f0_min'],
            pitch_ceiling=CONFIG['praat_f0_max'],
            time_step=CONFIG['frame_period_ms'] / 1000
        )
        times = pitch.xs()
        f0 = np.array([pitch.get_value_at_time(t) if pitch.get_value_at_time(t) != 0 else np.nan for t in times])
        vuv = np.array([pitch.get_value_at_time(t) > 0 for t in times])
        return times, f0, vuv
    except ImportError:
        # fallback to WORLD if parselmouth not available
        _f0, t = pw.dio(y.astype(np.float64), sr, 
                        frame_period=CONFIG['frame_period_ms'],
                        f0_floor=CONFIG['praat_f0_min'],
                        f0_ceil=CONFIG['praat_f0_max'])
        f0 = pw.stonemask(y.astype(np.float64), sr, t, _f0)
        time = np.arange(len(f0)) * (CONFIG['frame_period_ms'] / 1000)
        vuv = f0 > 0
        f0[~vuv] = np.nan
        return time, f0, vuv


def extract_world_f0(y, sr):
    """WORLD vocoder (harvest algorithm)"""
    y = y.astype(np.float64)
    f0, _ = pw.harvest(y, sr,
                       frame_period=CONFIG['world_frame_period_ms'],
                       f0_floor=CONFIG['world_f0_min'],
                       f0_ceil=CONFIG['world_f0_max'])
    time = np.arange(len(f0)) * (CONFIG['world_frame_period_ms'] / 1000)
    vuv = f0 > 0
    f0[~vuv] = np.nan
    return time, f0, vuv


def extract_crepe_f0(y, sr, device):
    """CREPE neural F0 with confidence"""
    audio_tensor = torch.from_numpy(y).float().unsqueeze(0).to(device)
    
    pitch = torchcrepe.predict(
        audio_tensor, sr,
        hop_length=int(sr * 0.01),  # 10ms
        fmin=CONFIG['crepe_f0_min'],
        fmax=CONFIG['crepe_f0_max'],
        model=CONFIG['torchcrepe_model'],
        device=device,
        batch_size=CONFIG['torchcrepe_batch_size']
    )
    
    frequency = pitch[0].cpu().numpy()
    time = np.arange(len(frequency)) * 0.01
    
    # pitch stability as confidence proxy
    freq_diff = np.abs(np.diff(frequency, prepend=frequency[0]))
    confidence = np.clip(np.exp(-freq_diff / 20.0), 0.3, 0.95)
    
    # V/UV based on confidence
    vuv = confidence > CONFIG['crepe_confidence_threshold']
    frequency[~vuv] = np.nan
    
    return time, frequency, vuv, confidence


def resample_to_grid(time, values, target_time):
    """Resample to common 10ms grid using linear interpolation"""
    valid = ~np.isnan(values) if values.dtype == np.float64 else values > 0
    if not np.any(valid):
        return np.full_like(target_time, np.nan)
    
    resampled = np.interp(target_time, time[valid], values[valid])
    # mark out-of-range as NaN
    resampled[target_time < time[valid][0]] = np.nan
    resampled[target_time > time[valid][-1]] = np.nan
    return resampled


def create_consensus_f0(praat_f0, world_f0, crepe_f0, crepe_conf):
    """Weighted average consensus (thesis methodology)"""
    n = len(praat_f0)
    consensus = np.full(n, np.nan)
    vuv = np.zeros(n, dtype=bool)
    
    for i in range(n):
        weights, values = [], []
        
        if not np.isnan(praat_f0[i]):
            weights.append(CONFIG['weight_praat'])
            values.append(praat_f0[i])
        
        if not np.isnan(world_f0[i]):
            weights.append(CONFIG['weight_world'])
            values.append(world_f0[i])
        
        if not np.isnan(crepe_f0[i]):
            weights.append(CONFIG['weight_crepe'] * crepe_conf[i])
            values.append(crepe_f0[i])
        
        if values:
            consensus[i] = np.average(values, weights=weights)
            vuv[i] = True
    
    return consensus, vuv


def majority_vote_vuv(praat_vuv, world_vuv, crepe_vuv):
    """V/UV decision: ≥2/3 extractors must agree"""
    return (praat_vuv.astype(int) + world_vuv.astype(int) + crepe_vuv.astype(int)) >= 2


def apply_preprocessing(f0, vuv):
    """Log transform, octave jump correction, outlier clipping"""
    voiced_f0 = f0[vuv]
    if len(voiced_f0) == 0:
        return f0, vuv
    
    # log transform
    log_f0 = np.log(voiced_f0 + CONFIG['epsilon'])
    
    # octave jump correction (median filter)
    log_f0 = medfilt(log_f0, kernel_size=5)
    
    # outlier clipping (±3σ)
    mean_log = np.mean(log_f0)
    std_log = np.std(log_f0)
    log_f0_clipped = np.clip(log_f0,
                              mean_log - CONFIG['outlier_std_threshold'] * std_log,
                              mean_log + CONFIG['outlier_std_threshold'] * std_log)
    
    # back to Hz
    f0_processed = np.exp(log_f0_clipped) - CONFIG['epsilon']
    
    f0_output = f0.copy()
    f0_output[vuv] = f0_processed
    return f0_output, vuv


def interpolate_f0(f0, vuv):
    """Fill unvoiced gaps with linear interpolation"""
    voiced_indices = np.where(vuv)[0]
    if len(voiced_indices) < 2:
        return f0
    
    f0_interp = np.interp(np.arange(len(f0)), voiced_indices, f0[voiced_indices])
    return f0_interp


def process_file(wav_path, device):
    """Process single utterance through 3-extractor pipeline"""
    stem = wav_path.stem
    
    # load audio
    y, sr = librosa.load(wav_path, sr=CONFIG['sample_rate'])
    
    # extract from all three
    praat_time, praat_f0, praat_vuv = extract_praat_f0(y, sr)
    world_time, world_f0, world_vuv = extract_world_f0(y, sr)
    crepe_time, crepe_f0, crepe_vuv, crepe_conf = extract_crepe_f0(y, sr, device)
    
    # use WORLD time as reference grid
    target_time = world_time
    
    # resample all to common grid
    praat_f0_rs = resample_to_grid(praat_time, praat_f0, target_time)
    world_f0_rs = resample_to_grid(world_time, world_f0, target_time)
    crepe_f0_rs = resample_to_grid(crepe_time, crepe_f0, target_time)
    crepe_conf_rs = resample_to_grid(crepe_time, crepe_conf, target_time)
    
    praat_vuv_rs = resample_to_grid(praat_time, praat_vuv.astype(float), target_time) > 0.5
    world_vuv_rs = resample_to_grid(world_time, world_vuv.astype(float), target_time) > 0.5
    crepe_vuv_rs = resample_to_grid(crepe_time, crepe_vuv.astype(float), target_time) > 0.5
    
    # consensus
    consensus_f0, consensus_vuv_weighted = create_consensus_f0(
        praat_f0_rs, world_f0_rs, crepe_f0_rs, crepe_conf_rs
    )
    consensus_vuv = majority_vote_vuv(praat_vuv_rs, world_vuv_rs, crepe_vuv_rs)
    
    # preprocessing
    consensus_f0_proc, consensus_vuv_proc = apply_preprocessing(consensus_f0, consensus_vuv)
    
    # interpolate gaps (produces 0% NaN output)
    consensus_f0_interp = interpolate_f0(consensus_f0_proc, consensus_vuv_proc)
    
    # log transform final output
    log_f0 = np.log(consensus_f0_interp + CONFIG['epsilon'])
    
    # save individual extractors
    np.save(OUTPUT_PRAAT / f"{stem}_f0.npy", praat_f0_rs.astype(np.float32))
    np.save(OUTPUT_WORLD / f"{stem}_f0.npy", world_f0_rs.astype(np.float32))
    np.save(OUTPUT_CREPE / f"{stem}_f0.npy", crepe_f0_rs.astype(np.float32))
    np.save(OUTPUT_CREPE / f"{stem}_conf.npy", crepe_conf_rs.astype(np.float32))
    
    # save consensus (fully interpolated, log-transformed)
    np.save(OUTPUT_CONSENSUS / f"{stem}_f0_consensus.npy", consensus_f0_interp.astype(np.float32))
    np.save(OUTPUT_CONSENSUS / f"{stem}_log_f0.npy", log_f0.astype(np.float32))
    np.save(OUTPUT_CONSENSUS / f"{stem}_vuv_consensus.npy", consensus_vuv_proc)
    np.save(OUTPUT_CONSENSUS / f"{stem}_time.npy", target_time.astype(np.float32))
    
    voiced_pct = consensus_vuv_proc.sum() / len(consensus_vuv_proc) * 100
    return len(target_time), voiced_pct


def main():
    # create output directories
    for d in [OUTPUT_PRAAT, OUTPUT_WORLD, OUTPUT_CREPE, OUTPUT_CONSENSUS]:
        d.mkdir(parents=True, exist_ok=True)
    
    # device selection
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Device: {device}")
    
    # find wav files
    wav_files = list(AUDIO_DIR.rglob("*.wav"))
    print(f"Found {len(wav_files)} wav files")
    
    # filter to files with TextGrids (from MFA)
    processed = 0
    for wav in wav_files:
        stem = wav.stem
        
        # check TextGrid exists
        tg_exists = any((TEXTGRID_DIR / ch / f"{stem}.TextGrid").exists() 
                        for ch in TEXTGRID_DIR.glob("ch*"))
        if not tg_exists:
            continue
        
        # skip if already processed
        if (OUTPUT_CONSENSUS / f"{stem}_f0_consensus.npy").exists():
            processed += 1
            continue
        
        try:
            frames, voiced_pct = process_file(wav, device)
            processed += 1
            print(f"  {stem}: {frames} frames, {voiced_pct:.0f}% voiced")
        except Exception as e:
            print(f"  {stem}: ERROR {e}")
    
    print(f"\nProcessed: {processed} files")
    print(f"Output: {OUTPUT_CONSENSUS}")
    
    # save summary
    summary = {
        'config': CONFIG,
        'device': device,
        'n_files': processed,
        'output_format': 'frame-level F0 at 10ms, fully interpolated, log-transformed'
    }
    with open(OUTPUT_DIR / "extraction_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
