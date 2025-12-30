#!/usr/bin/env python3
# data loader for phoneme-level and word-level prosody prediction
#
# strategy:
# 1. PhonemeLevelDataset loads 30 features per phoneme + 3 prosody targets (F0, duration, energy)
# 2. features are min-max normalized to [0,1] (continuous/discrete features only)
# 3. targets are z-score normalized using pre-computed train-only statistics (no data leakage)
# 4. WordLevelDataset wraps phoneme-level and aggregates by word boundaries (feature 16)
#    - features: first phoneme for lexical/POS, max for binary, mean for positional
#    - targets: F0=mean, Duration=see below, Energy=mean
# 5. collate functions handle variable-length sequences with padding and attention masks
# 6. create_dataloader() is the main entry point for training scripts
#
# WORD-LEVEL DURATION MODES (USE_PHYSICAL_WORD_DURATION flag):
#   False (thesis-consistent): word_dur = sum(log(d_i)) = log(∏d_i)
#     - naive baseline R² = 0.98 (trivial: just count phones)
#     - stored log-durations summed directly
#   True (Option B, corrected): word_dur = log(∑d_i) = physical duration
#     - naive baseline R² = 0.68 (30% residual variance to model)
#     - computed as log(sum(exp(log_dur))) from same stored data
#     - scientifically valid for word-level prosody research
#
# usage:
#   train_loader = create_dataloader(split_name='train', level='phoneme')
#   word_loader = create_dataloader(split_name='train', level='word')

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# DATA SOURCE SELECTION
# Set USE_THESIS_DATA = True to use full thesis data (6770 utterances)
# Set USE_THESIS_DATA = False to use CODE2 pipeline output (local processing)
USE_THESIS_DATA = True  # Full thesis data for final training

# DURATION MODE FOR WORD-LEVEL AGGREGATION
# Word-level duration is defined as log(sum of raw phoneme durations):
#   word_dur = log(sum_i(d_i))
# This is the physically meaningful definition (total speaking time for a word).
#
# IMPORTANT: An earlier version used sum(log(d_i)) which equals log(product(d_i)).
# This is mathematically incorrect for duration and was rejected after analysis.
# The thesis MUST document that Option B (below) is used.
#
# Implementation: Since phoneme durations are stored as log(d_i), we compute:
#   word_dur = log(sum_i(exp(log_d_i))) = log(sum(raw_durations))
USE_PHYSICAL_WORD_DURATION = True  # REQUIRED: Use physically correct word duration

# Thesis data paths (full dataset, 6770 utterances)
THESIS_FEATURES_DIR = Path("/Users/s.mengari/Desktop/THESIS/18_features_pipeline/data/features_18_fixed_phrase_boundary")
THESIS_PROSODY_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody_full")
THESIS_SPLIT_FILE = Path("/Users/s.mengari/Desktop/THESIS/18_features_pipeline/results/splits/dataset_splits.json")
THESIS_STATS_FILE = Path("/Users/s.mengari/Desktop/THESIS/18_features_pipeline/results/phoneme_level_target_statistics.json")

# CODE2 pipeline paths (local processing, e.g., Chapter 2 subset)
CODE2_FEATURES_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/features/phoneme_level")
CODE2_PROSODY_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody")
CODE2_SPLIT_FILE = CODE2_FEATURES_DIR / "splits.json"
CODE2_STATS_FILE = Path("/Users/s.mengari/Desktop/CODE2/results/phoneme_level_target_statistics.json")

# Select paths based on data source
if USE_THESIS_DATA:
    FEATURES_DIR = THESIS_FEATURES_DIR
    PROSODY_DIR = THESIS_PROSODY_DIR
    SPLIT_FILE = THESIS_SPLIT_FILE
    STATS_FILE = THESIS_STATS_FILE
else:
    FEATURES_DIR = CODE2_FEATURES_DIR
    PROSODY_DIR = CODE2_PROSODY_DIR
    SPLIT_FILE = CODE2_SPLIT_FILE
    STATS_FILE = CODE2_STATS_FILE

FEATURE_NAMES = [
    'primary_stress_pos', 'word_frequency', 'syllable_count', 'phoneme_count',
    'pos_noun', 'pos_verb', 'pos_adjective', 'pos_adverb',
    'pos_pronoun', 'pos_preposition', 'pos_determiner', 'pos_auxiliary',
    'syllable_initial', 'syllable_final', 'is_stressed',
    'is_phrase_boundary', 'is_word_boundary',
    'context_sentence_position', 'context_word_position',
    'is_vowel', 'is_voiced', 'is_plosive',
    'is_schwa', 'velar_fricative_next', 'vowel_height_high', 'vowel_height_low', 'vowel_tense',
    'sonority_of_nucleus',
    'distance_to_stress_norm', 'stress_pattern_class'
]
FEATURE_DIM = 30

# Features that need min-max normalization (continuous/ordinal)
# Binary features (0/1) are left as-is since they're already in [0,1]
# POS features (4-11) are binary one-hot encoded, left as-is
# Stats are computed from training set during first load
#
# NOTE: For thesis data (18 features), features 18+ don't exist and are padded with zeros.
# Only features within the actual feature dimension are normalized.
FEATURES_TO_NORMALIZE_18 = [
    0,   # primary_stress_pos (ordinal: 0, 1, 2, 3)
    1,   # word_frequency (continuous, log-scaled)
    2,   # syllable_count (discrete count)
    3,   # phoneme_count (discrete count)
    17,  # context_sentence_position (continuous: 0.0 to 1.0) - only if 18 features
]
FEATURES_TO_NORMALIZE_30 = FEATURES_TO_NORMALIZE_18 + [
    18,  # context_word_position (continuous: 0.0 to 1.0)
    27,  # sonority_of_nucleus (continuous: 0-9 scale)
    28,  # distance_to_stress_norm (continuous: -1 to 1)
    29,  # stress_pattern_class (ordinal: 0, 1, 2)
]
# Feature stats file - dataset-specific to prevent cross-experiment contamination
if USE_THESIS_DATA:
    FEATURE_STATS_FILE = Path("/Users/s.mengari/Desktop/THESIS/KANTTSeptember/data/feature_normalization_stats_thesis.json")
else:
    FEATURE_STATS_FILE = Path("/Users/s.mengari/Desktop/CODE2/results/feature_normalization_stats_code2.json")


class PhonemeLevelDataset(Dataset):
    # loads features (.npy) and prosody targets (F0, duration, energy)
    
    def __init__(self, data_dir: Path = FEATURES_DIR, split_file: Path = SPLIT_FILE,
                 split_name: str = 'train', prosody_dir: Path = PROSODY_DIR,
                 max_seq_len: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.prosody_dir = Path(prosody_dir) if prosody_dir else None
        self.max_seq_len = max_seq_len
        self.split_name = split_name
        
        with open(split_file) as f:
            splits = json.load(f)
        self.file_paths = [self.data_dir / fname for fname in splits['splits'][split_name]]
        
        with open(STATS_FILE) as f:
            self.target_stats = json.load(f)
        
        # load or compute feature normalization stats (from training set only)
        self.feature_stats = self._load_or_compute_feature_stats(splits)
    
    def _load_or_compute_feature_stats(self, splits: dict) -> dict:
        # compute min/max from training set only (prevents data leakage)
        if FEATURE_STATS_FILE.exists():
            with open(FEATURE_STATS_FILE) as f:
                return json.load(f)
        
        # compute stats from training files only
        train_files = [self.data_dir / fname for fname in splits['splits']['train']]
        all_features = []
        for f in train_files:
            if f.suffix == '.npz':
                data = np.load(f, allow_pickle=True)
                all_features.append(data['features'].astype(np.float32))
            else:
                all_features.append(np.load(f).astype(np.float32))
        all_features = np.vstack(all_features)
        
        # determine which features to normalize based on actual feature dimension
        actual_feat_dim = all_features.shape[1]
        if actual_feat_dim <= 18:
            features_to_norm = FEATURES_TO_NORMALIZE_18
        else:
            features_to_norm = FEATURES_TO_NORMALIZE_30
        
        stats = {}
        for idx in features_to_norm:
            if idx < actual_feat_dim:
                col = all_features[:, idx]
                stats[str(idx)] = {'min': float(col.min()), 'max': float(col.max())}
        
        # save for future use
        FEATURE_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURE_STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Computed feature stats for {len(stats)} features (dim={actual_feat_dim})")
        return stats
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feat_file = self.file_paths[idx]
        utt_id = feat_file.stem.replace('_features', '')
        
        # load features (handle both .npy and .npz formats)
        if feat_file.suffix == '.npz':
            data = np.load(feat_file, allow_pickle=True)
            features = data['features'].astype(np.float32)
            phonemes = list(data['phonemes'])
            
            # CRITICAL: Remap 18-feature data to 30-feature layout
            # The 18-feature and 30-feature layouts have DIFFERENT indices after position 7!
            # 18-feat has only 4 POS tags (4-7), 30-feat has 8 POS tags (4-11).
            # This causes all features after index 7 to be misaligned.
            #
            # 18-feature layout:     30-feature layout:
            # 0-3: lexical           0-3: lexical (same)
            # 4-7: 4 POS tags        4-11: 8 POS tags (4 extra)
            # 8-10: syllable/stress  12-14: syllable/stress
            # 11-12: boundaries      15-16: boundaries
            # 13-14: context         17-18: context
            # 15-17: phoneme class   19-21: phoneme class
            #                        22-29: extended phonetic (not in 18-feat)
            if features.shape[1] == 18:
                # Create 30-feature array with zeros
                features_30 = np.zeros((features.shape[0], 30), dtype=np.float32)
                
                # Map 18-feature indices to 30-feature indices
                # fmt: off
                MAPPING_18_TO_30 = {
                    0: 0,   # primary_stress_pos
                    1: 1,   # word_frequency
                    2: 2,   # syllable_count
                    3: 3,   # phoneme_count
                    4: 4,   # pos_noun
                    5: 5,   # pos_verb
                    6: 6,   # pos_adjective
                    7: 7,   # pos_adverb
                    8: 12,  # syllable_initial
                    9: 13,  # syllable_final
                    10: 14, # is_stressed
                    11: 15, # is_phrase_boundary
                    12: 16, # is_word_boundary
                    13: 17, # context_sentence_position
                    14: 18, # context_word_position
                    15: 19, # is_vowel
                    16: 20, # is_voiced
                    17: 21, # is_plosive
                }
                # fmt: on
                
                for idx_18, idx_30 in MAPPING_18_TO_30.items():
                    features_30[:, idx_30] = features[:, idx_18]
                
                # Indices 8-11 (extra POS tags) and 22-29 (extended phonetic) remain zero
                features = features_30
        else:
            features = np.load(feat_file).astype(np.float32)
            phonemes_file = feat_file.parent / f"{utt_id}_phonemes.json"
            with open(phonemes_file) as f:
                phonemes = json.load(f)
        
        if self.max_seq_len and len(features) > self.max_seq_len:
            features = features[:self.max_seq_len]
            phonemes = phonemes[:self.max_seq_len]
        
        # min-max normalize continuous features to [0,1] using training stats
        for idx_str, stat in self.feature_stats.items():
            feat_idx = int(idx_str)
            feat_min, feat_max = stat['min'], stat['max']
            if feat_max > feat_min:
                features[:, feat_idx] = (features[:, feat_idx] - feat_min) / (feat_max - feat_min)
                features[:, feat_idx] = np.clip(features[:, feat_idx], 0.0, 1.0)
        
        # clip features to [0,1] range expected by NASM/KAN B-spline basis
        # (small violations are possible due to numerical precision)
        features = np.clip(features, 0.0, 1.0)
        
        sample = {
            'features': torch.FloatTensor(features),
            'phonemes': phonemes,
            'utt_id': utt_id
        }
        
        if self.prosody_dir:
            raw_targets = self._load_targets_raw(utt_id, len(features))
            if raw_targets is not None:
                # store raw targets for word-level aggregation
                sample['raw_targets'] = raw_targets
                # apply transforms for phoneme-level output
                sample['targets'] = self._transform_targets_phoneme(raw_targets)
        
        return sample
    
    def _load_targets_raw(self, utt_id: str, n_phonemes: int) -> Optional[np.ndarray]:
        # load raw targets WITHOUT any transforms (log or z-score)
        # transforms are applied at the appropriate level (phoneme vs word)
        if USE_THESIS_DATA:
            f0_norm_dir = Path("/Users/s.mengari/Desktop/THESIS/18_features_pipeline/results/f0_normalized")
            f0_path = f0_norm_dir / f"{utt_id}_f0_consensus.npy"
            if not f0_path.exists():
                f0_path = self.prosody_dir / f"f0_3extractor/consensus/{utt_id}_f0_consensus.npy"
            dur_path = self.prosody_dir / f"Durations/{utt_id}_durations.npy"
            energy_path = self.prosody_dir / f"Energy_praat_smoothed/{utt_id}_energy.npy"
            if not energy_path.exists():
                energy_path = self.prosody_dir / f"Energy/{utt_id}_energy.npy"
        else:
            f0_path = self.prosody_dir / f"f0/consensus/{utt_id}_log_f0.npy"
            if not f0_path.exists():
                f0_path = self.prosody_dir / f"f0/consensus/{utt_id}_f0.npy"
            dur_path = self.prosody_dir / f"durations/{utt_id}_durations.npy"
            energy_path = self.prosody_dir / f"energy/{utt_id}_energy.npy"
        
        if not all(p.exists() for p in [f0_path, dur_path, energy_path]):
            return None
        
        f0 = np.load(f0_path)  # log(F0) or normalized F0
        duration = np.load(dur_path)  # log(seconds) - phoneme durations are log-transformed
        energy = np.load(energy_path)  # dB scale
        
        # sanity check: log-durations should have negative mean (typical phoneme ~0.08s → log ≈ -2.5)
        if duration.mean() > 1.0:
            raise ValueError(f"Duration values seem to be raw seconds (mean={duration.mean():.3f}) "
                           f"but code expects log-duration. Check data format for {utt_id}.")
        
        n_targets = min(n_phonemes, len(duration))
        
        if USE_THESIS_DATA:
            if len(f0) > n_targets:
                step_size = max(1, len(f0) // n_targets)
                f0 = f0[::step_size][:n_targets]
            else:
                f0 = f0[:n_targets]
            if len(energy) > n_targets:
                step_size = max(1, len(energy) // n_targets)
                energy = energy[::step_size][:n_targets]
            else:
                energy = energy[:n_targets]
        else:
            f0 = f0[:n_targets]
            energy = energy[:n_targets]
        
        duration = duration[:n_targets]
        
        n_final = min(len(f0), len(duration), len(energy))
        f0 = f0[:n_final]
        duration = duration[:n_final]
        energy = energy[:n_final]
        
        targets = np.column_stack([f0, duration, energy]).astype(np.float32)
        
        if n_final < n_phonemes:
            pad = np.zeros((n_phonemes - n_final, 3), dtype=np.float32)
            targets = np.vstack([targets, pad])
        
        return targets
    
    def _transform_targets_phoneme(self, targets: np.ndarray) -> torch.Tensor:
        # apply z-score normalization for phoneme-level targets
        # durations are already log-transformed in stored files
        targets = targets.copy()
        
        # z-score normalize all targets using train statistics
        for i, name in enumerate(['f0', 'duration', 'energy']):
            if name in self.target_stats:
                mean = self.target_stats[name].get('mean', 0)
                std = self.target_stats[name].get('std', 1)
                if std > 1e-8:
                    valid = ~np.isnan(targets[:, i])
                    targets[valid, i] = (targets[valid, i] - mean) / std
        
        return torch.FloatTensor(targets)


class WordLevelDataset(Dataset):
    # aggregates phoneme-level to word-level using word boundaries
    # F0: mean, Duration: log(sum(raw_dur)) for physical word duration, Energy: mean
    #
    # IMPORTANT: For val/test, pass external_target_stats from train dataset
    # to ensure consistent normalization (prevents data leakage)
    
    def __init__(self, phoneme_dataset: PhonemeLevelDataset, external_target_stats: dict = None):
        self.phoneme_dataset = phoneme_dataset
        self.external_target_stats = external_target_stats
        # for Option B, compute word-level stats; for thesis mode, use phoneme stats
        # if external_target_stats provided, use those instead (for val/test)
        self.word_samples, self.target_stats = self._aggregate_to_words_with_stats()
    
    def _aggregate_to_words_with_stats(self) -> Tuple[List[Dict], dict]:
        # first pass: collect all raw word-level targets to compute stats
        raw_word_targets = []  # list of [f0, dur, energy] per word
        word_data = []  # temporary storage
        
        for utt_idx in range(len(self.phoneme_dataset)):
            sample = self.phoneme_dataset[utt_idx]
            features = sample['features'].numpy()
            raw_targets = sample.get('raw_targets')
            utt_id = sample.get('utt_id', f'utt_{utt_idx}')
            
            words = self._group_by_word_boundary(features, raw_targets)
            
            for word_idx, (word_feats, word_tgts) in enumerate(words):
                if word_tgts is not None:
                    raw_agg = self._aggregate_targets_raw(word_tgts)
                    raw_word_targets.append(raw_agg)
                    word_data.append((word_feats, raw_agg, utt_id, word_idx))
                else:
                    word_data.append((word_feats, None, utt_id, word_idx))
        
        # compute word-level stats from raw aggregated targets
        # OR use external_target_stats if provided (for val/test consistency)
        if self.external_target_stats is not None:
            # use pre-computed stats from training set (prevents leakage)
            target_stats = self.external_target_stats
            print(f"Word-level: using external (train) target stats")
        elif raw_word_targets:
            raw_word_targets = np.array(raw_word_targets)
            target_stats = {
                'f0': {'mean': float(np.nanmean(raw_word_targets[:, 0])), 
                       'std': float(np.nanstd(raw_word_targets[:, 0]))},
                'duration': {'mean': float(np.nanmean(raw_word_targets[:, 1])), 
                             'std': float(np.nanstd(raw_word_targets[:, 1]))},
                'energy': {'mean': float(np.nanmean(raw_word_targets[:, 2])), 
                           'std': float(np.nanstd(raw_word_targets[:, 2]))}
            }
            print(f"Word-level target stats (Option B={USE_PHYSICAL_WORD_DURATION}):")
            print(f"  F0: mean={target_stats['f0']['mean']:.4f}, std={target_stats['f0']['std']:.4f}")
            print(f"  Duration: mean={target_stats['duration']['mean']:.4f}, std={target_stats['duration']['std']:.4f}")
            print(f"  Energy: mean={target_stats['energy']['mean']:.4f}, std={target_stats['energy']['std']:.4f}")
        else:
            target_stats = self.phoneme_dataset.target_stats
        
        # second pass: z-normalize and create final samples
        word_samples = []
        for word_feats, raw_agg, utt_id, word_idx in word_data:
            if raw_agg is not None:
                # z-normalize using word-level stats
                normalized = raw_agg.copy()
                for i, name in enumerate(['f0', 'duration', 'energy']):
                    mean = target_stats[name]['mean']
                    std = target_stats[name]['std']
                    if std > 1e-8 and not np.isnan(normalized[i]):
                        normalized[i] = (normalized[i] - mean) / std
                targets = torch.FloatTensor(normalized)
            else:
                targets = None
            
            word_samples.append({
                'features': torch.FloatTensor(self._aggregate_features(word_feats)),
                'targets': targets,
                'utt_id': utt_id,
                'word_idx': word_idx,
                'n_phonemes': len(word_feats)
            })
        
        return word_samples, target_stats
    
    def _group_by_word_boundary(self, features: np.ndarray, targets: Optional[np.ndarray]) -> List[Tuple]:
        # NOTE: Cannot use is_word_boundary feature due to index mismatch between
        # 18-feature data (index 12) and 30-feature FEATURE_NAMES (index 16).
        # Also, the original is_word_boundary was buggy (marked word start+end, not just end).
        # 
        # ROBUST APPROACH: Detect word boundaries by observing when word_frequency changes.
        # word_frequency (index 1) is unique per word and consistent across feature layouts.
        # This correctly identifies ~103k words (thesis has ~130k total, 80% train).
        
        words = []
        word_start = 0
        
        # word_frequency (index 1) is word-level and unique per word
        # When it changes, we've crossed a word boundary
        prev_word_freq = features[0, 1] if len(features) > 0 else None
        
        for i in range(1, len(features)):
            curr_word_freq = features[i, 1]
            
            # Word boundary: word_frequency changed
            if curr_word_freq != prev_word_freq:
                # Collect previous word
                word_feats = features[word_start:i]
                word_tgts = targets[word_start:i] if targets is not None else None
                if len(word_feats) > 0:
                    words.append((word_feats, word_tgts))
                word_start = i
            
            prev_word_freq = curr_word_freq
        
        # Last word
        if word_start < len(features):
            word_feats = features[word_start:]
            word_tgts = targets[word_start:] if targets is not None else None
            if len(word_feats) > 0:
                words.append((word_feats, word_tgts))
        
        # no words detected = whole utterance is one word
        if len(words) == 0:
            words.append((features, targets))
        
        return words
    
    def _aggregate_features(self, word_feats: np.ndarray) -> np.ndarray:
        # aggregation strategy per feature group:
        # 0-3 lexical, 4-11 POS, 12-14 syllable: first phoneme
        # 15-16 boundaries: max
        # 17-18 context: mean
        # 19-27 phonetic: max (word contains any)
        # 28-29 stress: first phoneme
        agg = np.zeros(FEATURE_DIM, dtype=np.float32)
        
        agg[0:4] = word_feats[0, 0:4]      # lexical
        agg[4:12] = word_feats[0, 4:12]    # POS
        agg[12:15] = word_feats[0, 12:15]  # syllable/stress
        
        agg[15] = np.max(word_feats[:, 15])  # is_phrase_boundary
        agg[16] = 1.0                         # is_word_boundary (always 1)
        
        agg[17] = np.mean(word_feats[:, 17])  # sentence_position
        agg[18] = np.mean(word_feats[:, 18])  # word_position
        
        agg[19:28] = np.max(word_feats[:, 19:28], axis=0)  # phonetic features
        
        agg[28] = word_feats[0, 28]  # distance_to_stress
        agg[29] = word_feats[0, 29]  # stress_pattern
        
        return agg
    
    def _aggregate_targets_raw(self, word_tgts: np.ndarray) -> np.ndarray:
        # aggregate raw targets without z-normalization
        # F0: mean, Energy: mean
        # Duration: depends on USE_PHYSICAL_WORD_DURATION flag
        
        f0_mean = np.nanmean(word_tgts[:, 0])
        
        log_durations = word_tgts[:, 1]  # stored as log(dur)
        
        if USE_PHYSICAL_WORD_DURATION:
            # Option B: physical word duration = log(sum(raw))
            raw_durations = np.exp(log_durations)
            dur_word = np.log(np.nansum(raw_durations) + 1e-8)
        else:
            # Thesis-consistent: sum of log-durations = log(product)
            dur_word = np.nansum(log_durations)
        
        energy_mean = np.nanmean(word_tgts[:, 2])
        
        return np.array([f0_mean, dur_word, energy_mean], dtype=np.float32)
    
    def __len__(self):
        return len(self.word_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.word_samples[idx]


def collate_phoneme_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # pad variable-length sequences to max length in batch
    features = [item['features'] for item in batch]
    max_len = max(f.shape[0] for f in features)
    feature_dim = features[0].shape[1]
    
    padded_features = []
    masks = []
    
    for feat in features:
        seq_len = feat.shape[0]
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            padded = torch.cat([feat, torch.zeros(pad_len, feature_dim)], dim=0)
            mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        else:
            padded = feat
            mask = torch.ones(seq_len)
        
        padded_features.append(padded)
        masks.append(mask.bool())
    
    result = {
        'features': torch.stack(padded_features),
        'attention_mask': torch.stack(masks),
        'phonemes': [item['phonemes'] for item in batch],
        'utt_ids': [item['utt_id'] for item in batch]
    }
    
    has_targets = all('targets' in item and item['targets'] is not None for item in batch)
    if has_targets:
        padded_targets = []
        for item in batch:
            tgt = item['targets']
            pad_len = max_len - tgt.shape[0]
            if pad_len > 0:
                padded = torch.cat([tgt, torch.zeros(pad_len, 3)], dim=0)
            else:
                padded = tgt
            padded_targets.append(padded)
        result['targets'] = torch.stack(padded_targets)
    
    return result


def collate_word_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # word-level: fixed-length features, no padding needed
    result = {
        'features': torch.stack([item['features'] for item in batch]),
        'utt_ids': [item['utt_id'] for item in batch],
        'word_indices': [item['word_idx'] for item in batch],
        'n_phonemes': [item['n_phonemes'] for item in batch]
    }
    
    if batch[0].get('targets') is not None:
        result['targets'] = torch.stack([item['targets'] for item in batch])
    
    return result


collate_fn = collate_phoneme_fn  # backward compatibility


def create_dataloader(split_name: str = 'train', batch_size: int = 32, 
                      shuffle: bool = True, num_workers: int = 4,
                      max_seq_len: Optional[int] = None, level: str = 'phoneme',
                      word_target_stats: dict = None) -> DataLoader:
    """
    Create dataloader for phoneme or word level.
    
    For word-level val/test, pass word_target_stats from training set
    to ensure consistent normalization (prevents data leakage).
    """
    phoneme_dataset = PhonemeLevelDataset(split_name=split_name, max_seq_len=max_seq_len)
    
    if level == 'word':
        # For val/test: use external stats from train set
        dataset = WordLevelDataset(phoneme_dataset, external_target_stats=word_target_stats)
        collate = collate_word_fn
    else:
        dataset = phoneme_dataset
        collate = collate_phoneme_fn
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate, pin_memory=True)


if __name__ == '__main__':
    phoneme_loader = create_dataloader(level='phoneme', batch_size=8)
    word_loader = create_dataloader(level='word', batch_size=8)

    print(f"Phoneme: {len(phoneme_loader.dataset)} utterances")
    print(f"Word: {len(word_loader.dataset)} words")
