#!/usr/bin/env python3
# data loader for phoneme-level and word-level prosody prediction
#
# strategy:
# 1. PhonemeLevelDataset loads 30 features per phoneme + 3 prosody targets (F0, duration, energy)
# 2. features are min-max normalized to [0,1] (continuous/discrete features only)
# 3. targets are z-score normalized using pre-computed train-only statistics (no data leakage)
# 4. WordLevelDataset wraps phoneme-level and aggregates by word boundaries (feature 16)
#    - features: first phoneme for lexical/POS, max for binary, mean for positional
#    - targets: F0=mean, Duration=sum, Energy=mean
# 5. collate functions handle variable-length sequences with padding and attention masks
# 6. create_dataloader() is the main entry point for training scripts
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
USE_THESIS_DATA = True  # Switch to thesis data for comparison

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

# Features that need min-max normalization (continuous/discrete)
# Binary features (0/1) are left as-is
# Stats are computed from training set during first load
FEATURES_TO_NORMALIZE = [1, 2, 3]  # word_frequency, syllable_count, phoneme_count
FEATURE_STATS_FILE = Path("/Users/s.mengari/Desktop/CODE2/results/feature_normalization_stats.json")


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
        
        stats = {}
        for idx in FEATURES_TO_NORMALIZE:
            col = all_features[:, idx]
            stats[str(idx)] = {'min': float(col.min()), 'max': float(col.max())}
        
        # save for future use
        FEATURE_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURE_STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        
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
            
            # convert 18 features to 30 by padding with zeros
            # thesis used 18 base features, CODE2 expects 30
            if features.shape[1] == 18:
                # pad with 12 zero columns (features 18-29)
                padding = np.zeros((features.shape[0], 12), dtype=np.float32)
                features = np.hstack([features, padding])
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
        
        sample = {
            'features': torch.FloatTensor(features),
            'phonemes': phonemes,
            'utt_id': utt_id
        }
        
        if self.prosody_dir:
            targets = self._load_targets(utt_id, len(features))
            if targets is not None:
                sample['targets'] = targets
        
        return sample
    
    def _load_targets(self, utt_id: str, n_phonemes: int) -> Optional[torch.Tensor]:
        # handle both thesis data format and CODE2 pipeline format
        if USE_THESIS_DATA:
            # Thesis format: frame-level F0/energy, phoneme-level duration
            f0_norm_dir = Path("/Users/s.mengari/Desktop/THESIS/18_features_pipeline/results/f0_normalized")
            f0_path = f0_norm_dir / f"{utt_id}_f0_consensus.npy"
            if not f0_path.exists():
                f0_path = self.prosody_dir / f"f0_3extractor/consensus/{utt_id}_f0_consensus.npy"
            
            dur_path = self.prosody_dir / f"Durations/{utt_id}_durations.npy"
            
            # prefer smoothed Praat energy
            energy_path = self.prosody_dir / f"Energy_praat_smoothed/{utt_id}_energy.npy"
            if not energy_path.exists():
                energy_path = self.prosody_dir / f"Energy/{utt_id}_energy.npy"
        else:
            # CODE2 pipeline format: frame-level F0 (matches thesis after fix)
            # use log_f0 (already log-transformed and interpolated)
            f0_path = self.prosody_dir / f"f0/consensus/{utt_id}_log_f0.npy"
            if not f0_path.exists():
                # fallback to old naming if not yet re-extracted
                f0_path = self.prosody_dir / f"f0/consensus/{utt_id}_f0.npy"
            dur_path = self.prosody_dir / f"durations/{utt_id}_durations.npy"
            energy_path = self.prosody_dir / f"energy/{utt_id}_energy.npy"
        
        if not all(p.exists() for p in [f0_path, dur_path, energy_path]):
            return None
        
        f0 = np.load(f0_path)
        duration = np.load(dur_path)
        energy = np.load(energy_path)
        
        # align to phoneme count
        n_targets = min(n_phonemes, len(duration))
        
        if USE_THESIS_DATA:
            # thesis: subsample F0 from frame-level to phoneme-level
            if len(f0) > n_targets:
                step_size = max(1, len(f0) // n_targets)
                f0 = f0[::step_size][:n_targets]
            else:
                f0 = f0[:n_targets]
            
            # thesis: subsample energy from frame-level to phoneme-level
            if len(energy) > n_targets:
                step_size = max(1, len(energy) // n_targets)
                energy = energy[::step_size][:n_targets]
            else:
                energy = energy[:n_targets]
        else:
            # CODE2: already phoneme-level, just truncate
            f0 = f0[:n_targets]
            energy = energy[:n_targets]
        
        duration = duration[:n_targets]
        
        # ensure all arrays are same length
        n_final = min(len(f0), len(duration), len(energy))
        f0 = f0[:n_final]
        duration = duration[:n_final]
        energy = energy[:n_final]
        
        targets = np.column_stack([f0, duration, energy]).astype(np.float32)
        n_targets = n_final
        
        # pad if targets shorter than features
        if n_targets < n_phonemes:
            pad = np.zeros((n_phonemes - n_targets, 3), dtype=np.float32)
            targets = np.vstack([targets, pad])
        
        # z-score normalize using train statistics
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
    # F0: mean, Duration: sum, Energy: mean
    
    def __init__(self, phoneme_dataset: PhonemeLevelDataset):
        self.phoneme_dataset = phoneme_dataset
        self.word_samples = self._aggregate_to_words()
    
    def _aggregate_to_words(self) -> List[Dict]:
        word_samples = []
        
        for utt_idx in range(len(self.phoneme_dataset)):
            sample = self.phoneme_dataset[utt_idx]
            features = sample['features'].numpy()
            targets = sample.get('targets')
            if targets is not None:
                targets = targets.numpy()
            utt_id = sample.get('utt_id', f'utt_{utt_idx}')
            
            # split by is_word_boundary (feature 16)
            words = self._group_by_word_boundary(features, targets)
            
            for word_idx, (word_feats, word_tgts) in enumerate(words):
                word_samples.append({
                    'features': torch.FloatTensor(self._aggregate_features(word_feats)),
                    'targets': torch.FloatTensor(self._aggregate_targets(word_tgts)) if word_tgts is not None else None,
                    'utt_id': utt_id,
                    'word_idx': word_idx,
                    'n_phonemes': len(word_feats)
                })
        
        return word_samples
    
    def _group_by_word_boundary(self, features: np.ndarray, targets: Optional[np.ndarray]) -> List[Tuple]:
        is_word_boundary = features[:, 16]
        boundary_indices = np.where(is_word_boundary > 0.5)[0]
        
        words = []
        prev_idx = 0
        
        for boundary_idx in boundary_indices:
            if boundary_idx > prev_idx:
                word_feats = features[prev_idx:boundary_idx]
                word_tgts = targets[prev_idx:boundary_idx] if targets is not None else None
                words.append((word_feats, word_tgts))
            prev_idx = boundary_idx + 1
        
        # last word after final boundary
        if prev_idx < len(features):
            word_feats = features[prev_idx:]
            word_tgts = targets[prev_idx:] if targets is not None else None
            words.append((word_feats, word_tgts))
        
        # no boundaries = whole utterance is one word
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
    
    def _aggregate_targets(self, word_tgts: np.ndarray) -> np.ndarray:
        # F0: mean, Duration: sum, Energy: mean
        return np.array([
            np.nanmean(word_tgts[:, 0]),
            np.nansum(word_tgts[:, 1]),
            np.nanmean(word_tgts[:, 2])
        ], dtype=np.float32)
    
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
                      max_seq_len: Optional[int] = None, level: str = 'phoneme') -> DataLoader:
    phoneme_dataset = PhonemeLevelDataset(split_name=split_name, max_seq_len=max_seq_len)
    
    if level == 'word':
        dataset = WordLevelDataset(phoneme_dataset)
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
