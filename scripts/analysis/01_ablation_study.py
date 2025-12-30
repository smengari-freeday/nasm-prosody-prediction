#!/usr/bin/env python3
"""
Ablation Study for Duration Prediction
=======================================

IEEE-standard ablation study to demonstrate that NASM's duration predictions
capture prosodic timing effects (boundaries, stress) rather than lexical length.

Key question: Is duration prediction dominated by phone count, or does NASM
learn genuine prosodic structure?

Usage:
    python 01_ablation_study.py

Output:
    - results/ablation_results.json
    - results/ablation_table.txt
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Add training scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

# Feature indices (thesis 18-feature format)
FEATURE_INDICES = {
    'primary_stress_pos': 0,
    'word_frequency': 1,
    'syllable_count': 2,
    'phoneme_count': 3,
    'pos_noun': 4,
    'pos_verb': 5,
    'pos_adjective': 6,
    'pos_adverb': 7,
    'syllable_initial': 8,
    'syllable_final': 9,
    'is_stressed': 10,
    'is_phrase_boundary': 11,
    'is_word_boundary': 12,
}

# Ablation groups to test
ABLATION_GROUPS = {
    'phone_count': [3],
    'syllable_count': [2],
    'phone+syllable': [2, 3],
    'stress_features': [0, 10],
    'boundary_features': [11, 12],
    'position_features': [8, 9],
    'all_prosody_cues': [0, 8, 9, 10, 11, 12],
    'lexical_features': [1, 4, 5, 6, 7],
}


class NASM(nn.Module):
    """Neural Additive Spline Model for ablation study"""
    def __init__(self, in_dim, num_basis=8):
        super().__init__()
        self.coef = nn.Parameter(torch.randn(in_dim, num_basis) * 0.01)
        self.linear = nn.Linear(in_dim, 1)
        self.register_buffer('knots', torch.linspace(0, 1, num_basis))
    
    def forward(self, x):
        x_exp = x.unsqueeze(2)
        knots_exp = self.knots.unsqueeze(0).unsqueeze(0)
        basis = torch.exp(-10 * (x_exp - knots_exp) ** 2)
        spline_out = (basis * self.coef.unsqueeze(0)).sum(dim=[1, 2])
        linear_out = self.linear(x).squeeze(1)
        return spline_out + linear_out


def load_data():
    """Load word-level data using the data loader"""
    _dl = {'__file__': str(Path(__file__).parent.parent / 'training' / '01_data_loader.py')}
    exec(open(Path(__file__).parent.parent / 'training' / '01_data_loader.py').read(), _dl)
    
    def collect(split_name):
        phoneme_ds = _dl['PhonemeLevelDataset'](split_name=split_name)
        word_ds = _dl['WordLevelDataset'](phoneme_ds)
        features, targets = [], []
        for i in range(len(word_ds)):
            sample = word_ds[i]
            if sample['targets'] is not None:
                features.append(sample['features'].numpy())
                targets.append(sample['targets'].numpy())
        return np.array(features), np.array(targets)
    
    X_train, y_train = collect('train')
    X_test, y_test = collect('test')
    return X_train, y_train[:, 1], X_test, y_test[:, 1]


def normalize_features(X_train, X_test):
    """Min-max normalize using train statistics only"""
    feat_min = X_train.min(axis=0)
    feat_max = X_train.max(axis=0)
    denom = feat_max - feat_min + 1e-8
    return (X_train - feat_min) / denom, (X_test - feat_min) / denom


def ablate_features(X, remove_indices):
    """Remove specified feature columns"""
    keep = [i for i in range(X.shape[1]) if i not in remove_indices]
    return X[:, keep]


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='linear', epochs=200):
    """Train model and return test R²"""
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X_train_norm, y_train)
        pred = model.predict(X_test_norm)
        return r2_score(y_test, pred)
    
    elif model_type == 'nasm':
        torch.manual_seed(42)
        model = NASM(X_train.shape[1], num_basis=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        X_t = torch.FloatTensor(X_train_norm)
        y_t = torch.FloatTensor(y_train)
        
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_t), y_t)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_test_norm)).numpy()
        return r2_score(y_test, pred)


def run_ablation_study():
    """Run complete ablation study"""
    print("=" * 70)
    print("ABLATION STUDY: Duration Prediction")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    X_train, dur_train, X_test, dur_test = load_data()
    print(f"Train: {len(X_train)} words, Test: {len(X_test)} words")
    
    results = []
    
    # Full model
    print("\nRunning: Full model...")
    r2_lin = train_and_evaluate(X_train, X_test, dur_train, dur_test, 'linear')
    r2_nasm = train_and_evaluate(X_train, X_test, dur_train, dur_test, 'nasm')
    results.append({'name': 'Full (all features)', 'linear': r2_lin, 'nasm': r2_nasm})
    print(f"  Linear: {r2_lin:.4f}, NASM: {r2_nasm:.4f}")
    
    # Ablations
    for abl_name, remove_idx in ABLATION_GROUPS.items():
        print(f"Running: – {abl_name}...")
        X_train_abl = ablate_features(X_train, remove_idx)
        X_test_abl = ablate_features(X_test, remove_idx)
        
        r2_lin = train_and_evaluate(X_train_abl, X_test_abl, dur_train, dur_test, 'linear')
        r2_nasm = train_and_evaluate(X_train_abl, X_test_abl, dur_train, dur_test, 'nasm')
        results.append({'name': f'– {abl_name}', 'linear': r2_lin, 'nasm': r2_nasm})
        print(f"  Linear: {r2_lin:.4f}, NASM: {r2_nasm:.4f}")
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS TABLE")
    print("=" * 70)
    
    full_nasm = results[0]['nasm']
    print(f"\n{'Features Removed':<30} {'Linear':<10} {'NASM':<10} {'Δ NASM':<10} {'Drop':<10}")
    print("-" * 70)
    
    for r in results:
        delta = r['nasm'] - r['linear']
        drop = r['nasm'] - full_nasm if r['name'] != 'Full (all features)' else 0
        marker = '***' if drop < -0.04 else ''
        print(f"{r['name']:<30} {r['linear']:<10.4f} {r['nasm']:<10.4f} {delta:+.4f}     {drop:+.4f} {marker}")
    
    # Save table
    with open(output_dir / 'ablation_table.txt', 'w') as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Features Removed':<30} {'Linear':<10} {'NASM':<10} {'Δ NASM':<10} {'Drop':<10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            delta = r['nasm'] - r['linear']
            drop = r['nasm'] - full_nasm if r['name'] != 'Full (all features)' else 0
            f.write(f"{r['name']:<30} {r['linear']:<10.4f} {r['nasm']:<10.4f} {delta:+.4f}     {drop:+.4f}\n")
    
    print(f"\nResults saved to: {output_dir}")
    return results


if __name__ == '__main__':
    run_ablation_study()


