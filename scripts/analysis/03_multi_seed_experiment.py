#!/usr/bin/env python3
"""
Multi-Seed Experiment for Statistical Significance
===================================================

Runs NASM and MLP training with multiple random seeds to compute:
- Mean and standard deviation of test R²
- 95% confidence intervals
- Paired statistical tests (NASM vs Linear, NASM vs MLP)

Usage:
    python 03_multi_seed_experiment.py

Output:
    - results/multi_seed_results.json
    - results/multi_seed_table.txt
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

# Add training scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

SEEDS = [42, 43, 44, 45, 46, 47]
EPOCHS = 200
LR = 1e-3


class NASM(nn.Module):
    """Neural Additive Spline Model"""
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
        return spline_out + self.linear(x).squeeze(1)


class MLP(nn.Module):
    """Parameter-matched MLP baseline"""
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(1)


def load_data():
    """Load word-level data"""
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


def normalize(X_train, X_test):
    """Min-max normalize using train statistics"""
    feat_min = X_train.min(axis=0)
    feat_max = X_train.max(axis=0)
    denom = feat_max - feat_min + 1e-8
    return (X_train - feat_min) / denom, (X_test - feat_min) / denom


def train_model(model, X_train, y_train, X_test, y_test):
    """Train model and return test R²"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_test)).numpy()
    return r2_score(y_test, pred)


def run_multi_seed_experiment():
    """Run multi-seed experiment"""
    print("=" * 70)
    print("MULTI-SEED EXPERIMENT")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    X_train, dur_train, X_test, dur_test = load_data()
    X_train_norm, X_test_norm = normalize(X_train, X_test)
    print(f"Train: {len(X_train)} words, Test: {len(X_test)} words")
    
    results = {'nasm': [], 'mlp': []}
    
    # Linear baseline (deterministic)
    model_linear = LinearRegression()
    model_linear.fit(X_train_norm, dur_train)
    r2_linear = r2_score(dur_test, model_linear.predict(X_test_norm))
    results['linear'] = [r2_linear]
    
    print(f"\nLinear baseline: R² = {r2_linear:.4f}")
    
    # Multi-seed runs
    print(f"\nRunning {len(SEEDS)} seeds...")
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        nasm = NASM(X_train.shape[1])
        r2_nasm = train_model(nasm, X_train_norm, dur_train, X_test_norm, dur_test)
        results['nasm'].append(r2_nasm)
        
        mlp = MLP(X_train.shape[1])
        r2_mlp = train_model(mlp, X_train_norm, dur_train, X_test_norm, dur_test)
        results['mlp'].append(r2_mlp)
        
        print(f"  Seed {seed}: NASM={r2_nasm:.4f}, MLP={r2_mlp:.4f}")
    
    # Compute statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    summary = {}
    print(f"\n{'Model':<12} {'Mean R²':<12} {'Std':<12} {'95% CI':<25}")
    print("-" * 65)
    
    for name in ['linear', 'nasm', 'mlp']:
        scores = np.array(results[name])
        mean = scores.mean()
        std = scores.std() if len(scores) > 1 else 0
        n = len(scores)
        ci_low = mean - 1.96 * std / np.sqrt(n) if n > 1 else mean
        ci_high = mean + 1.96 * std / np.sqrt(n) if n > 1 else mean
        
        summary[name] = {'mean': float(mean), 'std': float(std), 'ci': [float(ci_low), float(ci_high)]}
        print(f"{name.upper():<12} {mean:<12.4f} {std:<12.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Statistical tests
    print("\n" + "-" * 65)
    print("STATISTICAL COMPARISONS")
    print("-" * 65)
    
    nasm_arr = np.array(results['nasm'])
    mlp_arr = np.array(results['mlp'])
    
    # NASM vs Linear
    diff = nasm_arr - r2_linear
    t_stat, p_val = stats.ttest_1samp(diff, 0)
    print(f"\nNASM vs Linear: diff={np.mean(diff):+.4f}, t={t_stat:.2f}, p={p_val:.2e}")
    
    # NASM vs MLP
    diff = nasm_arr - mlp_arr
    t_stat, p_val = stats.ttest_rel(nasm_arr, mlp_arr)
    print(f"NASM vs MLP: diff={np.mean(diff):+.4f}, t={t_stat:.2f}, p={p_val:.2e}")
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'multi_seed_results.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'results': results, 'summary': summary}, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'multi_seed_results.json'}")
    return results, summary


if __name__ == '__main__':
    run_multi_seed_experiment()


