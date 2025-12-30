#!/usr/bin/env python3 -u
# hyperparameter sweep for NASM and MLP
# -u flag forces unbuffered output
#
# fixes applied:
# - loss matches training scripts exactly (energy filtered separately)
# - var_reg only for F0/duration (not energy)
# - multi-seed sweep (2 seeds for stability)
# - expanded LR grid
#
# sweep design:
# - LR: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
# - weight_decay: [0, 1e-5]
# - NASM: var_reg [0, 1e-3]
# - MLP: dropout [0, 0.1]
# - Seeds: [13, 42] (mean RMSE for selection)

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random
import sys
from itertools import product

# paths
SCRIPTS_DIR = Path("/Users/s.mengari/Desktop/CODE2/scripts/training")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/sweeps")

# load modules
_dl = {'__file__': str(SCRIPTS_DIR / "01_data_loader.py")}
exec(open(SCRIPTS_DIR / "01_data_loader.py").read(), _dl)
create_dataloader = _dl['create_dataloader']

_kan = {}
exec(open(SCRIPTS_DIR / "true_kan_heads_vectorized.py").read(), _kan)
TrueKANHead = _kan['TrueKANHead']


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# === NASM Model ===
class NASMModel(nn.Module):
    def __init__(self, num_basis=8, spline_degree=3):
        super().__init__()
        self.kan = TrueKANHead(
            in_dim=30, out_dim=3, num_basis=num_basis, degree=spline_degree,
            grid_range=(0, 1), enable_interpretability=True, learn_base_linear=False
        )
    
    def forward(self, features, mask=None):
        output = self.kan(features)
        if mask is not None:
            output = output * mask.unsqueeze(-1).float()
        return {'f0': output[:,:,0:1], 'duration': output[:,:,1:2], 'energy': output[:,:,2:3]}


# === MLP Model ===
class MLPModel(nn.Module):
    def __init__(self, hidden_dim=8, dropout=0.1):
        super().__init__()
        self.f0 = nn.Sequential(nn.Linear(30, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.duration = nn.Sequential(nn.Linear(30, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.energy = nn.Sequential(nn.Linear(30, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    
    def forward(self, features, mask=None):
        return {'f0': self.f0(features), 'duration': self.duration(features), 'energy': self.energy(features)}


def compute_loss(predictions, targets, mask, lambda_var=0.0, device=None):
    """
    MSE loss matching training scripts exactly:
    - F0, duration: masked by padding only (assume finite)
    - Energy: masked by padding AND isfinite (handles NaN)
    - Normalize by number of active targets (2 or 3)
    - Var reg only on F0/duration (not energy)
    """
    if device is None:
        device = targets.device
    
    mask = mask.unsqueeze(-1).bool()
    losses = []
    active_targets = 0
    
    # F0 and Duration: assume finite, just use mask
    for i, name in enumerate(['f0', 'duration']):
        pred = predictions[name]
        tgt = targets[:, :, i:i+1]
        target_mask = mask.float()
        denom = target_mask.sum() + 1e-8
        
        if denom > 1:
            loss = ((pred - tgt) ** 2 * target_mask).sum() / denom
            losses.append(loss)
            active_targets += 1
    
    # Energy: also filter by isfinite (handles NaN/missing)
    en_pred = predictions['energy']
    en_tgt = targets[:, :, 2:3]
    en_valid = torch.isfinite(en_tgt) & torch.isfinite(en_pred) & mask
    en_mask = en_valid.float()
    en_denom = en_mask.sum() + 1e-8
    
    if en_denom > 1:
        en_loss = ((en_pred - en_tgt) ** 2 * en_mask).sum() / en_denom
        losses.append(en_loss)
        active_targets += 1
    
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # normalize by active targets
    total = sum(losses) / active_targets
    
    # variance regularization (F0 and duration ONLY, not energy)
    if lambda_var > 0:
        flat_mask = mask.squeeze(-1)
        for i, name in enumerate(['f0', 'duration']):  # NOT energy
            pred_flat = predictions[name].squeeze(-1)[flat_mask]
            tgt_flat = targets[:, :, i][flat_mask]
            
            if len(pred_flat) > 1 and len(tgt_flat) > 1:
                var_pred = torch.var(pred_flat) + 0.01
                var_tgt = torch.var(tgt_flat) + 0.01
                var_loss = torch.clamp((torch.log(var_pred / var_tgt)) ** 2, max=10.0)
                total = total + lambda_var * var_loss
    
    return total


def compute_metrics(model, dataloader, device, level='phoneme'):
    """Compute RMSE and R² for each target."""
    model.eval()
    all_preds = {'f0': [], 'duration': [], 'energy': []}
    all_targets = {'f0': [], 'duration': [], 'energy': []}
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            if level == 'word':
                features = features.unsqueeze(1)
                targets = targets.unsqueeze(1)
                mask = torch.ones(features.shape[0], 1, dtype=torch.bool).to(device)
            else:
                mask = batch['attention_mask'].to(device)
            
            predictions = model(features, mask)
            mask_np = mask.bool().cpu().numpy()
            
            for i, name in enumerate(['f0', 'duration', 'energy']):
                pred = predictions[name].squeeze(-1).cpu().numpy()
                tgt = targets[:, :, i].cpu().numpy()
                for b in range(pred.shape[0]):
                    m = mask_np[b]
                    valid = m & np.isfinite(tgt[b]) & np.isfinite(pred[b])
                    if valid.sum() > 0:
                        all_preds[name].extend(pred[b][valid])
                        all_targets[name].extend(tgt[b][valid])
    
    metrics = {}
    rmse_sum = 0.0
    
    for name in ['f0', 'duration', 'energy']:
        if len(all_targets[name]) > 0:
            p = np.array(all_preds[name])
            t = np.array(all_targets[name])
            
            rmse = np.sqrt(np.mean((t - p) ** 2))
            rmse_sum += rmse
            
            ss_res = np.sum((t - p) ** 2)
            ss_tot = np.sum((t - np.mean(t)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            metrics[name] = {'r2': float(r2), 'rmse': float(rmse)}
        else:
            metrics[name] = {'r2': 0.0, 'rmse': float('inf')}
            rmse_sum += float('inf')
    
    metrics['rmse_sum'] = rmse_sum
    return metrics


def train_model(model, config, train_loader, val_loader, device, level='phoneme', verbose=False):
    """Train model, return best validation metrics."""
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_rmse_sum = float('inf')
    best_metrics = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch}/{config['epochs']}", flush=True)
        model.train()
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            if level == 'word':
                features = features.unsqueeze(1)
                targets = targets.unsqueeze(1)
                mask = torch.ones(features.shape[0], 1, dtype=torch.bool).to(device)
            else:
                mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            predictions = model(features, mask)
            loss = compute_loss(predictions, targets, mask, 
                              lambda_var=config.get('lambda_var', 0.0),
                              device=device)
            
            if loss is not None and torch.isfinite(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        scheduler.step()
        
        val_metrics = compute_metrics(model, val_loader, device, level)
        
        if val_metrics['rmse_sum'] < best_rmse_sum:
            best_rmse_sum = val_metrics['rmse_sum']
            best_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    return best_metrics


def run_sweep(level: str):
    """Run hyperparameter sweep for given level."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # base config (reduced epochs for faster sweep - full training uses 100)
    base = {'epochs': 30, 'patience': 10, 'batch_size': 32}
    
    # sweep grid
    lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # expanded LR grid
    wd_values = [0.0, 1e-5]
    var_values = [0.0, 1e-3]      # NASM only
    dropout_values = [0.0, 0.1]   # MLP only
    sweep_seeds = [42]            # single seed for speed
    
    # load data once
    print(f"Loading {level}-level data...")
    train_loader = create_dataloader('train', batch_size=32, shuffle=True, num_workers=0, level=level)
    word_stats = train_loader.dataset.target_stats if level == 'word' else None
    val_loader = create_dataloader('val', batch_size=32, shuffle=False, num_workers=0, level=level, word_target_stats=word_stats)
    
    results = {'level': level, 'base_config': base, 'sweep_seeds': sweep_seeds, 'nasm': [], 'mlp': []}
    
    # === NASM Sweep (5 lr × 2 wd × 2 var = 20 configs) ===
    print(f"\n=== NASM {level}-level sweep (20 configs, seed={sweep_seeds[0]}) ===")
    nasm_configs = list(product(lr_values, wd_values, var_values))
    
    for idx, (lr, wd, var_reg) in enumerate(nasm_configs):
        print(f"  [{idx+1}/20] lr={lr:.0e}, wd={wd:.0e}, var={var_reg:.0e}...", flush=True)
        seed_results = []
        for seed in sweep_seeds:
            set_seed(seed)
            config = {**base, 'lr': lr, 'weight_decay': wd, 'lambda_var': var_reg}
            model = NASMModel(num_basis=8, spline_degree=3).to(device)
            verbose = (idx == 0)  # verbose only for first config
            metrics = train_model(model, config, train_loader, val_loader, device, level, verbose=verbose)
            seed_results.append(metrics['rmse_sum'])
        
        mean_rmse = np.mean(seed_results)
        result = {
            'lr': lr, 'weight_decay': wd, 'lambda_var': var_reg,
            'rmse_per_seed': seed_results, 'mean_rmse': mean_rmse
        }
        results['nasm'].append(result)
        print(f"  lr={lr:.0e}, wd={wd:.0e}, var={var_reg:.0e} → RMSE={mean_rmse:.3f}", flush=True)
    
    # === MLP Sweep (5 lr × 2 wd × 2 drop = 20 configs) ===
    print(f"\n=== MLP {level}-level sweep (20 configs, seed={sweep_seeds[0]}) ===")
    mlp_configs = list(product(lr_values, wd_values, dropout_values))
    
    for idx, (lr, wd, dropout) in enumerate(mlp_configs):
        print(f"  [{idx+1}/20] lr={lr:.0e}, wd={wd:.0e}, drop={dropout}...", flush=True)
        seed_results = []
        for seed in sweep_seeds:
            set_seed(seed)
            config = {**base, 'lr': lr, 'weight_decay': wd}
            model = MLPModel(hidden_dim=8, dropout=dropout).to(device)
            metrics = train_model(model, config, train_loader, val_loader, device, level)
            seed_results.append(metrics['rmse_sum'])
        
        mean_rmse = np.mean(seed_results)
        result = {
            'lr': lr, 'weight_decay': wd, 'dropout': dropout,
            'rmse_per_seed': seed_results, 'mean_rmse': mean_rmse
        }
        results['mlp'].append(result)
        print(f"  lr={lr:.0e}, wd={wd:.0e}, drop={dropout} → RMSE={mean_rmse:.3f}", flush=True)
    
    # === Best Configs ===
    print("\n" + "=" * 70)
    print(f"BEST CONFIGS FOR {level.upper()}-LEVEL (seed={sweep_seeds[0]})")
    print("=" * 70)
    
    best_nasm = min(results['nasm'], key=lambda x: x['mean_rmse'])
    best_mlp = min(results['mlp'], key=lambda x: x['mean_rmse'])
    
    print(f"\nNASM best: lr={best_nasm['lr']:.0e}, wd={best_nasm['weight_decay']:.0e}, var_reg={best_nasm['lambda_var']:.0e}")
    print(f"           mean_RMSE={best_nasm['mean_rmse']:.3f}, per_seed={best_nasm['rmse_per_seed']}")
    
    print(f"\nMLP best:  lr={best_mlp['lr']:.0e}, wd={best_mlp['weight_decay']:.0e}, dropout={best_mlp['dropout']}")
    print(f"           mean_RMSE={best_mlp['mean_rmse']:.3f}, per_seed={best_mlp['rmse_per_seed']}")
    
    results['best_nasm'] = best_nasm
    results['best_mlp'] = best_mlp
    
    # save
    out_file = OUTPUT_DIR / f"sweep_{level}_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep')
    parser.add_argument('--level', type=str, default='word', choices=['phoneme', 'word'])
    args = parser.parse_args()
    
    run_sweep(args.level)


if __name__ == "__main__":
    main()
