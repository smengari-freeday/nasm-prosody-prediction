#!/usr/bin/env python3
# KAN-style training for prosody prediction
# NASM = Neural Additive Spline Model (single-layer KAN without stacking)
#
# architecture:
# - each input feature has its own learned spline curve plus per-feature linear residual
# - outputs are simply the sum of all feature contributions (additive, no interactions)
# - fully interpretable: can plot how each feature affects each prosody target
# - 813 parameters in default config: 30×3×8 spline coefficients + 30×3 linear residuals + 3 bias
#
# training strategy:
# 1. MSE loss on z-normalized targets (train-only stats, variance-balanced by active target count)
# 2. Adam optimizer with cosine annealing LR schedule
# 3. early stopping on sum of per-target RMSE (not a single joint RMSE)
# 4. gradient clipping for stability
# 5. masking applied in both forward and loss; NaN targets filtered per-target
#
# usage:
#   python 02_train_kan.py --epochs 50 --lr 1e-4 --seed 42
#   python 02_train_kan.py --epochs 50 --lr 1e-4 --seed 42 --level word  # faster, word-level

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import random

# paths
SCRIPTS_DIR = Path("/Users/s.mengari/Desktop/CODE2/scripts/training")
DATA_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/features/phoneme_level")
PROSODY_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody")
SPLIT_FILE = DATA_DIR / "splits.json"
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/training/kan")

# load data loader and KAN module via exec (avoids import path issues)
_dl = {'__file__': str(SCRIPTS_DIR / "01_data_loader.py")}
exec(open(SCRIPTS_DIR / "01_data_loader.py").read(), _dl)
create_dataloader = _dl['create_dataloader']
collate_fn = _dl['collate_fn']
collate_word_fn = _dl['collate_word_fn']

_kan = {}
exec(open(SCRIPTS_DIR / "true_kan_heads_vectorized.py").read(), _kan)
TrueKANHead = _kan['TrueKANHead']


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KANProsodyPredictor(nn.Module):
    # 30 features → KAN → 3 targets
    
    def __init__(self, num_basis: int = 8, spline_degree: int = 3):
        super().__init__()
        self.kan = TrueKANHead(
            in_dim=30, out_dim=3, num_basis=num_basis, degree=spline_degree,
            grid_range=(0, 1),  # explicit: features are normalized to [0,1]
            enable_interpretability=True, learn_base_linear=False
        )
    
    def forward(self, features, attention_mask=None):
        output = self.kan(features)
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).float()
        return {
            'f0': output[:, :, 0:1],
            'duration': output[:, :, 1:2],
            'energy': output[:, :, 2:3]
        }


def compute_loss(predictions, targets, mask):
    # targets are z-normalized by data loader (train-only stats)
    # loss is normalized by number of active targets (variance-balanced)
    # all targets use finite filtering for robustness (one bad file won't break training)
    mask = mask.unsqueeze(-1).bool()
    losses = []
    
    for i, name in enumerate(['f0', 'duration', 'energy']):
        pred = predictions[name]
        tgt = targets[:, :, i:i+1]
        
        # filter NaN/Inf in both targets AND predictions (fully robust)
        valid = torch.isfinite(tgt) & torch.isfinite(pred)
        target_mask = (mask & valid).float()
        denom = target_mask.sum() + 1e-8
        
        if denom > 1:
            loss = ((pred - tgt) ** 2 * target_mask).sum() / denom
            losses.append(loss)
    
    if not losses:
        return None  # caller handles (skip batch)
    return sum(losses) / len(losses)


def train_epoch(model, dataloader, optimizer, device, level='phoneme'):
    model.train()
    total_loss = 0.0
    n = 0
    
    for batch in tqdm(dataloader, desc="Train", leave=False):
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        
        # word-level: no sequence dimension, add it for model compatibility
        if level == 'word':
            features = features.unsqueeze(1)  # (B, 30) -> (B, 1, 30)
            targets = targets.unsqueeze(1)    # (B, 3) -> (B, 1, 3)
            mask = torch.ones(features.shape[0], 1, dtype=torch.bool, device=device)
        else:
            mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        predictions = model(features, mask)
        loss = compute_loss(predictions, targets, mask)
        
        # skip batch if no valid targets or invalid loss
        if loss is None or not torch.isfinite(loss) or loss.item() <= 0:
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    
    return total_loss / max(n, 1)


def validate(model, dataloader, device, level='phoneme'):
    model.eval()
    all_preds = {'f0': [], 'duration': [], 'energy': []}
    all_targets = {'f0': [], 'duration': [], 'energy': []}
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            # word-level: add sequence dimension
            if level == 'word':
                features = features.unsqueeze(1)
                targets = targets.unsqueeze(1)
                mask = torch.ones(features.shape[0], 1, dtype=torch.bool, device=device)
            else:
                mask = batch['attention_mask'].to(device)
            
            predictions = model(features, mask)
            mask_np = mask.bool().cpu().numpy()
            
            for i, name in enumerate(['f0', 'duration', 'energy']):
                pred = predictions[name].squeeze(-1).cpu().numpy()
                tgt = targets[:, :, i].cpu().numpy()
                for b in range(pred.shape[0]):
                    m = mask_np[b]
                    # filter NaNs to prevent RMSE/R² from becoming NaN
                    valid = m & np.isfinite(tgt[b]) & np.isfinite(pred[b])
                    if valid.sum() > 0:
                        all_preds[name].extend(pred[b][valid])
                        all_targets[name].extend(tgt[b][valid])
    
    # compute R² and RMSE per target
    metrics = {}
    for name in ['f0', 'duration', 'energy']:
        y_pred = np.array(all_preds[name])
        y_true = np.array(all_targets[name])
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        
        if valid.sum() > 1:
            y_p, y_t = y_pred[valid], y_true[valid]
            ss_res = np.sum((y_t - y_p) ** 2)
            ss_tot = np.sum((y_t - y_t.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
            metrics[name] = {'r2': float(r2), 'rmse': float(rmse)}
        else:
            metrics[name] = {'r2': 0, 'rmse': 0}
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-basis', type=int, default=8)
    parser.add_argument('--spline-degree', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--level', type=str, default='phoneme', choices=['phoneme', 'word'])
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # per-seed output directory (prevents overwriting)
    run_dir = OUTPUT_DIR / f"{args.level}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    
    # model
    model = KANProsodyPredictor(args.num_basis, args.spline_degree).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    
    print("=" * 60)
    print("NASM Training (Neural Additive Spline Model)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Parameters: {n_params:,}")
    print(f"Config: basis={args.num_basis}, degree={args.spline_degree}, lr={args.lr}")
    print(f"Training: {args.epochs} epochs, patience={args.patience}")
    print("=" * 60)
    
    # data loaders
    # For word-level: pass train target stats to val/test to prevent data leakage
    train_loader = create_dataloader(split_name='train', batch_size=args.batch_size, 
                                      shuffle=True, num_workers=0, level=args.level)
    
    word_stats = None
    if args.level == 'word':
        # extract train stats for consistent val/test normalization
        word_stats = train_loader.dataset.target_stats
    
    val_loader = create_dataloader(split_name='val', batch_size=args.batch_size,
                                    shuffle=False, num_workers=0, level=args.level,
                                    word_target_stats=word_stats)
    test_loader = create_dataloader(split_name='test', batch_size=args.batch_size,
                                     shuffle=False, num_workers=0, level=args.level,
                                     word_target_stats=word_stats)
    
    print(f"Train: {len(train_loader.dataset)} samples ({args.level})")
    print(f"Val: {len(val_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")
    print("-" * 60)
    
    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # early stopping state
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []
    
    # training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.level)
        val_metrics = validate(model, val_loader, device, args.level)
        scheduler.step()
        
        history.append({'epoch': epoch+1, 'train_loss': train_loss, **val_metrics})
        
        # check for improvement (early stopping on sum of RMSE)
        improved = ""
        val_loss = sum(m['rmse'] for m in val_metrics.values())
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / 'best_model.pt')
            improved = " ✓"
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Train: {train_loss:.4f} | RMSE_sum: {val_loss:.3f} | "
              f"F0={val_metrics['f0']['r2']:.3f}, "
              f"Dur={val_metrics['duration']['r2']:.3f}, "
              f"En={val_metrics['energy']['r2']:.3f}{improved}")
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (best: {best_epoch})")
            break
    
    # final evaluation on best model (BOTH val and test)
    print("-" * 60)
    model.load_state_dict(torch.load(run_dir / 'best_model.pt', map_location=device))
    val_final = validate(model, val_loader, device, args.level)
    test_final = validate(model, test_loader, device, args.level)
    
    # save results with full reproducibility info
    results = {
        'model': 'nasm',  # Neural Additive Spline Model
        'seed': args.seed,
        'level': args.level,
        'config': {
            'num_basis': args.num_basis,
            'spline_degree': args.spline_degree,
            'grid_range': [0, 1],
            'lr': args.lr,
            'batch_size': args.batch_size,
            'patience': args.patience,
        },
        'data': {
            'use_thesis_data': _dl.get('USE_THESIS_DATA', 'unknown'),
            'use_physical_word_duration': _dl.get('USE_PHYSICAL_WORD_DURATION', 'unknown'),
            'n_train': len(train_loader.dataset),
            'n_val': len(val_loader.dataset),
            'n_test': len(test_loader.dataset),
        },
        'best_epoch': best_epoch,
        'val_metrics': val_final,
        'test_metrics': test_final,
        'history': history
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Best model (epoch {best_epoch}):")
    print(f"  VAL  - F0={val_final['f0']['r2']:.4f}, Dur={val_final['duration']['r2']:.4f}, En={val_final['energy']['r2']:.4f}")
    print(f"  TEST - F0={test_final['f0']['r2']:.4f}, Dur={test_final['duration']['r2']:.4f}, En={test_final['energy']['r2']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
