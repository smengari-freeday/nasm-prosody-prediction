#!/usr/bin/env python3
# MLP baseline for prosody prediction (parameter-matched with KAN)
#
# architecture:
# - three independent MLPs: one per target (F0, Duration, Energy)
# - each MLP: 30 → 8 → 1 with ReLU and dropout
# - ~771 parameters total (vs KAN's ~813)
#
# training strategy:
# 1. MSE loss per target (z-normalized, variance-balanced by active target count)
# 2. Adam optimizer with weight decay
# 3. early stopping on sum of per-target RMSE (not a single joint RMSE)
# 4. padding handled by masking; NaN targets filtered per-target (matches NASM)
# 5. gradient clipping (1.0) for parity with NASM
#
# usage:
#   python 03_train_mlp_baseline.py --epochs 50 --lr 1e-4 --seed 42

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# paths
SCRIPTS_DIR = Path("/Users/s.mengari/Desktop/CODE2/scripts/training")
DATA_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/features/phoneme_level")
PROSODY_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/prosody")
SPLIT_FILE = DATA_DIR / "splits.json"
STATS_FILE = Path("/Users/s.mengari/Desktop/CODE2/results/phoneme_level_target_statistics.json")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/training/mlp")

# load data loader via exec (avoids import path issues)
_dl = {'__file__': str(SCRIPTS_DIR / "01_data_loader.py")}
exec(open(SCRIPTS_DIR / "01_data_loader.py").read(), _dl)
PhonemeLevelDataset = _dl['PhonemeLevelDataset']
WordLevelDataset = _dl['WordLevelDataset']
collate_phoneme_fn = _dl['collate_phoneme_fn']
collate_word_fn = _dl['collate_word_fn']


class MLPProsodyPredictor(nn.Module):
    # 30 → 8 → 1 per target, ~771 params total
    
    def __init__(self, hidden_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.f0 = nn.Sequential(
            nn.Linear(30, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.duration = nn.Sequential(
            nn.Linear(30, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.energy = nn.Sequential(
            nn.Linear(30, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features, attention_mask=None):
        # NOTE: masking is applied in loss only (not here)
        # this keeps model architecture independent of batching/padding
        # and ensures identical gradient treatment to NASM
        f0 = self.f0(features)
        dur = self.duration(features)
        en = self.energy(features)
        return {'f0': f0, 'duration': dur, 'energy': en}


def compute_loss(predictions, targets, mask):
    # targets are z-normalized by data loader, variance-balanced by active target count
    # all targets use finite filtering for robustness (matches NASM exactly)
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


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    
    for batch in tqdm(dataloader, desc="Train", leave=False):
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        
        # handle word-level (2D) vs phoneme-level (3D)
        is_word_level = features.dim() == 2
        if is_word_level:
            features = features.unsqueeze(1)  # (B, F) -> (B, 1, F)
            targets = targets.unsqueeze(1)    # (B, 3) -> (B, 1, 3)
            mask = torch.ones(features.shape[0], 1, device=device).bool()
        else:
            mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        predictions = model(features, mask)
        loss = compute_loss(predictions, targets, mask)
        
        # skip batch if no valid targets or invalid loss (matches NASM)
        if loss is None or not torch.isfinite(loss) or loss.item() <= 0:
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    
    return total_loss / n


def validate(model, dataloader, device):
    model.eval()
    all_preds = {'f0': [], 'duration': [], 'energy': []}
    all_targets = {'f0': [], 'duration': [], 'energy': []}
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            # handle word-level (2D) vs phoneme-level (3D)
            is_word_level = features.dim() == 2
            if is_word_level:
                features = features.unsqueeze(1)
                targets = targets.unsqueeze(1)
                mask = torch.ones(features.shape[0], 1, device=device).bool()
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
    
    # compute R² and RMSE per target
    metrics = {}
    for name in ['f0', 'duration', 'energy']:
        y_pred = np.array(all_preds[name])
        y_true = np.array(all_targets[name])
        
        if len(y_true) > 1:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            metrics[name] = {'r2': float(r2), 'rmse': float(rmse)}
        else:
            metrics[name] = {'r2': 0, 'rmse': 0}
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--level', type=str, default='phoneme', choices=['phoneme', 'word'])
    args = parser.parse_args()
    
    # seed all random sources for reproducibility (matches NASM script)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # per-seed output directory (prevents overwriting)
    run_dir = OUTPUT_DIR / f"{args.level}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # device selection (match NASM script)
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    
    # model
    model = MLPProsodyPredictor(args.hidden_dim, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    
    print("=" * 60)
    print("MLP Baseline Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Parameters: {n_params:,}")
    print(f"Config: hidden={args.hidden_dim}, dropout={args.dropout}, lr={args.lr}")
    print(f"Training: {args.epochs} epochs")
    print("=" * 60)
    
    # data loaders
    # For word-level: pass train target stats to val/test to prevent data leakage
    train_ds_phoneme = PhonemeLevelDataset(split_name='train')
    val_ds_phoneme = PhonemeLevelDataset(split_name='val')
    test_ds_phoneme = PhonemeLevelDataset(split_name='test')
    
    if args.level == 'word':
        train_ds = WordLevelDataset(train_ds_phoneme)
        word_stats = train_ds.target_stats  # extract train stats
        val_ds = WordLevelDataset(val_ds_phoneme, external_target_stats=word_stats)
        test_ds = WordLevelDataset(test_ds_phoneme, external_target_stats=word_stats)
        collate_fn = collate_word_fn
    else:
        train_ds = train_ds_phoneme
        val_ds = val_ds_phoneme
        test_ds = test_ds_phoneme
        collate_fn = collate_phoneme_fn
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Train: {len(train_ds)} samples ({args.level})")
    print(f"Val: {len(val_ds)} samples, Test: {len(test_ds)} samples")
    print("-" * 60)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # best model tracking + early stopping
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []
    
    # training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        
        history.append({'epoch': epoch+1, 'train_loss': train_loss, **val_metrics})
        
        # check for improvement (save best on sum of RMSE)
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
    val_final = validate(model, val_loader, device)
    test_final = validate(model, test_loader, device)
    
    # save results with full reproducibility info
    results = {
        'model': 'mlp',
        'seed': args.seed,
        'level': args.level,
        'config': {
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'patience': args.patience,
        },
        'data': {
            'use_thesis_data': _dl.get('USE_THESIS_DATA', 'unknown'),
            'use_physical_word_duration': _dl.get('USE_PHYSICAL_WORD_DURATION', 'unknown'),
            'n_train': len(train_ds),
            'n_val': len(val_ds),
            'n_test': len(test_ds),
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
