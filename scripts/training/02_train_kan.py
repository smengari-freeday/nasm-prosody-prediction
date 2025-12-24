#!/usr/bin/env python3
# KAN-style training for prosody prediction
# NASM = Neural Additive Spline Model (single-layer KAN without stacking)
#
# architecture:
# - each input feature has its own learned curve (B-spline) plus linear term
# - outputs are simply the sum of all feature contributions (additive, no interactions)
# - fully interpretable: can plot how each feature affects each prosody target
# - ~813 parameters for default config, dropout(0.1) during training
#
# training strategy:
# 1. MSE loss with variance normalization per target
# 2. Adam optimizer with cosine annealing LR schedule
# 3. early stopping on validation RMSE sum (test evaluation is separate)
# 4. gradient clipping for stability
# 5. masking applied in both forward and loss (redundant but safe)
#
# usage:
#   python 02_train_kan.py --epochs 50 --lr 1e-4 --seed 42

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
STATS_FILE = Path("/Users/s.mengari/Desktop/CODE2/results/phoneme_level_target_statistics.json")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/training/kan")

# load data loader and KAN module via exec (avoids import path issues)
_dl = {'__file__': str(SCRIPTS_DIR / "01_data_loader.py")}
exec(open(SCRIPTS_DIR / "01_data_loader.py").read(), _dl)
PhonemeLevelDataset = _dl['PhonemeLevelDataset']
collate_fn = _dl['collate_fn']

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


def compute_loss(predictions, targets, mask, target_stats):
    mask = mask.unsqueeze(-1).float()
    mask_sum = mask.sum() + 1e-8
    total_loss = 0.0
    
    for i, name in enumerate(['f0', 'duration', 'energy']):
        pred = predictions[name]
        tgt = targets[:, :, i:i+1]
        mse = (pred - tgt) ** 2
        
        # variance normalization
        if target_stats and name in target_stats:
            std = target_stats[name].get('std', 1.0)
            if std > 1e-6:
                mse = mse / (std ** 2)
        
        loss = (mse * mask).sum() / mask_sum
        if torch.isfinite(loss):
            total_loss += loss
    
    return total_loss / 3.0


def train_epoch(model, dataloader, optimizer, device, target_stats):
    model.train()
    total_loss = 0.0
    n = 0
    
    for batch in tqdm(dataloader, desc="Train", leave=False):
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        predictions = model(features, mask)
        loss = compute_loss(predictions, targets, mask, target_stats)
        
        if torch.isfinite(loss) and loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
    
    return total_loss / max(n, 1)


def validate(model, dataloader, device, target_stats):
    model.eval()
    all_preds = {'f0': [], 'duration': [], 'energy': []}
    all_targets = {'f0': [], 'duration': [], 'energy': []}
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['attention_mask'].to(device)
            
            predictions = model(features, mask)
            mask_np = mask.bool().cpu().numpy()
            
            for i, name in enumerate(['f0', 'duration', 'energy']):
                pred = predictions[name].squeeze(-1).cpu().numpy()
                tgt = targets[:, :, i].cpu().numpy()
                for b in range(pred.shape[0]):
                    m = mask_np[b]
                    if m.sum() > 0:
                        all_preds[name].extend(pred[b][m])
                        all_targets[name].extend(tgt[b][m])
    
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
    args = parser.parse_args()
    
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
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
    train_ds = PhonemeLevelDataset(split_name='train')
    val_ds = PhonemeLevelDataset(split_name='val')
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Train: {len(train_ds)} utterances, Val: {len(val_ds)} utterances")
    print("-" * 60)
    
    # target statistics for loss normalization
    with open(STATS_FILE) as f:
        target_stats = json.load(f)
    
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
        train_loss = train_epoch(model, train_loader, optimizer, device, target_stats)
        val_metrics = validate(model, val_loader, device, target_stats)
        scheduler.step()
        
        history.append({'epoch': epoch+1, 'train_loss': train_loss, **val_metrics})
        
        # check for improvement (early stopping on sum of RMSE)
        improved = ""
        val_loss = sum(m['rmse'] for m in val_metrics.values())
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pt')
            improved = " ✓"
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Train: {train_loss:.4f} | "
              f"F0={val_metrics['f0']['r2']:.3f}, "
              f"Dur={val_metrics['duration']['r2']:.3f}, "
              f"En={val_metrics['energy']['r2']:.3f}{improved}")
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (best: {best_epoch})")
            break
    
    # final evaluation on best model
    print("-" * 60)
    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pt', map_location=device))
    final = validate(model, val_loader, device, target_stats)
    
    # save results
    results = {
        'model': 'kan', 'seed': args.seed,
        'config': {'num_basis': args.num_basis, 'spline_degree': args.spline_degree},
        'best_epoch': best_epoch,
        'metrics': final,
        'history': history
    }
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Best model (epoch {best_epoch}):")
    print(f"  F0 R²={final['f0']['r2']:.4f}, RMSE={final['f0']['rmse']:.4f}")
    print(f"  Duration R²={final['duration']['r2']:.4f}, RMSE={final['duration']['rmse']:.4f}")
    print(f"  Energy R²={final['energy']['r2']:.4f}, RMSE={final['energy']['rmse']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
