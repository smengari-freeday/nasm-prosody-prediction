#!/usr/bin/env python3
# MLP baseline for prosody prediction (parameter-matched with KAN)
#
# architecture:
# - three independent MLPs: one per target (F0, Duration, Energy)
# - each MLP: 30 → 8 → 1 with ReLU and dropout
# - ~771 parameters total (vs KAN's ~813)
#
# training strategy:
# 1. MSE loss per target (no variance normalization)
# 2. Adam optimizer with weight decay
# 3. no early stopping (fixed epochs, save best on RMSE)
# 4. padding handled by masking to 0.5 (neutral value)
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
        # mask padded positions to 0.5 (neutral value for normalized features)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            features = features * mask + (1 - mask) * 0.5
        
        f0 = self.f0(features)
        dur = self.duration(features)
        en = self.energy(features)
        
        if attention_mask is not None:
            f0 = f0 * mask
            dur = dur * mask
            en = en * mask
        
        return {'f0': f0, 'duration': dur, 'energy': en}


def compute_loss(predictions, targets, mask):
    mask = mask.unsqueeze(-1).float()
    mask_sum = mask.sum() + 1e-8
    
    f0_loss = ((predictions['f0'] - targets[:, :, 0:1]) ** 2 * mask).sum() / mask_sum
    dur_loss = ((predictions['duration'] - targets[:, :, 1:2]) ** 2 * mask).sum() / mask_sum
    
    # energy: skip NaN values (some utterances may have missing energy)
    en_tgt = targets[:, :, 2:3]
    en_valid = torch.isfinite(en_tgt)
    en_mask = (mask.bool() & en_valid).float()
    en_sum = en_mask.sum() + 1e-8
    en_loss = ((predictions['energy'] - en_tgt) ** 2 * en_mask).sum() / en_sum if en_sum > 10 else 0
    
    return f0_loss + dur_loss + en_loss


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
        loss.backward()
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--level', type=str, default='phoneme', choices=['phoneme', 'word'])
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    train_ds_phoneme = PhonemeLevelDataset(split_name='train')
    val_ds_phoneme = PhonemeLevelDataset(split_name='val')
    
    if args.level == 'word':
        train_ds = WordLevelDataset(train_ds_phoneme)
        val_ds = WordLevelDataset(val_ds_phoneme)
        collate_fn = collate_word_fn
    else:
        train_ds = train_ds_phoneme
        val_ds = val_ds_phoneme
        collate_fn = collate_phoneme_fn
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Train: {len(train_ds)} samples ({args.level}), Val: {len(val_ds)} samples")
    print("-" * 60)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # best model tracking
    best_loss = float('inf')
    best_epoch = 0
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
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pt')
            improved = " ✓"
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Train: {train_loss:.4f} | "
              f"F0={val_metrics['f0']['r2']:.3f}, "
              f"Dur={val_metrics['duration']['r2']:.3f}, "
              f"En={val_metrics['energy']['r2']:.3f}{improved}")
    
    # final evaluation on best model
    print("-" * 60)
    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pt', map_location=device))
    final = validate(model, val_loader, device)
    
    # save results
    results = {
        'model': 'mlp', 'seed': args.seed,
        'config': {'hidden_dim': args.hidden_dim},
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
