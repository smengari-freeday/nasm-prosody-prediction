#!/usr/bin/env python3
# rigorous hyperparameter sweep addressing IEEE reviewer concerns:
#
# 1. convergence curves (train/val loss per epoch) to prove LR effect
# 2. multiple seeds (3) for reliability
# 3. fixed epoch budget (no early stopping for sweep fairness)
# 4. proper variance regularization (clamp, not add)
# 5. additional diagnostics: Pearson r, calibration slope, RMSE
# 6. separate val selection from test evaluation

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random
from scipy import stats

SCRIPTS_DIR = Path("/Users/s.mengari/Desktop/CODE2/scripts/training")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/sweeps")

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


class KANModel(nn.Module):
    def __init__(self, num_basis=8, spline_degree=3):
        super().__init__()
        self.kan = TrueKANHead(in_dim=30, out_dim=3, num_basis=num_basis, 
                               degree=spline_degree, enable_interpretability=True)
    
    def forward(self, x, mask=None):
        out = self.kan(x)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return {'f0': out[:,:,0:1], 'duration': out[:,:,1:2], 'energy': out[:,:,2:3]}


def compute_loss_with_var_reg(preds, targets, mask, target_stats, lambda_var=0.0):
    """Corrected loss with proper variance regularization (clamp, not add)."""
    mask = mask.unsqueeze(-1).float()
    mask_sum = mask.sum() + 1e-8
    total_loss = 0.0
    
    for i, name in enumerate(['f0', 'duration', 'energy']):
        pred = preds[name]
        tgt = targets[:, :, i:i+1]
        mse = (pred - tgt) ** 2
        
        # variance normalization per target
        if target_stats and name in target_stats:
            std = target_stats[name].get('std', 1.0)
            if std > 1e-6:
                mse = mse / (std ** 2)
        
        loss = (mse * mask).sum() / mask_sum
        if torch.isfinite(loss):
            total_loss += loss
        
        # CORRECTED variance regularization: clamp instead of add
        if lambda_var > 0:
            pred_flat = pred[mask.squeeze(-1).bool()]
            tgt_flat = tgt[mask.squeeze(-1).bool()]
            if len(pred_flat) > 10:
                eps = 1e-6
                var_pred = torch.clamp(torch.var(pred_flat), min=eps)
                var_tgt = torch.clamp(torch.var(tgt_flat), min=eps)
                var_loss = (torch.log(var_pred / var_tgt)) ** 2
                var_loss = torch.clamp(var_loss, max=10.0)
                total_loss += lambda_var * var_loss
    
    return total_loss


def compute_diagnostics(model, dataloader, device, level='word'):
    """Comprehensive diagnostics: R², Pearson r, RMSE, calibration slope, var ratio."""
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
            
            preds = model(features, mask)
            mask_np = mask.bool().cpu().numpy()
            
            for i, name in enumerate(['f0', 'duration', 'energy']):
                p = preds[name].squeeze(-1).cpu().numpy()
                t = targets[:, :, i].cpu().numpy()
                for b in range(p.shape[0]):
                    m = mask_np[b]
                    valid = m & np.isfinite(t[b]) & np.isfinite(p[b])
                    if valid.sum() > 0:
                        all_preds[name].extend(p[b][valid])
                        all_targets[name].extend(t[b][valid])
    
    metrics = {}
    for name in ['f0', 'duration', 'energy']:
        p = np.array(all_preds[name])
        t = np.array(all_targets[name])
        
        if len(t) > 10:
            # R²
            ss_res = np.sum((t - p) ** 2)
            ss_tot = np.sum((t - np.mean(t)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # Pearson r
            pearson_r, _ = stats.pearsonr(p, t)
            
            # RMSE
            rmse = np.sqrt(np.mean((t - p) ** 2))
            
            # Calibration slope (regression of target on prediction)
            slope, intercept, _, _, _ = stats.linregress(p, t)
            
            # Variance ratio
            var_ratio = np.var(p) / (np.var(t) + 1e-8)
            
            metrics[name] = {
                'r2': float(r2),
                'pearson_r': float(pearson_r),
                'rmse': float(rmse),
                'calibration_slope': float(slope),
                'var_ratio': float(var_ratio),
                'pred_std': float(np.std(p)),
                'tgt_std': float(np.std(t))
            }
        else:
            metrics[name] = {'r2': 0, 'pearson_r': 0, 'rmse': 0, 
                           'calibration_slope': 0, 'var_ratio': 0}
    
    return metrics


def train_with_curves(config, train_loader, val_loader, device, level='word'):
    """Train and return full convergence curves (no early stopping)."""
    set_seed(config['seed'])
    
    model = KANModel(config['num_basis'], config['spline_degree']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    if hasattr(train_loader.dataset, 'target_stats'):
        target_stats = train_loader.dataset.target_stats
    elif hasattr(train_loader.dataset, 'phoneme_dataset'):
        target_stats = train_loader.dataset.phoneme_dataset.target_stats
    else:
        target_stats = {}
    
    history = {'train_loss': [], 'val_r2_f0': [], 'val_r2_dur': [], 'val_r2_en': []}
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        epoch_loss = 0
        n_batches = 0
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
            preds = model(features, mask)
            loss = compute_loss_with_var_reg(preds, targets, mask, target_stats, 
                                             config.get('lambda_var', 0.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validate
        val_metrics = compute_diagnostics(model, val_loader, device, level)
        
        history['train_loss'].append(epoch_loss / n_batches)
        history['val_r2_f0'].append(val_metrics['f0']['r2'])
        history['val_r2_dur'].append(val_metrics['duration']['r2'])
        history['val_r2_en'].append(val_metrics['energy']['r2'])
    
    # Final diagnostics
    final_metrics = compute_diagnostics(model, val_loader, device, level)
    
    return final_metrics, history


def run_rigorous_sweep():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Key comparison: LR=1e-3 vs LR=1e-4 with multiple seeds
    configs = [
        {'lr': 1e-3, 'lambda_var': 0.0, 'spline_degree': 3, 'label': 'lr1e-3'},
        {'lr': 1e-4, 'lambda_var': 0.0, 'spline_degree': 3, 'label': 'lr1e-4'},
        {'lr': 1e-3, 'lambda_var': 1e-3, 'spline_degree': 3, 'label': 'lr1e-3+var'},
    ]
    
    seeds = [42, 123, 456]
    epochs = 30
    level = 'word'  # faster for sweep
    
    results = {}
    all_histories = {}
    
    print("=" * 70)
    print("RIGOROUS HYPERPARAMETER SWEEP")
    print(f"Seeds: {seeds}, Epochs: {epochs}, Level: {level}")
    print("=" * 70)
    
    for cfg in configs:
        label = cfg['label']
        results[label] = {'seeds': [], 'metrics': []}
        all_histories[label] = []
        
        for seed in tqdm(seeds, desc=label):
            config = {
                'num_basis': 8,
                'spline_degree': cfg['spline_degree'],
                'lr': cfg['lr'],
                'lambda_var': cfg['lambda_var'],
                'epochs': epochs,
                'seed': seed,
                'batch_size': 32,
            }
            
            train_loader = create_dataloader(split_name='train', batch_size=config['batch_size'], 
                                             shuffle=True, num_workers=0, level=level)
            val_loader = create_dataloader(split_name='val', batch_size=config['batch_size'], 
                                           shuffle=False, num_workers=0, level=level)
            
            metrics, history = train_with_curves(config, train_loader, val_loader, device, level)
            
            results[label]['seeds'].append(seed)
            results[label]['metrics'].append(metrics)
            all_histories[label].append(history)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    
    print("\n1. FINAL METRICS (mean ± std across seeds)")
    print("-" * 50)
    
    for label in results:
        metrics_list = results[label]['metrics']
        
        f0_r2 = [m['f0']['r2'] for m in metrics_list]
        dur_r2 = [m['duration']['r2'] for m in metrics_list]
        en_r2 = [m['energy']['r2'] for m in metrics_list]
        
        print(f"\n{label}:")
        print(f"  F0:  R²={np.mean(f0_r2):.3f}±{np.std(f0_r2):.3f}")
        print(f"  Dur: R²={np.mean(dur_r2):.3f}±{np.std(dur_r2):.3f}")
        print(f"  En:  R²={np.mean(en_r2):.3f}±{np.std(en_r2):.3f}")
        
        # Additional diagnostics for first seed
        m0 = metrics_list[0]
        print(f"  Diagnostics (seed 42):")
        print(f"    F0:  Pearson r={m0['f0']['pearson_r']:.3f}, "
              f"calib_slope={m0['f0']['calibration_slope']:.2f}, "
              f"var_ratio={m0['f0']['var_ratio']:.2f}")
        print(f"    Dur: Pearson r={m0['duration']['pearson_r']:.3f}, "
              f"calib_slope={m0['duration']['calibration_slope']:.2f}, "
              f"var_ratio={m0['duration']['var_ratio']:.2f}")
        print(f"    En:  Pearson r={m0['energy']['pearson_r']:.3f}, "
              f"calib_slope={m0['energy']['calibration_slope']:.2f}, "
              f"var_ratio={m0['energy']['var_ratio']:.2f}")
    
    print("\n" + "-" * 50)
    print("2. CONVERGENCE PROOF: Final epoch val R² for F0")
    print("-" * 50)
    
    for label in results:
        histories = all_histories[label]
        final_f0_r2 = [h['val_r2_f0'][-1] for h in histories]
        epoch_10_f0 = [h['val_r2_f0'][9] if len(h['val_r2_f0']) > 9 else 0 for h in histories]
        
        print(f"{label}:")
        print(f"  Epoch 10: F0 R²={np.mean(epoch_10_f0):.3f}±{np.std(epoch_10_f0):.3f}")
        print(f"  Epoch 30: F0 R²={np.mean(final_f0_r2):.3f}±{np.std(final_f0_r2):.3f}")
        improvement = np.mean(final_f0_r2) - np.mean(epoch_10_f0)
        print(f"  Improvement epoch 10→30: {improvement:+.3f}")
    
    print("\n" + "-" * 50)
    print("3. VARIANCE COLLAPSE DIAGNOSIS")
    print("-" * 50)
    
    for label in results:
        metrics_list = results[label]['metrics']
        f0_var_ratios = [m['f0']['var_ratio'] for m in metrics_list]
        f0_slopes = [m['f0']['calibration_slope'] for m in metrics_list]
        
        avg_var_ratio = np.mean(f0_var_ratios)
        avg_slope = np.mean(f0_slopes)
        
        collapse = "⚠️ COLLAPSE" if avg_var_ratio < 0.5 else "✓ OK"
        print(f"{label}: var_ratio={avg_var_ratio:.2f}, calib_slope={avg_slope:.2f} {collapse}")
    
    # Save convergence curves for plotting
    curves_data = {}
    for label in all_histories:
        # Average curves across seeds
        n_epochs = len(all_histories[label][0]['train_loss'])
        avg_train_loss = np.mean([[h['train_loss'][e] for h in all_histories[label]] 
                                   for e in range(n_epochs)], axis=1)
        avg_val_f0 = np.mean([[h['val_r2_f0'][e] for h in all_histories[label]] 
                              for e in range(n_epochs)], axis=1)
        
        curves_data[label] = {
            'epochs': list(range(1, n_epochs + 1)),
            'train_loss': avg_train_loss.tolist(),
            'val_r2_f0': avg_val_f0.tolist()
        }
    
    # Save all results
    output = {
        'summary': {},
        'curves': curves_data,
        'full_results': {}
    }
    
    for label in results:
        metrics_list = results[label]['metrics']
        output['summary'][label] = {
            'f0_r2': f"{np.mean([m['f0']['r2'] for m in metrics_list]):.3f}±{np.std([m['f0']['r2'] for m in metrics_list]):.3f}",
            'dur_r2': f"{np.mean([m['duration']['r2'] for m in metrics_list]):.3f}±{np.std([m['duration']['r2'] for m in metrics_list]):.3f}",
            'en_r2': f"{np.mean([m['energy']['r2'] for m in metrics_list]):.3f}±{np.std([m['energy']['r2'] for m in metrics_list]):.3f}",
        }
        output['full_results'][label] = results[label]
    
    with open(OUTPUT_DIR / "rigorous_sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    # Compare LR
    lr1e3_f0 = np.mean([m['f0']['r2'] for m in results['lr1e-3']['metrics']])
    lr1e4_f0 = np.mean([m['f0']['r2'] for m in results['lr1e-4']['metrics']])
    
    print(f"\n✅ LR=1e-3 vs LR=1e-4:")
    print(f"   F0 R²: {lr1e3_f0:.3f} vs {lr1e4_f0:.3f} (Δ={lr1e3_f0-lr1e4_f0:+.3f})")
    if lr1e3_f0 > lr1e4_f0 + 0.05:
        print("   → LR=1e-3 significantly better. LR=1e-4 under-converges.")
    
    print(f"\nResults saved to: {OUTPUT_DIR / 'rigorous_sweep_results.json'}")
    
    return output


if __name__ == "__main__":
    run_rigorous_sweep()

