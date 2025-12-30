#!/usr/bin/env python3
# FAIR comparison: NASM vs MLP with identical LR=1e-3
# Word-level, full thesis data, 6 seeds

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random

SCRIPTS_DIR = Path("/Users/s.mengari/Desktop/CODE2/scripts/training")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/fair_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load modules
_dl = {'__file__': str(SCRIPTS_DIR / '01_data_loader.py')}
exec(open(SCRIPTS_DIR / '01_data_loader.py').read(), _dl)
create_dataloader = _dl['create_dataloader']

_kan = {}
exec(open(SCRIPTS_DIR / 'true_kan_heads_vectorized.py').read(), _kan)
TrueKANHead = _kan['TrueKANHead']

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# =========== NASM Model ===========
class NASM(nn.Module):
    def __init__(self, num_basis=8, spline_degree=3):
        super().__init__()
        self.kan = TrueKANHead(in_dim=30, out_dim=3, num_basis=num_basis, 
                               degree=spline_degree, enable_interpretability=True)
    
    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.kan(x)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return {'f0': out[:,:,0:1], 'duration': out[:,:,1:2], 'energy': out[:,:,2:3]}

# =========== MLP Model ===========
class MLP(nn.Module):
    def __init__(self, hidden_dim=8, dropout=0.1):
        super().__init__()
        self.f0 = nn.Sequential(nn.Linear(30, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.duration = nn.Sequential(nn.Linear(30, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.energy = nn.Sequential(nn.Linear(30, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    
    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return {'f0': self.f0(x), 'duration': self.duration(x), 'energy': self.energy(x)}

def compute_metrics(model, dataloader, device):
    model.eval()
    all_preds = {'f0': [], 'duration': [], 'energy': []}
    all_targets = {'f0': [], 'duration': [], 'energy': []}
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            preds = model(features)
            
            for i, n in enumerate(['f0', 'duration', 'energy']):
                p = preds[n].squeeze().cpu().numpy()
                t = targets[:, i].cpu().numpy()
                valid = np.isfinite(p) & np.isfinite(t)
                all_preds[n].extend(p[valid])
                all_targets[n].extend(t[valid])
    
    metrics = {}
    for n in ['f0', 'duration', 'energy']:
        p, t = np.array(all_preds[n]), np.array(all_targets[n])
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((t - p) ** 2))
        var_ratio = np.var(p) / (np.var(t) + 1e-8)
        metrics[n] = {'r2': r2, 'rmse': rmse, 'var_ratio': var_ratio}
    
    return metrics

def train_model(model_class, lr, seed, epochs=30, device='cpu'):
    set_seed(seed)
    
    train_loader = create_dataloader('train', batch_size=32, shuffle=True, num_workers=0, level='word')
    val_loader = create_dataloader('val', batch_size=32, shuffle=False, num_workers=0, level='word')
    
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            preds = model(features)
            loss = 0
            for i, name in enumerate(['f0', 'duration', 'energy']):
                loss += ((preds[name].squeeze() - targets[:, i])**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validate
        val_metrics = compute_metrics(model, val_loader, device)
        val_loss = sum(m['rmse'] for m in val_metrics.values())
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_epoch = epoch + 1
    
    return best_metrics, n_params, best_epoch

# =========== Main ===========
if __name__ == "__main__":
    print('=' * 70)
    print('FAIR COMPARISON: NASM vs MLP')
    print('=' * 70)
    print(f'Data: Full thesis data (USE_THESIS_DATA=True)')
    print(f'Level: Word')
    print(f'LR: 1e-3 (both models)')
    print(f'Epochs: 30')
    print(f'Seeds: 42, 123, 456, 789, 1337, 2024')
    print('=' * 70)

    device = torch.device('cpu')
    seeds = [42, 123, 456, 789, 1337, 2024]
    lr = 1e-3
    epochs = 30

    results = {'nasm': [], 'mlp': []}

    for model_name, model_class in [('nasm', NASM), ('mlp', MLP)]:
        print(f'\nTraining {model_name.upper()}...')
        for seed in tqdm(seeds, desc=model_name.upper()):
            metrics, n_params, best_epoch = train_model(model_class, lr, seed, epochs, device)
            results[model_name].append({
                'seed': seed,
                'params': n_params,
                'best_epoch': best_epoch,
                'metrics': metrics
            })

    # =========== Results ===========
    print('\n' + '=' * 70)
    print('RESULTS')
    print('=' * 70)

    for model_name in ['nasm', 'mlp']:
        metrics_list = [r['metrics'] for r in results[model_name]]
        n_params = results[model_name][0]['params']
        
        f0_r2 = [m['f0']['r2'] for m in metrics_list]
        dur_r2 = [m['duration']['r2'] for m in metrics_list]
        en_r2 = [m['energy']['r2'] for m in metrics_list]
        
        print(f'\n{model_name.upper()} ({n_params} params):')
        print(f'  F0:       R²={np.mean(f0_r2):.4f} ± {np.std(f0_r2):.4f}')
        print(f'  Duration: R²={np.mean(dur_r2):.4f} ± {np.std(dur_r2):.4f}')
        print(f'  Energy:   R²={np.mean(en_r2):.4f} ± {np.std(en_r2):.4f}')

    # Comparison
    print('\n' + '-' * 70)
    print('HEAD-TO-HEAD COMPARISON (NASM - MLP)')
    print('-' * 70)

    for target in ['f0', 'duration', 'energy']:
        nasm_r2 = np.mean([r['metrics'][target]['r2'] for r in results['nasm']])
        mlp_r2 = np.mean([r['metrics'][target]['r2'] for r in results['mlp']])
        delta = nasm_r2 - mlp_r2
        winner = 'NASM' if delta > 0.005 else ('MLP' if delta < -0.005 else 'TIE')
        print(f'{target.upper():10s}: NASM={nasm_r2:.4f}, MLP={mlp_r2:.4f}, Δ={delta:+.4f} → {winner}')

    # Save results
    output = {
        'config': {'lr': lr, 'epochs': epochs, 'seeds': seeds, 'level': 'word', 'data': 'thesis'},
        'results': results,
        'summary': {}
    }

    for model_name in ['nasm', 'mlp']:
        metrics_list = [r['metrics'] for r in results[model_name]]
        output['summary'][model_name] = {
            'f0_r2': f"{np.mean([m['f0']['r2'] for m in metrics_list]):.4f}±{np.std([m['f0']['r2'] for m in metrics_list]):.4f}",
            'dur_r2': f"{np.mean([m['duration']['r2'] for m in metrics_list]):.4f}±{np.std([m['duration']['r2'] for m in metrics_list]):.4f}",
            'en_r2': f"{np.mean([m['energy']['r2'] for m in metrics_list]):.4f}±{np.std([m['energy']['r2'] for m in metrics_list]):.4f}",
            'params': results[model_name][0]['params']
        }

    with open(OUTPUT_DIR / 'fair_comparison_lr1e-3.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nResults saved to: {OUTPUT_DIR / "fair_comparison_lr1e-3.json"}')
    print('=' * 70)


