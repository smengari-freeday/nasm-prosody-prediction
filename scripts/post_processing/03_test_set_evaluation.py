#!/usr/bin/env python3
# evaluate trained KAN models on held-out test set
#
# loads best_model.pt from each seed folder and evaluates on test split
# outputs R², RMSE, MAE per target
#
# usage:
#   python 03_test_set_evaluation.py

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# paths (hardcoded)
SCRIPTS_DIR = Path("/Users/s.mengari/Desktop/CODE2/scripts/training")
MODEL_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/training_full")
OUTPUT_JSON = Path("/Users/s.mengari/Desktop/CODE2/results/test_set_evaluation.json")
SEEDS = [42, 43, 44, 45, 46, 47]

# config (using Annotated KAN naming)
FEATURE_DIM = 30
GRID_SIZE = 8
SPLINE_ORDER = 2  # saved models use quadratic B-splines

# load modules via exec
_dl = {'__file__': str(SCRIPTS_DIR / "01_data_loader.py")}
exec(open(SCRIPTS_DIR / "01_data_loader.py").read(), _dl)
PhonemeLevelDataset = _dl['PhonemeLevelDataset']
collate_fn = _dl['collate_fn']

_kan = {}
exec(open(SCRIPTS_DIR / "true_kan_heads_vectorized.py").read(), _kan)
TrueKANHead = _kan['TrueKANHead']


class SimpleKAN(torch.nn.Module):
    def __init__(self, in_features=30, grid_size=8, spline_order=2):
        super().__init__()
        self.kan = TrueKANHead(
            in_features=in_features, out_features=3, grid_size=grid_size, spline_order=spline_order
        )
    
    def forward(self, x, mask=None):
        out = self.kan(x)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return {'f0': out[:, :, 0:1], 'duration': out[:, :, 1:2], 'energy': out[:, :, 2:3]}


def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    all_preds = {'f0': [], 'duration': [], 'energy': []}
    all_targets = {'f0': [], 'duration': [], 'energy': []}
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['attention_mask'].to(device)
            
            preds = model(features, mask)
            
            for i, name in enumerate(['f0', 'duration', 'energy']):
                pred = preds[name].squeeze(-1)
                tgt = targets[:, :, i]
                m = mask.bool()
                
                all_preds[name].extend(pred[m].cpu().numpy().tolist())
                all_targets[name].extend(tgt[m].cpu().numpy().tolist())
    
    results = {}
    for name in ['f0', 'duration', 'energy']:
        p = np.array(all_preds[name])
        t = np.array(all_targets[name])
        valid = np.isfinite(p) & np.isfinite(t)
        p, t = p[valid], t[valid]
        
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        results[name] = {
            'r2': float(r2),
            'rmse': float(np.sqrt(np.mean((p - t) ** 2))),
            'mae': float(np.mean(np.abs(p - t))),
            'n_samples': int(len(p))
        }
    
    return results


def main():
    print("=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    
    # create test dataloader
    test_ds = PhonemeLevelDataset(split_name='test')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Test samples: {len(test_ds)}")
    print()
    
    # evaluate each seed
    all_results = {}
    
    for seed in SEEDS:
        model_path = MODEL_DIR / f'kan_seed{seed}' / 'best_model.pt'
        
        if not model_path.exists():
            print(f"Seed {seed}: model not found")
            continue
        
        model = SimpleKAN(FEATURE_DIM, GRID_SIZE, SPLINE_ORDER)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # handle old parameter names (coefficients → coef, per_feature_linear_weights → scale)
        renamed_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('coefficients', 'coef').replace('per_feature_linear_weights', 'scale')
            renamed_state_dict[new_k] = v
        
        model.load_state_dict(renamed_state_dict, strict=False)
        
        results = evaluate_model(model, test_loader)
        all_results[str(seed)] = results
        
        print(f"Seed {seed}: F0 R²={results['f0']['r2']:.3f}, "
              f"Dur R²={results['duration']['r2']:.3f}, "
              f"En R²={results['energy']['r2']:.3f}")
    
    # summary
    print()
    print("=" * 60)
    print("Test Set Summary (KAN)")
    print("=" * 60)
    
    for target in ['f0', 'duration', 'energy']:
        r2s = [all_results[s][target]['r2'] for s in all_results]
        print(f"{target.upper():10} R² = {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    
    # save results
    output = {
        'test_set_size': len(test_ds),
        'config': {'in_features': FEATURE_DIM, 'grid_size': GRID_SIZE, 'spline_order': SPLINE_ORDER},
        'kan_results': all_results
    }
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print(f"Results saved to: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
