#!/usr/bin/env python3
# aggregate training results from KAN and MLP models into unified CSV
#
# collects per_target_results.json from each seed folder and outputs:
# - model_comparison.csv with columns: model, seed, target, r2, rmse, mae, correlation
#
# usage:
#   python 01_aggregate_results.py

import json
import pandas as pd
from pathlib import Path

# paths (hardcoded)
RESULTS_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/training_full")
OUTPUT_CSV = Path("/Users/s.mengari/Desktop/CODE2/results/model_comparison.csv")
SEEDS = [42, 43, 44, 45, 46, 47]


def aggregate_results(results_dir: Path, seeds: list) -> pd.DataFrame:
    results = []
    
    for model_type in ['kan', 'mlp']:
        for seed in seeds:
            result_path = results_dir / f'{model_type}_seed{seed}' / 'per_target_results.json'
            
            if not result_path.exists():
                continue
            
            with open(result_path) as f:
                data = json.load(f)
            
            for target in ['f0', 'duration', 'energy']:
                m = data['per_target_metrics'][target]
                results.append({
                    'model': model_type.upper(),
                    'seed': seed,
                    'target': target,
                    'r2': m['r2'],
                    'rmse': m['rmse'],
                    'mae': m['mae'],
                    'correlation': m.get('correlation', None)
                })
    
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Aggregate Training Results")
    print("=" * 60)
    
    df = aggregate_results(RESULTS_DIR, SEEDS)
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print()
    print("Summary (mean R² by model and target):")
    print("-" * 40)
    summary = df.groupby(['model', 'target'])['r2'].agg(['mean', 'std']).round(4)
    print(summary)


if __name__ == '__main__':
    main()
