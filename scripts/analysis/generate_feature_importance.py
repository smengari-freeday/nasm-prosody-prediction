#!/usr/bin/env python3
# Generate feature importance table for thesis (Table 5.3)
# Uses NASM spline coefficients to compute per-feature contribution magnitude

import sys
sys.path.insert(0, '/Users/s.mengari/Desktop/CODE2/scripts/training')

import torch
import numpy as np
import json
from pathlib import Path

# Feature names (must match data loader)
FEATURE_NAMES = [
    'primary_stress_pos', 'word_frequency', 'syllable_count', 'phoneme_count',
    'pos_noun', 'pos_verb', 'pos_adjective', 'pos_adverb',
    'pos_pronoun', 'pos_preposition', 'pos_determiner', 'pos_auxiliary',
    'syllable_initial', 'syllable_final', 'is_stressed',
    'is_phrase_boundary', 'is_word_boundary',
    'context_sentence_position', 'context_word_position',
    'is_vowel', 'is_voiced', 'is_plosive',
    'is_schwa', 'velar_fricative_next', 'vowel_height_high', 'vowel_height_low', 'vowel_tense',
    'sonority_of_nucleus',
    'distance_to_stress_norm', 'stress_pattern_class'
]

TARGET_NAMES = ['F0', 'Duration', 'Energy']

MODEL_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/training/kan")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_feature_importance(model_path: Path) -> dict:
    """
    Compute feature importance from NASM spline coefficients.
    Importance = mean absolute coefficient magnitude per feature.
    """
    state = torch.load(model_path, map_location='cpu')
    
    # KAN coefficients: shape (in_features, out_features, grid_size)
    coef = state['kan.coef']  # (30, 3, 8)
    scale = state['kan.scale']  # (30, 3) - linear residual term
    
    # Importance per feature per target: |spline coef| + |linear scale|
    # Mean over grid dimension for spline coefficients
    spline_importance = torch.abs(coef).mean(dim=2)  # (30, 3)
    linear_importance = torch.abs(scale)  # (30, 3)
    
    # Combined importance (spline + linear)
    total_importance = spline_importance + linear_importance  # (30, 3)
    
    # Normalize per target (percentage)
    importance_pct = total_importance / total_importance.sum(dim=0, keepdim=True) * 100
    
    return importance_pct.numpy()


def main():
    # Find all seed models
    seed_dirs = sorted(MODEL_DIR.glob("word_seed*"))
    if not seed_dirs:
        # Try single model file
        if (MODEL_DIR / "best_model.pt").exists():
            seed_dirs = [MODEL_DIR]
        else:
            print("No models found!")
            return
    
    print(f"Found {len(seed_dirs)} model(s)")
    
    # Collect importance across seeds
    all_importance = []
    for seed_dir in seed_dirs:
        model_path = seed_dir / "best_model.pt" if seed_dir != MODEL_DIR else MODEL_DIR / "best_model.pt"
        if model_path.exists():
            importance = compute_feature_importance(model_path)
            all_importance.append(importance)
            print(f"  Loaded: {model_path}")
    
    if not all_importance:
        print("No valid models found!")
        return
    
    # Average across seeds
    mean_importance = np.mean(all_importance, axis=0)  # (30, 3)
    
    # Create results dict
    results = {
        'feature_names': FEATURE_NAMES,
        'target_names': TARGET_NAMES,
        'importance_matrix': mean_importance.tolist(),
        'n_seeds': len(all_importance)
    }
    
    # Save JSON
    with open(OUTPUT_DIR / "feature_importance.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print LaTeX table (Top 5 per target)
    print("\n" + "=" * 80)
    print("TABLE 5.3: Feature Contributions (% of total, top 5 per target)")
    print("=" * 80)
    
    print("\n% LaTeX format:")
    print("\\begin{tabular}{clr|clr|clr}")
    print("\\toprule")
    print("\\multicolumn{3}{c|}{$F_0$} & \\multicolumn{3}{c|}{Duration} & \\multicolumn{3}{c}{Energy} \\\\")
    print("Rank & Feature & \\% & Rank & Feature & \\% & Rank & Feature & \\% \\\\")
    print("\\midrule")
    
    # Get top 5 for each target
    top_features = []
    for t in range(3):
        sorted_idx = np.argsort(mean_importance[:, t])[::-1]
        top_features.append([(FEATURE_NAMES[i], mean_importance[i, t]) for i in sorted_idx[:5]])
    
    for rank in range(5):
        row = []
        for t in range(3):
            feat, pct = top_features[t][rank]
            # Clean feature name for LaTeX
            feat_clean = feat.replace('_', '\\_')
            row.append(f"{rank+1} & {feat_clean} & {pct:.1f}")
        print(" & ".join(row) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    
    # Also print plain text version
    print("\n" + "-" * 80)
    print("Plain text version:")
    print("-" * 80)
    
    for t, target in enumerate(TARGET_NAMES):
        print(f"\n{target}:")
        sorted_idx = np.argsort(mean_importance[:, t])[::-1]
        for rank, i in enumerate(sorted_idx[:10]):
            print(f"  {rank+1:2d}. {FEATURE_NAMES[i]:30s} {mean_importance[i, t]:5.1f}%")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary: Top contributors per target")
    print("=" * 80)
    
    for t, target in enumerate(TARGET_NAMES):
        sorted_idx = np.argsort(mean_importance[:, t])[::-1]
        top3_pct = sum(mean_importance[i, t] for i in sorted_idx[:3])
        print(f"{target}: Top 3 features account for {top3_pct:.1f}% of contribution")
    
    print(f"\nResults saved to: {OUTPUT_DIR / 'feature_importance.json'}")


if __name__ == "__main__":
    main()


