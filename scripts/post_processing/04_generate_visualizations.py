#!/usr/bin/env python3
# generate thesis-quality visualizations for model comparison
#
# creates:
# 1. learning curves (train/val loss per seed)
# 2. R² comparison bar chart
# 3. effect size plot (Cohen's d)
# 4. seed stability boxplots
# 5. B-spline plots (90 total: 30 features × 3 targets) with ±1 SD bands
# 6. multi-panel spline summary (6 key features per target)
# 7. seed overlay "spaghetti" plots :) 
#
# usage:
#   python 04_generate_visualizations.py

import json
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import Dict, List

# paths (hardcoded)
PROJECT_ROOT = Path("/Users/s.mengari/Desktop/CODE")
RESULTS_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/training_full")
COMPARISON_CSV = Path("/Users/s.mengari/Desktop/CODE2/results/model_comparison.csv")
STATS_FILE = Path("/Users/s.mengari/Desktop/CODE2/results/statistical_tests/kan_vs_mlp_tests.csv")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/visualizations")
SEEDS = [42, 43, 44, 45, 46, 47]

# feature names (30 features)
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

# display names for thesis figures
DISPLAY_NAMES = {
    'primary_stress_pos': 'Primary Stress Position',
    'word_frequency': 'Word Frequency',
    'syllable_count': 'Syllable Count',
    'phoneme_count': 'Phoneme Count',
    'pos_noun': 'POS: Noun',
    'pos_verb': 'POS: Verb',
    'pos_adjective': 'POS: Adjective',
    'pos_adverb': 'POS: Adverb',
    'pos_pronoun': 'POS: Pronoun',
    'pos_preposition': 'POS: Preposition',
    'pos_determiner': 'POS: Determiner',
    'pos_auxiliary': 'POS: Auxiliary',
    'syllable_initial': 'Syllable-Initial',
    'syllable_final': 'Syllable-Final',
    'is_stressed': 'Stressed Syllable',
    'is_phrase_boundary': 'Phrase Boundary',
    'is_word_boundary': 'Word Boundary',
    'context_sentence_position': 'Sentence Position',
    'context_word_position': 'Word Position',
    'is_vowel': 'Vowel',
    'is_voiced': 'Voiced',
    'is_plosive': 'Plosive',
    'is_schwa': 'Schwa',
    'velar_fricative_next': 'Velar Fricative Next',
    'vowel_height_high': 'High Vowel',
    'vowel_height_low': 'Low Vowel',
    'vowel_tense': 'Tense Vowel',
    'sonority_of_nucleus': 'Sonority',
    'distance_to_stress_norm': 'Distance to Stress',
    'stress_pattern_class': 'Stress Pattern'
}

TARGETS = ['f0', 'duration', 'energy']
TARGET_NAMES = {'f0': 'F0', 'duration': 'Duration', 'energy': 'Energy'}

# key features for multi-panel (6 per target)
KEY_FEATURES = {
    'f0': [0, 14, 28, 24, 4, 17],      # stress pos, is_stressed, dist_to_stress, high_vowel, noun, sentence_pos
    'duration': [2, 3, 14, 15, 13, 17],  # syl_count, phone_count, is_stressed, phrase_boundary, syl_final, sentence_pos
    'energy': [14, 19, 27, 4, 15, 17]    # is_stressed, is_vowel, sonority, noun, phrase_boundary, sentence_pos
}

# style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'kan': '#2E86AB', 'mlp': '#A23B72'}


def create_learning_curves(results_dir: Path, output_dir: Path, seeds: list):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for model_type, color in [('kan', COLORS['kan']), ('mlp', COLORS['mlp'])]:
        col = 0 if model_type == 'kan' else 1
        
        for seed in seeds:
            hist_path = results_dir / f'{model_type}_seed{seed}' / 'training_history.json'
            
            if not hist_path.exists():
                continue
            
            with open(hist_path) as f:
                hist = json.load(f)
            
            if model_type == 'kan':
                train_losses = [h['loss'] for h in hist.get('train', [])]
                val_losses = [h['loss'] for h in hist.get('val', [])]
            else:
                train_losses = hist.get('train_losses', [])
                val_losses = hist.get('val_losses', [])
            
            epochs = range(1, len(train_losses) + 1)
            axes[0, col].plot(epochs, train_losses, alpha=0.6, color=color, label=f'Seed {seed}')
            axes[1, col].plot(epochs, val_losses, alpha=0.6, color=color, label=f'Seed {seed}')
    
    titles = ['KAN Training Loss', 'MLP Training Loss']
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold')
        axes[0, col].set_xlabel('Epoch')
        axes[0, col].set_ylabel('Loss')
        axes[0, col].legend(fontsize=8, loc='upper right')
        
        axes[1, col].set_title(title.replace('Training', 'Validation'), fontsize=14, fontweight='bold')
        axes[1, col].set_xlabel('Epoch')
        axes[1, col].set_ylabel('Loss')
        axes[1, col].legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'learning_curves.pdf', bbox_inches='tight')
    plt.close()


def create_r2_comparison(comparison_csv: Path, output_dir: Path):
    df = pd.read_csv(comparison_csv)
    
    kan_means = df[df['model'] == 'KAN'].groupby('target')['r2'].mean()
    mlp_means = df[df['model'] == 'MLP'].groupby('target')['r2'].mean()
    kan_stds = df[df['model'] == 'KAN'].groupby('target')['r2'].std()
    mlp_stds = df[df['model'] == 'MLP'].groupby('target')['r2'].std()
    
    targets = ['f0', 'duration', 'energy']
    x = np.arange(len(targets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, [kan_means[t] for t in targets], width,
                   yerr=[kan_stds[t] for t in targets],
                   label='KAN (NASM)', color=COLORS['kan'], capsize=5)
    bars2 = ax.bar(x + width/2, [mlp_means[t] for t in targets], width,
                   yerr=[mlp_stds[t] for t in targets],
                   label='MLP', color=COLORS['mlp'], capsize=5)
    
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Model Comparison: KAN vs MLP (6 seeds)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['F0', 'Duration', 'Energy'], fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'r2_comparison.pdf', bbox_inches='tight')
    plt.close()


def create_effect_size_plot(stats_csv: Path, output_dir: Path):
    stats_df = pd.read_csv(stats_csv)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    targets = stats_df['target'].values
    cohens_d = stats_df['cohens_d'].values
    
    colors = [COLORS['kan'] if d > 0 else COLORS['mlp'] for d in cohens_d]
    y_pos = np.arange(len(targets))
    
    ax.barh(y_pos, cohens_d, color=colors, alpha=0.7, height=0.6)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    for thresh in [0.2, 0.5, 0.8]:
        ax.axvline(x=thresh, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=-thresh, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.upper() for t in targets], fontsize=12)
    ax.set_xlabel("Cohen's d (effect size)", fontsize=12)
    ax.set_title('Effect Size: KAN vs MLP (positive = KAN better)', fontsize=14, fontweight='bold')
    
    for i, (t, d) in enumerate(zip(targets, cohens_d)):
        label = 'KAN' if d > 0 else 'MLP'
        offset = 5 if d >= 0 else -5
        ha = 'left' if d >= 0 else 'right'
        ax.annotate(f'{d:.2f} ({label})', xy=(d, i), xytext=(offset, 0),
                   textcoords='offset points', va='center', ha=ha, fontsize=10)
    
    ax.text(0.02, 0.98, 'Effect sizes: |d|<0.2=negligible, 0.2-0.5=small, 0.5-0.8=medium, >0.8=large',
            transform=ax.transAxes, fontsize=8, va='top', style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'effect_sizes.pdf', bbox_inches='tight')
    plt.close()


def create_seed_stability_boxplots(comparison_csv: Path, output_dir: Path):
    df = pd.read_csv(comparison_csv)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for i, target in enumerate(['f0', 'duration', 'energy']):
        target_df = df[df['target'] == target]
        
        kan_vals = target_df[target_df['model'] == 'KAN']['r2'].values
        mlp_vals = target_df[target_df['model'] == 'MLP']['r2'].values
        
        bp = axes[i].boxplot([kan_vals, mlp_vals], labels=['KAN', 'MLP'],
                             patch_artist=True, widths=0.6)
        
        bp['boxes'][0].set_facecolor(COLORS['kan'])
        bp['boxes'][1].set_facecolor(COLORS['mlp'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        axes[i].set_title(f'{target.upper()}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('R²', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        for j, (vals, color) in enumerate([(kan_vals, COLORS['kan']), (mlp_vals, COLORS['mlp'])]):
            x = np.random.normal(j+1, 0.04, size=len(vals))
            axes[i].scatter(x, vals, alpha=0.6, color=color, s=50, zorder=3)
    
    plt.suptitle('R² Distribution Across Seeds (n=6)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'seed_stability_boxplots.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'seed_stability_boxplots.pdf', bbox_inches='tight')
    plt.close()


# ============================================================================
# B-SPLINE VISUALIZATIONS (KAN interpretability)
# ============================================================================

def load_kan_model(model_path: Path, in_features: int = 30, grid_size: int = 8, spline_order: int = 2):
    """Load trained KAN model from checkpoint."""
    # import model class dynamically
    training_dir = PROJECT_ROOT / 'scripts' / 'training'
    sys.path.insert(0, str(training_dir))
    
    from true_kan_heads_vectorized import TrueKANHead
    
    class KANProsodyPredictor(torch.nn.Module):
        def __init__(self, in_features, grid_size, spline_order):
            super().__init__()
            self.kan = TrueKANHead(
                in_features=in_features,
                out_features=3,
                grid_size=grid_size,
                spline_order=spline_order
            )
            self.dropout = torch.nn.Dropout(0.1)
        
        def forward(self, x, attention_mask=None):
            return self.kan(x)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    # handle old parameter names (coefficients → coef, per_feature_linear_weights → scale)
    renamed_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('coefficients', 'coef').replace('per_feature_linear_weights', 'scale')
        renamed_state_dict[new_k] = v
    
    model = KANProsodyPredictor(in_features, grid_size, spline_order)
    model.load_state_dict(renamed_state_dict, strict=False)
    model.eval()
    return model


def extract_spline_curve(model, feature_idx: int, target_idx: int, n_points: int = 200):
    """Extract spline curve for a feature-target pair."""
    x_eval, y_curve = model.kan.get_curve_for_plotting(feature_idx, target_idx, n_points=n_points)
    return x_eval.numpy(), y_curve.numpy()


def load_all_kan_models(results_dir: Path, seeds: list) -> Dict[int, torch.nn.Module]:
    """Load all KAN models across seeds."""
    models = {}
    
    for seed in seeds:
        model_dir = results_dir / f'kan_seed{seed}'
        model_path = model_dir / 'best_model.pt'
        
        if not model_path.exists():
            continue
        
        # get config from training history (using Annotated KAN naming)
        config_path = model_dir / 'training_history.json'
        grid_size, spline_order = 8, 2  # default: spline_order=2 (quadratic B-splines)
        if config_path.exists():
            with open(config_path) as f:
                hist = json.load(f)
                if 'config' in hist:
                    # support both old (num_basis) and new (grid_size) naming
                    grid_size = hist['config'].get('grid_size', hist['config'].get('num_basis', 8))
                    spline_order = hist['config'].get('spline_order', hist['config'].get('spline_degree', 3))
        
        try:
            model = load_kan_model(model_path, 30, grid_size, spline_order)
            models[seed] = model
        except Exception as e:
            print(f"  ⚠️  Could not load seed {seed}: {e}")
    
    return models


def extract_all_splines(models: Dict[int, torch.nn.Module]) -> Dict[str, Dict]:
    """Extract all spline curves across seeds, features, and targets."""
    splines = {}
    n_features = 30
    
    for target_idx, target in enumerate(TARGETS):
        splines[target] = {}
        
        for feature_idx in range(n_features):
            curves = []
            x_grid = None
            
            for seed, model in models.items():
                try:
                    x, y = extract_spline_curve(model, feature_idx, target_idx)
                    curves.append(y)
                    if x_grid is None:
                        x_grid = x
                except Exception:
                    pass
            
            if len(curves) > 0:
                curves_array = np.array(curves)
                splines[target][feature_idx] = {
                    'x': x_grid,
                    'curves': curves,
                    'mean': np.mean(curves_array, axis=0),
                    'std': np.std(curves_array, axis=0),
                    'n_seeds': len(curves)
                }
    
    return splines


def create_individual_spline_plots(splines: Dict, output_dir: Path):
    """Create 90 individual spline plots (30 features × 3 targets) with blue ±1 SD bands."""
    spline_dir = output_dir / 'splines'
    spline_dir.mkdir(parents=True, exist_ok=True)
    
    for target in TARGETS:
        target_dir = spline_dir / target
        target_dir.mkdir(exist_ok=True)
        
        for feature_idx in range(30):
            if feature_idx not in splines[target]:
                continue
            
            data = splines[target][feature_idx]
            x = data['x']
            mean = data['mean']
            std = data['std']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # blueish hue style with ±1 SD band
            ax.fill_between(x, mean - std, mean + std, alpha=0.3, color='#2E86AB', label='±1 SD')
            ax.plot(x, mean, color='#2E86AB', linewidth=2, label='Mean')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            feature_name = FEATURE_NAMES[feature_idx]
            display_name = DISPLAY_NAMES.get(feature_name, feature_name)
            
            ax.set_title(f'{display_name} → {TARGET_NAMES[target]}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Value (normalized)', fontsize=11)
            ax.set_ylabel('Contribution to Target', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(target_dir / f'feature_{feature_idx:02d}_{feature_name}.png', dpi=150, bbox_inches='tight')
            plt.close()


def create_spline_summary_grid(splines: Dict, output_dir: Path):
    """Create 6×5 summary grid of all 30 splines per target."""
    spline_dir = output_dir / 'splines'
    spline_dir.mkdir(parents=True, exist_ok=True)
    
    for target in TARGETS:
        fig, axes = plt.subplots(6, 5, figsize=(20, 24))
        axes = axes.flatten()
        
        for feature_idx in range(30):
            ax = axes[feature_idx]
            
            if feature_idx in splines[target]:
                data = splines[target][feature_idx]
                x = data['x']
                mean = data['mean']
                std = data['std']
                
                ax.fill_between(x, mean - std, mean + std, alpha=0.3, color='#2E86AB')
                ax.plot(x, mean, color='#2E86AB', linewidth=1.5)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                
                feature_name = FEATURE_NAMES[feature_idx]
                display_name = DISPLAY_NAMES.get(feature_name, feature_name)
                ax.set_title(display_name, fontsize=9, fontweight='bold')
                ax.tick_params(axis='both', labelsize=7)
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')
        
        plt.suptitle(f'{TARGET_NAMES[target]} - Per-Feature B-Splines (Mean ± 1 SD)', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(spline_dir / f'summary_grid_{target}.png', dpi=150, bbox_inches='tight')
        plt.savefig(spline_dir / f'summary_grid_{target}.pdf', bbox_inches='tight')
        plt.close()


def create_multipanel_splines(splines: Dict, output_dir: Path):
    """Create 3×2 multi-panel figures with 6 key features per target."""
    spline_dir = output_dir / 'splines'
    spline_dir.mkdir(parents=True, exist_ok=True)
    
    for target in TARGETS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        key_indices = KEY_FEATURES[target]
        
        for panel_idx, feature_idx in enumerate(key_indices):
            ax = axes[panel_idx]
            
            if feature_idx in splines[target]:
                data = splines[target][feature_idx]
                x = data['x']
                mean = data['mean']
                std = data['std']
                
                ax.fill_between(x, mean - std, mean + std, alpha=0.3, color='#2E86AB', label='±1 SD')
                ax.plot(x, mean, color='#2E86AB', linewidth=2, label='Mean')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                feature_name = FEATURE_NAMES[feature_idx]
                display_name = DISPLAY_NAMES.get(feature_name, feature_name)
                ax.set_title(display_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('Feature Value', fontsize=10)
                ax.set_ylabel('Contribution', fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{TARGET_NAMES[target]} - Key Feature Effects', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(spline_dir / f'multipanel_{target}.png', dpi=150, bbox_inches='tight')
        plt.savefig(spline_dir / f'multipanel_{target}.pdf', bbox_inches='tight')
        plt.close()


def create_seed_overlay_plots(splines: Dict, output_dir: Path):
    """Create seed overlay 'spaghetti' plots showing individual seed curves."""
    spline_dir = output_dir / 'splines' / 'seed_overlays'
    spline_dir.mkdir(parents=True, exist_ok=True)
    
    seed_colors = plt.cm.tab10(np.linspace(0, 1, len(SEEDS)))
    
    for target in TARGETS:
        for feature_idx in range(30):
            if feature_idx not in splines[target]:
                continue
            
            data = splines[target][feature_idx]
            x = data['x']
            curves = data['curves']
            mean = data['mean']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # individual seed curves
            for i, curve in enumerate(curves):
                ax.plot(x, curve, color=seed_colors[i % len(seed_colors)], 
                       alpha=0.5, linewidth=1, label=f'Seed {SEEDS[i]}')
            
            # mean curve (bold black)
            ax.plot(x, mean, 'k-', linewidth=3, label='Mean', zorder=10)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            feature_name = FEATURE_NAMES[feature_idx]
            display_name = DISPLAY_NAMES.get(feature_name, feature_name)
            
            ax.set_title(f'{display_name} → {TARGET_NAMES[target]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature Value (normalized)', fontsize=10)
            ax.set_ylabel('Contribution', fontsize=10)
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(spline_dir / f'{target}_feature_{feature_idx:02d}_{feature_name}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()


def create_bspline_visualizations(results_dir: Path, output_dir: Path, seeds: list):
    """Main function to create all B-spline visualizations."""
    print("Loading KAN models...")
    models = load_all_kan_models(results_dir, seeds)
    
    if len(models) == 0:
        print("  ⚠️  No KAN models found, skipping B-spline visualizations")
        return
    
    print(f"  Loaded {len(models)}/{len(seeds)} models")
    
    print("Extracting spline curves...")
    splines = extract_all_splines(models)
    
    print("Creating individual spline plots (90 total)...")
    create_individual_spline_plots(splines, output_dir)
    
    print("Creating summary grids...")
    create_spline_summary_grid(splines, output_dir)
    
    print("Creating multi-panel figures...")
    create_multipanel_splines(splines, output_dir)
    
    print("Creating seed overlay plots...")
    create_seed_overlay_plots(splines, output_dir)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generate Visualizations")
    print("=" * 60)
    
    # model comparison plots
    print("\n[1/5] Creating learning curves...")
    create_learning_curves(RESULTS_DIR, OUTPUT_DIR, SEEDS)
    
    print("[2/5] Creating R² comparison bar chart...")
    create_r2_comparison(COMPARISON_CSV, OUTPUT_DIR)
    
    print("[3/5] Creating effect size plot...")
    create_effect_size_plot(STATS_FILE, OUTPUT_DIR)
    
    print("[4/5] Creating seed stability boxplots...")
    create_seed_stability_boxplots(COMPARISON_CSV, OUTPUT_DIR)
    
    # B-spline interpretability plots (KAN-specific)
    print("\n[5/5] Creating B-spline visualizations...")
    create_bspline_visualizations(RESULTS_DIR, OUTPUT_DIR, SEEDS)
    
    print()
    print("=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nOutput structure:")
    print("  visualizations/")
    print("    learning_curves.png/pdf")
    print("    r2_comparison.png/pdf")
    print("    effect_sizes.png/pdf")
    print("    seed_stability_boxplots.png/pdf")
    print("    splines/")
    print("      f0/, duration/, energy/  (90 individual plots)")
    print("      summary_grid_*.png/pdf   (30 features per target)")
    print("      multipanel_*.png/pdf     (6 key features per target)")
    print("      seed_overlays/           (spaghetti plots)")


if __name__ == '__main__':
    main()
