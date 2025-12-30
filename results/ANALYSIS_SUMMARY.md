# Duration Prediction Analysis Summary

**Date:** December 27, 2025  
**Dataset:** Dutch audiobook corpus (71,330 train / 9,197 test words)  
**Target:** Word-level log duration

---

## 1. Multi-Seed Experiment Results

### Test Set R² (Duration)

| Model  | Mean R² | Std    | 95% CI              |
|--------|---------|--------|---------------------|
| Linear | 0.188   | 0.000  | [0.188, 0.188]      |
| NASM   | 0.296   | 0.005  | [0.292, 0.300]      |
| MLP    | 0.197   | 0.017  | [0.183, 0.210]      |

### Statistical Tests

- **NASM vs Linear:** Δ = +0.108, t = 50.8, p < 10⁻⁷ ✓
- **NASM vs MLP:** Δ = +0.100, t = 16.4, p < 10⁻⁵ ✓

### Key Finding
NASM significantly outperforms both baselines with very low variance across seeds.

---

## 2. Ablation Study Results

### Ablation Table

| Features Removed            | Linear R² | NASM R² | NASM Drop |
|-----------------------------|-----------|---------|-----------|
| Full (all features)         | 0.188     | 0.291   | baseline  |
| – phone_count               | 0.189     | 0.300   | +0.009    |
| – syllable_count            | 0.187     | 0.300   | +0.009    |
| – phone+syllable            | 0.186     | 0.304   | +0.013    |
| – stress_features           | 0.187     | 0.304   | +0.013    |
| – boundary_features         | 0.130     | 0.257   | **-0.034**|
| – position_features         | 0.188     | 0.305   | +0.014    |
| – all_prosody_cues          | 0.129     | 0.243   | **-0.048**|
| – lexical_features          | 0.181     | 0.289   | -0.002    |

### Key Findings

1. **Lexical length is NOT the main driver:**  
   Removing phone_count + syllable_count causes **no drop** (even slight improvement).

2. **Boundary features are most important:**  
   Removing boundary features causes 3.4% drop (-0.034).

3. **Prosodic cues are critical:**  
   Removing all prosody cues causes 4.8% drop (-0.048).

4. **NASM advantage is robust:**  
   NASM maintains +0.10 to +0.13 advantage over linear across all ablations.

---

## 3. Phrase-Final Lengthening Analysis

### Residual Duration by Phrase Boundary

| Group              | n      | Mean Residual | Std    |
|--------------------|--------|---------------|--------|
| Non-boundary words | 7,478  | -0.142        | 0.822  |
| Phrase-boundary    | 1,719  | +0.616        | 1.392  |

### Statistical Test

- **t-statistic:** 29.70
- **p-value:** < 10⁻¹⁸⁵
- **Cohen's d:** 0.66 (medium-large effect)
- **Difference:** 0.76 log-seconds

### Interpretation

After controlling for lexical length (phone + syllable count), words at phrase boundaries are significantly longer. This confirms **phrase-final lengthening**, a well-documented prosodic phenomenon (Wightman et al., 1992).

---

## 4. Conclusions

### Evidence That NASM Learns Prosodic Timing

1. **Duration prediction is NOT simply word length:**  
   Ablation shows phone/syllable count contribute minimally.

2. **NASM captures phrase-final lengthening:**  
   Boundary features cause largest performance drop.

3. **Nonlinear effects are real:**  
   NASM consistently outperforms linear model by ~10%.

### Implications for Thesis

- Duration prediction captures genuine prosodic timing
- NASM's interpretability advantage is meaningful
- Feature importance aligns with linguistic expectations

---

## 5. Scripts Used

| Script | Purpose |
|--------|---------|
| `scripts/analysis/01_ablation_study.py` | IEEE-standard ablation study |
| `scripts/analysis/02_residual_analysis.py` | Phrase-final lengthening analysis |
| `scripts/analysis/03_multi_seed_experiment.py` | Multi-seed statistical validation |
| `scripts/training/01_data_loader.py` | Data loading and preprocessing |

### Output Files

| File | Description |
|------|-------------|
| `results/ablation_results.json` | Ablation study raw results |
| `results/ablation_table.txt` | Formatted ablation table |
| `results/residual_analysis.json` | Phrase-final lengthening stats |
| `results/phrase_final_lengthening.png` | Visualization |
| `results/multi_seed_results.json` | Multi-seed experiment results |

---

## 6. Reproducibility

All experiments use:
- **Seeds:** 42, 43, 44, 45, 46, 47
- **Train/test split:** Thesis-defined split file
- **Normalization:** Min-max on features, z-score on targets
- **Model:** NASM with 8 basis functions, degree 2 B-splines
- **Training:** Adam optimizer, lr=1e-3, 200 epochs

