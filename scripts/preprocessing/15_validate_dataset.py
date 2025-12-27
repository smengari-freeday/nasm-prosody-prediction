#!/usr/bin/env python3
# validate phoneme feature dataset (optional QA step)
import json
import numpy as np
from pathlib import Path

# paths
FEATURES_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/features/phoneme_level")
FEATURE_DIM = 30


def validate_file(npy_path):
    """Check single feature file for issues."""
    issues = []
    try:
        features = np.load(npy_path)
        
        if features.ndim != 2:
            issues.append(f"wrong ndim: {features.ndim}")
        elif features.shape[1] != FEATURE_DIM:
            issues.append(f"wrong dim: {features.shape[1]}")
        
        if np.any(np.isnan(features)):
            issues.append(f"NaN: {np.sum(np.isnan(features))}")
        if np.any(np.isinf(features)):
            issues.append(f"Inf: {np.sum(np.isinf(features))}")
        
        # check matching phoneme labels
        json_path = npy_path.with_name(npy_path.stem.replace("_features", "_phonemes") + ".json")
        if json_path.exists():
            with open(json_path) as f:
                phonemes = json.load(f)
        if len(phonemes) != features.shape[0]:
            issues.append(f"phoneme mismatch: {len(phonemes)} vs {features.shape[0]}")
        
    except Exception as e:
        issues.append(f"error: {e}")
    
    return issues


def main():
    npy_files = list(FEATURES_DIR.glob("*_features.npy"))
    print(f"Validating {len(npy_files)} files...")
    
    valid, invalid = 0, 0
    all_issues = []
    total_phonemes = 0
    
    for npy in npy_files:
        issues = validate_file(npy)
        if issues:
            invalid += 1
            all_issues.append((npy.name, issues))
        else:
            valid += 1
            total_phonemes += np.load(npy).shape[0]
    
    print(f"\nValid: {valid}, Invalid: {invalid}")
    print(f"Total phonemes: {total_phonemes:,}")
    
    if all_issues:
        print("\nIssues:")
        for name, issues in all_issues[:10]:
            print(f"  {name}: {', '.join(issues)}")


if __name__ == "__main__":
    main()
