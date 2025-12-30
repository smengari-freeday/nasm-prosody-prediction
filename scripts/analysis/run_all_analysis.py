#!/usr/bin/env python3
"""
Run All Analysis Scripts
========================

Master script to run the complete analysis pipeline:
1. Ablation study
2. Residual analysis (phrase-final lengthening)
3. Multi-seed experiment

Usage:
    python run_all_analysis.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

def run_script(name):
    """Run a script and check for errors"""
    script_path = SCRIPT_DIR / name
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run([sys.executable, str(script_path)], 
                          capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: {name} failed with exit code {result.returncode}")
        return False
    return True

def main():
    print("="*70)
    print("RUNNING COMPLETE ANALYSIS PIPELINE")
    print("="*70)
    
    scripts = [
        '01_ablation_study.py',
        '02_residual_analysis.py',
        '03_multi_seed_experiment.py',
    ]
    
    success = True
    for script in scripts:
        if not run_script(script):
            success = False
            break
    
    print("\n" + "="*70)
    if success:
        print("ALL ANALYSIS SCRIPTS COMPLETED SUCCESSFULLY")
        print("\nOutput files in results/:")
        print("  - ablation_results.json")
        print("  - ablation_table.txt")
        print("  - residual_analysis.json")
        print("  - phrase_final_lengthening.png")
        print("  - multi_seed_results.json")
        print("  - ANALYSIS_SUMMARY.md")
    else:
        print("ANALYSIS PIPELINE FAILED")
    print("="*70)

if __name__ == '__main__':
    main()


