#!/bin/bash
# KAN Prosody Prediction Pipeline
# Runs all scripts in sequence from raw data to final visualizations

set -e  # Exit on any error

# Configuration
EPOCHS=50
SEEDS="42 43 44 45 46 47"
SKIP_PREPROCESSING=false
SKIP_TRAINING=false
SKIP_POSTPROCESSING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-preprocessing) SKIP_PREPROCESSING=true; shift ;;
        --skip-training) SKIP_TRAINING=true; shift ;;
        --skip-postprocessing) SKIP_POSTPROCESSING=true; shift ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        --seeds) SEEDS=$2; shift 2 ;;
        -h|--help)
            echo "Usage: ./run_pipeline.sh [options]"
            echo "Options:"
            echo "  --skip-preprocessing    Skip steps 01-15"
            echo "  --skip-training         Skip KAN and MLP training"
            echo "  --skip-postprocessing   Skip evaluation and visualizations"
            echo "  --epochs N              Training epochs (default: 50)"
            echo "  --seeds \"42 43 ...\"     Seeds for multi-seed training"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " KAN PROSODY PREDICTION PIPELINE"
echo "============================================================"
echo "Config: epochs=$EPOCHS, seeds=$SEEDS"
echo ""

# ============================================================
# PREPROCESSING
# ============================================================
if [ "$SKIP_PREPROCESSING" = false ]; then
    echo "[STAGE 1/3] PREPROCESSING"
    echo "------------------------------------------------------------"
    
    cd scripts/preprocessing
    
    echo "[01] Splitting chapters..."
    python3 01_split_gutenberg_into_chapters.py
    
    echo "[02] Extracting unique words..."
    python3 02_extract_unique_words.py
    
    echo "[03] Converting G2P to dictionary..."
    python3 03_convert_g2p_to_dict.py
    
    echo "[04] Applying soft-g corrections..."
    python3 04_apply_soft_g_corrections.py
    
    echo "[05] Whisper transcription..."
    python3 05_whisper_transcribe.py
    
    echo "[06] Extracting Whisper text..."
    python3 06_extract_whisper_text.py
    
    echo "[07] Aligning words to CSV..."
    python3 07_align_words_to_csv.py
    
    echo "[08] Building MFA utterances..."
    python3 08_build_mfa_utterances.py
    
    echo "[09] Running MFA alignment..."
    python3 09_run_mfa.py
    
    echo "[10] Extracting F0..."
    python3 10_extract_f0.py
    
    echo "[11] Extracting duration and energy..."
    python3 11_extract_duration_energy.py
    
    echo "[12] Matching WebCelex..."
    python3 12_match_webcelex.py
    
    echo "[13] Building features..."
    python3 13_build_features.py
    
    echo "[14] Validating dataset..."
    python3 14_validate_dataset.py
    
    echo "[15] Creating splits..."
    python3 15_create_splits.py
    
    cd ../..
    echo "✓ Preprocessing complete"
    echo ""
fi

# ============================================================
# TRAINING
# ============================================================
if [ "$SKIP_TRAINING" = false ]; then
    echo "[STAGE 2/3] TRAINING"
    echo "------------------------------------------------------------"
    
    cd scripts/training
    
    # Train KAN for each seed
    echo "Training KAN models..."
    for seed in $SEEDS; do
        echo "  [KAN] seed=$seed"
        python3 02_train_kan.py --epochs $EPOCHS --seed $seed
    done
    
    # Train MLP for each seed
    echo "Training MLP baselines..."
    for seed in $SEEDS; do
        echo "  [MLP] seed=$seed"
        python3 03_train_mlp_baseline.py --epochs $EPOCHS --seed $seed
    done
    
    cd ../..
    echo "✓ Training complete"
    echo ""
fi

# ============================================================
# POST-PROCESSING
# ============================================================
if [ "$SKIP_POSTPROCESSING" = false ]; then
    echo "[STAGE 3/3] POST-PROCESSING"
    echo "------------------------------------------------------------"
    
    cd scripts/post_processing
    
    echo "[01] Aggregating results..."
    python3 01_aggregate_results.py
    
    echo "[02] Statistical tests..."
    python3 02_statistical_tests.py
    
    echo "[03] Test set evaluation..."
    python3 03_test_set_evaluation.py
    
    echo "[04] Generating visualizations..."
    python3 04_generate_visualizations.py
    
    cd ../..
    echo "✓ Post-processing complete"
    echo ""
fi

# ============================================================
# SUMMARY
# ============================================================
echo "============================================================"
echo " PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  results/model_comparison.csv          - Metrics by model/seed"
echo "  results/statistical_tests/            - KAN vs MLP comparisons"
echo "  results/test_set_evaluation.json      - Test set R²"
echo "  results/visualizations/               - All plots"
echo ""
echo "Key outputs:"
ls -la results/visualizations/*.png 2>/dev/null | head -5 || echo "  (visualizations in results/visualizations/)"
echo ""

