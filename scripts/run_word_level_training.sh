#!/bin/bash
# WORD-LEVEL ONLY training script
# Phoneme-level is already complete
# This script runs word-level training with FIXED word boundary detection

cd /Users/s.mengari/Desktop/CODE2/scripts
mkdir -p logs

SEEDS=(13 22 42 111 222 333)
EPOCHS=100
PATIENCE=15
BATCH=32

# Learning rates (justified from hyperparameter sweep)
LR_NASM=1e-2
LR_MLP=1e-3

echo "=============================================="
echo "WORD-LEVEL Training (with FIXED word boundaries)"
echo "Bug fix: Word boundary detection now uses syllable_count changes"
echo "Expected ~103k word samples (was incorrectly 345k before fix)"
echo "Seeds: ${SEEDS[*]}"
echo "=============================================="

# --- WORD LEVEL ---
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- NASM word seed=$seed ---"
    python3 training/02_train_kan.py \
        --epochs $EPOCHS --level word --seed $seed --patience $PATIENCE \
        --lr $LR_NASM --batch-size $BATCH --num-basis 8 --spline-degree 3 \
        2>&1 | tee logs/nasm_word_seed${seed}.log
    
    echo ""
    echo "--- MLP word seed=$seed ---"
    python3 training/03_train_mlp_baseline.py \
        --epochs $EPOCHS --level word --seed $seed --patience $PATIENCE \
        --lr $LR_MLP --batch-size $BATCH --hidden-dim 8 --dropout 0.0 \
        2>&1 | tee logs/mlp_word_seed${seed}.log
done

echo ""
echo "=============================================="
echo "Word-level training complete!"
echo "Logs saved to: logs/"
echo "=============================================="

