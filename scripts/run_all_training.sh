#!/bin/bash
# Sequential training script for thesis experiments
# Seeds: 13, 22, 42, 111, 222, 333
# Runs NASM and MLP for phoneme-level, then word-level

cd /Users/s.mengari/Desktop/CODE2/scripts
mkdir -p logs

SEEDS=(13 22 42 111 222 333)
EPOCHS=100
PATIENCE=15
BATCH=32

# Learning rates (justified from hyperparameter sweep + methodological considerations)
# NASM: lr=1e-2 works well for additive spline models (validated on Chapter 2 data)
# MLP:  lr=1e-3 is standard for Adam with small MLPs (1e-2 is too aggressive)
LR_NASM=1e-2
LR_MLP=1e-3

echo "=============================================="
echo "Starting Sequential Training"
echo "Seeds: ${SEEDS[*]}"
echo "NASM LR: $LR_NASM, MLP LR: $LR_MLP"
echo "=============================================="

# --- PHONEME LEVEL ---
echo ""
echo "=== PHONEME LEVEL TRAINING ==="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- NASM phoneme seed=$seed ---"
    python3 training/02_train_kan.py \
        --epochs $EPOCHS --level phoneme --seed $seed --patience $PATIENCE \
        --lr $LR_NASM --batch-size $BATCH --num-basis 8 --spline-degree 3 \
        2>&1 | tee logs/nasm_phoneme_seed${seed}.log
    
    echo ""
    echo "--- MLP phoneme seed=$seed ---"
    python3 training/03_train_mlp_baseline.py \
        --epochs $EPOCHS --level phoneme --seed $seed --patience $PATIENCE \
        --lr $LR_MLP --batch-size $BATCH --hidden-dim 8 --dropout 0.0 \
        2>&1 | tee logs/mlp_phoneme_seed${seed}.log
done

# --- WORD LEVEL ---
echo ""
echo "=== WORD LEVEL TRAINING ==="

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
echo "All training complete!"
echo "Logs saved to: logs/"
echo "=============================================="
