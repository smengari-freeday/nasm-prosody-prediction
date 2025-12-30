#!/bin/bash
# FULL TRAINING SCRIPT - Both Phoneme and Word Level
# With CORRECT feature mapping (fixed 2024-12-28 21:08)
# Optimal hyperparameters from sweep

cd /Users/s.mengari/Desktop/CODE2/scripts
mkdir -p logs

SEEDS=(13 22 42 111 222 333)
EPOCHS=100
PATIENCE=15
BATCH=32

# Optimal learning rates
LR_NASM=1e-2   # From hyperparameter sweep
LR_MLP=1e-3    # Standard for Adam

echo "=============================================="
echo "FULL TRAINING: Phoneme + Word Level"
echo "=============================================="
echo "Seeds: ${SEEDS[*]}"
echo "NASM LR: $LR_NASM, MLP LR: $LR_MLP"
echo "Epochs: $EPOCHS, Patience: $PATIENCE"
echo "=============================================="
echo ""

# ========================================
# PHONEME LEVEL TRAINING
# ========================================
echo "=== PHONEME LEVEL ==="
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

# ========================================
# WORD LEVEL TRAINING
# ========================================
echo ""
echo "=== WORD LEVEL ==="
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
echo "ALL TRAINING COMPLETE!"
echo "=============================================="
echo "Results: results/training/kan/ and results/training/mlp/"
echo "Logs: scripts/logs/"
