# Duration & Energy Pipeline Options

## Word-Level Duration: Critical Choice

### Option A: Thesis-Consistent (DEFAULT)
**Setting:** `USE_PHYSICAL_WORD_DURATION = False` in `01_data_loader.py`

**Formula:** `word_dur = sum(log(d_i)) = log(∏d_i)`

**Problem:** Naive baseline R² = **0.98** (just count phones!)
- Duration prediction becomes trivial
- Model learns phone count, not prosody

**Use when:** Reproducing exact thesis numbers

---

### Option B: Physically Correct (RECOMMENDED for research)
**Setting:** `USE_PHYSICAL_WORD_DURATION = True` in `01_data_loader.py`

**Formula:** `word_dur = log(∑d_i) = log(physical word duration)`

**Advantage:** Naive baseline R² = **0.68** (30% residual variance to model)
- Duration prediction is meaningful
- Model learns speaking rate, final lengthening, etc.

**Implementation:** Computed from same stored log-durations:
```python
raw_durations = np.exp(log_durations)  # Convert back to seconds
word_duration = np.log(np.sum(raw_durations) + 1e-8)  # Physical duration
```

**No re-extraction needed!**

---

## Results Comparison (Chapter 2, word-level)

| Mode | Model R² | Baseline R² | Above Baseline |
|------|----------|-------------|----------------|
| Option A (thesis) | 0.743 | 0.98 | **-0.24 (worse!)** |
| Option B (physical) | **0.854** | 0.68 | **+0.17** |

---

## Energy Normalization

### Per-Chapter (thesis-consistent)
**Setting:** `SKIP_CHAPTER_NORM = False` in `12_extract_energy.py`
- Z-normalizes energy per speaker (chapter) before saving
- Minor leakage risk if splits are utterance-wise

### Raw dB (Option B)
**Setting:** `SKIP_CHAPTER_NORM = True` in `12_extract_energy.py`
- Saves raw dB values
- Z-normalization at training time using train-only stats
- No leakage

---

## Quick Switch

For word-level research, just change one flag:
```python
# In scripts/training/01_data_loader.py
USE_PHYSICAL_WORD_DURATION = True  # Enable Option B
```

For phoneme-level: Both options give same results (no aggregation).
