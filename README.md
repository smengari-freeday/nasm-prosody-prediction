# KAN Prosody Prediction Pipeline

Neural Additive Spline Model (NASM) for interpretable prosody prediction from linguistic features.

## Overview

This pipeline extracts linguistic features from Dutch audiobook recordings and trains interpretable KAN models to predict prosody (F0, Duration, Energy).

**Key Features:**
- 30 linguistically-motivated features extracted from text and WebCelex
- Multi-extractor F0 consensus (Praat, WORLD, CREPE)
- Word modernization for 19th-century Dutch text
- Fully interpretable B-spline visualizations

## Repository Structure

```
CODE2/
├── scripts/
│   ├── preprocessing/          # 01-15: Data pipeline
│   │   └── modernization/      # Optional diagnostic scripts
│   ├── training/               # 01-03: Model training
│   └── post_processing/        # 01-04: Evaluation & visualization
├── data/
│   ├── raw/                    # Your input files go here
│   │   ├── audio/              # Audio files (.mp3/.wav)
│   │   ├── text/               # Source transcript
│   │   └── lexicon/            # G2P dictionary
│   ├── intermediate/           # Pipeline outputs
│   └── features/               # Final feature matrices
└── results/                    # Training outputs & visualizations
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download nl_core_news_sm
```

### 2. Install Montreal Forced Aligner (Docker)

```bash
docker pull mmcauliffe/montreal-forced-aligner:latest
```

### 3. Download External Resources

**WebCelex Database** (required for Dutch lexical features):
- Register at: http://celex.mpi.nl/
- Download Dutch wordforms
- Place as: `data/webcelex-every-word.txt`

### 4. Prepare Your Data

Place your files in:
- `data/raw/audio/` - Audio chapters (.mp3 or .wav)
- `data/raw/text/original_text.txt` - Full transcript

### 5. Run Pipeline

```bash
# Preprocessing (run in order)
cd scripts/preprocessing
python 01_split_gutenberg_into_chapters.py
python 02_extract_unique_words.py
# ... continue through 15_create_splits.py

# Training
cd ../training
python 02_train_kan.py --epochs 50 --seed 42

# Evaluation
cd ../post_processing
python 01_aggregate_results.py
python 04_generate_visualizations.py
```

## Pipeline Steps

| # | Script | Description |
|---|--------|-------------|
| **Preprocessing** | | |
| 01 | split_gutenberg_into_chapters | Split text into chapters |
| 02 | extract_unique_words | Extract vocabulary for G2P |
| 03 | convert_g2p_to_dict | Convert phonemization to MFA format |
| 04 | apply_soft_g_corrections | Dutch soft-g phoneme corrections |
| 05 | whisper_transcribe | ASR transcription |
| 06 | extract_whisper_text | Parse Whisper output |
| 07 | align_words_to_csv | Align original to ASR (modernization) |
| 08 | build_mfa_utterances | Prepare MFA input |
| 09 | run_mfa | Montreal Forced Aligner |
| 10 | extract_f0 | Multi-extractor F0 consensus |
| 11 | extract_duration_energy | Duration and energy extraction |
| 12 | match_webcelex | WebCelex feature lookup |
| 13 | build_features | 30-feature matrix construction |
| 14 | validate_dataset | Quality checks |
| 15 | create_splits | Train/val/test split (80/10/10) |
| **Training** | | |
| 01 | data_loader | PyTorch data loading |
| 02 | train_kan | KAN (NASM) training |
| 03 | train_mlp_baseline | MLP baseline comparison |
| **Post-processing** | | |
| 01 | aggregate_results | Collect metrics |
| 02 | statistical_tests | KAN vs MLP comparison |
| 03 | test_set_evaluation | Final evaluation |
| 04 | generate_visualizations | B-spline plots |

## External Dependencies

| Resource | Source | Notes |
|----------|--------|-------|
| WebCelex | http://celex.mpi.nl/ | Academic registration required |
| LibriVox Audio | librivox.org | Public domain |
| Whisper | OpenAI | `pip install openai-whisper` |
| MFA | montreal-forced-aligner.readthedocs.io | Docker recommended |
| espeak-ng | espeak.sourceforge.net | For G2P phonemization |

## Citation

If you use this pipeline, please cite:

```bibtex
@thesis{mengari2025kan,
  title={Interpretable Prosody Prediction with Kolmogorov-Arnold Networks},
  author={Mengari, S.},
  year={2025}
}
```

## License

[Your license here]

