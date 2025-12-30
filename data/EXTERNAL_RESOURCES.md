# External Resources

These files are required but NOT included in the repository due to size/licensing.

## 1. WebCelex Database

**Required for:** Dutch lexical features (stress, syllables, phonemes)

**Download:**
1. Register at http://celex.mpi.nl/ (academic email required)
2. Download: Dutch Wordforms (tab-separated)
3. Place as: `webcelex-every-word.txt` in this directory

**Size:** ~79 MB

## 2. Audio Files

**Required for:** Prosody extraction (F0, duration, energy)

**Options:**
- Use your own audio recordings
- Download from LibriVox: https://librivox.org/

**Place in:** `raw/audio/`

**Format:** .mp3 or .wav (16kHz+ recommended)

## 3. G2P Phonemization

**Required for:** MFA pronunciation dictionary

**Option A:** Use provided `raw/lexicon/g2p_output.xlsx`

**Option B:** Generate yourself:
```bash
# Install phonemizer
pip install phonemizer

# Generate (requires espeak-ng)
phonemizer --language nl-be input_words.txt -o phonemes.txt
```

## 4. Montreal Forced Aligner

**Required for:** Phoneme-level alignment

**Install via Docker (recommended):**
```bash
docker pull mmcauliffe/montreal-forced-aligner:latest
```

**Or via conda:**
```bash
conda create -n mfa -c conda-forge montreal-forced-aligner
```
