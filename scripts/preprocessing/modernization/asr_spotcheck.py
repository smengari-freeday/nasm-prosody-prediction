#!/usr/bin/env python3
"""Random spot-check of utterance ASR quality."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import random
from difflib import SequenceMatcher
from pathlib import Path

try:
    import whisper
except ImportError:
    print("pip install openai-whisper")
    exit(1)

# Hardcoded paths
UTTERANCES_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/utterances")
REPORT_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/alignment/diagnostics/spotcheck_report.json")

# Config
N_SAMPLES = 20
SIM_THRESHOLD = 0.7
MODEL = "tiny"
SEED = 42


def find_pairs():
    """Find wav/txt pairs."""
    wav_dir = UTTERANCES_DIR / "wav"
    txt_dir = UTTERANCES_DIR / "txt"
    pairs = []
    if wav_dir.exists():
        for wav in wav_dir.glob("*.wav"):
            txt = txt_dir / f"{wav.stem}.txt"
            if txt.exists():
                pairs.append((wav, txt))
    return sorted(pairs)


def main():
    random.seed(SEED)
    
    pairs = find_pairs()
    if not pairs:
        print(f"No pairs in {UTTERANCES_DIR}")
        return
    
    print(f"Found {len(pairs)} pairs, sampling {min(N_SAMPLES, len(pairs))}")
    print(f"Loading Whisper {MODEL}...")
    model = whisper.load_model(MODEL)
    
    samples = random.sample(pairs, min(N_SAMPLES, len(pairs)))
    results = []
    warnings = []
    
    for wav, txt in samples:
        ref = txt.read_text(encoding="utf-8").strip().lower()
        hyp = model.transcribe(str(wav), language="nl", fp16=False)["text"].strip().lower()
        sim = SequenceMatcher(None, ref, hyp).ratio()
        
        results.append({"file": wav.name, "sim": sim})
        if sim < SIM_THRESHOLD:
            warnings.append(wav.name)
        print(f"  {wav.name}: {sim:.2f}" + (" [LOW]" if sim < SIM_THRESHOLD else ""))
    
    mean_sim = sum(r["sim"] for r in results) / len(results) if results else 0
    print(f"\nMean: {mean_sim:.2%}, Low: {len(warnings)}")
    
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps({"mean": mean_sim, "warnings": warnings, "results": results}, indent=2))


if __name__ == "__main__":
    main()
