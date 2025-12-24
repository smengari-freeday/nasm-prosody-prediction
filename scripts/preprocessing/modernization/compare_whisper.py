#!/usr/bin/env python3
# compare whisper ASR output against original text

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

WHISPER_JSON = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/whisper/json/hoofdstuk_2.json")
ORIGINAL_TXT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/text/chapters_cleaned/ch02.txt")
REPORT_OUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/alignment/diagnostics/whisper_vs_original_ch02.txt")

SIM_THRESHOLD = 0.5 # similarity threshold for sentences


def normalize(text):
    # normalize text by removing whitespace and special characters
    return re.sub(r"\s+", " ", re.sub(r"[\W_]+", " ", text.lower())).strip()


def words(text):
    # return set of words in text
    return set(normalize(text).split())


def sentences(text):
    # return list of sentences in text
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def main():
    # load whisper and original text
    whisper_text = json.loads(WHISPER_JSON.read_text(encoding="utf-8")).get("text", "")
    original_text = ORIGINAL_TXT.read_text(encoding="utf-8")
    
    original_words = words(original_text)
    whisper_words = words(whisper_text)
    common = original_words & whisper_words
    missing = original_words - whisper_words
    extra = whisper_words - original_words
    
    original_sentences = sentences(original_text)
    whisper_sentences = sentences(whisper_text)
    
    # find anomalous sentences
    anomalies = []
    for original_sentence in original_sentences:
        norm_original_sentence = normalize(original_sentence)
        best_sim = max((SequenceMatcher(None, norm_original_sentence, normalize(w)).ratio() for w in whisper_sentences), default=0)
        if best_sim < SIM_THRESHOLD:
            anomalies.append((original_sentence[:100], best_sim))
    
    # write report
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUT.open("w", encoding="utf-8") as f:
        f.write("ASR COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Original words: {len(original_words)}\n")
        f.write(f"Whisper words:  {len(whisper_words)}\n")
        f.write(f"Overlap:        {len(common)/len(original_words):.1%}\n\n")
        f.write(f"Missing ({len(missing)}): {', '.join(sorted(missing)[:50])}\n\n")
        f.write(f"Extra ({len(extra)}): {', '.join(sorted(extra)[:50])}\n\n")
        f.write(f"Low-similarity sentences ({len(anomalies)}):\n")
        for sent, sim in anomalies[:20]:
            f.write(f"  [{sim:.2f}] {sent}...\n")
    
    print(f"Overlap: {len(common)/len(original_words):.1%}")
    print(f"Anomalous: {len(anomalies)}/{len(original_sentences)} sentences")
    print(f"Report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
