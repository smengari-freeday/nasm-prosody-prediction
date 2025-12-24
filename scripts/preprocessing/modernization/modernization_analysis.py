#!/usr/bin/env python3
# Analyze modernization patterns from alignment data.
# Finds cases where original text differs from Whisper transcription (archaic → modern).

import csv
import json
from collections import Counter
from pathlib import Path

ALIGNMENT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/alignment/word_level")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/alignment/diagnostics")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    csvs = sorted(ALIGNMENT_DIR.glob("*_aligned.csv"))
    print(f"Analyzing {len(csvs)} alignment files")
    
    # Collect fuzzy matches (original != whisper but similar)
    modernizations = Counter()  # (original, whisper) -> count
    
    for csv_file in csvs:
        with csv_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both old and new column names
                align_type = row.get("type") or row.get("alignment_type", "")
                orig = row.get("original") or row.get("original_word", "")
                whis = row.get("whisper") or row.get("whisper_word", "")
                
                if align_type == "fuzzy" and orig and whis and orig != whis:
                    modernizations[(orig, whis)] += 1
    
    # Report
    report = OUTPUT_DIR / "modernization_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("MODERNIZATION ANALYSIS (from alignment data)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Unique fuzzy pairs: {len(modernizations)}\n")
        f.write(f"Total occurrences: {sum(modernizations.values())}\n\n")
        f.write("Most common (original → whisper):\n")
        for (orig, whis), count in modernizations.most_common(50):
            f.write(f"  {orig} → {whis}: {count}\n")
    
    # JSON
    mappings = {f"{o}→{w}": c for (o, w), c in modernizations.most_common()}
    (OUTPUT_DIR / "modernization_mappings.json").write_text(
        json.dumps(mappings, indent=2, ensure_ascii=False)
    )
    
    print(f"\nFound {len(modernizations)} unique modernization pairs")
    print(f"Total occurrences: {sum(modernizations.values())}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
