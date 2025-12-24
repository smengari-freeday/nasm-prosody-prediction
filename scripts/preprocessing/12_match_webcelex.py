#!/usr/bin/env python3
# match corpus words to WebCelex for lexical features
# extracts: stress pattern, syllable count, POS, phoneme count

import csv
import json
from pathlib import Path

ALIGNED_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/alignment/word_level")
WEBCELEX = Path("/Users/s.mengari/Desktop/THESIS/Data/webcelex-every-word.txt")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/lexicon")


def parse_webcelex():
    # parse WebCelex backslash-separated file into {word: features}
    lexicon = {}
    header = None
    
    for line in WEBCELEX.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        if header is None and "\\" in line:
            header = line.split("\\")
            continue
        if header is None:
            continue
        
        fields = line.split("\\")
        if len(fields) < len(header):
            continue
        
        row = dict(zip(header, fields))
        word = row.get("Word", "").lower().strip()
        if not word or word in lexicon:
            continue
        
        phonemes = row.get("PhonCLX", "").strip()
        if phonemes:
            # syllable count
            syl_raw = row.get("SylCnt", "1")
            syllables = int(syl_raw) if syl_raw.isdigit() else syl_raw.count("-") + 1
            
            # phoneme count
            phon_cnt = row.get("PhonCnt", "1")
            phoneme_count = int(phon_cnt) if phon_cnt.isdigit() else len(phonemes)
            
            lexicon[word] = {
                "stress": row.get("StrsPat", ""),
                "syllables": syllables,
                "phoneme_count": phoneme_count,
                "pos": row.get("Lemma Class", ""),
            }
    
    return lexicon


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    lexicon = parse_webcelex()
    print(f"WebCelex entries: {len(lexicon)}")
    
    # collect corpus words from alignment CSVs
    # prefer whisper (modernized), fall back to original if whisper not in lexicon
    corpus_words = set()
    csv_files = sorted(ALIGNED_DIR.glob("*_aligned.csv"))
    print(f"Found {len(csv_files)} alignment CSVs")
    
    for csv_file in csv_files:
        with csv_file.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pref = row.get("preferred_word", "").strip().lower()
                orig = row.get("original_word", "").strip().lower()
                if pref and pref in lexicon:
                    corpus_words.add(pref)
                elif orig and orig in lexicon:
                    corpus_words.add(orig)
                elif pref:
                    corpus_words.add(pref)
                elif orig:
                    corpus_words.add(orig)
    
    print(f"Unique corpus words: {len(corpus_words)}")
    
    # match against lexicon
    matched = {}
    oov = []
    for word in sorted(corpus_words):
        if word in lexicon:
            matched[word] = lexicon[word]
        else:
            oov.append(word)
    
    coverage = len(matched) / len(corpus_words) * 100 if corpus_words else 0
    print(f"Matched: {len(matched)} ({coverage:.1f}%)")
    print(f"OOV: {len(oov)}")
    
    # save outputs
    (OUTPUT_DIR / "webcelex_matched.json").write_text(json.dumps(matched, indent=2, ensure_ascii=False))
    (OUTPUT_DIR / "oov_words.txt").write_text("\n".join(sorted(oov)))


if __name__ == "__main__":
    main()
