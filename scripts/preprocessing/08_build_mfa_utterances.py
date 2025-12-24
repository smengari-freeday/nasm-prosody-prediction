#!/usr/bin/env python3
# build sentence-level utterances, optimal for MFA alignment:
# 1 sentence tokenization
# 2 word tokenization (only keep words that match the regex \w+)
# 3 match book words to aligned words using SequenceMatcher
# 4 find sentence boundaries by finding the closest aligned word to the start and end of the sentence

import csv
import re
import subprocess
from pathlib import Path

try:
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError:
    print("pip install nltk && python -m nltk.downloader punkt")
    exit(1)

AUDIO_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/raw/audio")
TEXT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/text/chapters_cleaned")
ALIGNED_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/alignment/word_level")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/utterances")

BUFFER = 0.5  # seconds before/after
MIN_DUR, MAX_DUR = 0.2, 60.0


def get_duration(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    return float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())


def load_aligned(csv_path):
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Handle both column name formats
            word = (row.get("original") or row.get("original_word", "")).strip().lower()
            start_val = row.get("start") or row.get("start_time", "")
            end_val = row.get("end") or row.get("end_time", "")
            start = float(start_val) if start_val else None
            end = float(end_val) if end_val else None
            rows.append({"word": word, "start": start, "end": end})
    return rows


def extract_segment(audio_path, start, end, output_path):
    cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-ss", str(start), "-to", str(end), "-c", "copy", str(output_path)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def process_chapter(ch_num):
    audio = AUDIO_DIR / f"hoofdstuk_{ch_num}.mp3"
    text = TEXT_DIR / f"ch{ch_num:02d}.txt"
    aligned_csv = ALIGNED_DIR / f"ch{ch_num:02d}_aligned.csv"
    output = OUTPUT_DIR / f"ch{ch_num:02d}"
    
    if not all(p.exists() for p in [audio, text, aligned_csv]):
        return 0, 0
    
    output.mkdir(parents=True, exist_ok=True)
    duration = get_duration(audio)
    
    # Load text and alignment
    sentences = [s.strip() for s in sent_tokenize(text.read_text(encoding="utf-8"), language="dutch") if s.strip()]
    aligned = load_aligned(aligned_csv)
    
    # Fix missing timestamps
    for i, a in enumerate(aligned):
        if a["end"] is None:
            a["end"] = aligned[i+1]["end"] if i+1 < len(aligned) and aligned[i+1]["end"] else duration
        if a["start"] is None:
            a["start"] = aligned[i-1]["end"] if i > 0 and aligned[i-1]["end"] else 0.0
    
    # Build word index
    from difflib import SequenceMatcher
    aligned_tokens = [a["word"] for a in aligned]
    book_words = []
    sent_spans = []
    idx = 0
    for sent in sentences:
        tokens = [w.lower() for w in word_tokenize(sent, language="dutch") if re.match(r"\w+", w)] # takes regex \w+ such as café, etc.
        book_words.extend(tokens)
        sent_spans.append((idx, idx + len(tokens)))
        idx += len(tokens)
    
    # Match
    matcher = SequenceMatcher(None, aligned_tokens, book_words)
    book_to_aligned = [None] * len(book_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for ai, bi in zip(range(i1, i2), range(j1, j2)):
                book_to_aligned[bi] = ai
    
    # Extract segments
    success, fail = 0, 0
    csv_rows = []
    
    for i, (sent, (start_idx, end_idx)) in enumerate(zip(sentences, sent_spans), 1):
        indices = [book_to_aligned[j] for j in range(start_idx, end_idx) if book_to_aligned[j] is not None]
        
        wav_name = f"ch{ch_num:02d}_sent_{i:03d}.wav"
        txt_name = f"ch{ch_num:02d}_sent_{i:03d}.txt"
        
        if indices:
            start = max(0, aligned[min(indices)]["start"] - BUFFER)
            end = min(duration, aligned[max(indices)]["end"] + BUFFER)
            
            if MIN_DUR <= end - start <= MAX_DUR:
                if extract_segment(audio, start, end, output / wav_name):
                    (output / txt_name).write_text(sent + "\n", encoding="utf-8")
                    csv_rows.append({"idx": i, "start": start, "end": end, "wav": wav_name, "text": sent[:50]})
                    success += 1
                    continue
        
        fail += 1
    
    # Write CSV
    with (output / f"sentences.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["idx", "start", "end", "wav", "text"]).writeheader()
        csv.DictWriter(f, fieldnames=["idx", "start", "end", "wav", "text"]).writerows(csv_rows)
    
    return success, fail


def main():
    total_success, total_fail = 0, 0
    
    for ch in range(1, 48):  # Chapters 1-47
        s, f = process_chapter(ch)
        if s or f:
            print(f"ch{ch:02d}: {s} success, {f} failed")
            total_success += s
            total_fail += f
    
    print(f"\nTotal: {total_success} success, {total_fail} failed")


if __name__ == "__main__":
    main()
