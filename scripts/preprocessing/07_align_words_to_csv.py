# align original transcript words with whisper ASR words for manual review and further processing
# output: CSV with timestamps and acceptance states
# logic for script:
# 1. tokenize original transcript
# 2. load whisper words
# 3. align original words to whisper words
# 4. write aligned words to CSV
# 5. calculate statistics

# alignment types:
# equal: original and whisper words are the same
# insertion: whisper has a word that original transcript missed
# deletion: original has a word that whisper missed
# fuzzy: original and whisper words are similar but not the same (fuzzy matching)
# lemma: original and whisper words are the same but have different forms (lemma matching)
# mismatch: original and whisper words are not the same (no matching)



import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

try:
    import spacy
    NLP = spacy.load("nl_core_news_sm")
except:
    NLP = None
    print("Note: spacy not loaded, lemma matching disabled")

TRANSCRIPT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/text/chapters_cleaned")
WHISPER_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/whisper/json")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/alignment/word_level")

# thresholds
MIN_DUR_MS = 20 # 20ms minimum duration for word in order to be considered valid
MAX_DUR_MS = 2000 # 2000ms maximum duration for word in order to be considered valid
FUZZY_THRESH = 0.6 # 0.6 fuzzy threshold for word in order to be considered similar


def tokenize(text):
    # extract words from text
    return re.findall(r"\b[\w']+\b", text.lower()) # find all words in text and convert to lowercase


def load_whisper(path):
    # load whisper JSON and return words and timings
    data = json.loads(path.read_text(encoding="utf-8"))
    words, times = [], []
    for segment in data.get("segments", []):
        for word_data in segment.get("words", []):
            cleaned = re.sub(r"[^\w']", "", word_data.get("word", "").strip().lower())
            if cleaned:
                words.append(cleaned)
                times.append((word_data.get("start"), word_data.get("end")))
    return words, times


def calc_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


# lemmatize word using spacy if available
def lemma(word): 
    if NLP and word: # if spacy is loaded and word is not empty
        doc = NLP(word) # lemmatize word
        return doc[0].lemma_ if doc else word # return lemma if available, otherwise return word
    return word

# align original words to whisper words using SequenceMatcher
def align(orig, whis, times):
    matcher = SequenceMatcher(None, orig, whis) # create matcher 
    result = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes(): # get opcodes for matcher
        if tag == "equal": 
            for k, (o, w) in enumerate(zip(orig[i1:i2], whis[j1:j2])): 
                t = times[j1 + k] if j1 + k < len(times) else (None, None) # get timing for word
                result.append(make_row(o, w, "equal", 1.0, t)) # add word and timing to result
        
        elif tag == "replace":
            # match within chunks
            orig_chunk, whis_chunk, time_chunk = orig[i1:i2], whis[j1:j2], times[j1:j2]
            used = set()
            
            for original_word in orig_chunk:
                best_index, best_sim, is_lemma = None, 0, False
                
                # find best match for this original word
                for i, whisper_word in enumerate(whis_chunk):
                    if i in used:
                        continue
                    sim = calc_similarity(original_word, whisper_word)
                    lm = lemma(original_word) == lemma(whisper_word)
                    if lm and not is_lemma:
                        best_index, best_sim, is_lemma = i, sim, True
                    elif sim > best_sim and not is_lemma:
                        best_index, best_sim = i, sim
                
                # after inner loop: emit exactly one row
                if best_index is not None:
                    used.add(best_index)
                    t = time_chunk[best_index] if best_index < len(time_chunk) else (None, None)
                    atype = "lemma" if is_lemma else ("fuzzy" if best_sim >= FUZZY_THRESH else "mismatch")
                    result.append(make_row(original_word, whis_chunk[best_index], atype, best_sim, t))
                else:
                    result.append(make_row(original_word, "", "deletion", 0, (None, None)))
            
            # unused whisper words
            for i, whisper_word in enumerate(whis_chunk):
                if i not in used:
                    t = time_chunk[i] if i < len(time_chunk) else (None, None)
                    result.append(make_row("", whisper_word, "insertion", 0, t))
        
        elif tag == "delete":
            for original_word in orig[i1:i2]:
                result.append(make_row(original_word, "", "deletion", 0, (None, None)))
        
        elif tag == "insert":
            for k, whisper_word in enumerate(whis[j1:j2]):
                t = times[j1 + k] if j1 + k < len(times) else (None, None)
                result.append(make_row("", whisper_word, "insertion", 0, t))
    
    return result # return result

def make_row(original_word, whisper_word, alignment_type, similarity_score, timing):
    start_time, end_time = timing
    duration = (end_time - start_time) * 1000 if start_time is not None and end_time is not None else None
    
    # acceptance logic
    if alignment_type in ("deletion", "insertion"):
        acceptance_status = "needs_review"
    elif duration is None:
        acceptance_status = "needs_review"
    elif not (MIN_DUR_MS <= duration <= MAX_DUR_MS):
        acceptance_status = "needs_review"
    elif alignment_type in ("equal", "lemma"):
        acceptance_status = "auto_accept"
    elif alignment_type == "fuzzy" and similarity_score >= FUZZY_THRESH:
        acceptance_status = "auto_accept"
    else:
        acceptance_status = "needs_review"
    
    return {
        "original_word": original_word,
        "whisper_word": whisper_word,
        "preferred_word": whisper_word if alignment_type in ("equal", "lemma", "fuzzy") else original_word,
        "alignment_type": alignment_type,
        "start_time": f"{start_time:.3f}" if start_time is not None else "",
        "end_time": f"{end_time:.3f}" if end_time is not None else "",
        "duration_ms": f"{duration:.0f}" if duration is not None else "",
        "similarity_score": f"{similarity_score:.2f}",
        "acceptance_status": acceptance_status,
    }


def process_chapter(transcript, whisper, output):
    # process chapter by tokenizing original transcript, loading whisper words, and aligning them
    original_words = tokenize(transcript.read_text(encoding="utf-8"))
    whisper_words, timings = load_whisper(whisper)
    
    rows = align(original_words, whisper_words, timings) # align original words to whisper words
    
    with output.open("w", newline="", encoding="utf-8") as f: # write aligned words to CSV
        w = csv.DictWriter(f, fieldnames=["original_word", "whisper_word", "preferred_word", "alignment_type", "start_time", "end_time", "duration_ms", "similarity_score", "acceptance_status"])
        w.writeheader()
        w.writerows(rows) # write rows to CSV
    
    # statistics
    auto = sum(1 for r in rows if r["acceptance_status"] == "auto_accept")
    return len(rows), auto # return number of rows and number of auto accepted rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # find matching pairs (transcripts <-> whisper files)
    transcripts = sorted(TRANSCRIPT_DIR.glob("ch*.txt"))
    print(f"Found {len(transcripts)} transcripts")
    
    for transcript in transcripts:
        # extract chapter number
        num = re.search(r"ch(\d+)", transcript.stem)
        if not num:
            continue
        ch = int(num.group(1))
        
        # find matching Whisper file
        whisper = WHISPER_DIR / f"hoofdstuk_{ch}.json"
        if not whisper.exists():
            print(f"  {transcript.name}: no whisper file")
            continue
        
        output = OUTPUT_DIR / f"ch{ch:02d}_aligned.csv"
        total, auto = process_chapter(transcript, whisper, output)
        acceptance_rate = auto / total * 100 if total else 0
        print(f"  {transcript.name}: {total} tokens, {acceptance_rate:.0f}% accepted")
    

if __name__ == "__main__":
    main()
