#!/usr/bin/env python3
# build 30 phoneme-level linguistic features
# features extracted:
# 0-3:   Lexical (stress position, word frequency, syllable count, phoneme count)
# 4-11:  POS tags (4 content + 4 function word classes)
# 12-14: Syllable position & stress (actual syllable boundaries via vowel-based heuristic)
# 15-16: Prosodic boundaries (phrase, word)
# 17-18: Context (sentence position, word position)
# 19-21: Phoneme class (vowel, voiced, plosive)
# 22-26: Dutch phonetic (schwa, velar context, vowel height, tense)
# 27:    Sonority of nucleus
# 28-29: Stress features (distance to stress, stress pattern class)

# source for phonetic classification:  ...... 
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter
from praatio import textgrid as tgio

# paths
DATA_DIR = Path("/Users/s.mengari/Desktop/CODE2/data")
CORPUS_DIR = DATA_DIR / "intermediate" / "text" / "chapters_cleaned"
TEXTGRID_DIR = DATA_DIR / "mfa" / "output"
LEXICON_FILE = DATA_DIR / "intermediate" / "lexicon" / "corpus_webcelex_features.json"
OUTPUT_DIR = DATA_DIR / "features" / "phoneme_level"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIM = 30

# Celex POS to Universal Dependencies mapping
CELEX_TO_UD = {
    'N': 'NOUN', 'V': 'VERB', 'A': 'ADJ', 'ADV': 'ADV',
    'PRON': 'PRON', 'PREP': 'ADP', 'ADP': 'ADP', 'ART': 'DET', 'DET': 'DET',
    'C': 'CCONJ', 'NUM': 'NUM', 'I': 'INTJ', 'EXP': 'X'
}

# Dutch phoneme inventory
DUTCH_VOWELS = {'a', 'e', 'i', 'o', 'u', 'y', 'ə', 'ɛ', 'ɪ', 'ɔ', 'ʏ', 
                'aː', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː', 'œy', 'ɛi', 'ɑu', 'ɑ', 'œ'}
VOICED = {'b', 'd', 'g', 'v', 'z', 'ɣ', 'm', 'n', 'ŋ', 'ɲ', 'l', 'r', 'j', 'w', 'ɡ', 'ʒ'}
VOICELESS = {'p', 't', 'k', 'f', 's', 'x', 'h', 'ʃ'}
PLOSIVES = {'p', 'b', 't', 'd', 'k', 'g', 'ɡ'}
SCHWA = {'ə'}
HIGH_VOWELS = {'i', 'iː', 'y', 'yː', 'u', 'uː', 'ɪ', 'ʏ'}
LOW_VOWELS = {'a', 'aː', 'ɑ', 'ɑː', 'ɔ', 'œ'}
TENSE_VOWELS = {'iː', 'eː', 'aː', 'oː', 'uː', 'yː', 'øː'}
VELAR_FRICATIVES = {'x', 'ɣ', 'ɡ', 'g'} # although corpus does not contain /x/ 

# Sonority scale (thesis convention): 0=consonant, 0.25=schwa, 0.75=short vowel, 1.0=long/tense vowel

SHORT_VOWELS = {'ɛ', 'ɪ', 'ɔ', 'ʏ', 'ɑ', 'i', 'u', 'y', 'a', 'e', 'o', 'œ'}
LONG_VOWELS = {'aː', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː', 'œy', 'ɛi', 'ɑu'}

# POS mapping (Universal Dependencies)
POS_CONTENT = {'NOUN': 0, 'VERB': 1, 'ADJ': 2, 'ADV': 3}
POS_FUNCTION = {'PRON': 0, 'ADP': 1, 'DET': 2, 'AUX': 3}


def is_vowel(ph):
    """Check if phoneme is vowel (including long vowels)."""
    ph = ph.strip().lower()
    if ph in DUTCH_VOWELS:
        return True
    return ph and ph[0] in 'aeiouyəɛɪɔʏɑœ'


def syllabify(phones):
    """
    Vowel-based syllabification: each vowel marks a new syllable nucleus.
    Returns list of syllable indices (1-based) for each phone.
    """
    indices = []
    current = 0
    for ph in phones:
        if is_vowel(ph):
            current += 1
        indices.append(max(1, current))
    return indices, max(1, current)


def parse_stress(stress_str):
    """
    Parse WebCelex StrsPat to get primary stress position (1-based syllable).
    Handles both binary strings ('100', '010') and integer strings.
    """
    if not stress_str:
        return 0
    s = str(stress_str).strip()
    positions = [i + 1 for i, c in enumerate(s) if c == '1']
    return positions[0] if positions else 0


def get_stress_class(stress_str, syl_cnt):
    """
    Classify stress pattern: 0=initial, 1=medial, 2=final, 3=compound.
    """
    if not stress_str or syl_cnt <= 0:
        return 0
    s = str(stress_str).strip()
    positions = [i for i, c in enumerate(s) if c == '1']
    if len(positions) == 0:
        return 0
    if len(positions) > 1:
        return 3  # compound
    pos = positions[0] + 1
    if pos == 1:
        return 0  # initial
    if pos == syl_cnt:
        return 2  # final
    return 1  # medial


def compute_word_frequency(corpus_dir):
    """Extract word frequency from corpus (log-transformed)."""
    words = []
    for f in sorted(corpus_dir.glob("ch*.txt")):
        text = f.read_text()
        words.extend(re.findall(r'\b\w+\b', text.lower()))
    counts = Counter(words)
    total = len(words)
    return {w: np.log1p(c / total) for w, c in counts.items()}


def build_features(tg_path, lexicon, word_freq, sentence_text):
    """
    Build feature vectors for all phonemes in a TextGrid.
    Uses vowel-based syllabification for phoneme→syllable mapping.
    """
    tg = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    words_tier = tg.getTier("words")
    phones_tier = tg.getTier("phones")
    
    # Get word intervals
    word_intervals = [(e.label, e.start, e.end) for e in words_tier.entries if e.label.strip()]
    
    # Sentence-level context
    sentence_words = re.findall(r'\b\w+\b', sentence_text.lower())
    total_words = len(sentence_words)
    
    features_list = []
    phoneme_labels = []
    
    for word_idx, (word_text, word_start, word_end) in enumerate(word_intervals):
        word_lower = word_text.lower()
        lex = lexicon.get(word_lower, {})
        
        # Get phones for this word (matching prosody extraction MIN_DUR threshold)
        MIN_DUR = 0.02  # 20ms, same as 11_extract_duration_energy.py
        word_phones = []
        for e in phones_tier.entries:
            if e.start >= word_start - 0.001 and e.end <= word_end + 0.001:
                dur = e.end - e.start
                if e.label.strip() and e.label not in ('sp', 'sil', '') and dur >= MIN_DUR:
                    word_phones.append(e.label)
        
        if not word_phones:
            continue
        
        # Vowel-based syllabification
        syl_indices, syl_count = syllabify(word_phones)
        
        # WebCelex data (with fallbacks)
        stress_str = lex.get('stress_pattern', '')
        primary_stress = parse_stress(stress_str)
        syl_cnt_celex = lex.get('syl_cnt', syl_count)
        phone_cnt = lex.get('phoneme_count', len(word_phones))
        pos_raw = lex.get('pos', '').upper()
        pos = CELEX_TO_UD.get(pos_raw, pos_raw)  # map Celex POS to Universal Dependencies
        
        # Word frequency
        wfreq = word_freq.get(word_lower, 0.0)
        
        # Sentence/word position (normalized)
        sent_pos = word_idx / max(1, total_words - 1) if total_words > 1 else 0.5
        
        # Stress pattern class
        stress_class = get_stress_class(stress_str, syl_cnt_celex)
        
        # Build feature vector for each phoneme
        for phone_idx, phone in enumerate(word_phones):
            ph_norm = phone.strip().lower()
            current_syl = syl_indices[phone_idx]
            
            # Per-phoneme: velar fricative in next syllable onset (stress magnet feature)
            # Look ahead up to 3 phonemes; if velar fricative follows a vowel, it's likely onset
            velar_next = 0.0
            for j in range(phone_idx + 1, min(phone_idx + 4, len(word_phones))):
                if word_phones[j].lower() in VELAR_FRICATIVES:
                    if j > 0 and is_vowel(word_phones[j - 1]):
                        velar_next = 1.0
                        break
            
            # Phoneme-level stress: is this phoneme in the stressed syllable?
            is_stressed = 1.0 if current_syl == primary_stress else 0.0
            
            # Distance to stress (normalized)
            dist_to_stress = (current_syl - primary_stress) / max(1, syl_count) if primary_stress > 0 else 0.0
            
            # Syllable position within word (mutual exclusion: initial takes priority for monosyllables)
            if current_syl == 1:
                syl_initial, syl_final = 1.0, 0.0
            elif current_syl == syl_count:
                syl_initial, syl_final = 0.0, 1.0
            else:
                syl_initial, syl_final = 0.0, 0.0
            
            # Word position (normalized within word)
            word_pos = phone_idx / max(1, len(word_phones) - 1) if len(word_phones) > 1 else 0.5
            
            # Boundaries
            is_word_boundary = 1.0 if phone_idx == len(word_phones) - 1 else 0.0
            is_phrase_boundary = 1.0 if word_idx == len(word_intervals) - 1 else 0.0
            
            # Phoneme class
            ph_is_vowel = 1.0 if is_vowel(ph_norm) else 0.0
            ph_is_voiced = 1.0 if (ph_norm in VOICED or is_vowel(ph_norm)) else 0.0
            ph_is_plosive = 1.0 if ph_norm in PLOSIVES else 0.0
            
            # Dutch phonetic features
            ph_is_schwa = 1.0 if ph_norm in SCHWA else 0.0
            ph_high = 1.0 if ph_norm in HIGH_VOWELS else 0.0
            ph_low = 1.0 if ph_norm in LOW_VOWELS else 0.0
            ph_tense = 1.0 if ph_norm in TENSE_VOWELS else 0.0
            
            # Sonority (thesis scale): 0=consonant, 0.25=schwa, 0.75=short, 1.0=long
            if ph_norm in SCHWA:
                sonority = 0.25
            elif ph_norm in LONG_VOWELS:
                sonority = 1.0
            elif ph_norm in SHORT_VOWELS or is_vowel(ph_norm):
                sonority = 0.75
            else:
                sonority = 0.0
            
            # POS features (4 content + 4 function)
            pos_content = [0.0] * 4
            pos_function = [0.0] * 4
            if pos in POS_CONTENT:
                pos_content[POS_CONTENT[pos]] = 1.0
            elif pos in POS_FUNCTION:
                pos_function[POS_FUNCTION[pos]] = 1.0
            
            # Build 30-feature vector
            feat = [
                # 0-3: Lexical
                primary_stress / max(1, syl_cnt_celex),  # normalized stress position
                wfreq,                                   # word frequency (log)
                float(syl_cnt_celex),                    # syllable count
                float(phone_cnt),                        # phoneme count
                # 4-11: POS (content + function)
                *pos_content,                            # noun, verb, adj, adv
                *pos_function,                           # pron, prep, det, aux
                # 12-14: Syllable & stress
                syl_initial,                             # syllable initial in word
                syl_final,                               # syllable final in word
                is_stressed,                             # phoneme in stressed syllable
                # 15-16: Prosodic boundaries
                is_phrase_boundary,
                is_word_boundary,
                # 17-18: Context
                sent_pos,                                # sentence position
                word_pos,                                # word position
                # 19-21: Phoneme class
                ph_is_vowel,
                ph_is_voiced,
                ph_is_plosive,
                # 22-26: Dutch phonetic
                ph_is_schwa,
                velar_next,                              # velar fricative in next syllable onset
                ph_high,                                 # vowel height high
                ph_low,                                  # vowel height low
                ph_tense,                                # tense vowel
                # 27: Sonority
                sonority,                                # already 0-1 scale
                # 28-29: Stress features
                dist_to_stress,                          # distance to stress (normalized)
                float(stress_class) / 3.0,               # stress pattern class (normalized)
            ]
            
            assert len(feat) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(feat)}"
            features_list.append(feat)
            phoneme_labels.append(ph_norm)
    
    return np.array(features_list, dtype=np.float32), phoneme_labels


def main():
    # Load lexicon
    with open(LEXICON_FILE) as f:
        lexicon = json.load(f)
    print(f"Loaded lexicon: {len(lexicon)} words")
    
    # Compute corpus word frequency
    word_freq = compute_word_frequency(CORPUS_DIR)
    print(f"Computed word frequencies: {len(word_freq)} unique words")
    
    # Process all TextGrids
    tg_files = sorted(TEXTGRID_DIR.glob("**/*.TextGrid"))
    print(f"Found {len(tg_files)} TextGrid files")
    
    UTT_DIR = DATA_DIR / "intermediate" / "utterances"
    
    total_phonemes = 0
    for tg_path in tg_files:
        utt_id = tg_path.stem
        chapter = utt_id.split('_')[0]  # e.g., "ch02" from "ch02_sent_001"
        
        # Load sentence text from utterance file
        utt_text_file = UTT_DIR / chapter / f"{utt_id}.txt"
        if utt_text_file.exists():
            sentence_text = utt_text_file.read_text().strip()
        else:
            sentence_text = ""  # fallback
        
        features, labels = build_features(tg_path, lexicon, word_freq, sentence_text)
        
        if len(features) > 0:
            # Save features
            np.save(OUTPUT_DIR / f"{utt_id}_features.npy", features)
            
            # Save phoneme labels (for validation)
            with open(OUTPUT_DIR / f"{utt_id}_phonemes.json", 'w') as f:
                json.dump(labels, f)
            
            total_phonemes += len(features)
    
    print(f"\nExtracted {total_phonemes} phoneme features")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
