#!/usr/bin/env python3
# Verify modernization pairs against lexicon.
# Checks if whisper (modern) form exists in lexicon while original (archaic) doesn't.

import json
from pathlib import Path

MAPPINGS = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/alignment/diagnostics/modernization_mappings.json")
LEXICON = Path("/Users/s.mengari/Desktop/THESIS/Data/webcelex-every-word.txt")
OUTPUT = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/alignment/diagnostics/verified_mappings.json")


def load_lexicon():
    if not LEXICON.exists():
        return set()
    words = set()
    for line in LEXICON.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        # WebCelex uses backslash separator, Word is first field
        parts = line.split("\\")
        if parts:
            words.add(parts[0].lower().strip())
    return words


def main():
    if not MAPPINGS.exists():
        print(f"Not found: {MAPPINGS}")
        return
    
    # Format: {"orig→whis": count, ...}
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    lexicon = load_lexicon()
    print(f"Pairs: {len(mappings)}, Lexicon: {len(lexicon)} words")
    
    verified = {}
    stats = {"verified": 0, "both_valid": 0, "no_lexicon": 0, "not_found": 0}
    
    for pair, count in mappings.items():
        if "→" not in pair:
            continue
        orig, whis = pair.split("→", 1)
        
        if not lexicon:
            status = "no_lexicon"
        elif whis in lexicon and orig not in lexicon:
            status = "verified"  # Modern in lexicon, archaic not
        elif whis in lexicon and orig in lexicon:
            status = "both_valid"  # Both valid words
        else:
            status = "not_found"
        
        stats[status] += 1
        verified[pair] = {"original": orig, "whisper": whis, "count": count, "status": status}
    
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(verified, indent=2, ensure_ascii=False))
    
    print(f"\nOutput: {OUTPUT}")
    for s, c in sorted(stats.items()):
        print(f"  {s}: {c}")


if __name__ == "__main__":
    main()
