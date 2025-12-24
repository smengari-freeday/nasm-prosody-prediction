#!/usr/bin/env python3
"""
utils.py - Shared utilities for the preprocessing pipeline

Common functions for file handling, path management, and data loading.
All scripts in this folder can import from here:
    from utils import get_project_root, load_json, save_csv, etc.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional


def get_project_root() -> Path:
    """Get the project root directory (hardcoded for CODE2)."""
    return Path("/Users/s.mengari/Desktop/CODE2")


def get_paths() -> Dict[str, Path]:
    """Get all standard data directory paths."""
    root = Path("/Users/s.mengari/Desktop/CODE2")
    return {
        # Raw data (never modified)
        "raw_audio": root / "data" / "raw" / "audio",
        "raw_text": root / "data" / "raw" / "text",
        
        # Intermediate: text
        "chapters_cleaned": root / "data" / "intermediate" / "text" / "chapters_cleaned",
        "chapters_whisper": root / "data" / "intermediate" / "text" / "chapters_from_whisper",
        
        # Intermediate: whisper
        "whisper_json": root / "data" / "intermediate" / "whisper" / "json",
        "whisper_logs": root / "data" / "intermediate" / "whisper" / "logs",
        
        # Intermediate: alignment
        "alignment_words": root / "data" / "intermediate" / "alignment" / "word_level",
        "alignment_diagnostics": root / "data" / "intermediate" / "alignment" / "diagnostics",
        
        # Intermediate: utterances
        "utterances_wav": root / "data" / "intermediate" / "utterances" / "wav",
        "utterances_txt": root / "data" / "intermediate" / "utterances" / "txt",
        
        # MFA
        "mfa_input": root / "data" / "mfa" / "input",
        "mfa_output": root / "data" / "mfa" / "output",
        "mfa_logs": root / "data" / "mfa" / "logs",
        
        # Final
        "final_textgrids": root / "data" / "final" / "textgrids",
        "final": root / "data" / "final",
    }


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# JSON utilities
def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """Save data as JSON file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


# Text utilities
def load_text(path: Path) -> str:
    """Load text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_text(text: str, path: Path) -> None:
    """Save text to file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# CSV utilities
def load_csv(path: Path) -> List[Dict[str, str]]:
    """Load CSV file as list of dictionaries."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_csv(data: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Save list of dictionaries as CSV."""
    ensure_dir(path.parent)
    if not data:
        return
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# File finding utilities
def find_audio_files(directory: Path, extensions: tuple = (".wav", ".mp3", ".flac")) -> List[Path]:
    """Find all audio files in directory."""
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def find_chapter_files(directory: Path, pattern: str = "ch*.txt") -> List[Path]:
    """Find all chapter text files."""
    return sorted(directory.glob(pattern))


def find_matching_pairs(dir_a: Path, dir_b: Path, ext_a: str = ".wav", ext_b: str = ".txt") -> List[tuple]:
    """Find matching file pairs by stem name across two directories."""
    pairs = []
    for file_a in dir_a.glob(f"*{ext_a}"):
        file_b = dir_b / f"{file_a.stem}{ext_b}"
        if file_b.exists():
            pairs.append((file_a, file_b))
    return sorted(pairs)


# Logging utilities
def print_header(title: str, width: int = 60) -> None:
    """Print a formatted header."""
    print("=" * width)
    print(title)
    print("=" * width)


def print_section(title: str, width: int = 40) -> None:
    """Print a section header."""
    print(f"\n{title}")
    print("-" * width)
