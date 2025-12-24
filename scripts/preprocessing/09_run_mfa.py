#!/usr/bin/env python3
# run MFA alignment with speaker adaptive training (SAT) using Docker
# 
# As described in thesis Chapter 3:
# "Forced alignment was performed using MFA's speaker adaptive training mode,
#  which trains a custom acoustic model for the narrator's voice."
#
# Pipeline:
# 1. Train custom acoustic model using mfa train --single_speaker
# 2. Align corpus with trained model using mfa align
# 3. Output TextGrids
# 
# Important checklist I realised far too late:
# make sure all .wav's have the same sample rate as the corpus (22050 Hz)
# I used Docker: mmcauliffe/montreal-forced-aligner:latest
# i used linux because Kaldi was hardly useable on macos (installation issues)
# source: https://montreal-forced-aligner.readthedocs.io/en/v3.2.3/user_guide/workflows/train_acoustic_model.html

import subprocess
import shutil
from pathlib import Path

# paths
CORPUS_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/intermediate/utterances")
DICTIONARY = Path("/Users/s.mengari/Desktop/CODE2/data/raw/lexicon/custom_dictionary_soft_g.dict")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/data/mfa")

TRAINED_MODEL = OUTPUT_DIR / "trained_model.zip" 
TEXTGRID_OUTPUT = OUTPUT_DIR / "output" # TextGrid output directory

# my own docker image
DOCKER_IMAGE = "mmcauliffe/montreal-forced-aligner:latest"



def check_corpus():
   # check if corpus has wav/txt pairs
    chapters = sorted(CORPUS_DIR.glob("ch*")) # get all chapter directories
    total_wavs = sum(len(list(ch.glob("*.wav"))) for ch in chapters) # get total number of wav files
    total_txts = sum(len(list(ch.glob("*.txt"))) for ch in chapters) # get total number of txt files
    print(f"Corpus: {len(chapters)} chapters, {total_wavs} wav, {total_txts} txt")
    return total_wavs > 0 # return True if there are more than 0 wav files

def run_train():
    # train custom acoustic model using mfa train --single_speaker
    print("\n=== TRAINING ACOUSTIC MODEL ===")
    print(f"Corpus: {CORPUS_DIR}") # print corpus directory
    print(f"Dictionary: {DICTIONARY}") # print dictionary directory
    print(f"Output model: {TRAINED_MODEL}") # print trained model directory
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # create output directory if it doesn't exist

    # command to train custom acoustic model using mfa train --single_speaker
    cmd = [ 
        "docker", "run", "--rm", "--platform", "linux/amd64", # i used linux because Kaldi was hardly useable on macos
        "-v", f"{CORPUS_DIR}:/mnt/corpus",
        "-v", f"{DICTIONARY}:/mnt/dict.dict",
        "-v", f"{OUTPUT_DIR}:/mnt/output",
        DOCKER_IMAGE,
        "mfa", "train",
        "--single_speaker",  # single narrator, disables cross-speaker adaptation
        "--clean",
        "--overwrite",
        "/mnt/corpus",
        "/mnt/dict.dict",
        "/mnt/output/trained_model.zip"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0 and TRAINED_MODEL.exists():
        print(f"\n✓ Model trained: {TRAINED_MODEL}")
        return True
    else:
        print("\n✗ Training failed")
        return False


def run_align():
    # align corpus with trained model using mfa align

    # creating output dir for TextGrids
    TEXTGRID_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # command to align corpus with trained model using mfa align
    cmd = [ 
        "docker", "run", "--rm", "--platform", "linux/amd64", # i used linux because Kaldi was hardly useable on macos
        "-v", f"{CORPUS_DIR}:/mnt/corpus",
        "-v", f"{DICTIONARY}:/mnt/dict.dict",
        "-v", f"{TRAINED_MODEL}:/mnt/model.zip",
        "-v", f"{TEXTGRID_OUTPUT}:/mnt/output",
        DOCKER_IMAGE,
        "mfa", "align",
        "--clean",
        "--overwrite",
        "/mnt/corpus",
        "/mnt/dict.dict",
        "/mnt/model.zip",
        "/mnt/output"
    ]
    
    result = subprocess.run(cmd) # run command
    
    if result.returncode == 0:
        tg_count = len(list(TEXTGRID_OUTPUT.rglob("*.TextGrid"))) # get total number of TextGrid files
        print(f"\n✓ Alignment complete: {tg_count} TextGrids") # print success message
        return True
    else:
        print("\n✗ Alignment failed") # print failure message
        return False


def main():
    print("=" * 60)
    print("MFA SPEAKER ADAPTIVE TRAINING")
    print("=" * 60)
    
    # Check prerequisites
    if not DICTIONARY.exists(): # check if dictionary exists
        print(f"No dictionary found:: {DICTIONARY}") # print failure message
        return
    
    if not check_corpus(): # check if corpus exists
        print(f"No corpus found: {CORPUS_DIR}") # print failure message
        return
    
    # train model if not already trained
    if TRAINED_MODEL.exists():
        print(f"\nTrained model exists: {TRAINED_MODEL}") # print success message
        print("Delete it to retrain, or proceeding to alignment...") # print instructions
    else:
        if not run_train(): # train model
            return
    
    # align with trained model
    run_align()
    print("run completed")

    print(f"Model: {TRAINED_MODEL}") # print model name for clarity
    print(f"TextGrids: {TEXTGRID_OUTPUT}") # print TextGrid output directory for clarity

if __name__ == "__main__":
    main() # run main function
