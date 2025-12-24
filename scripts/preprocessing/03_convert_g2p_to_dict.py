#!/usr/bin/env python3
# Convert G2P spreadsheet output to MFA dictionary format

from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pip install pandas openpyxl")
    exit(1)

# Hardcoded paths
G2P_XLSX = Path("/Users/s.mengari/Desktop/CODE2/data/raw/lexicon/g2p_output.xlsx")
OUTPUT = Path("/Users/s.mengari/Desktop/CODE2/data/raw/lexicon/custom_dictionary.dict")


def convert_g2p_to_dict():

    df = pd.read_excel(G2P_XLSX) # read excel file
    
    # Get columns word, phonemes
    columns = df.columns.tolist() # get columns
    word_col = columns[0]
    phon_col = columns[1]
    
    entries = [] # init list
    for _, row in df.iterrows(): # loop through rows
        word = str(row[word_col]).strip().lower() # get word and convert to lowercase
        phonemes = str(row[phon_col]).strip() # get phonemes and convert to string
        if word and phonemes and phonemes != "nan": # if word and phonemes are not empty
            entries.append(f"{word} {phonemes}") # add to entries
    print(f"Length of unique words: {len(entries)}") # print number of entries
    