"""
It takes every parsed .txt file from data/processed/, reads the full text, and breaks it into smaller overlapping pieces that an embedding model can handle.
"""

import tiktoken
import json
import re 
import os 

PROCESSED_DIR = "data/processed"
CHUNKS_PATH = "data/processed/chunks.json"
CHUNK_SIZE = 512
OVERLAP = 50

enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, size = CHUNK_SIZE, overlap = OVERLAP):
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + size]
        chunks.append(enc.decode(chunk_tokens))
        i += size - overlap
    return chunks 

def main():
    all_chunks = []
    txt_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".txt")]

    for fname in txt_files:
        paper_id = fname.replace(".txt", "")
        with open(os.path.join(PROCESSED_DIR, fname)) as f:
            text = f.read()

        chunks = chunk_text(text)
        
        for idx, c in enumerate(chunks):
            all_chunks.append({
                "paper_id": paper_id,
                "chunk_idx": idx,
                "text": c,
            })

    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Done. {len(all_chunks)} chunks from {len(txt_files)} papers.")

if __name__ == "__main__":
    main()