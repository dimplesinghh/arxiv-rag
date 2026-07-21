"""
eval/build_qasper_index.py

Downloads the QASPER dataset from HuggingFace, chunks each paper's
full text, embeds with all-MiniLM-L6-v2, and stores in Qdrant
collection 'qasper_chunks' for eval.

Run once before running run_eval.py.
"""

import json
import os
import tiktoken
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

COLLECTION_NAME = "qasper_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
OVERLAP = 50
BATCH_SIZE = 64

enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping token-based chunks."""
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(enc.decode(tokens[i:i + size]))
        i += size - overlap
    return chunks

def extract_paper_text(paper):
    """
    Extract full text from a QASPER paper entry.

    QASPER stores paper content in sections. Each section has
    a heading and paragraphs. We join them all into one string.

    Args:
        paper (dict): One row from the QASPER dataset.

    Returns:
        str: Full concatenated paper text.
    """
    parts = []

    # title + abstract
    if paper.get("title"):
        parts.append(paper["title"])
    if paper.get("abstract"):
        parts.append(paper["abstract"])

    # body sections
    full_text = paper.get("full_text", {})
    section_names = full_text.get("section_name", [])
    paragraphs_list = full_text.get("paragraphs", [])

    for section, paragraphs in zip(section_names, paragraphs_list):
        if section:
            parts.append(section)
        if isinstance(paragraphs, list):
            parts.extend(paragraphs)
        elif isinstance(paragraphs, str):
            parts.append(paragraphs)

    return "\n".join(parts)

def main():
    print("Loading QASPER dataset...")
    # loads the train split — ~888 papers with QA pairs
    dataset = load_dataset("allenai/qasper", split="train")
    print(f"Loaded {len(dataset)} papers from QASPER")

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host="localhost", port=6333)

    # fresh collection each run
    client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    all_chunks = []

    for paper in dataset:
        paper_id = paper["id"]
        text = extract_paper_text(paper)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for idx, c in enumerate(chunks):
            all_chunks.append({
                "paper_id": paper_id,
                "chunk_idx": idx,
                "text": c,
            })

    print(f"Total chunks to embed: {len(all_chunks)}")

    # embed and upsert in batches
    # embed and upsert in batches — never accumulate all points at once
    
    total_upserted = 0
    UPSERT_BATCH = 256  # upsert 256 points at a time, well under 33MB limit

    
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = model.encode(texts, show_progress_bar=True)

        points = []

        for j, c in enumerate(batch):
            points.append(PointStruct(
                id=i + j,
                vector=vectors[j].tolist(),
                payload={
                    "paper_id": c["paper_id"],
                    "chunk_idx": c["chunk_idx"],
                    "text": c["text"],
                }
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_upserted += len(points)

        if total_upserted % 1000 == 0:
            print(f"Upserted {total_upserted} / {len(all_chunks)} chunks...")

    print(f"Done. Upserted {total_upserted} chunks into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()