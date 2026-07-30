"""
eval/build_qasper_index_v2.py

Rebuilds the QASPER Qdrant index using section-aware chunking.
Stores in collection 'qasper_chunks_v2' — baseline collection
'qasper_chunks' is preserved for comparison.

Run before eval/run_eval_v2.py.
"""

import json
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.chunk_sections import chunk_paper_by_sections

COLLECTION_NAME = "qasper_chunks_v2"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

def extract_abstract_chunks(paper):
    """
    Extract title and abstract as a dedicated chunk.

    Title and abstract are the most summary-dense text in a paper.
    Keeping them as a standalone chunk improves retrieval for
    high-level questions.

    Args:
        paper (dict): QASPER paper row.

    Returns:
        list[dict]: One chunk containing title + abstract.
    """

    parts = []
    if paper.get("title"):
        parts.append(paper["title"])
    if paper.get("abstract"):
        parts.append(paper["abstract"])

    if not parts:
        return []

    return [{
        "paper_id": paper["id"],
        "chunk_idx": 0,
        "section": "abstract",
        "text": " ".join(parts),
    }]

def main():
    print("Loading QASPER dataset...")
    dataset = load_dataset("allenai/qasper", split="validation")
    print(f"Loaded {len(dataset)} papers")

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host="localhost", port=6333)

    client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    all_chunks = []

    for paper in dataset:
        paper_id = paper["id"]

        # abstract chunk first
        abs_chunks = extract_abstract_chunks(paper)
        all_chunks.extend(abs_chunks)

        # section-aware body chunks
        full_text = paper.get("full_text", {})
        if full_text:
            body_chunks = chunk_paper_by_sections(paper_id, full_text)

            # re-index chunk_idx to continue from abstract
            offset = len(abs_chunks)
            for c in body_chunks:
                c["chunk_idx"] += offset
            all_chunks.extend(body_chunks)

    print(f"Total chunks: {len(all_chunks)}")

    total_upserted = 0
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = model.encode(texts, show_progress_bar=False)

        points = []
        for j, c in enumerate(batch):
            points.append(PointStruct(
                id=i + j,
                vector=vectors[j].tolist(),
                payload={
                    "paper_id": c["paper_id"],
                    "chunk_idx": c["chunk_idx"],
                    "section": c["section"],
                    "text": c["text"],
                }
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_upserted += len(points)

        if total_upserted % 2000 == 0:
            print(f"Upserted {total_upserted} / {len(all_chunks)}")

    print(f"Done. {total_upserted} chunks in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()