"""
Takes every text chunk, converts it into a vector (list of numbers), and stores it in Qdrant so it can be searched later.
"""

import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

CHUNKS_PATH = "data/processed/chunks.json"
COLLECTION_NAME = "arxiv_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, local

def main():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    
    model = SentenceTransformer(MODEL_NAME)
    # Connects to your Qdrant Docker container. localhost:6333 is where Docker exposes it, matching docker-compose.yml.
    client = QdrantClient(host="localhost", port=6333)

    client.delete_collection(collection_name = COLLECTION_NAME)
    client.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config = VectorParams(size=384, distance=Distance.COSINE)
    )

    batch_size = 64
    points = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c['text'] for c in batch]
        vectors = model.encode(texts, show_progress_bar=True)

        for j, c in enumerate(batch):
            points.append(PointStruct(
                id = i + j,
                vector = vectors[j].tolist(),
                payload = {
                    "paper_id": c["paper_id"],
                    "chunk_idx": c["chunk_idx"],
                    "text": c["text"],
                }
            ))

    # Qdrant builds an index over the vectors so future similarity searches are fast.
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Done. Upserted {len(points)} chunks into Qdrant.")

if __name__ == "__main__":
    main()



