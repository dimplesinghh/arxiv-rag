"""
src/hybrid_retrieve.py

Hybrid retrieval combining dense vector search (Qdrant) with
sparse BM25 keyword search.

Dense retrieval finds semantically similar chunks even when
exact words differ. BM25 finds chunks with exact keyword
overlap. Together they cover cases each misses individually.

Fusion formula:
    hybrid_score = alpha * dense_score + (1 - alpha) * bm25_score
    default alpha = 0.6
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "qasper_chunks_v2"

model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(host="localhost", port=6333)


def load_all_chunks(collection_name=COLLECTION_NAME, limit=100000):
    """
    Load all chunks from Qdrant collection into memory for BM25 indexing.

    BM25 requires the full corpus in memory. For this scale
    (~50k chunks) this is fine. Would not scale to millions.

    Args:
        collection_name (str): Qdrant collection to load from.
        limit           (int): Max points to load.

    Returns:
        list[dict]: All chunk payloads with text and metadata.
    """
    results, _ = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [r.payload for r in results]


def build_bm25_index(chunks):
    """
    Build a BM25 index from chunk texts.

    Tokenizes each chunk by whitespace for BM25.
    More sophisticated tokenization (stemming, stopword removal)
    would improve results but adds complexity.

    Args:
        chunks (list[dict]): Chunks with 'text' field.

    Returns:
        BM25Okapi: Fitted BM25 index.
    """
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def hybrid_retrieve(query, chunks, bm25_index, k=10, alpha=0.6):
    """
    Retrieve top-k chunks using hybrid dense + BM25 scoring.

    Args:
        query       (str):        User query string.
        chunks      (list[dict]): All chunks (same order as BM25 index).
        bm25_index  (BM25Okapi):  Fitted BM25 index.
        k           (int):        Number of results to return.
        alpha       (float):      Weight for dense score (0-1).
                                  1-alpha is weight for BM25.

    Returns:
        list[dict]: Top-k chunks sorted by hybrid score, highest first.
                    Each dict has chunk payload + 'hybrid_score'.
    """
    # --- Dense retrieval ---
    qvec = model.encode(query).tolist()
    dense_hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec,
        limit=min(k * 5, 100),  # fetch more candidates, rerank below
    )

    # map chunk text → dense score for fast lookup
    dense_scores = {}
    for hit in dense_hits:
        dense_scores[hit.payload["text"]] = hit.score

    # --- BM25 retrieval ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # normalize bm25 scores to 0-1
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    bm25_norm = [s / max_bm25 for s in bm25_scores]

    # normalize dense scores (already cosine 0-1, but clamp)
    max_dense = max(dense_scores.values()) if dense_scores else 1.0

    # --- Fusion ---
    scored = []
    for idx, chunk in enumerate(chunks):
        d_score = dense_scores.get(chunk["text"], 0.0) / max_dense
        b_score = bm25_norm[idx]
        hybrid = alpha * d_score + (1 - alpha) * b_score
        scored.append((hybrid, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, chunk in scored[:k]:
        result = dict(chunk)
        result["hybrid_score"] = round(score, 4)
        results.append(result)

    return results


if __name__ == "__main__":
    print("Loading chunks from Qdrant...")
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks. Building BM25 index...")
    bm25 = build_bm25_index(chunks)
    print("Ready.")

    query = input("Query: ")
    results = hybrid_retrieve(query, chunks, bm25, k=5)
    for r in results:
        print(f"\nscore={r['hybrid_score']} | paper={r['paper_id']} | section={r.get('section','?')}")
        print(r["text"][:300])