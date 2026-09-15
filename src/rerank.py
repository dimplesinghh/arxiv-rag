"""
Cross-encoder reranker for RAG pipeline.

Retrieval (dense/hybrid) finds candidate chunks fast but uses
approximate similarity. A cross-encoder reranker scores each
(query, chunk) pair jointly — much more accurate but too slow
to run over the full corpus. So we retrieve top-20 candidates
first, then rerank to get the best top-10.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking
    - Fast inference, strong performance
    - Runs locally, no API cost
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# load once at module level — expensive to reload per query
reranker = CrossEncoder(MODEL_NAME)


def rerank(query, chunks, top_k=10):
    """
    Rerank retrieved chunks using a cross-encoder model.

    The cross-encoder reads query and chunk together in one
    forward pass, producing a relevance score that is more
    accurate than embedding cosine similarity.

    Args:
        query  (str):        The user query.
        chunks (list[dict]): Candidate chunks from retrieval.
                             Each dict must have a 'text' field.
        top_k  (int):        Number of top chunks to return
                             after reranking.

    Returns:
        list[dict]: Top-k chunks sorted by reranker score,
                    highest first. Each dict has original
                    chunk fields plus 'rerank_score'.
    """
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    scored = list(zip(scores, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, chunk in scored[:top_k]:
        result = dict(chunk)
        result["rerank_score"] = round(float(score), 4)
        results.append(result)

    return results