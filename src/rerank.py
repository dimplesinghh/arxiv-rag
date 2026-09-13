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
