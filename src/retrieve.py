"""
Semantic retrieval over indexed arXiv paper chunks.

This script queries a Qdrant vector database to find the most semantically
similar text chunks to a given natural language query.

"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "arxiv_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(host="localhost", port=6333)

def retrieve(query, k = 5):
    qvec = model.encode(query).tolist()
    """
    Sends the query vector to Qdrant. Qdrant computes cosine similarity between qvec and every stored vector, returns the top k closest ones. This happens entirely inside Qdrant — we not pulling all vectors into Python and comparing them yourself.
    """
    results = client.search(
        collection_name = COLLECTION_NAME,
        query_vector = qvec,
        limit = k 
    )
    return results 

if __name__ == "__main__":
    query = input("query: ")
    results = retrieve(query)

    for r in results:
        print(f"\nscore={r.score:.4f} paper={r.payload['paper_id']} chunk={r.payload['chunk_idx']}")
        print(r.payload['text'][:300])
