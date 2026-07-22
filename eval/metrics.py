"""
eval/metrics.py

Retrieval evaluation metrics for the RAG eval harness.

Metrics implemented:
    - Recall@k : fraction of relevant chunks found in top-k results
    - MRR      : Mean Reciprocal Rank across all queries
"""

def recall_at_k(retrieved_ids, relevant_ids, k):
    """
    Compute Recall@k for a single query.

    Recall@k = (relevant items in top-k) / (total relevant items)

    Args:
        retrieved_ids (list[str]): Ordered list of retrieved chunk IDs,
                                   best match first.
        relevant_ids  (set[str]):  Set of ground-truth relevant chunk IDs.
        k             (int):       Cutoff rank.

    Returns:
        float: Recall@k score between 0.0 and 1.0.

    Example:
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant  = {"c2", "c5"}
        recall_at_k(retrieved, relevant, k=3) → 1/2 = 0.5
        recall_at_k(retrieved, relevant, k=5) → 2/2 = 1.0
    """

    if not relevant_ids:
        return 0.0
    top_k = set(relevant_ids[:k])
    hits = top_k & relevant_ids
    return len(hits)/len(relevant_ids)

def reciprocal_rank(retrieved_ids, relevant_ids):
     """
    Compute Reciprocal Rank for a single query.

    RR = 1 / rank of first relevant result.
    If no relevant result is found, RR = 0.

    Args:
        retrieved_ids (list[str]): Ordered list of retrieved chunk IDs.
        relevant_ids  (set[str]):  Set of ground-truth relevant chunk IDs.

    Returns:
        float: Reciprocal rank between 0.0 and 1.0.

    Example:
        retrieved = ["c1", "c2", "c3"]
        relevant  = {"c2"}
        reciprocal_rank → 1/2 = 0.5  (c2 is at rank 2)
    """
     
     for rank, cid in enumerate(retrieved_ids, start=1):
         if cid in relevant_ids:
             return 1.0/ rank
     return 0.0
             
    
def mean_reciprocal_rank(all_retrieved, all_relevant):
    """
    Compute MRR across all queries.

    MRR = mean of reciprocal ranks over all queries.

    Args:
        all_retrieved (list[list[str]]): Retrieved IDs per query.
        all_relevant  (list[set[str]]): Relevant IDs per query.

    Returns:
        float: MRR score between 0.0 and 1.0.
    """
    rrs = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    return sum(rrs)/len(rrs)



